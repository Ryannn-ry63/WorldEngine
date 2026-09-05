"""Shared schema-v2 cache utilities for DiffusionDrive selector GRPO V3."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


COMPONENT_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def _load_selector_module():
    algengine_root = Path(__file__).resolve().parents[2]
    path = (
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
    )
    spec = importlib.util.spec_from_file_location("diffusion_grpo_scene_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SELECTOR_MODULE = _load_selector_module()
SceneConditionedTrajectorySetSelector = (
    SELECTOR_MODULE.SceneConditionedTrajectorySetSelector
)
TrajectorySetReasoningResidualSelector = (
    SELECTOR_MODULE.TrajectorySetReasoningResidualSelector
)
ReferenceAnchoredPreferenceGraphSelector = (
    SELECTOR_MODULE.ReferenceAnchoredPreferenceGraphSelector
)
ProposalConditionedCounterfactualEvaluator = (
    SELECTOR_MODULE.ProposalConditionedCounterfactualEvaluator
)
CapacityMatchedUnaryResidualSelector = (
    SELECTOR_MODULE.CapacityMatchedUnaryResidualSelector
)
InteractionAwareResidualSelector = SELECTOR_MODULE.InteractionAwareResidualSelector
build_scene_selector = SELECTOR_MODULE.build_scene_selector


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path, expected_split=None):
    path = Path(path).expanduser().resolve()
    manifest_path = path.parent / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(path if not path.is_file() else manifest_path)
    manifest = json.loads(manifest_path.read_text())
    schema_version = manifest.get("schema_version")
    if manifest.get("status") != "PASS" or schema_version not in (2, 3, 4):
        raise RuntimeError(
            f"context cache manifest did not pass schema v2/v3/v4: {manifest_path}"
        )
    if manifest.get("cache_sha256") != sha256_file(path):
        raise RuntimeError(f"context cache SHA256 mismatch: {path}")
    if expected_split is not None and manifest.get("split") != expected_split:
        raise RuntimeError(f"context cache split mismatch: {path}")
    cache = torch.load(path, map_location="cpu")
    if cache.get("schema_version") != schema_version:
        raise RuntimeError(f"context cache payload schema drifted: {path}")
    if schema_version == 3 and manifest.get("source_kind") != "base_policy_rollout":
        raise RuntimeError(f"unknown schema-v3 cache source: {manifest_path}")
    if schema_version == 4 and manifest.get("source_kind") != "frozen_track_interaction":
        raise RuntimeError(f"unknown schema-v4 cache source: {manifest_path}")
    count = len(cache["tokens"])
    required = {
        "candidate_features": (count, 20, 256),
        "candidate_trajectories_8": (count, 20, 8, 3),
        "route_bev_features": (count, 20, 8, 256),
        "status_tokens": (count, 1, 256),
        "ego_queries": (count, 1, 256),
        "agents_queries": (count, 30, 256),
        "candidate_rewards": (count, 20),
        "candidate_reward_components": (count, 20, 6),
        "candidate_reward_valid_mask": (count, 20),
        "reference_logits": (count, 20),
    }
    if schema_version == 4:
        required.update(
            frozen_track_states=(count, 30, 8),
            frozen_track_classes=(count, 30),
            frozen_track_mask=(count, 30),
        )
    for key, shape in required.items():
        if tuple(cache[key].shape) != shape:
            raise RuntimeError(f"{path}: {key} shape {tuple(cache[key].shape)} != {shape}")
    if len(cache["scenes"]) != count or len(set(cache["tokens"])) != count:
        raise RuntimeError(f"{path}: token/scene provenance drifted")
    return cache, manifest


def model_from_cache(
    cache,
    ablation="full",
    architecture="scene_conditioned_v3",
    architecture_config=None,
):
    """Build either frozen V3 or the trajectory-set reasoner from one cache."""
    config = dict(cache["scene_selector_config"])
    shared_v3_keys = {
        "feature_dim",
        "model_dim",
        "route_bev_dim",
        "context_dim",
        "geometry_dim",
        "geometry_hidden_dim",
        "num_heads",
        "feedforward_dim",
        "num_route_steps",
        "num_set_layers",
    }
    if architecture in {"scene_conditioned_v3", "v3"}:
        allowed = {
            "full",
            "feature_only",
            "feature_geometry",
            "feature_geometry_route",
        }
        if ablation not in allowed:
            raise ValueError(f"unknown V3 ablation: {ablation}")
        config = {key: value for key, value in config.items() if key in shared_v3_keys}
        config.update(
            use_trajectory_geometry=ablation != "feature_only",
            use_route_bev=ablation in {"full", "feature_geometry_route"},
            use_scene_context=ablation == "full",
            use_set_attention=ablation == "full",
        )
    elif architecture in {
        "trajectory_set_reasoner",
        "reference_anchored_preference_graph",
        "capacity_matched_unary",
        "incumbent_verified_preference",
        "proposal_conditioned_regret_arbitration",
        "proposal_conditioned_counterfactual_evaluator",
    }:
        allowed = {"full", "temporal_only", "relational_only", "no_scene_context"}
        if ablation not in allowed:
            raise ValueError(f"unknown trajectory-set ablation: {ablation}")
        shared_keys = {
            "feature_dim",
            "model_dim",
            "route_bev_dim",
            "context_dim",
            "num_heads",
            "feedforward_dim",
            "num_route_steps",
        }
        config = {key: value for key, value in config.items() if key in shared_keys}
        config.update(
            architecture=architecture,
            num_temporal_layers=2,
            num_relation_layers=2,
            use_temporal_reasoning=ablation != "relational_only",
            use_relational_reasoning=ablation != "temporal_only",
            use_route_bev=True,
            use_scene_context=ablation != "no_scene_context",
        )
    elif architecture in {"interaction_generic", "interaction_relation"}:
        if ablation != "full":
            raise ValueError("interaction selector only supports the locked full input")
        interaction_keys = shared_v3_keys | {
            "interaction_dim",
            "interaction_hidden_dim",
            "num_agent_classes",
            "max_agents",
            "seconds_per_step",
            "num_interaction_heads",
        }
        config = {key: value for key, value in config.items() if key in interaction_keys}
        config.update(
            architecture=architecture,
            use_set_attention=True,
            use_trajectory_geometry=True,
            use_route_bev=True,
            use_scene_context=True,
        )
    else:
        raise ValueError(f"unknown selector architecture: {architecture}")
    if architecture_config:
        config.update(dict(architecture_config))
    return build_scene_selector(config), config


def model_from_config(config):
    """Rebuild the exact checkpoint architecture without consulting a cache."""

    return build_scene_selector(dict(config))


def batch_inputs(cache, indices, device):
    inputs = {
        "candidate_features": cache["candidate_features"][indices].to(
            device=device, dtype=torch.float32
        ),
        "candidate_trajectories": cache["candidate_trajectories_8"][indices].to(
            device=device, dtype=torch.float32
        ),
        "route_bev_features": cache["route_bev_features"][indices].to(
            device=device, dtype=torch.float32
        ),
        "status_token": cache["status_tokens"][indices].to(
            device=device, dtype=torch.float32
        ),
        "ego_query": cache["ego_queries"][indices].to(
            device=device, dtype=torch.float32
        ),
        "agents_query": cache["agents_queries"][indices].to(
            device=device, dtype=torch.float32
        ),
    }
    if "frozen_track_states" in cache:
        inputs.update(
            frozen_track_states=cache["frozen_track_states"][indices].to(
                device=device, dtype=torch.float32
            ),
            frozen_track_classes=cache["frozen_track_classes"][indices].to(
                device=device, dtype=torch.long
            ),
            frozen_track_mask=cache["frozen_track_mask"][indices].to(
                device=device, dtype=torch.bool
            ),
        )
    return inputs


def current_logits(model, cache, indices, device):
    reference = cache["reference_logits"][indices].to(device=device, dtype=torch.float32)
    inputs = batch_inputs(cache, indices, device)
    if not getattr(model, "requires_frozen_track_states", False):
        inputs.pop("frozen_track_states", None)
        inputs.pop("frozen_track_classes", None)
        inputs.pop("frozen_track_mask", None)
    if getattr(model, "requires_reference_logits", False):
        inputs["reference_logits"] = reference
    return reference + model(**inputs), reference


def clip_grad_norm_cpu_(parameters, max_norm):
    """Clip a CUDA model by a host-computed global L2 norm.

    PyTorch 2.0.1+cu118 can dispatch an illegal vector_norm kernel on H100.
    Copying detached gradients to the host preserves global-norm clipping while
    avoiding that incompatible reduction kernel. The selector is small enough
    that the synchronization cost is acceptable for cached V3 training.
    """
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0
    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError(f"max_norm must be finite and positive, got {max_norm}")

    total_squared = 0.0
    for gradient in gradients:
        host_gradient = gradient.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(host_gradient).all()):
            raise RuntimeError("non-finite V3 gradient")
        total_squared += float(host_gradient.square().sum())

    total_norm = math.sqrt(total_squared)
    if not math.isfinite(total_norm):
        raise RuntimeError("non-finite V3 global gradient norm")
    clip_coefficient = min(1.0, max_norm / (total_norm + 1e-6))
    if clip_coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(clip_coefficient)
    return total_norm


def normalized_advantage(rewards, valid):
    count = valid.sum(dim=-1).clamp_min(1).to(rewards.dtype)
    safe = torch.where(valid, rewards, torch.zeros_like(rewards))
    mean = safe.sum(dim=-1) / count
    centered = torch.where(valid, rewards - mean[:, None], torch.zeros_like(rewards))
    std = (centered.square().sum(dim=-1) / count).sqrt()
    active = (valid.sum(dim=-1) >= 2) & (std > 1e-6)
    advantage = centered / std.clamp_min(1e-6)[:, None]
    return torch.where(valid & active[:, None], advantage, torch.zeros_like(advantage)), active


def exact_group_loss(logits, reference_logits, rewards, valid, temperature, kl_weight):
    current_masked = (logits / temperature).masked_fill(~valid, -1e4)
    reference_masked = (reference_logits / temperature).masked_fill(~valid, -1e4)
    current_logp = F.log_softmax(current_masked, dim=-1)
    reference_logp = F.log_softmax(reference_masked, dim=-1)
    probability = current_logp.exp()
    advantage, active = normalized_advantage(rewards, valid)
    if not active.any():
        zero = logits.sum() * 0.0
        return zero, zero, zero
    policy = -(probability * advantage).sum(dim=-1)[active].mean()
    kl = (
        probability * (current_logp - reference_logp) * valid.to(logits.dtype)
    ).sum(dim=-1)[active].mean()
    return policy + kl_weight * kl, policy, kl


def scalar_preference_targets(rewards, valid):
    """Build scalar-only, tie-aware pair targets from official PDM rewards.

    This helper deliberately accepts no reward-component tensor.  Invalid
    candidates and diagonal pairs are masked, while equal advantages map to a
    neutral target of 0.5.
    """

    if rewards.ndim != 2 or valid.shape != rewards.shape:
        raise ValueError("reward and valid tensors must have shape [B, K]")
    valid = valid.to(dtype=torch.bool) & torch.isfinite(rewards)
    advantage, active = normalized_advantage(rewards, valid)
    quality_mass = F.softmax(advantage.masked_fill(~valid, -1e4), dim=-1)
    advantage_difference = advantage[:, :, None] - advantage[:, None, :]
    targets = torch.sigmoid(advantage_difference)
    weights = quality_mass[:, :, None] + quality_mass[:, None, :]
    num_candidates = rewards.shape[-1]
    diagonal = torch.eye(
        num_candidates, dtype=torch.bool, device=rewards.device
    )[None]
    pair_mask = valid[:, :, None] & valid[:, None, :] & ~diagonal
    weights = weights * pair_mask.to(weights.dtype)
    return targets, weights, pair_mask, advantage, active


def scalar_preference_loss(logits, rewards, valid, temperature=1.0):
    """Weighted pairwise BCE for the official scalar PDM ordering only."""

    if logits.shape != rewards.shape:
        raise ValueError("selector logits and rewards must have the same shape")
    if temperature <= 0.0:
        raise ValueError("preference temperature must be positive")
    targets, weights, pair_mask, advantage, active = scalar_preference_targets(
        rewards, valid
    )
    pair_logits = (
        logits[:, :, None] - logits[:, None, :]
    ) / float(temperature)
    pair_bce = F.binary_cross_entropy_with_logits(
        pair_logits, targets, reduction="none"
    )
    weight_sum = weights.sum(dim=(1, 2))
    active = active & (weight_sum > 0.0)
    if active.any():
        group_loss = (pair_bce * weights).sum(dim=(1, 2)) / weight_sum.clamp_min(
            1e-8
        )
        loss = group_loss[active].mean()
    else:
        loss = logits.sum() * 0.0
    diagnostics = {
        "targets": targets,
        "weights": weights,
        "pair_mask": pair_mask,
        "advantage": advantage,
        "active": active,
    }
    return loss, diagnostics


def exact_group_preference_loss(
    logits,
    reference_logits,
    rewards,
    valid,
    temperature,
    kl_weight,
    preference_weight,
):
    """Exact complete-action GRPO plus scalar tie-aware preferences."""

    if preference_weight < 0.0:
        raise ValueError("preference_weight must be non-negative")
    exact, policy, kl = exact_group_loss(
        logits,
        reference_logits,
        rewards,
        valid,
        temperature,
        kl_weight,
    )
    preference, diagnostics = scalar_preference_loss(
        logits, rewards, valid, temperature
    )
    return (
        exact + float(preference_weight) * preference,
        policy,
        preference,
        kl,
        diagnostics,
    )


def incumbent_verification_targets(
    rewards,
    reference_logits,
    valid,
    reward_margin=0.0,
    reward_temperature=0.05,
):
    """Scalar-only soft targets for candidate improvement over the incumbent."""

    if rewards.ndim != 2 or reference_logits.shape != rewards.shape:
        raise ValueError("reward and reference tensors must have shape [B, K]")
    if valid.shape != rewards.shape:
        raise ValueError("valid mask and rewards are not aligned")
    if reward_margin < 0.0 or reward_temperature <= 0.0:
        raise ValueError("invalid incumbent-verification target parameters")
    valid = valid.to(dtype=torch.bool) & torch.isfinite(rewards)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("every verification group needs a valid candidate")
    incumbent = reference_logits.detach().masked_fill(~valid, -torch.inf).argmax(dim=-1)
    incumbent_reward = rewards.gather(1, incumbent[:, None])
    reward_difference = rewards - incumbent_reward
    targets = torch.sigmoid(
        (reward_difference - float(reward_margin)) / float(reward_temperature)
    )
    candidate_index = torch.arange(rewards.shape[-1], device=rewards.device)[None]
    pair_mask = valid & candidate_index.ne(incumbent[:, None])
    active = pair_mask.any(dim=-1)
    return targets, pair_mask, reward_difference, incumbent, active


def incumbent_verification_loss(
    verification_logits,
    reference_logits,
    proposal_logits,
    rewards,
    valid,
    reward_margin=0.0,
    reward_temperature=0.05,
    proposal_temperature=1.0,
):
    """Calibrate one-sided scalar improvement evidence for proposal overrides."""

    if not (
        verification_logits.shape
        == reference_logits.shape
        == proposal_logits.shape
        == rewards.shape
        == valid.shape
    ):
        raise ValueError("incumbent-verification tensors are not aligned")
    if proposal_temperature <= 0.0:
        raise ValueError("proposal temperature must be positive")
    targets, pair_mask, reward_difference, incumbent, active = (
        incumbent_verification_targets(
            rewards,
            reference_logits,
            valid,
            reward_margin,
            reward_temperature,
        )
    )
    valid = valid.to(dtype=torch.bool) & torch.isfinite(rewards)
    advantage, _ = normalized_advantage(rewards, valid)
    quality_mass = F.softmax(advantage.masked_fill(~valid, -1e4), dim=-1)
    incumbent_quality = quality_mass.gather(1, incumbent[:, None])
    proposal_mass = F.softmax(
        (proposal_logits.detach() / float(proposal_temperature)).masked_fill(
            ~valid, -1e4
        ),
        dim=-1,
    )
    weights = (
        quality_mass + incumbent_quality + proposal_mass
    ) * pair_mask.to(rewards.dtype)
    element_loss = F.binary_cross_entropy_with_logits(
        verification_logits, targets, reduction="none"
    )
    weight_sum = weights.sum(dim=-1)
    active = active & weight_sum.gt(0.0)
    if active.any():
        group_loss = (element_loss * weights).sum(dim=-1) / weight_sum.clamp_min(1e-8)
        loss = group_loss[active].mean()
    else:
        loss = verification_logits.sum() * 0.0
    diagnostics = {
        "targets": targets,
        "weights": weights,
        "pair_mask": pair_mask,
        "reward_difference": reward_difference,
        "incumbent_index": incumbent,
        "active": active,
    }
    return loss, diagnostics


def proposal_conditioned_arbitration_loss(
    proposal_evidence,
    incumbent_index,
    proposal_index,
    rewards,
    valid,
    loss_kind="regret",
    risk_aggregation="global",
    source_labels=None,
):
    """Optimize the realized proposal/incumbent decision under scalar PDM.

    Regret mode uses absolute raw-PDM gain as the BCE weight. At the population
    optimum, a zero evidence threshold is equivalent to accepting the proposal
    exactly when its conditional expected scalar gain is positive. Sign mode is
    the unweighted causal control. Ties and unchanged proposals have no
    deployment decision and contribute no gradient.

    CPV risk aggregation changes only how active decision strata contribute to
    the batch risk. Source labels are sampler/loss metadata and are never model
    inputs.
    """

    if rewards.ndim != 2 or valid.shape != rewards.shape:
        raise ValueError("reward and valid tensors must have shape [B, K]")
    batch_size = rewards.shape[0]
    expected = (batch_size,)
    if any(
        value.shape != expected
        for value in (proposal_evidence, incumbent_index, proposal_index)
    ):
        raise ValueError("proposal arbitration tensors are not aligned")
    if loss_kind not in ("sign", "regret"):
        raise ValueError(f"unsupported proposal arbitration loss: {loss_kind}")
    allowed_aggregations = ("global", "source", "sign", "source_sign")
    if risk_aggregation not in allowed_aggregations:
        raise ValueError(
            f"unsupported proposal arbitration risk: {risk_aggregation}"
        )

    valid = valid.to(dtype=torch.bool) & torch.isfinite(rewards)
    incumbent_valid = valid.gather(1, incumbent_index[:, None]).squeeze(1)
    proposal_valid = valid.gather(1, proposal_index[:, None]).squeeze(1)
    if not bool((incumbent_valid & proposal_valid).all()):
        raise ValueError("proposal arbitration selected an invalid candidate")
    incumbent_reward = rewards.gather(1, incumbent_index[:, None]).squeeze(1)
    proposal_reward = rewards.gather(1, proposal_index[:, None]).squeeze(1)
    gain = proposal_reward - incumbent_reward
    changed = proposal_index.ne(incumbent_index)
    active = changed & gain.abs().gt(1e-8)
    targets = gain.gt(0.0).to(dtype=proposal_evidence.dtype)
    weights = gain.abs() if loss_kind == "regret" else torch.ones_like(gain)
    element_loss = F.binary_cross_entropy_with_logits(
        proposal_evidence, targets, reduction="none"
    )
    weighted_loss = element_loss * weights

    allowed_sources = ("common", "hard_real_rare", "hard_synthetic")
    source_masks = None
    if risk_aggregation in ("source", "source_sign"):
        if source_labels is None or len(source_labels) != batch_size:
            raise ValueError(
                f"{risk_aggregation} risk requires one source label per sample"
            )
        unknown = set(source_labels).difference(allowed_sources)
        if unknown:
            raise ValueError(
                f"unknown proposal arbitration sources: {sorted(unknown)}"
            )
        source_masks = {
            label: torch.tensor(
                [value == label for value in source_labels],
                dtype=torch.bool,
                device=active.device,
            )
            for label in allowed_sources
        }

    if risk_aggregation == "global":
        group_masks = {"global": active}
    elif risk_aggregation == "source":
        group_masks = {
            f"source:{label}": active & source_masks[label]
            for label in allowed_sources
        }
    elif risk_aggregation == "sign":
        group_masks = {
            "sign:beneficial": active & gain.gt(0.0),
            "sign:degrading": active & gain.lt(0.0),
        }
    else:
        group_masks = {
            f"source_sign:{label}/{sign}": (
                active
                & source_masks[label]
                & (gain.gt(0.0) if sign == "beneficial" else gain.lt(0.0))
            )
            for label in allowed_sources
            for sign in ("beneficial", "degrading")
        }

    terms = []
    group_losses = {}
    group_active_counts = {}
    group_regret_mass = {}
    missing_groups = []
    for name, mask in group_masks.items():
        count = int(mask.sum().detach().cpu())
        group_active_counts[name] = count
        group_regret_mass[name] = float(weights[mask].sum().detach().cpu())
        if count:
            term = weighted_loss[mask].mean()
            terms.append(term)
            group_losses[name] = term.detach()
        else:
            missing_groups.append(name)
    loss = (
        torch.stack(terms).mean()
        if terms
        else proposal_evidence.sum() * 0.0
    )
    diagnostics = {
        "targets": targets,
        "weights": weights,
        "gain": gain,
        "changed": changed,
        "active": active,
        "incumbent_index": incumbent_index,
        "proposal_index": proposal_index,
        "beneficial": active & gain.gt(0.0),
        "degrading": active & gain.lt(0.0),
        "risk_aggregation": risk_aggregation,
        "group_losses": group_losses,
        "group_active_counts": group_active_counts,
        "group_regret_mass": group_regret_mass,
        "missing_groups": missing_groups,
    }
    return loss, diagnostics



def proposal_conditioned_counterfactual_loss(
    opportunity,
    risk,
    evidence,
    incumbent_index,
    proposal_index,
    rewards,
    valid,
    loss_kind="opportunity_risk",
    source_labels=None,
    source_risk="original_mixture",
):
    """Learn expected deployed-pair gain without reward-component inputs.

    Opportunity-risk mode estimates the positive and negative parts of scalar
    PDM gain separately. Signed-gain is a same-capacity regression control and
    regret-BCE is the V1 objective under the new independent representation.
    """
    if rewards.ndim != 2 or valid.shape != rewards.shape:
        raise ValueError("reward and valid tensors must have shape [B, K]")
    batch_size = rewards.shape[0]
    expected = (batch_size,)
    if any(
        value.shape != expected
        for value in (
            opportunity, risk, evidence, incumbent_index, proposal_index
        )
    ):
        raise ValueError("counterfactual evaluator tensors are not aligned")
    if loss_kind not in {"regret_bce", "signed_gain", "opportunity_risk"}:
        raise ValueError(f"unsupported counterfactual loss: {loss_kind}")
    if source_risk not in {"original_mixture", "equal_strata"}:
        raise ValueError(f"unsupported source risk: {source_risk}")

    valid = valid.to(dtype=torch.bool) & torch.isfinite(rewards)
    incumbent_valid = valid.gather(1, incumbent_index[:, None]).squeeze(1)
    proposal_valid = valid.gather(1, proposal_index[:, None]).squeeze(1)
    if not bool((incumbent_valid & proposal_valid).all()):
        raise ValueError("counterfactual evaluator selected an invalid candidate")
    incumbent_reward = rewards.gather(1, incumbent_index[:, None]).squeeze(1)
    proposal_reward = rewards.gather(1, proposal_index[:, None]).squeeze(1)
    gain = proposal_reward - incumbent_reward
    changed = proposal_index.ne(incumbent_index)
    positive_target = gain.clamp_min(0.0)
    negative_target = (-gain).clamp_min(0.0)

    if loss_kind == "opportunity_risk":
        if bool((opportunity < 0.0).any()) or bool((risk < 0.0).any()):
            raise ValueError("opportunity and risk must be non-negative")
        element_loss = 0.5 * (
            (opportunity - positive_target).square()
            + (risk - negative_target).square()
        )
        active = changed
    elif loss_kind == "signed_gain":
        element_loss = (evidence - gain).square()
        active = changed
    else:
        targets = gain.gt(0.0).to(dtype=evidence.dtype)
        element_loss = F.binary_cross_entropy_with_logits(
            evidence, targets, reduction="none"
        ) * gain.abs()
        active = changed & gain.abs().gt(1e-8)

    source_losses = {}
    if source_risk == "equal_strata":
        if source_labels is None or len(source_labels) != batch_size:
            raise ValueError("equal-strata risk requires one source label per sample")
        allowed = ("common", "hard_real_rare", "hard_synthetic")
        unknown = set(source_labels).difference(allowed)
        if unknown:
            raise ValueError(f"unknown counterfactual source labels: {sorted(unknown)}")
        terms = []
        for label in allowed:
            source_mask = torch.tensor(
                [value == label for value in source_labels],
                device=active.device,
                dtype=torch.bool,
            ) & active
            if source_mask.any():
                source_loss = element_loss[source_mask].mean()
                terms.append(source_loss)
                source_losses[label] = source_loss.detach()
        loss = (
            torch.stack(terms).mean()
            if terms
            else evidence.sum() * 0.0
        )
    elif active.any():
        loss = element_loss[active].mean()
    else:
        loss = evidence.sum() * 0.0

    diagnostics = {
        "gain": gain,
        "changed": changed,
        "active": active,
        "beneficial": active & gain.gt(0.0),
        "degrading": active & gain.lt(0.0),
        "ties": changed & gain.abs().le(1e-8),
        "positive_target": positive_target,
        "negative_target": negative_target,
        "element_loss": element_loss,
        "source_losses": source_losses,
        "incumbent_index": incumbent_index,
        "proposal_index": proposal_index,
    }
    return loss, diagnostics


def gate_conditioned_exact_group_loss(
    logits,
    reference_logits,
    rewards,
    components,
    valid,
    temperature,
    kl_weight,
):
    """Optimize official PDM plus quality ranking conditioned on safety.

    NAVSIM PDM uses no-collision and drivable-area compliance as
    multiplicative gates.  A single group normalization over the final score
    consequently devotes most of its dynamic range to safe-vs-unsafe
    separation.  The conditional term below renormalizes the official
    weighted quality metrics *only among gated-safe candidates*.  It changes
    neither the official reward nor the probability mass assigned to the safe
    set directly; it only supplies a relative ranking inside that set.
    """
    if components.ndim != 3 or components.shape[-1] != 6:
        raise ValueError("candidate reward components must have shape [B, K, 6]")
    if components.shape[:2] != rewards.shape or valid.shape != rewards.shape:
        raise ValueError("gate-conditioned reward tensor shapes disagree")

    finite_components = torch.isfinite(components).all(dim=-1)
    valid = valid & torch.isfinite(rewards) & finite_components
    current_masked = (logits / temperature).masked_fill(~valid, -1e4)
    reference_masked = (reference_logits / temperature).masked_fill(~valid, -1e4)
    current_logp = F.log_softmax(current_masked, dim=-1)
    reference_logp = F.log_softmax(reference_masked, dim=-1)
    probability = current_logp.exp()

    official_advantage, official_active = normalized_advantage(rewards, valid)
    zero = logits.sum() * 0.0
    if official_active.any():
        official_policy = -(
            probability * official_advantage
        ).sum(dim=-1)[official_active].mean()
        kl = (
            probability
            * (current_logp - reference_logp)
            * valid.to(logits.dtype)
        ).sum(dim=-1)[official_active].mean()
    else:
        official_policy = zero
        kl = zero

    # Component order is NC, DAC, EP, TTC, comfort, DDC.  Only NC and DAC
    # are multiplicative gates in the NAVSIM-v1 scorer.  DDC has zero weight
    # in the frozen [5, 5, 2, 0] weighted-metric contract.
    safe = valid & (components[..., 0] >= 1.0 - 1e-6) & (
        components[..., 1] >= 1.0 - 1e-6
    )
    quality = (
        5.0 * components[..., 2]
        + 5.0 * components[..., 3]
        + 2.0 * components[..., 4]
    ) / 12.0
    quality_advantage, quality_active = normalized_advantage(quality, safe)
    if quality_active.any():
        safe_logits = (logits / temperature).masked_fill(~safe, -1e4)
        safe_probability = F.softmax(safe_logits, dim=-1)
        quality_policy = -(
            safe_probability * quality_advantage
        ).sum(dim=-1)[quality_active].mean()
    else:
        quality_policy = zero

    loss = official_policy + quality_policy + kl_weight * kl
    diagnostics = {
        "official_active": official_active,
        "quality_active": quality_active,
        "safe": safe,
    }
    return loss, official_policy, quality_policy, kl, diagnostics


def selector_metrics(logits, reference_logits, rewards, components, valid, temperature):
    current_masked = (logits / temperature).masked_fill(~valid, -1e4)
    reference_masked = (reference_logits / temperature).masked_fill(~valid, -1e4)
    current_logp = F.log_softmax(current_masked, dim=-1)
    reference_logp = F.log_softmax(reference_masked, dim=-1)
    probability = current_logp.exp()
    reference_probability = reference_logp.exp()
    current_index = current_masked.argmax(dim=-1)
    reference_index = reference_masked.argmax(dim=-1)
    oracle_index = rewards.masked_fill(~valid, -torch.inf).argmax(dim=-1)

    def gather(values, index):
        suffix = (1,) * (values.ndim - 2)
        gather_index = index.reshape(-1, 1, *suffix).expand(-1, 1, *values.shape[2:])
        return values.gather(1, gather_index).squeeze(1)

    safe_rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
    current_reward = gather(rewards, current_index)
    reference_reward = gather(rewards, reference_index)
    oracle_reward = gather(rewards, oracle_index)
    return {
        "current_reward": current_reward,
        "reference_reward": reference_reward,
        "top1_reward_gain": current_reward - reference_reward,
        "current_expected_reward": (probability * safe_rewards).sum(dim=-1),
        "reference_expected_reward": (reference_probability * safe_rewards).sum(dim=-1),
        "oracle_match": current_index.eq(oracle_index).float(),
        "reference_oracle_match": reference_index.eq(oracle_index).float(),
        "oracle_regret": oracle_reward - current_reward,
        "selection_disagreement": current_index.ne(reference_index).float(),
        "kl": (
            probability * (current_logp - reference_logp) * valid.to(logits.dtype)
        ).sum(dim=-1),
        "component_delta": gather(components, current_index)
        - gather(components, reference_index),
    }


def evaluate(model, cache, device, temperature, batch_size):
    output = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            indices = slice(start, min(start + batch_size, len(cache["tokens"])))
            logits, reference = current_logits(model, cache, indices, device)
            values = selector_metrics(
                logits,
                reference,
                cache["candidate_rewards"][indices].to(device),
                cache["candidate_reward_components"][indices].to(device),
                cache["candidate_reward_valid_mask"][indices].to(device),
                temperature,
            )
            for key, value in values.items():
                output.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(value) for key, value in output.items()}


def summarize(values):
    summary = {
        key: float(value.mean()) for key, value in values.items() if key != "component_delta"
    }
    for index, name in enumerate(COMPONENT_NAMES):
        summary[f"delta_{name}"] = float(values["component_delta"][:, index].mean())
    return summary
