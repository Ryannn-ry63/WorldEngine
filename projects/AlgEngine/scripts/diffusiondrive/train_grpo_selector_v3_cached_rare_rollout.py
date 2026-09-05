#!/usr/bin/env python3
"""Train a fresh zero-residual V3 selector on 50/50 common/hard rollout data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import cpv_e2_common_contract as e2_common
import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare
from build_grpo_selector_v3_rare_rollout_data import (
    BASELINE_SHA256,
    EXPECTED_REAL_CACHE_SEEDS,
    METHOD as DATA_METHOD,
    parse_seed_path,
)


METHOD = "scene_conditioned_exact_group_grpo_v3_rare_rollout_v1"
GATE_CONDITIONED_METHOD = (
    "scene_conditioned_gate_conditioned_group_grpo_v3_rare_rollout_v1"
)
OBJECTIVE_OFFICIAL = "official_pdm"
OBJECTIVE_GATE_CONDITIONED = "gate_conditioned_pdm"
OBJECTIVE_SCALAR_PREFERENCE = "official_pdm_plus_scalar_preference"
OBJECTIVE_INCUMBENT_VERIFICATION = "official_pdm_incumbent_verification"
OBJECTIVE_PROPOSAL_REGRET = "official_pdm_proposal_regret_arbitration"
OBJECTIVE_COUNTERFACTUAL_EVALUATION = "official_pdm_counterfactual_evaluation"
COUNTERFACTUAL_LOSSES = ("regret_bce", "signed_gain", "opportunity_risk")
SOURCE_RISKS = ("original_mixture", "equal_strata")
ARBITER_RISKS = ("global", "source", "sign", "source_sign")
SUPPORTED_ARCHITECTURES = (
    "scene_conditioned_v3",
    "trajectory_set_reasoner",
    "reference_anchored_preference_graph",
    "capacity_matched_unary",
    "incumbent_verified_preference",
    "proposal_conditioned_regret_arbitration",
    "proposal_conditioned_counterfactual_evaluator",
)
FORMAL_EPOCHS = 16
FORMAL_EXAMPLES_PER_CACHE_EPOCH = 6339
FORMAL_BATCH_SIZE = 64
FORMAL_TOTAL_EXAMPLES = 304272
FORMAL_OPTIMIZER_STEPS = 4800
E2_DATA_METHOD = "cpv_e2_existing_navtrain_common_coverage_v1"
E2_ARMS = ("D1_diverse_matched", "D2_diverse_20k")


def implementation_provenance() -> dict[str, str]:
    algengine_root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        Path(standard.__file__).resolve(),
        Path(e2_common.__file__).resolve(),
        Path(rare.__file__).resolve(),
        Path(__file__).with_name("build_grpo_selector_v3_rare_rollout_data.py"),
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py",
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py",
    )
    return {str(path): common.sha256_file(path) for path in paths}


def load_hard_pool(path: Path, expected_sha: str) -> list[dict]:
    if common.sha256_file(path) != expected_sha:
        raise RuntimeError("hard-pool SHA256 drifted")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["hard_id"] for row in rows}) != len(rows):
        raise RuntimeError("hard pool is empty or repeats identities")
    for index, row in enumerate(rows):
        if row.get("hard_kind") not in ("real_rare", "synthetic_rollout"):
            raise RuntimeError(f"hard row {index} has an invalid source kind")
        if row.get("split") not in ("train", "development", "certification"):
            raise RuntimeError(f"hard row {index} has an invalid log split")
    return rows


def validate_synthetic_cache(cache: dict, manifest: dict) -> None:
    count = int(manifest["filtered_synthetic_records"])
    if cache.get("schema_version") != 3 or cache.get("source_kind") != (
        "base_policy_rollout"
    ):
        raise RuntimeError("synthetic cache schema drifted")
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
    for key, shape in required.items():
        if tuple(cache[key].shape) != shape:
            raise RuntimeError(f"synthetic cache {key} shape drifted")
    if len(cache["tokens"]) != count or len(set(cache["tokens"])) != count:
        raise RuntimeError("synthetic cache identities drifted")
    if count and not bool(cache["candidate_reward_valid_mask"].all()):
        raise RuntimeError("synthetic cache contains invalid candidates")


def load_contract(args):
    manifest_path = args.data_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "PASS"
        or manifest.get("method") != DATA_METHOD
        or manifest.get("baseline_checkpoint_sha256") != BASELINE_SHA256
        or manifest.get("behavior_policy_is_trained_v3") is not False
        or manifest.get("mixture_contract", {}).get("overall_common_fraction") != 0.5
        or manifest.get("mixture_contract", {}).get("overall_hard_fraction") != 0.5
    ):
        raise RuntimeError("rare-rollout data manifest did not pass")
    hard_pool_path = args.hard_pool.expanduser().resolve()
    if str(hard_pool_path) != str(Path(manifest["hard_pool"]).resolve()):
        raise RuntimeError("hard-pool path disagrees with data manifest")
    hard_rows = load_hard_pool(hard_pool_path, manifest["hard_pool_sha256"])
    if len(hard_rows) != int(manifest["hard_pool_rows"]):
        raise RuntimeError("hard-pool row count drifted")

    cache_paths = dict(args.real_cache)
    if set(cache_paths) != set(EXPECTED_REAL_CACHE_SEEDS):
        raise RuntimeError("exact real cache seeds 0,1,2 are required")
    pairs = [
        common.load_cache(cache_paths[seed], "train")
        for seed in EXPECTED_REAL_CACHE_SEEDS
    ]
    real_caches, real_manifests = zip(*pairs)
    standard.assert_cache_group(
        real_caches, real_manifests, "train", EXPECTED_REAL_CACHE_SEEDS
    )
    for seed, path in cache_paths.items():
        expected = manifest["real_caches"][str(seed)]
        if common.sha256_file(path) != expected["cache_sha256"]:
            raise RuntimeError(f"real cache seed {seed} drifted")

    synthetic_path = args.synthetic_cache.expanduser().resolve()
    if (
        str(synthetic_path) != str(Path(manifest["synthetic_cache"]).resolve())
        or common.sha256_file(synthetic_path) != manifest["synthetic_cache_sha256"]
    ):
        raise RuntimeError("synthetic cache provenance drifted")
    synthetic = torch.load(synthetic_path, map_location="cpu")
    validate_synthetic_cache(synthetic, manifest)
    if synthetic["scene_selector_config"] != real_caches[0]["scene_selector_config"]:
        raise RuntimeError("synthetic/real selector architecture drifted")

    token_maps = [
        {str(token): index for index, token in enumerate(cache["tokens"])}
        for cache in real_caches
    ]
    synthetic_count = len(synthetic["tokens"])
    for position, row in enumerate(hard_rows):
        for token_map in token_maps:
            if row["paired_common_token"] not in token_map:
                raise RuntimeError(f"hard row {position} has no paired common cache row")
            if row["hard_kind"] == "real_rare" and row["rare_token"] not in token_map:
                raise RuntimeError(f"hard row {position} has no real-rare cache row")
        if row["hard_kind"] == "synthetic_rollout":
            index = int(row.get("synthetic_index", -1))
            if not 0 <= index < synthetic_count:
                raise RuntimeError(f"hard row {position} has invalid synthetic index")
            if synthetic["origin_rare_tokens"][index] != row["rare_token"]:
                raise RuntimeError(f"hard row {position} synthetic origin drifted")
    return (
        manifest_path,
        manifest,
        hard_pool_path,
        hard_rows,
        real_caches,
        real_manifests,
        token_maps,
        synthetic_path,
        synthetic,
    )


def gather_source(cache: dict, indices: list[int], device: torch.device):
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    inputs = common.batch_inputs(cache, tensor_indices, device)
    reference = cache["reference_logits"][tensor_indices].to(
        device=device, dtype=torch.float32
    )
    rewards = cache["candidate_rewards"][tensor_indices].to(
        device=device, dtype=torch.float32
    )
    components = cache["candidate_reward_components"][tensor_indices].to(
        device=device, dtype=torch.float32
    )
    valid = cache["candidate_reward_valid_mask"][tensor_indices].to(device=device)
    return inputs, reference, rewards, components, valid


def mixed_batch(
    rows: list[tuple[str, int]],
    hard_rows: list[dict],
    real_cache: dict,
    real_token_map: dict[str, int],
    common_cache: dict | None,
    common_token_map: dict[str, int] | None,
    common_tokens: list[str] | None,
    synthetic: dict,
    device: torch.device,
):
    sources = {"real": [], "common": [], "synthetic": []}
    source_kinds = Counter()
    for class_name, position in rows:
        if class_name == "common":
            if common_tokens is None:
                row = hard_rows[position]
                sources["real"].append(real_token_map[row["paired_common_token"]])
            else:
                sources["common"].append(
                    common_token_map[common_tokens[position]]
                )
            source_kinds["common"] += 1
            continue
        row = hard_rows[position]
        if row["hard_kind"] == "real_rare":
            sources["real"].append(real_token_map[row["rare_token"]])
            source_kinds["hard_real_rare"] += 1
        else:
            sources["synthetic"].append(int(row["synthetic_index"]))
            source_kinds["hard_synthetic"] += 1

    batches = []
    if sources["real"]:
        batches.append(gather_source(real_cache, sources["real"], device))
    if sources["common"]:
        if common_cache is None:
            raise RuntimeError("E2 common rows have no sidecar cache")
        batches.append(gather_source(common_cache, sources["common"], device))
    if sources["synthetic"]:
        batches.append(gather_source(synthetic, sources["synthetic"], device))
    if not batches:
        raise RuntimeError("empty mixed batch")
    input_keys = tuple(batches[0][0])
    inputs = {
        key: torch.cat([batch[0][key] for batch in batches], dim=0)
        for key in input_keys
    }
    reference = torch.cat([batch[1] for batch in batches], dim=0)
    rewards = torch.cat([batch[2] for batch in batches], dim=0)
    components = torch.cat([batch[3] for batch in batches], dim=0)
    valid = torch.cat([batch[4] for batch in batches], dim=0)
    return inputs, reference, rewards, components, valid, source_kinds



def ordered_batch_source_labels(
    rows: list[tuple[str, int]],
    hard_rows: list[dict],
    separate_common: bool = False,
) -> list[str]:
    """Match source labels to mixed_batch's real/common/synthetic order."""
    real_labels = []
    common_labels = []
    synthetic_labels = []
    for class_name, position in rows:
        if class_name == "common":
            (common_labels if separate_common else real_labels).append("common")
            continue
        row = hard_rows[position]
        if row["hard_kind"] == "real_rare":
            real_labels.append("hard_real_rare")
        else:
            synthetic_labels.append("hard_synthetic")
    return real_labels + common_labels + synthetic_labels


DECISION_SOURCES = ("common", "hard_real_rare", "hard_synthetic")
DECISION_SIGNS = ("beneficial", "degrading")


def balanced_group_counts(
    total: int, group_names: tuple[str, ...], global_block: int
) -> dict[str, int]:
    """Distribute a fixed budget exactly across groups over successive blocks."""

    if total <= 0 or not group_names or len(set(group_names)) != len(group_names):
        raise ValueError("invalid balanced-group count request")
    base, remainder = divmod(total, len(group_names))
    counts = {name: base for name in group_names}
    if remainder:
        start = (global_block * remainder) % len(group_names)
        for offset in range(remainder):
            counts[group_names[(start + offset) % len(group_names)]] += 1
    if sum(counts.values()) != total:
        raise RuntimeError("balanced-group counts drifted")
    return counts


def balanced_batch_ranges(total: int, maximum_batch_size: int):
    """Avoid a tiny final CPV batch while preserving optimizer-step count."""

    if total <= 0 or maximum_batch_size <= 0:
        raise ValueError("invalid balanced-batch request")
    steps = (total + maximum_batch_size - 1) // maximum_batch_size
    base, remainder = divmod(total, steps)
    sizes = [base + (index < remainder) for index in range(steps)]
    if max(sizes) > maximum_batch_size or sum(sizes) != total:
        raise RuntimeError("balanced-batch construction drifted")
    start = 0
    ranges = []
    for size in sizes:
        ranges.append((start, start + size))
        start += size
    return ranges


def decision_group_names(risk_aggregation: str) -> tuple[str, ...]:
    if risk_aggregation == "source":
        return DECISION_SOURCES
    if risk_aggregation == "sign":
        return DECISION_SIGNS
    if risk_aggregation == "source_sign":
        return tuple(
            f"{source}/{sign}"
            for source in DECISION_SOURCES
            for sign in DECISION_SIGNS
        )
    if risk_aggregation == "global":
        return ("global",)
    raise ValueError(f"unsupported proposal arbitration risk: {risk_aggregation}")


def decision_source_specs(
    hard_rows: list[dict],
    common_count: int | None = None,
) -> dict[str, list[tuple[str, int]]]:
    common_rows = len(hard_rows) if common_count is None else common_count
    return {
        "common": [("common", index) for index in range(common_rows)],
        "hard_real_rare": [
            ("hard", index)
            for index, row in enumerate(hard_rows)
            if row["hard_kind"] == "real_rare"
        ],
        "hard_synthetic": [
            ("hard", index)
            for index, row in enumerate(hard_rows)
            if row["hard_kind"] == "synthetic_rollout"
        ],
    }


@torch.no_grad()
def build_proposal_decision_pools(
    model,
    hard_rows,
    real_caches,
    token_maps,
    common_caches,
    common_token_maps,
    common_tokens,
    synthetic,
    device,
    risk_aggregation,
    batch_size,
):
    """Mine active source/sign cells using the immutable proposal only."""

    if risk_aggregation not in ("source", "sign", "source_sign"):
        raise ValueError("decision-pool mining is only valid for CPV E1 arms")
    source_specs = decision_source_specs(
        hard_rows, None if common_tokens is None else len(common_tokens)
    )
    if any(not rows for rows in source_specs.values()):
        raise RuntimeError("CPV decision mining requires every source")
    group_names = decision_group_names(risk_aggregation)
    pools = {}
    audit = {}
    was_training = model.training
    model.eval()
    try:
        for cache_index, (real_cache, token_map, common_cache, common_token_map) in enumerate(
            zip(real_caches, token_maps, common_caches, common_token_maps)
        ):
            source_sign_rows = {
                f"{source}/{sign}": []
                for source in DECISION_SOURCES
                for sign in DECISION_SIGNS
            }
            cache_audit = {}
            for source in DECISION_SOURCES:
                rows = source_specs[source]
                source_audit = {
                    "rows": len(rows),
                    "active": 0,
                    "beneficial": 0,
                    "degrading": 0,
                    "inactive": 0,
                    "absolute_regret": 0.0,
                }
                for start in range(0, len(rows), batch_size):
                    batch_rows = rows[start : start + batch_size]
                    (
                        inputs,
                        reference,
                        rewards,
                        _,
                        valid,
                        _,
                    ) = mixed_batch(
                        batch_rows,
                        hard_rows,
                        real_cache,
                        token_map,
                        common_cache,
                        common_token_map,
                        common_tokens,
                        synthetic,
                        device,
                    )
                    inputs["reference_logits"] = reference
                    inputs["candidate_mask"] = valid
                    _, diagnostics = model(
                        **inputs, return_override_diagnostics=True
                    )
                    _, pair = common.proposal_conditioned_arbitration_loss(
                        diagnostics["proposal_evidence"],
                        diagnostics["incumbent_index"],
                        diagnostics["proposal_index"],
                        rewards,
                        valid,
                        "regret",
                    )
                    gains = pair["gain"].detach().cpu().tolist()
                    active = pair["active"].detach().cpu().tolist()
                    for row_spec, is_active, gain in zip(
                        batch_rows, active, gains
                    ):
                        if not is_active:
                            source_audit["inactive"] += 1
                            continue
                        sign = "beneficial" if gain > 0.0 else "degrading"
                        source_sign_rows[f"{source}/{sign}"].append(row_spec)
                        source_audit["active"] += 1
                        source_audit[sign] += 1
                        source_audit["absolute_regret"] += abs(float(gain))
                cache_audit[source] = source_audit

            if risk_aggregation == "source":
                grouped = {
                    source: (
                        source_sign_rows[f"{source}/beneficial"]
                        + source_sign_rows[f"{source}/degrading"]
                    )
                    for source in DECISION_SOURCES
                }
            elif risk_aggregation == "sign":
                grouped = {
                    sign: [
                        row
                        for source in DECISION_SOURCES
                        for row in source_sign_rows[f"{source}/{sign}"]
                    ]
                    for sign in DECISION_SIGNS
                }
            else:
                grouped = source_sign_rows
            empty = [name for name in group_names if not grouped[name]]
            if empty:
                raise RuntimeError(
                    f"CPV decision cells are empty for cache {cache_index}: {empty}"
                )
            pools[cache_index] = grouped
            cache_audit["pool_groups"] = {
                name: len(grouped[name]) for name in group_names
            }
            audit[str(cache_index)] = cache_audit
    finally:
        model.train(was_training)
    return pools, audit


def expected_method(args) -> str:
    if args.objective == OBJECTIVE_GATE_CONDITIONED:
        return GATE_CONDITIONED_METHOD
    architecture = getattr(args, "architecture", "scene_conditioned_v3")
    ablation = getattr(args, "ablation", "full")
    if architecture == "scene_conditioned_v3" and args.objective == OBJECTIVE_OFFICIAL:
        return METHOD
    if args.objective == OBJECTIVE_INCUMBENT_VERIFICATION:
        margin = int(round(float(args.verifier_reward_margin) * 1000.0))
        return f"ivps_v1_incumbent_verified_preference_m{margin:03d}"
    if args.objective == OBJECTIVE_PROPOSAL_REGRET:
        common_arm = getattr(args, "common_arm", None)
        if common_arm is not None:
            return f"cpv_e2_{common_arm.lower()}_source_sign_pair_regret"
        context = "context" if args.use_decision_context else "pair"
        if args.arbiter_risk == "global":
            return (
                f"pcra_v1_{context}_{args.arbiter_loss}_proposal_conditioned_regret"
            )
        return f"cpv_v1_{args.arbiter_risk}_{context}_regret"
    if args.objective == OBJECTIVE_COUNTERFACTUAL_EVALUATION:
        encoder = "independent" if args.train_evaluator_encoder else "frozen"
        source = "equal_source" if args.source_risk == "equal_strata" else "rarelog"
        return f"pcra_v2_{encoder}_{args.counterfactual_loss}_{source}"
    objective = (
        "exact_group_grpo"
        if args.objective == OBJECTIVE_OFFICIAL
        else "exact_group_grpo_scalar_preference"
    )
    return f"rapg_v1_{architecture}_{ablation}_{objective}"


def paper_role(args) -> str:
    if args.objective == OBJECTIVE_GATE_CONDITIONED:
        return "diagnostic_pareto_ablation"
    if args.objective == OBJECTIVE_PROPOSAL_REGRET:
        if args.arbiter_risk == "source_sign":
            return "counterfactual_proposal_verification_mainline"
        if args.arbiter_risk in ("source", "sign"):
            return "counterfactual_proposal_verification_causal_control"
        return "proposal_conditioned_regret_arbitration_mainline"
    if args.objective == OBJECTIVE_COUNTERFACTUAL_EVALUATION:
        if (
            args.counterfactual_loss == "opportunity_risk"
            and args.train_evaluator_encoder
            and args.source_risk == "original_mixture"
        ):
            return "decoupled_counterfactual_opportunity_risk_mainline"
        return "counterfactual_evaluator_causal_control"
    if args.objective == OBJECTIVE_INCUMBENT_VERIFICATION:
        return "candidate_wise_verification_causal_control"
    return "scalar_only_mainline_or_causal_control"


def validate_formal_args(args) -> None:
    expected_data_split = (
        "train"
        if (
            args.objective == OBJECTIVE_PROPOSAL_REGRET
            and args.arbiter_risk != "global"
        )
        else "all"
    )
    expected = {
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "kl_weight": 1e-3,
        "epochs": FORMAL_EPOCHS,
        "examples_per_cache_epoch": FORMAL_EXAMPLES_PER_CACHE_EPOCH,
        "batch_size": FORMAL_BATCH_SIZE,
        "method_name": expected_method(args),
        "data_split": expected_data_split,
    }
    drift = {
        key: {"actual": getattr(args, key), "expected": value}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if drift:
        raise RuntimeError("formal rare-rollout training contract drifted: " + repr(drift))


def save_selector_state(path, model, model_config, args, epoch, manifest_path, pool_path):
    payload = {
        "schema_version": (
            6 if args.objective == OBJECTIVE_COUNTERFACTUAL_EVALUATION else 3
        ),
        "method": args.method_name,
        "source_kind": "full_navtrain_real_rare_plus_filtered_online_rollout_v1",
        "implementation_files": implementation_provenance(),
        "scene_selector_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scene_selector_config": model_config,
        "ablation": args.ablation,
        "selector_architecture": args.architecture,
        "objective": args.objective,
        "objective_paper_name": (
            "conditional_quality_reweighting_cqr"
            if args.objective == OBJECTIVE_GATE_CONDITIONED
            else args.objective
        ),
        "paper_role": paper_role(args),
        "preference_weight": args.preference_weight,
        "proposal_checkpoint": (
            None if args.proposal_checkpoint is None else str(args.proposal_checkpoint)
        ),
        "proposal_checkpoint_sha256": args.proposal_checkpoint_sha256,
        "verifier_reward_margin": args.verifier_reward_margin,
        "verifier_reward_temperature": args.verifier_reward_temperature,
        "override_threshold": args.override_threshold,
        "arbiter_loss": args.arbiter_loss,
        "arbiter_risk": args.arbiter_risk,
        "use_decision_context": args.use_decision_context,
        "arbiter_target": args.arbiter_target,
        "counterfactual_loss": args.counterfactual_loss,
        "train_evaluator_encoder": args.train_evaluator_encoder,
        "source_risk": args.source_risk,
        "evaluator_initialization_max_abs_delta": (
            args.evaluator_initialization_max_abs_delta
        ),
        "reward_components_consumed": False,
        "training_data_split": args.data_split,
        "training_epochs": args.epochs,
        "training_examples_per_cache_epoch": args.examples_per_cache_epoch,
        "training_batch_size": args.batch_size,
        "formal_contract": bool(args.formal_contract),
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "sampling_mode": (
            f"cpv_decision_{args.arbiter_risk}"
            if (
                args.objective == OBJECTIVE_PROPOSAL_REGRET
                and args.arbiter_risk != "global"
            )
            else (
                "three_source_equal_risk"
                if args.source_risk == "equal_strata"
                else "common_hard_balanced"
            )
        ),
        "train_seed": args.seed,
        "epoch": epoch,
        "data_manifest": str(manifest_path),
        "common_arm": args.common_arm,
        "common_unique_tokens": args.common_unique_tokens,
        "common_data_manifest": (
            None
            if args.common_data_manifest is None
            else str(args.common_data_manifest.expanduser().resolve())
        ),
        "common_data_manifest_sha256": (
            None
            if args.common_data_manifest is None
            else common.sha256_file(args.common_data_manifest.expanduser().resolve())
        ),
        "common_caches": (
            None
            if args.common_cache is None
            else {
                str(seed): {
                    "path": str(path.expanduser().resolve()),
                    "sha256": common.sha256_file(path.expanduser().resolve()),
                }
                for seed, path in args.common_cache
            }
        ),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(pool_path),
        "hard_pool_sha256": common.sha256_file(pool_path),
    }
    torch.save(payload, path)


def load_and_freeze_proposal(model, checkpoint_path):
    """Load the immutable proposal and isolate arbitration/evaluation updates."""

    path = checkpoint_path.expanduser().resolve()
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") not in (3, 4, 5):
        raise RuntimeError("proposal checkpoint schema drifted")
    if payload.get("selector_architecture") != "trajectory_set_reasoner":
        raise RuntimeError("proposal must be a trajectory-set reasoner")
    if payload.get("objective") != OBJECTIVE_SCALAR_PREFERENCE:
        raise RuntimeError("proposal must use the scalar preference objective")
    if float(payload.get("preference_weight", 0.0)) != 1.0:
        raise RuntimeError("proposal preference weight must be exactly 1.0")
    state = payload.get("scene_selector_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("proposal state is empty")
    incompatible = model.load_state_dict(state, strict=False)

    if getattr(model, "is_counterfactual_evaluator", False):
        missing_prefixes = (
            "counterfactual_encoder.",
            "counterfactual_value_head.",
        )
        expected_missing = {
            name for name in model.state_dict()
            if name.startswith(missing_prefixes)
        }
        if (
            set(incompatible.missing_keys) != expected_missing
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "proposal/counterfactual architecture mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        initialization_delta = model.initialize_evaluator_from_proposal()
        if initialization_delta != 0.0:
            raise RuntimeError("counterfactual evaluator initialization drifted")
        trainable_prefixes = ["counterfactual_value_head."]
        if model.train_evaluator_encoder:
            trainable_prefixes.append("counterfactual_encoder.")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(tuple(trainable_prefixes)))
        model.evaluator_initialization_max_abs_delta = initialization_delta
    else:
        head_prefix = (
            "regret_arbiter_head."
            if getattr(model, "requires_candidate_mask", False)
            else "override_verifier_head."
        )
        expected_missing = {
            name for name in model.state_dict() if name.startswith(head_prefix)
        }
        if (
            set(incompatible.missing_keys) != expected_missing
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "proposal/arbitrator architecture mismatch: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(head_prefix))
        trainable_prefixes = [head_prefix]

    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(
        not name.startswith(tuple(trainable_prefixes)) for name in trainable
    ):
        raise RuntimeError("training did not isolate arbitration optimization")
    return path, common.sha256_file(path), payload["method"], trainable


def train(args) -> dict:
    args.objective = getattr(args, "objective", OBJECTIVE_OFFICIAL)
    args.architecture = getattr(args, "architecture", "scene_conditioned_v3")
    args.ablation = getattr(args, "ablation", "full")
    args.preference_weight = float(getattr(args, "preference_weight", 0.0))
    args.data_split = getattr(args, "data_split", "all")
    args.proposal_checkpoint = getattr(args, "proposal_checkpoint", None)
    args.proposal_checkpoint_sha256 = None
    args.proposal_method = None
    args.verifier_reward_margin = float(
        getattr(args, "verifier_reward_margin", 0.0)
    )
    args.verifier_reward_temperature = float(
        getattr(args, "verifier_reward_temperature", 0.05)
    )
    args.override_threshold = float(getattr(args, "override_threshold", 0.0))
    args.arbiter_loss = getattr(args, "arbiter_loss", None)
    args.arbiter_risk = getattr(args, "arbiter_risk", "global")
    args.use_decision_context = bool(
        getattr(args, "use_decision_context", False)
    )
    args.counterfactual_loss = getattr(args, "counterfactual_loss", None)
    args.train_evaluator_encoder = bool(
        getattr(args, "train_evaluator_encoder", False)
    )
    args.source_risk = getattr(args, "source_risk", "original_mixture")
    args.common_cache = getattr(args, "common_cache", None)
    args.common_data_manifest = getattr(args, "common_data_manifest", None)
    args.common_arm = getattr(args, "common_arm", None)
    args.evaluator_initialization_max_abs_delta = None
    args.arbiter_target = (
        "actual_top1_proposal_vs_reference_incumbent"
        if args.objective in (
            OBJECTIVE_PROPOSAL_REGRET,
            OBJECTIVE_COUNTERFACTUAL_EVALUATION,
        )
        else None
    )
    if args.objective not in (
        OBJECTIVE_OFFICIAL,
        OBJECTIVE_GATE_CONDITIONED,
        OBJECTIVE_SCALAR_PREFERENCE,
        OBJECTIVE_INCUMBENT_VERIFICATION,
        OBJECTIVE_PROPOSAL_REGRET,
        OBJECTIVE_COUNTERFACTUAL_EVALUATION,
    ):
        raise ValueError(f"unsupported objective: {args.objective}")
    if args.architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {args.architecture}")
    if args.data_split not in ("train", "all"):
        raise ValueError(f"unsupported training data split: {args.data_split}")
    if args.objective == OBJECTIVE_GATE_CONDITIONED and args.architecture != (
        "scene_conditioned_v3"
    ):
        raise ValueError("CQR remains a V3 diagnostic and is not a RAPG objective")
    if args.objective == OBJECTIVE_INCUMBENT_VERIFICATION:
        if args.architecture != "incumbent_verified_preference":
            raise ValueError("incumbent verification requires the IVPS architecture")
        if args.proposal_checkpoint is None:
            raise ValueError("incumbent verification requires --proposal-checkpoint")
        if args.verifier_reward_margin < 0.0:
            raise ValueError("verifier reward margin must be non-negative")
        if args.verifier_reward_temperature <= 0.0:
            raise ValueError("verifier reward temperature must be positive")
        if args.override_threshold != 0.0:
            raise ValueError("the frozen IVPS search requires threshold zero")
    elif args.objective == OBJECTIVE_PROPOSAL_REGRET:
        if args.architecture != "proposal_conditioned_regret_arbitration":
            raise ValueError("proposal regret requires the PCRA architecture")
        if args.proposal_checkpoint is None:
            raise ValueError("proposal regret requires --proposal-checkpoint")
        if args.arbiter_loss not in ("sign", "regret"):
            raise ValueError("PCRA arbiter loss must be sign or regret")
        if args.arbiter_risk not in ARBITER_RISKS:
            raise ValueError("PCRA requires a fixed risk-aggregation contract")
        if args.arbiter_risk != "global" and args.arbiter_loss != "regret":
            raise ValueError("CPV stratification requires regret-weighted BCE")
        if args.arbiter_risk != "global" and args.use_decision_context:
            raise ValueError("CPV E1 fixes the context-free PCRA V1 representation")
        if args.override_threshold != 0.0:
            raise ValueError("the frozen PCRA search requires threshold zero")
        e2_supplied = (
            args.common_cache is not None,
            args.common_data_manifest is not None,
            args.common_arm is not None,
        )
        if any(e2_supplied) and not all(e2_supplied):
            raise ValueError("E2 common inputs must be supplied together")
        if args.common_arm is not None and (
            args.arbiter_risk != "source_sign" or args.common_arm not in e2_common.ARMS
        ):
            raise ValueError("E2 is frozen to D1/D2 source-sign CPV")
    elif args.objective == OBJECTIVE_COUNTERFACTUAL_EVALUATION:
        if args.architecture != "proposal_conditioned_counterfactual_evaluator":
            raise ValueError("counterfactual evaluation requires the PCRA V2 architecture")
        if args.proposal_checkpoint is None:
            raise ValueError("counterfactual evaluation requires --proposal-checkpoint")
        if args.counterfactual_loss not in COUNTERFACTUAL_LOSSES:
            raise ValueError("PCRA V2 requires a fixed counterfactual loss")
        if args.source_risk not in SOURCE_RISKS:
            raise ValueError("PCRA V2 requires a fixed source-risk contract")
        if args.override_threshold != 0.0:
            raise ValueError("the frozen PCRA V2 search requires threshold zero")
        if (
            args.arbiter_loss is not None
            or args.arbiter_risk != "global"
            or args.use_decision_context
        ):
            raise ValueError("PCRA V1 controls are invalid for PCRA V2")
    elif args.proposal_checkpoint is not None:
        raise ValueError("proposal checkpoints are only valid for arbitration")
    elif (
        args.arbiter_loss is not None
        or args.arbiter_risk != "global"
        or args.use_decision_context
        or args.counterfactual_loss is not None
        or args.train_evaluator_encoder
        or args.source_risk != "original_mixture"
    ):
        raise ValueError("arbitration controls require PCRA V1 or V2")
    if args.objective == OBJECTIVE_SCALAR_PREFERENCE:
        if args.preference_weight <= 0.0:
            raise ValueError("scalar preference training requires a positive weight")
    elif args.preference_weight != 0.0:
        raise ValueError("preference weight is only valid for the scalar preference objective")
    if args.method_name is None:
        args.method_name = expected_method(args)
    if args.formal_contract:
        validate_formal_args(args)
    if args.temperature <= 0 or args.learning_rate <= 0 or args.kl_weight < 0:
        raise ValueError("invalid training hyperparameters")
    if min(args.epochs, args.examples_per_cache_epoch, args.batch_size) <= 0:
        raise ValueError("training budgets must be positive")

    (
        manifest_path,
        manifest,
        hard_pool_path,
        hard_rows,
        real_caches,
        real_manifests,
        token_maps,
        synthetic_path,
        synthetic,
    ) = load_contract(args)
    e2_contract = e2_common.load(args, real_caches)
    if e2_contract[0] is None:
        e2_manifest_path = None
        e2_manifest = None
        common_tokens = None
        common_caches = (None,) * len(real_caches)
        common_cache_manifests = ()
        common_token_maps = (None,) * len(real_caches)
    else:
        (
            e2_manifest_path,
            e2_manifest,
            common_tokens,
            common_caches,
            common_cache_manifests,
            common_token_maps,
        ) = e2_contract
    args.common_unique_tokens = (
        None if common_tokens is None else len(common_tokens)
    )
    if args.data_split != "all":
        hard_rows = [row for row in hard_rows if row["split"] == args.data_split]
        if not hard_rows:
            raise RuntimeError(f"hard pool has no rows for split {args.data_split}")
    decision_stratified = (
        args.objective == OBJECTIVE_PROPOSAL_REGRET
        and args.arbiter_risk != "global"
    )
    if args.smoke_limit_hard_pool is not None:
        if args.formal_contract:
            raise RuntimeError("smoke hard-pool limiting is forbidden in formal mode")
        if args.smoke_limit_hard_pool < 1:
            raise ValueError("--smoke-limit-hard-pool must be positive")
        real_rows = [row for row in hard_rows if row["hard_kind"] == "real_rare"]
        synthetic_rows = [
            row for row in hard_rows if row["hard_kind"] == "synthetic_rollout"
        ]
        if decision_stratified:
            hard_rows = (
                real_rows[: args.smoke_limit_hard_pool]
                + synthetic_rows[: args.smoke_limit_hard_pool]
            )
        else:
            hard_rows = real_rows[: args.smoke_limit_hard_pool]
        if (
            not decision_stratified
            and synthetic_rows
            and args.smoke_limit_hard_pool > 1
        ):
            selected_synthetic = synthetic_rows[0]
            if args.objective == OBJECTIVE_GATE_CONDITIONED:
                for candidate_row in synthetic_rows:
                    index = int(candidate_row["synthetic_index"])
                    components = synthetic["candidate_reward_components"][index]
                    valid = synthetic["candidate_reward_valid_mask"][index]
                    safe = valid & (components[:, 0] >= 1.0 - 1e-6) & (
                        components[:, 1] >= 1.0 - 1e-6
                    )
                    quality = (
                        5.0 * components[:, 2]
                        + 5.0 * components[:, 3]
                        + 2.0 * components[:, 4]
                    ) / 12.0
                    if int(safe.sum()) >= 2 and float(quality[safe].std()) > 1e-6:
                        selected_synthetic = candidate_row
                        break
                else:
                    raise RuntimeError(
                        "smoke pool has no synthetic gate-conditioned signal"
                    )
            hard_rows[-1] = selected_synthetic
    minimum_hard_draws = args.epochs * (args.examples_per_cache_epoch // 2)
    if not decision_stratified and len(hard_rows) > minimum_hard_draws:
        raise RuntimeError(
            "fixed compute cannot cover every hard row before reuse: "
            f"hard_rows={len(hard_rows)} hard_draws_per_cache={minimum_hard_draws}"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    architecture_config = {}
    if args.architecture in {
        "trajectory_set_reasoner",
        "reference_anchored_preference_graph",
        "capacity_matched_unary",
        "incumbent_verified_preference",
        "proposal_conditioned_regret_arbitration",
        "proposal_conditioned_counterfactual_evaluator",
    }:
        architecture_config.update(
            num_temporal_layers=2,
            num_relation_layers=2,
        )
    if args.architecture == "reference_anchored_preference_graph":
        architecture_config["pairwise_hidden_dim"] = 128
    elif args.architecture == "capacity_matched_unary":
        architecture_config["capacity_hidden_dim"] = 277
    elif args.architecture == "incumbent_verified_preference":
        architecture_config.update(
            verifier_hidden_dim=128,
            override_threshold=args.override_threshold,
        )
    elif args.architecture == "proposal_conditioned_regret_arbitration":
        architecture_config.update(
            arbiter_hidden_dim=128,
            use_decision_context=args.use_decision_context,
            override_threshold=args.override_threshold,
        )
    elif args.architecture == "proposal_conditioned_counterfactual_evaluator":
        architecture_config.update(
            counterfactual_hidden_dim=128,
            counterfactual_loss=args.counterfactual_loss,
            train_evaluator_encoder=args.train_evaluator_encoder,
            override_threshold=args.override_threshold,
        )
    model, model_config = common.model_from_cache(
        real_caches[0],
        args.ablation,
        args.architecture,
        architecture_config,
    )
    trainable_parameter_names = None
    arbitration_objective = args.objective in (
        OBJECTIVE_INCUMBENT_VERIFICATION,
        OBJECTIVE_PROPOSAL_REGRET,
        OBJECTIVE_COUNTERFACTUAL_EVALUATION,
    )
    if arbitration_objective:
        (
            args.proposal_checkpoint,
            args.proposal_checkpoint_sha256,
            args.proposal_method,
            trainable_parameter_names,
        ) = load_and_freeze_proposal(model, args.proposal_checkpoint)
        args.evaluator_initialization_max_abs_delta = getattr(
            model, "evaluator_initialization_max_abs_delta", None
        )
    frozen_parameter_state = (
        {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        if arbitration_objective
        else {}
    )
    trainable_parameter_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    model = model.to(device)
    with torch.no_grad():
        initial_inputs, reference, _, _, initial_valid, _ = mixed_batch(
            [("hard", 0)],
            hard_rows,
            real_caches[0],
            token_maps[0],
            common_caches[0],
            common_token_maps[0],
            common_tokens,
            synthetic,
            device,
        )
        if getattr(model, "requires_reference_logits", False):
            initial_inputs["reference_logits"] = reference
        if getattr(model, "requires_candidate_mask", False):
            initial_inputs["candidate_mask"] = initial_valid
        if arbitration_objective:
            initial_delta, initial_diagnostics = model(
                **initial_inputs, return_override_diagnostics=True
            )
            initial_proposal_max_abs_delta = float(
                initial_diagnostics["proposal_delta"].abs().max().cpu()
            )
        else:
            initial_delta = model(**initial_inputs)
            initial_proposal_max_abs_delta = initial_max_abs_delta = float(
                initial_delta.abs().max().cpu()
            )
        initial_max_abs_delta = float(initial_delta.abs().max().cpu())
    if initial_max_abs_delta != 0.0:
        raise RuntimeError("fresh V3 selector is not exact-zero at initialization")

    decision_pools = None
    decision_pool_audit = {}
    if decision_stratified:
        decision_pools, decision_pool_audit = build_proposal_decision_pools(
            model,
            hard_rows,
            real_caches,
            token_maps,
            common_caches,
            common_token_maps,
            common_tokens,
            synthetic,
            device,
            args.arbiter_risk,
            args.batch_size,
        )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("selector has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=1e-4, foreach=False
    )
    schedule = torch.Generator().manual_seed(args.seed + 910000)
    samplers = {
        (cache_index, class_name): rare.CyclingPairSampler(
            len(hard_rows),
            args.seed * 100000
            + cache_index * 1000
            + (17 if class_name == "hard" else 37),
        )
        for cache_index in range(3)
        for class_name in ("hard", "common")
    }
    real_hard_positions = [
        index for index, row in enumerate(hard_rows)
        if row["hard_kind"] == "real_rare"
    ]
    synthetic_hard_positions = [
        index for index, row in enumerate(hard_rows)
        if row["hard_kind"] == "synthetic_rollout"
    ]
    if not real_hard_positions or not synthetic_hard_positions:
        raise RuntimeError("PCRA training requires real-rare and synthetic sources")
    equal_source_sizes = {
        "common": len(hard_rows),
        "hard_real_rare": len(real_hard_positions),
        "hard_synthetic": len(synthetic_hard_positions),
    }
    equal_samplers = {
        (cache_index, source): rare.CyclingPairSampler(
            size,
            args.seed * 100000 + cache_index * 1000 + offset,
        )
        for cache_index in range(3)
        for source, size, offset in (
            ("common", equal_source_sizes["common"], 41),
            ("hard_real_rare", equal_source_sizes["hard_real_rare"], 43),
            ("hard_synthetic", equal_source_sizes["hard_synthetic"], 47),
        )
    }
    decision_groups = (
        decision_group_names(args.arbiter_risk) if decision_stratified else ()
    )
    decision_samplers = {
        (cache_index, group): rare.CyclingPairSampler(
            len(decision_pools[cache_index][group]),
            args.seed * 100000
            + cache_index * 1000
            + 101
            + group_index * 13,
        )
        for cache_index in range(3)
        for group_index, group in enumerate(decision_groups)
    }
    checkpoint_epochs = tuple(
        sorted({int(value) for value in args.checkpoint_epochs.split(",")})
    )
    if not checkpoint_epochs or checkpoint_epochs[-1] != args.epochs:
        raise ValueError("checkpoint epochs must end at --epochs")

    total_examples = 0
    optimizer_steps = 0
    class_examples = Counter()
    source_examples = Counter()
    risk_group_examples = Counter()
    accumulated = Counter()
    checkpoint_reports = []
    global_block = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        cache_order = torch.randperm(3, generator=schedule).tolist()
        for cache_index in cache_order:
            if decision_stratified:
                group_counts = balanced_group_counts(
                    args.examples_per_cache_epoch,
                    decision_groups,
                    global_block,
                )
                drawn = {}
                for group in decision_groups:
                    pool = decision_pools[cache_index][group]
                    positions = decision_samplers[(cache_index, group)].take(
                        group_counts[group]
                    )
                    drawn[group] = [pool[position] for position in positions]
                    risk_group_examples[group] += group_counts[group]
                selected = []
                for position in range(max(group_counts.values())):
                    order = torch.randperm(
                        len(decision_groups), generator=schedule
                    ).tolist()
                    for group_index in order:
                        group = decision_groups[group_index]
                        if position < len(drawn[group]):
                            selected.append(drawn[group][position])
                common_count = sum(
                    class_name == "common" for class_name, _ in selected
                )
                hard_count = len(selected) - common_count
            elif args.source_risk == "equal_strata":
                if args.examples_per_cache_epoch % 3:
                    raise RuntimeError("equal-source budget must be divisible by three")
                source_count = args.examples_per_cache_epoch // 3
                common_positions = equal_samplers[(cache_index, "common")].take(
                    source_count
                )
                rare_positions = [
                    real_hard_positions[index]
                    for index in equal_samplers[
                        (cache_index, "hard_real_rare")
                    ].take(source_count)
                ]
                synthetic_positions = [
                    synthetic_hard_positions[index]
                    for index in equal_samplers[
                        (cache_index, "hard_synthetic")
                    ].take(source_count)
                ]
                triplet_order = torch.randperm(
                    source_count, generator=schedule
                ).tolist()
                selected = []
                for index in triplet_order:
                    selected.extend(
                        (
                            ("common", common_positions[index]),
                            ("hard", rare_positions[index]),
                            ("hard", synthetic_positions[index]),
                        )
                    )
                common_count = source_count
                hard_count = 2 * source_count
            else:
                hard_count, common_count = rare.block_class_counts(
                    args.examples_per_cache_epoch, global_block
                )
                selected = [
                    ("hard", position)
                    for position in samplers[(cache_index, "hard")].take(hard_count)
                ]
                selected.extend(
                    ("common", position)
                    for position in samplers[(cache_index, "common")].take(common_count)
                )
                permutation = torch.randperm(
                    len(selected), generator=schedule
                ).tolist()
                selected = [selected[index] for index in permutation]
            batch_ranges = (
                balanced_batch_ranges(len(selected), args.batch_size)
                if decision_stratified
                else [
                    (start, min(start + args.batch_size, len(selected)))
                    for start in range(0, len(selected), args.batch_size)
                ]
            )
            for start, end in batch_ranges:
                batch_rows = selected[start:end]
                inputs, reference, rewards, components, valid, kinds = mixed_batch(
                    batch_rows,
                    hard_rows,
                    real_caches[cache_index],
                    token_maps[cache_index],
                    common_caches[cache_index],
                    common_token_maps[cache_index],
                    common_tokens,
                    synthetic,
                    device,
                )
                source_labels = ordered_batch_source_labels(
                    batch_rows, hard_rows, common_tokens is not None
                )
                if getattr(model, "requires_reference_logits", False):
                    inputs["reference_logits"] = reference
                if getattr(model, "requires_candidate_mask", False):
                    inputs["candidate_mask"] = valid
                if arbitration_objective:
                    residual, override_diagnostics = model(
                        **inputs, return_override_diagnostics=True
                    )
                    logits = reference + residual
                else:
                    logits = reference + model(**inputs)
                optimizer.zero_grad(set_to_none=True)
                if args.objective == OBJECTIVE_GATE_CONDITIONED:
                    loss, policy, quality_policy, kl, diagnostics = (
                        common.gate_conditioned_exact_group_loss(
                            logits,
                            reference,
                            rewards,
                            components,
                            valid,
                            args.temperature,
                            args.kl_weight,
                        )
                    )
                    accumulated["quality_policy"] += float(
                        quality_policy.detach().cpu()
                    )
                    accumulated["official_active_groups"] += int(
                        diagnostics["official_active"].sum().detach().cpu()
                    )
                    accumulated["quality_active_groups"] += int(
                        diagnostics["quality_active"].sum().detach().cpu()
                    )
                    accumulated["safe_candidates"] += int(
                        diagnostics["safe"].sum().detach().cpu()
                    )
                elif args.objective == OBJECTIVE_SCALAR_PREFERENCE:
                    loss, policy, preference, kl, diagnostics = (
                        common.exact_group_preference_loss(
                            logits,
                            reference,
                            rewards,
                            valid,
                            args.temperature,
                            args.kl_weight,
                            args.preference_weight,
                        )
                    )
                    accumulated["preference"] += float(
                        preference.detach().cpu()
                    )
                    accumulated["preference_active_groups"] += int(
                        diagnostics["active"].sum().detach().cpu()
                    )
                    accumulated["preference_pairs"] += int(
                        diagnostics["pair_mask"].sum().detach().cpu()
                    )
                elif args.objective == OBJECTIVE_INCUMBENT_VERIFICATION:
                    proposal_logits = (
                        reference + override_diagnostics["proposal_delta"]
                    )
                    loss, diagnostics = common.incumbent_verification_loss(
                        override_diagnostics["verification_logits"],
                        reference,
                        proposal_logits,
                        rewards,
                        valid,
                        args.verifier_reward_margin,
                        args.verifier_reward_temperature,
                        args.temperature,
                    )
                    policy = loss.detach() * 0.0
                    kl = loss.detach() * 0.0
                    accumulated["verification"] += float(loss.detach().cpu())
                    accumulated["verification_active_groups"] += int(
                        diagnostics["active"].sum().detach().cpu()
                    )
                    accumulated["verification_pairs"] += int(
                        diagnostics["pair_mask"].sum().detach().cpu()
                    )
                elif args.objective == OBJECTIVE_PROPOSAL_REGRET:
                    loss, diagnostics = common.proposal_conditioned_arbitration_loss(
                        override_diagnostics["proposal_evidence"],
                        override_diagnostics["incumbent_index"],
                        override_diagnostics["proposal_index"],
                        rewards,
                        valid,
                        args.arbiter_loss,
                        risk_aggregation=args.arbiter_risk,
                        source_labels=source_labels,
                    )
                    policy = loss.detach() * 0.0
                    kl = loss.detach() * 0.0
                    accumulated["arbitration"] += float(loss.detach().cpu())
                    accumulated["arbitration_active_groups"] += int(
                        diagnostics["active"].sum().detach().cpu()
                    )
                    accumulated["arbitration_beneficial"] += int(
                        diagnostics["beneficial"].sum().detach().cpu()
                    )
                    accumulated["arbitration_degrading"] += int(
                        diagnostics["degrading"].sum().detach().cpu()
                    )
                    accumulated["arbitration_absolute_regret"] += float(
                        diagnostics["gain"][diagnostics["active"]]
                        .abs().sum().detach().cpu()
                    )
                    if decision_stratified and diagnostics["missing_groups"]:
                        raise RuntimeError(
                            "CPV batch missed required decision groups: "
                            + repr(diagnostics["missing_groups"])
                        )
                    for group, group_loss in diagnostics["group_losses"].items():
                        report_group = (
                            group.split(":", 1)[1]
                            if decision_stratified
                            else group
                        )
                        accumulated[f"arbitration_group_loss:{report_group}"] += float(
                            group_loss.cpu()
                        )
                        accumulated[f"arbitration_group_batches:{report_group}"] += 1
                        accumulated[f"arbitration_group_active:{report_group}"] += (
                            diagnostics["group_active_counts"][group]
                        )
                        accumulated[f"arbitration_group_regret:{report_group}"] += (
                            diagnostics["group_regret_mass"][group]
                        )
                elif args.objective == OBJECTIVE_COUNTERFACTUAL_EVALUATION:
                    loss, diagnostics = (
                        common.proposal_conditioned_counterfactual_loss(
                            override_diagnostics["opportunity"],
                            override_diagnostics["risk"],
                            override_diagnostics["proposal_evidence"],
                            override_diagnostics["incumbent_index"],
                            override_diagnostics["proposal_index"],
                            rewards,
                            valid,
                            args.counterfactual_loss,
                            source_labels=source_labels,
                            source_risk=args.source_risk,
                        )
                    )
                    policy = loss.detach() * 0.0
                    kl = loss.detach() * 0.0
                    accumulated["counterfactual"] += float(loss.detach().cpu())
                    accumulated["counterfactual_active"] += int(
                        diagnostics["active"].sum().detach().cpu()
                    )
                    accumulated["counterfactual_beneficial"] += int(
                        diagnostics["beneficial"].sum().detach().cpu()
                    )
                    accumulated["counterfactual_degrading"] += int(
                        diagnostics["degrading"].sum().detach().cpu()
                    )
                    accumulated["counterfactual_ties"] += int(
                        diagnostics["ties"].sum().detach().cpu()
                    )
                    accumulated["counterfactual_absolute_regret"] += float(
                        diagnostics["gain"][diagnostics["active"]]
                        .abs().sum().detach().cpu()
                    )
                    for source, source_loss in diagnostics["source_losses"].items():
                        accumulated[f"counterfactual_source_loss_{source}"] += float(
                            source_loss.cpu()
                        )
                        accumulated[f"counterfactual_source_batches_{source}"] += 1
                else:
                    loss, policy, kl = common.exact_group_loss(
                        logits,
                        reference,
                        rewards,
                        valid,
                        args.temperature,
                        args.kl_weight,
                    )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite rare-rollout loss at epoch {epoch}")
                loss.backward()
                common.clip_grad_norm_cpu_(trainable_parameters, 10.0)
                optimizer.step()
                optimizer_steps += 1
                accumulated["loss"] += float(loss.detach().cpu())
                accumulated["policy"] += float(policy.detach().cpu())
                accumulated["kl"] += float(kl.detach().cpu())
                source_examples.update(kinds)
            class_examples["hard"] += hard_count
            class_examples["common"] += common_count
            total_examples += len(selected)
            global_block += 1

        if epoch in checkpoint_epochs:
            state_path = output_dir / f"epoch_{epoch}_scene_selector.pt"
            save_selector_state(
                state_path,
                model,
                model_config,
                args,
                epoch,
                manifest_path,
                hard_pool_path,
            )
            checkpoint_reports.append(
                {
                    "epoch": epoch,
                    "scene_selector_state": str(state_path),
                    "scene_selector_state_sha256": common.sha256_file(state_path),
                }
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "examples": total_examples,
                        "optimizer_steps": optimizer_steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if decision_stratified:
        group_totals = {risk_group_examples[group] for group in decision_groups}
        if len(group_totals) != 1:
            raise RuntimeError("global CPV decision-group balance drifted")
    elif args.source_risk == "original_mixture":
        if class_examples["hard"] != class_examples["common"]:
            raise RuntimeError("global 50/50 common/hard balance drifted")
    else:
        source_totals = {
            source_examples["common"],
            source_examples["hard_real_rare"],
            source_examples["hard_synthetic"],
        }
        if len(source_totals) != 1:
            raise RuntimeError("global equal-source balance drifted")
    if args.formal_contract:
        if total_examples != FORMAL_TOTAL_EXAMPLES:
            raise RuntimeError("formal total example budget drifted")
        if optimizer_steps != FORMAL_OPTIMIZER_STEPS:
            raise RuntimeError("formal optimizer-step budget drifted")
    final_parameters = dict(model.named_parameters())
    frozen_parameter_max_abs_delta = max(
        (
            float(
                (final_parameters[name].detach().cpu() - initial)
                .abs().max()
            )
            for name, initial in frozen_parameter_state.items()
        ),
        default=0.0,
    )
    if arbitration_objective and frozen_parameter_max_abs_delta != 0.0:
        raise RuntimeError("frozen proposal parameters changed during arbitration")
    trainable_parameter_max_abs_delta = max(
        (
            float(
                (final_parameters[name].detach().cpu() - initial)
                .abs().max()
            )
            for name, initial in trainable_parameter_state.items()
        ),
        default=0.0,
    )
    coverage = {}
    for cache_index in range(3):
        coverage[str(cache_index)] = {}
        if decision_stratified:
            for group in decision_groups:
                sampler = decision_samplers[(cache_index, group)]
                pool_rows = len(decision_pools[cache_index][group])
                coverage[str(cache_index)][group] = {
                    "unique_positions_visited": len(sampler.visited),
                    "pool_rows": pool_rows,
                    "completed_cycles": sampler.completed_cycles,
                    "full_pool_coverage": len(sampler.visited) == pool_rows,
                }
        elif args.source_risk == "original_mixture":
            for class_name in ("hard", "common"):
                sampler = samplers[(cache_index, class_name)]
                if len(sampler.visited) != len(hard_rows):
                    raise RuntimeError(
                        f"cache {cache_index} {class_name} did not cover the hard pool"
                    )
                coverage[str(cache_index)][class_name] = {
                    "unique_positions_visited": len(sampler.visited),
                    "pool_rows": len(hard_rows),
                    "completed_cycles": sampler.completed_cycles,
                }
        else:
            for source, pool_size in equal_source_sizes.items():
                sampler = equal_samplers[(cache_index, source)]
                if len(sampler.visited) != pool_size:
                    raise RuntimeError(
                        f"cache {cache_index} {source} did not cover its source pool"
                    )
                coverage[str(cache_index)][source] = {
                    "unique_positions_visited": len(sampler.visited),
                    "pool_rows": pool_size,
                    "completed_cycles": sampler.completed_cycles,
                }


    report = {
        "schema_version": (
            6 if args.objective == OBJECTIVE_COUNTERFACTUAL_EVALUATION else 3
        ),
        "status": "PASS",
        "method": args.method_name,
        "source_kind": "full_navtrain_real_rare_plus_filtered_online_rollout_v1",
        "source_policy": "immutable_epoch100_diffusiondrive",
        "baseline_checkpoint_sha256": BASELINE_SHA256,
        "fresh_selector_initialization": "exact_zero",
        "initial_max_abs_delta": initial_max_abs_delta,
        "initial_proposal_max_abs_delta": initial_proposal_max_abs_delta,
        "frozen_parameter_max_abs_delta": frozen_parameter_max_abs_delta,
        "trainable_parameter_max_abs_delta": trainable_parameter_max_abs_delta,
        "sampling_mode": (
            f"cpv_decision_{args.arbiter_risk}"
            if (
                args.objective == OBJECTIVE_PROPOSAL_REGRET
                and args.arbiter_risk != "global"
            )
            else (
                "three_source_equal_risk"
                if args.source_risk == "equal_strata"
                else "common_hard_balanced"
            )
        ),
        "objective": args.objective,
        "objective_paper_name": (
            "conditional_quality_reweighting_cqr"
            if args.objective == OBJECTIVE_GATE_CONDITIONED
            else args.objective
        ),
        "paper_role": paper_role(args),
        "preference_weight": args.preference_weight,
        "proposal_checkpoint": (
            None if args.proposal_checkpoint is None else str(args.proposal_checkpoint)
        ),
        "proposal_checkpoint_sha256": args.proposal_checkpoint_sha256,
        "proposal_method": args.proposal_method,
        "verifier_reward_margin": args.verifier_reward_margin,
        "verifier_reward_temperature": args.verifier_reward_temperature,
        "override_threshold": args.override_threshold,
        "arbiter_loss": args.arbiter_loss,
        "arbiter_risk": args.arbiter_risk,
        "use_decision_context": args.use_decision_context,
        "arbiter_target": args.arbiter_target,
        "counterfactual_loss": args.counterfactual_loss,
        "train_evaluator_encoder": args.train_evaluator_encoder,
        "source_risk": args.source_risk,
        "evaluator_initialization_max_abs_delta": (
            args.evaluator_initialization_max_abs_delta
        ),
        "trainable_parameter_names": trainable_parameter_names,
        "ablation": args.ablation,
        "selector_architecture": args.architecture,
        "training_data_split": args.data_split,
        "training_epochs": args.epochs,
        "training_examples_per_cache_epoch": args.examples_per_cache_epoch,
        "training_batch_size": args.batch_size,
        "formal_contract": bool(args.formal_contract),
        "scene_selector_config": model_config,
        "implementation_files": implementation_provenance(),
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_epochs": list(checkpoint_epochs),
        "fixed_budget_selection": "final_epoch",
        "early_stopping": False,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(hard_pool_path),
        "hard_pool_sha256": common.sha256_file(hard_pool_path),
        "synthetic_cache": str(synthetic_path),
        "synthetic_cache_sha256": common.sha256_file(synthetic_path),
        "real_cache_manifests": list(real_manifests),
        "common_arm": args.common_arm,
        "common_data_manifest": (
            None if e2_manifest_path is None else str(e2_manifest_path)
        ),
        "common_data_manifest_sha256": (
            None
            if e2_manifest_path is None
            else common.sha256_file(e2_manifest_path)
        ),
        "common_cache_manifests": list(common_cache_manifests),
        "common_unique_tokens": (
            None if common_tokens is None else len(common_tokens)
        ),
        "sampling": {
            "contract": (
                f"active proposal pairs balanced by {args.arbiter_risk}"
                if decision_stratified
                else (
                    "equal common/real-rare/synthetic source risk"
                    if args.source_risk == "equal_strata"
                    else "50% common; 50% hard=(real rare + senior-v1 filtered synthetic)"
                )
            ),
            "examples_per_cache_epoch": args.examples_per_cache_epoch,
            "total_examples": total_examples,
            "total_optimizer_steps": optimizer_steps,
            "class_examples": dict(class_examples),
            "source_examples": dict(source_examples),
            "risk_group_examples": dict(risk_group_examples),
            "decision_pool_audit": decision_pool_audit,
            "coverage": coverage,
        },
        "mean_training_loss": accumulated["loss"] / optimizer_steps,
        "mean_training_policy": accumulated["policy"] / optimizer_steps,
        "mean_training_quality_policy": (
            accumulated["quality_policy"] / optimizer_steps
        ),
        "mean_training_preference": (
            accumulated["preference"] / optimizer_steps
        ),
        "mean_training_verification": (
            accumulated["verification"] / optimizer_steps
        ),
        "mean_training_arbitration": accumulated["arbitration"] / optimizer_steps,
        "mean_training_counterfactual": (
            accumulated["counterfactual"] / optimizer_steps
        ),
        "mean_training_kl": accumulated["kl"] / optimizer_steps,
        "gate_conditioned_diagnostics": {
            "official_active_groups": accumulated["official_active_groups"],
            "quality_active_groups": accumulated["quality_active_groups"],
            "safe_candidates": accumulated["safe_candidates"],
        },
        "scalar_preference_diagnostics": {
            "active_groups": accumulated["preference_active_groups"],
            "ordered_valid_pairs": accumulated["preference_pairs"],
            "reward_source": "official_scalar_pdm_only",
            "reward_components_consumed": False,
        },
        "incumbent_verification_diagnostics": {
            "active_groups": accumulated["verification_active_groups"],
            "candidate_incumbent_pairs": accumulated["verification_pairs"],
            "reward_source": "official_scalar_pdm_only",
            "reward_components_consumed": False,
            "proposal_frozen": args.objective == OBJECTIVE_INCUMBENT_VERIFICATION,
        },
        "counterfactual_evaluation_diagnostics": {
            "active_groups": accumulated["counterfactual_active"],
            "beneficial_proposals": accumulated["counterfactual_beneficial"],
            "degrading_proposals": accumulated["counterfactual_degrading"],
            "tie_proposals": accumulated["counterfactual_ties"],
            "absolute_scalar_regret": accumulated[
                "counterfactual_absolute_regret"
            ],
            "target": args.arbiter_target,
            "loss_kind": args.counterfactual_loss,
            "source_risk": args.source_risk,
            "train_evaluator_encoder": args.train_evaluator_encoder,
            "evaluator_initialization_max_abs_delta": (
                args.evaluator_initialization_max_abs_delta
            ),
            "fixed_override_threshold": args.override_threshold,
            "reward_source": "official_scalar_pdm_only",
            "reward_components_consumed": False,
            "proposal_frozen": (
                args.objective == OBJECTIVE_COUNTERFACTUAL_EVALUATION
            ),
            "mean_source_losses": {
                source: (
                    accumulated[f"counterfactual_source_loss_{source}"]
                    / accumulated[f"counterfactual_source_batches_{source}"]
                    if accumulated[f"counterfactual_source_batches_{source}"]
                    else None
                )
                for source in (
                    "common", "hard_real_rare", "hard_synthetic"
                )
            },
        },
        "proposal_regret_arbitration_diagnostics": {
            "active_groups": accumulated["arbitration_active_groups"],
            "beneficial_proposals": accumulated["arbitration_beneficial"],
            "degrading_proposals": accumulated["arbitration_degrading"],
            "absolute_scalar_regret": accumulated[
                "arbitration_absolute_regret"
            ],
            "target": args.arbiter_target,
            "loss_kind": args.arbiter_loss,
            "risk_aggregation": args.arbiter_risk,
            "decision_pool_mined": decision_stratified,
            "mean_group_losses": {
                group: (
                    accumulated[f"arbitration_group_loss:{group}"]
                    / accumulated[f"arbitration_group_batches:{group}"]
                    if accumulated[f"arbitration_group_batches:{group}"]
                    else None
                )
                for group in (
                    decision_groups if decision_stratified else ("global",)
                )
            },
            "group_active_examples": {
                group: accumulated[f"arbitration_group_active:{group}"]
                for group in (
                    decision_groups if decision_stratified else ("global",)
                )
            },
            "group_regret_mass": {
                group: accumulated[f"arbitration_group_regret:{group}"]
                for group in (
                    decision_groups if decision_stratified else ("global",)
                )
            },
            "fixed_override_threshold": args.override_threshold,
            "reward_source": "official_scalar_pdm_only",
            "reward_components_consumed": False,
            "proposal_frozen": args.objective == OBJECTIVE_PROPOSAL_REGRET,
        },
        "checkpoints": checkpoint_reports,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS V3 rare-rollout cached selector training: {report_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-cache", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument(
        "--examples-per-cache-epoch",
        type=int,
        default=FORMAL_EXAMPLES_PER_CACHE_EPOCH,
    )
    parser.add_argument("--checkpoint-epochs", default=str(FORMAL_EPOCHS))
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method-name")
    parser.add_argument(
        "--objective",
        choices=(
            OBJECTIVE_OFFICIAL,
            OBJECTIVE_GATE_CONDITIONED,
            OBJECTIVE_SCALAR_PREFERENCE,
            OBJECTIVE_INCUMBENT_VERIFICATION,
            OBJECTIVE_PROPOSAL_REGRET,
            OBJECTIVE_COUNTERFACTUAL_EVALUATION,
        ),
        default=OBJECTIVE_OFFICIAL,
    )
    parser.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES, default="scene_conditioned_v3")
    parser.add_argument(
        "--ablation",
        choices=("full", "relational_only", "temporal_only", "no_scene_context"),
        default="full",
    )
    parser.add_argument("--preference-weight", type=float, default=0.0)
    parser.add_argument("--proposal-checkpoint", type=Path)
    parser.add_argument("--verifier-reward-margin", type=float, default=0.0)
    parser.add_argument("--verifier-reward-temperature", type=float, default=0.05)
    parser.add_argument("--override-threshold", type=float, default=0.0)
    parser.add_argument("--arbiter-loss", choices=("sign", "regret"))
    parser.add_argument("--arbiter-risk", choices=ARBITER_RISKS, default="global")
    parser.add_argument("--counterfactual-loss", choices=COUNTERFACTUAL_LOSSES)
    parser.add_argument(
        "--train-evaluator-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--source-risk", choices=SOURCE_RISKS, default="original_mixture")
    parser.add_argument(
        "--use-decision-context",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--data-split", choices=("train", "all"), default="all")
    parser.add_argument("--formal-contract", action="store_true")
    parser.add_argument("--smoke-limit-hard-pool", type=int)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
