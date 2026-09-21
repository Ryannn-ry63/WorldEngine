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


def _load_selector_class():
    algengine_root = Path(__file__).resolve().parents[2]
    path = (
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
    )
    spec = importlib.util.spec_from_file_location("diffusion_grpo_scene_selector_v3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SceneConditionedTrajectorySetSelector


SceneConditionedTrajectorySetSelector = _load_selector_class()


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
    if manifest.get("status") != "PASS" or schema_version not in (2, 3):
        raise RuntimeError(
            f"context cache manifest did not pass schema v2/v3: {manifest_path}"
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
    for key, shape in required.items():
        if tuple(cache[key].shape) != shape:
            raise RuntimeError(f"{path}: {key} shape {tuple(cache[key].shape)} != {shape}")
    if len(cache["scenes"]) != count or len(set(cache["tokens"])) != count:
        raise RuntimeError(f"{path}: token/scene provenance drifted")
    return cache, manifest


def model_from_cache(cache, ablation="full"):
    config = dict(cache["scene_selector_config"])
    allowed = {
        "full",
        "feature_only",
        "feature_geometry",
        "feature_geometry_route",
    }
    if ablation not in allowed:
        raise ValueError(f"unknown V3 ablation: {ablation}")
    config.update(
        use_trajectory_geometry=ablation != "feature_only",
        use_route_bev=ablation in {"full", "feature_geometry_route"},
        use_scene_context=ablation == "full",
        use_set_attention=ablation == "full",
    )
    return SceneConditionedTrajectorySetSelector(**config), config


def batch_inputs(cache, indices, device):
    return {
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


def current_logits(model, cache, indices, device):
    reference = cache["reference_logits"][indices].to(device=device, dtype=torch.float32)
    return reference + model(**batch_inputs(cache, indices, device)), reference


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
