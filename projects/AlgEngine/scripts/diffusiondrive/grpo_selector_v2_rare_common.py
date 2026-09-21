"""Shared audited utilities for DiffusionDrive selector V2 rare experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

import grpo_selector_v3_cached_common as v3_common


BASELINE_SHA256 = (
    "1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
)
REWARD_CONTRACT = "navsim_pairwise_raw_progress_then_candidate_gate_v1"
SELECTOR_PREFIX = (
    "planning_head.diff_decoder.layers.1.task_decoder.plan_cls_branch."
)
REFERENCE_PREFIX = "planning_head.reference_selector."
SELECTOR_TENSOR_COUNT = 10


class Selector(nn.Sequential):
    """Exact final DiffusionDrive plan-cls MLP trained by selector V2."""

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_from_cache(cache: dict, device: torch.device) -> Selector:
    state = cache.get("baseline_selector_state")
    if not isinstance(state, dict) or len(state) != SELECTOR_TENSOR_COUNT:
        raise RuntimeError("cache does not contain the exact 10-tensor V2 selector")
    model = Selector().to(device)
    model.load_state_dict(state, strict=True)
    baseline = Selector().to(device)
    baseline.load_state_dict(state, strict=True)
    baseline.requires_grad_(False).eval()
    # Bypass nn.Module.__setattr__ so the frozen anchor is not registered,
    # optimized, serialized, or counted among the exact 10 trainable tensors.
    object.__setattr__(model, "_v2_frozen_anchor", baseline)
    return model


def logits(model: Selector, features: torch.Tensor) -> torch.Tensor:
    """Raw plan-cls logits (used only with an explicit reference anchor)."""
    return model(features.float()).squeeze(-1)


def delta_logits(model: Selector, features: torch.Tensor) -> torch.Tensor:
    """Trainable V2 logit delta relative to its frozen epoch-100 clone."""
    baseline = getattr(model, "_v2_frozen_anchor", None)
    if baseline is None:
        raise RuntimeError("V2 model is missing its unregistered frozen anchor")
    with torch.no_grad():
        reference = logits(baseline, features)
    return logits(model, features) - reference


def current_logits(model: Selector, cache: dict, indices, device: torch.device):
    features = cache["candidate_features"][indices].to(
        device=device, dtype=torch.float32
    )
    reference = cache["reference_logits"][indices].to(
        device=device, dtype=torch.float32
    )
    # Cached features are FP16 while reference logits were produced online in
    # FP32. Delta anchoring makes step zero exactly the behavior policy while
    # preserving the gradient of the original 10-tensor V2 MLP.
    return reference + delta_logits(model, features), reference


def assert_reference_parity(
    model: Selector,
    cache: dict,
    device: torch.device,
    batch_size: int = 256,
    maximum_allowed_raw_error: float = 1e-2,
) -> dict:
    """Audit FP16 raw error and prove exact delta-anchored initialization."""
    baseline = getattr(model, "_v2_frozen_anchor", None)
    if baseline is None:
        raise RuntimeError("V2 model is missing its frozen anchor")
    maximum = 0.0
    total = 0.0
    count = 0
    mismatches = 0
    anchored_maximum = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            features = cache["candidate_features"][start:stop].to(
                device=device, dtype=torch.float32
            )
            reference = cache["reference_logits"][start:stop].to(
                device=device, dtype=torch.float32
            )
            raw = logits(baseline, features)
            difference = (raw - reference).abs()
            maximum = max(maximum, float(difference.max().cpu()))
            total += float(difference.sum().cpu())
            count += difference.numel()
            mismatches += int(raw.argmax(-1).ne(reference.argmax(-1)).sum().cpu())
            anchored = reference + (logits(model, features) - raw)
            anchored_maximum = max(
                anchored_maximum, float((anchored - reference).abs().max().cpu())
            )
    if maximum > maximum_allowed_raw_error:
        raise RuntimeError(
            "FP16 cache quantization error exceeded the audited V2 bound: "
            f"max_abs_error={maximum} bound={maximum_allowed_raw_error}"
        )
    if anchored_maximum != 0.0:
        raise RuntimeError("delta-anchored V2 initialization is not exact")
    return {
        "raw_max_abs_error": maximum,
        "raw_mean_abs_error": total / count,
        "raw_argmax_mismatches": mismatches,
        "rows": len(cache["tokens"]),
        "anchored_max_abs_error": anchored_maximum,
    }


def evaluate(
    model: Selector,
    cache: dict,
    device: torch.device,
    temperature: float,
    batch_size: int,
):
    output = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            indices = slice(start, min(start + batch_size, len(cache["tokens"])))
            current, reference = current_logits(model, cache, indices, device)
            values = selector_metrics(
                current,
                reference,
                cache["candidate_rewards"][indices].to(device),
                cache["candidate_reward_components"][indices].to(device),
                cache["candidate_reward_valid_mask"][indices].to(device),
                temperature,
            )
            for key, value in values.items():
                output.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(value) for key, value in output.items()}


exact_group_loss = v3_common.exact_group_loss
selector_metrics = v3_common.selector_metrics
clip_grad_norm_cpu_ = v3_common.clip_grad_norm_cpu_
load_cache = v3_common.load_cache
summarize = v3_common.summarize
COMPONENT_NAMES = v3_common.COMPONENT_NAMES
