#!/usr/bin/env python3
"""Outcome-label objectives for the V4 selector attribution matrix.

This module is intentionally independent of rollout and deployment code.  It
never exposes Q or reward as an inference feature; labels are consumed only by
the training loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IPCTDiagnostics:
    target_index: torch.Tensor
    incumbent_index: torch.Tensor
    incumbent_preserved: torch.Tensor
    active_comparisons: torch.Tensor


def _validate(logits: torch.Tensor, labels: torch.Tensor, v3_logits: torch.Tensor) -> None:
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (B, K) with K >= 2")
    if labels.shape != logits.shape or v3_logits.shape != logits.shape:
        raise ValueError("labels and frozen V3 logits must match logits")
    if not logits.is_floating_point() or not labels.is_floating_point() or not v3_logits.is_floating_point():
        raise TypeError("V4 objective inputs must be floating-point tensors")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain non-finite values")
    if not bool(torch.isfinite(labels).all()) or not bool(torch.isfinite(v3_logits).all()):
        raise ValueError("labels or frozen V3 logits contain non-finite values")


def incumbent_preserving_targets(
    causal_value: torch.Tensor,
    frozen_v3_logits: torch.Tensor,
    epsilon: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return IPCT target, incumbent, and incumbent-preservation indicators."""
    _validate(frozen_v3_logits, causal_value, frozen_v3_logits)
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    values = causal_value.detach()
    reference = frozen_v3_logits.detach()
    incumbent = reference.argmax(dim=-1)
    best = values.max(dim=-1, keepdim=True).values
    optimal = values >= best - float(epsilon)
    incumbent_is_optimal = optimal.gather(1, incumbent[:, None]).squeeze(1)
    # torch.argmax deterministically returns the first index on exact ties.
    masked_reference = reference.masked_fill(~optimal, -torch.inf)
    best_reference_optimal = masked_reference.argmax(dim=-1)
    target = torch.where(incumbent_is_optimal, incumbent, best_reference_optimal)
    return target, incumbent, incumbent_is_optimal


def incumbent_preserving_causal_top1_loss(
    logits: torch.Tensor,
    causal_value: torch.Tensor,
    frozen_v3_logits: torch.Tensor,
    epsilon: float = 0.01,
    reduction: str = "mean",
) -> tuple[torch.Tensor, IPCTDiagnostics]:
    """Non-vanishing, top-1-aligned preference loss from frozen causal values.

    Every target is compared with all candidates whose causal value is more
    than ``epsilon`` worse.  Gaps normalize to one within each example, so the
    example weight is independent of how many nearly duplicated candidates a
    diffusion basin contains.
    """
    _validate(logits, causal_value, frozen_v3_logits)
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction}")
    target, incumbent, preserved = incumbent_preserving_targets(
        causal_value, frozen_v3_logits, epsilon
    )
    labels = causal_value.detach()
    target_values = labels.gather(1, target[:, None])
    gaps = target_values - labels
    active = gaps > float(epsilon)
    weights = torch.where(active, gaps, torch.zeros_like(gaps))
    normalizer = weights.sum(dim=-1, keepdim=True)
    weights = weights / normalizer.clamp_min(torch.finfo(weights.dtype).eps)
    target_logits = logits.gather(1, target[:, None])
    pair_loss = F.softplus(logits - target_logits)
    per_example = (weights * pair_loss).sum(dim=-1)
    # Preserve a differentiable exact zero for a completely flat causal row.
    per_example = torch.where(
        normalizer.squeeze(-1) > 0.0,
        per_example,
        logits.sum(dim=-1) * 0.0,
    )
    if reduction == "mean":
        loss = per_example.mean()
    elif reduction == "sum":
        loss = per_example.sum()
    else:
        loss = per_example
    diagnostics = IPCTDiagnostics(
        target_index=target,
        incumbent_index=incumbent,
        incumbent_preserved=preserved,
        active_comparisons=active.sum(dim=-1),
    )
    return loss, diagnostics


def exact_group_grpo_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Exact-group scalar policy-gradient control used by A1 and A3.

    This is the full-candidate expectation ``-E_pi[A]`` with group-normalized
    detached labels.  It intentionally retains probability weighting, which
    is the mechanism IPCT is designed to contrast.
    """
    _validate(logits, labels, logits.detach())
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported reduction: {reduction}")
    detached = labels.detach()
    advantage = (detached - detached.mean(dim=-1, keepdim=True)) / detached.std(
        dim=-1, keepdim=True, unbiased=False
    ).clamp_min(float(epsilon))
    probability = logits.softmax(dim=-1)
    per_example = -(probability * advantage).sum(dim=-1)
    if reduction == "mean":
        return per_example.mean()
    if reduction == "sum":
        return per_example.sum()
    return per_example
