"""Full-feedback baselines. Rewards and incumbent IDs are training-only."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def validate(scores, values):
    if scores.ndim != 2 or scores.shape != values.shape or scores.shape[1] < 2:
        raise ValueError("Expected matching [B, K] score and label tensors")
    if not scores.is_floating_point() or not values.is_floating_point():
        raise TypeError("Scores and values must be floating point")
    if not bool(torch.isfinite(scores).all() and torch.isfinite(values).all()):
        raise ValueError("Scores and values must be finite")


def classification_weights(values, incumbent, delta=0.01):
    validate(values, values)
    if not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")
    if (incumbent.shape != values.shape[:1] or incumbent.dtype != torch.long
            or bool(((incumbent < 0) | (incumbent >= values.shape[1])).any())):
        raise ValueError("Invalid incumbent indices")
    weights = values.detach() - values.detach().min(dim=-1, keepdim=True).values
    return weights + delta * F.one_hot(incumbent, values.shape[1]).to(weights.dtype)


def return_weighted_classification(scores, values, incumbent, *, delta=0.01, normalizer=1.0):
    validate(scores, values)
    if not math.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("normalizer must be a positive training-fold constant")
    weights = classification_weights(values, incumbent, delta)
    return -(weights * scores.log_softmax(dim=-1)).sum(dim=-1).mean() / normalizer


def exact_group(scores, values, temperature=1.0):
    validate(scores, values)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid temperature")
    values = values.detach()
    advantage = (values - values.mean(dim=-1, keepdim=True)) / values.std(
        dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return -((scores / temperature).softmax(dim=-1) * advantage).sum(dim=-1).mean()


def q_regression(scores, values):
    validate(scores, values)
    return F.mse_loss(scores, values.detach())


METHODS = {
    "local_grpo_t1": dict(objective="grpo", label="local_official_pdm", temperature=1.0, delta=0.0),
    "q_grpo_t1": dict(objective="grpo", label="branch_returns", temperature=1.0, delta=0.0),
    "q_grpo_t5": dict(objective="grpo", label="branch_returns", temperature=5.0, delta=0.0),
    "q_mse": dict(objective="mse", label="branch_returns", temperature=1.0, delta=0.0),
    "q_ce_d0": dict(objective="ce", label="branch_returns", temperature=1.0, delta=0.0),
    "q_ce_d001": dict(objective="ce", label="branch_returns", temperature=1.0, delta=0.01),
}
