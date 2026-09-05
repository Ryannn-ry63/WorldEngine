"""Diffusion-aware, reference-anchored group objectives for selector post-training.

The inner action group is one diffusion draw with ``K`` trajectory candidates.
The outer group contains ``D`` independently sampled diffusion draws of the
same scene token.  Only the official scalar reward is consumed by this module.

This file is deliberately independent of the DiffusionDrive model code so the
objective can be unit-tested without importing MMCV or NAVSIM.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _validate_inputs(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    rewards: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("hierarchical selector tensors must have shape [B, D, K]")
    if not (
        logits.shape == anchor_logits.shape == rewards.shape == valid.shape
    ):
        raise ValueError("logits, anchor logits, rewards, and valid mask must align")
    if logits.shape[1] < 2:
        raise ValueError("an outer diffusion group requires at least two draws")
    if logits.shape[2] < 2:
        raise ValueError("an inner candidate group requires at least two actions")
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("policy temperature must be finite and positive")
    valid = valid.to(dtype=torch.bool) & torch.isfinite(rewards)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("every diffusion draw requires at least one valid reward")
    return valid


def masked_policy(
    logits: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a masked categorical policy and its log probabilities."""

    masked = (logits / float(temperature)).masked_fill(~valid, -1e4)
    log_probability = F.log_softmax(masked, dim=-1)
    probability = log_probability.exp() * valid.to(logits.dtype)
    return probability, log_probability


def straight_through_top1(
    probability: torch.Tensor,
    logits: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Use hard top-1 values in the forward pass and policy gradients backward."""

    index = logits.masked_fill(~valid, -torch.inf).argmax(dim=-1)
    hard = F.one_hot(index, num_classes=logits.shape[-1]).to(probability.dtype)
    return hard + probability - probability.detach()


def per_draw_reference_gain(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    rewards: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 1.0,
    selection_mode: str = "soft",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute scalar reward gain and KL to a frozen selector for every draw.

    ``selection_mode='soft'`` uses exact categorical expected reward.
    ``selection_mode='straight_through_top1'`` evaluates the deployed hard
    argmax in the forward pass while retaining the categorical policy gradient.
    The gain is exactly zero when current and anchor logits are equal.
    """

    valid = _validate_inputs(logits, anchor_logits, rewards, valid, temperature)
    probability, log_probability = masked_policy(logits, valid, temperature)
    anchor_probability, anchor_log_probability = masked_policy(
        anchor_logits.detach(), valid, temperature
    )
    safe_rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
    if selection_mode == "soft":
        value_distribution = probability
        anchor_distribution = anchor_probability
        current_value = (value_distribution * safe_rewards).sum(dim=-1)
        anchor_value = (anchor_distribution * safe_rewards).sum(dim=-1)
    elif selection_mode == "straight_through_top1":
        value_distribution = straight_through_top1(probability, logits, valid)
        current_index = logits.masked_fill(~valid, -torch.inf).argmax(dim=-1)
        anchor_index = anchor_logits.detach().masked_fill(~valid, -torch.inf).argmax(
            dim=-1
        )
        current_hard = safe_rewards.gather(-1, current_index[..., None]).squeeze(-1)
        anchor_value = safe_rewards.gather(-1, anchor_index[..., None]).squeeze(-1)
        current_soft = (probability * safe_rewards).sum(dim=-1)
        # The forward value is exactly the deployed hard top-1 reward.  Writing
        # the straight-through estimator at value level avoids an O(1e-8)
        # cancellation residue when current and anchor logits are identical.
        current_value = current_hard + (current_soft - current_soft.detach())
        anchor_distribution = F.one_hot(
            anchor_index, num_classes=logits.shape[-1]
        ).to(probability.dtype)
    else:
        raise ValueError(f"unknown selection mode: {selection_mode}")

    gain = current_value - anchor_value
    kl = (
        probability
        * (log_probability - anchor_log_probability)
        * valid.to(logits.dtype)
    ).sum(dim=-1)
    diagnostics = {
        "current_value": current_value,
        "anchor_value": anchor_value,
        "probability": probability,
        "anchor_probability": anchor_probability,
    }
    return gain, kl, diagnostics


def aggregate_draw_gains(
    gains: torch.Tensor,
    aggregation: str = "mean",
    risk_temperature: float = 0.02,
    risk_mix: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate same-token gains across diffusion draws.

    The normalized soft minimum is zero when all draw gains are zero and tends
    to the worst-draw gain as ``risk_temperature`` approaches zero.  Its
    gradient weights under-performing draws more heavily.
    """

    if gains.ndim != 2 or gains.shape[1] < 2:
        raise ValueError("draw gains must have shape [B, D] with D >= 2")
    if aggregation == "mean":
        weights = torch.full_like(gains, 1.0 / gains.shape[1])
        return gains.mean(dim=-1), weights
    if aggregation not in {"softmin", "bounded_mean_risk"}:
        raise ValueError(f"unknown draw aggregation: {aggregation}")
    if not math.isfinite(float(risk_temperature)) or risk_temperature <= 0.0:
        raise ValueError("soft-min risk temperature must be finite and positive")
    if not math.isfinite(float(risk_mix)) or not 0.0 <= risk_mix <= 1.0:
        raise ValueError("risk mix must be finite and lie in [0, 1]")
    tau = float(risk_temperature)
    normalizer = math.log(gains.shape[1])
    softmin_gain = -tau * (
        torch.logsumexp(-gains / tau, dim=-1) - normalizer
    )
    softmin_weights = F.softmax(-gains / tau, dim=-1)
    if aggregation == "softmin":
        return softmin_gain, softmin_weights

    # A pure soft minimum can collapse onto one unlucky diffusion draw and
    # reproduce the locking behavior of a hard minimum. Mixing it with the
    # uniform mean gives every draw a fixed gradient floor while preserving a
    # smooth lower-tail emphasis. For D=3 and risk_mix=0.5 the weights are
    # bounded to [1/6, 2/3].
    alpha = float(risk_mix)
    mean_gain = gains.mean(dim=-1)
    aggregate = (1.0 - alpha) * mean_gain + alpha * softmin_gain
    uniform = torch.full_like(gains, 1.0 / gains.shape[1])
    weights = (1.0 - alpha) * uniform + alpha * softmin_weights
    return aggregate, weights


def diffusion_robust_group_loss(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    rewards: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 1.0,
    kl_weight: float = 0.0,
    reward_scale: float = 1.0,
    selection_mode: str = "straight_through_top1",
    aggregation: str = "softmin",
    risk_temperature: float = 0.02,
    risk_mix: float = 0.5,
    objective_mode: str = "relative_gain",
):
    """Reference-anchored hierarchical objective over candidates and draws."""

    if not math.isfinite(float(kl_weight)) or kl_weight < 0.0:
        raise ValueError("KL weight must be finite and non-negative")
    if not math.isfinite(float(reward_scale)) or reward_scale <= 0.0:
        raise ValueError("reward scale must be finite and positive")
    gains, per_draw_kl, value_diagnostics = per_draw_reference_gain(
        logits,
        anchor_logits,
        rewards,
        valid,
        temperature=temperature,
        selection_mode=selection_mode,
    )
    if objective_mode == "relative_gain":
        objective_signal = gains
    elif objective_mode == "absolute_value":
        objective_signal = value_diagnostics["current_value"]
    else:
        raise ValueError(f"unknown objective mode: {objective_mode}")
    aggregate_gain, draw_weights = aggregate_draw_gains(
        objective_signal,
        aggregation=aggregation,
        risk_temperature=risk_temperature,
        risk_mix=risk_mix,
    )
    policy = -float(reward_scale) * aggregate_gain.mean()
    kl = per_draw_kl.mean()
    loss = policy + float(kl_weight) * kl
    diagnostics = {
        **value_diagnostics,
        "per_draw_gain": gains,
        "objective_signal": objective_signal,
        "objective_mode": objective_mode,
        "aggregate_gain": aggregate_gain,
        "mean_gain": gains.mean(dim=-1),
        "worst_gain": gains.min(dim=-1).values,
        "draw_weights": draw_weights,
        "per_draw_kl": per_draw_kl,
    }
    return loss, policy, kl, diagnostics
