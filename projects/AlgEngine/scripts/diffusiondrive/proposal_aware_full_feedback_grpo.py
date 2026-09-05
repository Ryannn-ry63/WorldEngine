#!/usr/bin/env python3
"""Proposal-aware full-feedback objectives for a diffusion candidate selector.

The generator has already paid for rewards for every candidate in a set.  This
module therefore separates the proposal distribution (the frozen V3 selector)
from the post-training target.  It consumes only the official scalar PDM reward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


ARMS = ("direct_grpo", "full_feedback", "opportunity_grpo", "paf_grpo")


def _validate(logits, anchor_logits, rewards, valid, temperature):
    if logits.shape != anchor_logits.shape or logits.shape != rewards.shape:
        raise ValueError("logits, anchor logits, and rewards must have one shape")
    if valid.shape != logits.shape or logits.ndim < 2:
        raise ValueError("valid mask must match a tensor with a candidate axis")
    if logits.shape[-1] < 2 or temperature <= 0.0:
        raise ValueError("each set needs at least two candidates and positive temperature")
    valid = valid.to(dtype=torch.bool) & torch.isfinite(rewards)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("every candidate set needs at least one finite reward")
    return valid


def masked_log_policy(logits, valid, temperature=1.0):
    """Return a normalized policy whose invalid entries carry exactly zero mass."""

    masked = (logits / temperature).masked_fill(~valid, -torch.inf)
    logp = F.log_softmax(masked, dim=-1)
    probability = torch.where(valid, logp.exp(), torch.zeros_like(logp))
    return logp, probability


def normalized_advantage(rewards, valid, epsilon=1e-6):
    """Population-zscore official scalar rewards with tie/invalid safeguards."""

    count = valid.sum(dim=-1).clamp_min(1).to(rewards.dtype)
    safe = torch.where(valid, rewards, torch.zeros_like(rewards))
    mean = safe.sum(dim=-1) / count
    centered = torch.where(valid, rewards - mean[..., None], torch.zeros_like(rewards))
    std = (centered.square().sum(dim=-1) / count).sqrt()
    active = (valid.sum(dim=-1) >= 2) & (std > epsilon)
    advantage = centered / std.clamp_min(epsilon)[..., None]
    advantage = torch.where(valid & active[..., None], advantage, torch.zeros_like(advantage))
    return advantage, active


def opportunity(rewards, valid, anchor_probability, lower=0.005, upper=0.1):
    """Measure V3-recoverable headroom and map it to a fixed opportunity weight."""

    if not 0.0 <= lower < upper:
        raise ValueError("opportunity bounds must satisfy 0 <= lower < upper")
    oracle_reward, oracle_index = rewards.masked_fill(~valid, -torch.inf).max(dim=-1)
    incumbent_index = anchor_probability.masked_fill(~valid, -1.0).argmax(dim=-1)
    incumbent_reward = rewards.gather(-1, incumbent_index[..., None]).squeeze(-1)
    headroom = oracle_reward - incumbent_reward
    weight = ((headroom - lower) / (upper - lower)).clamp(0.0, 1.0)
    return weight, headroom, oracle_index, incumbent_index


def full_feedback_distribution(advantage, valid):
    """Convert the complete set of observed scalar rewards into a quality target."""

    q = F.softmax(advantage.masked_fill(~valid, -torch.inf), dim=-1)
    return torch.where(valid, q, torch.zeros_like(q))


def arm_contract(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown PAF arm: {arm}")
    return {
        "direct_grpo": {
            "objective": "exact_expected_advantage_grpo",
            "full_candidate_feedback": False,
            "opportunity_conditioned": False,
        },
        "full_feedback": {
            "objective": "quality_distribution_cross_entropy",
            "full_candidate_feedback": True,
            "opportunity_conditioned": False,
        },
        "opportunity_grpo": {
            "objective": "opportunity_weighted_exact_grpo",
            "full_candidate_feedback": False,
            "opportunity_conditioned": True,
        },
        "paf_grpo": {
            "objective": "proposal_aware_full_feedback_cross_entropy",
            "full_candidate_feedback": True,
            "opportunity_conditioned": True,
        },
    }[arm]


def proposal_aware_loss(
    logits,
    anchor_logits,
    rewards,
    valid,
    *,
    arm,
    temperature=1.0,
    kl_weight=1e-3,
    opportunity_lower=0.005,
    opportunity_upper=0.1,
):
    """Compute one of four pre-registered, matched-budget selector objectives.

    Leading dimensions (for example batch and diffusion draw) are treated as
    independent candidate sets.  Returned diagnostics retain those dimensions.
    """

    arm_contract(arm)
    if kl_weight < 0.0:
        raise ValueError("KL weight must be non-negative")
    valid = _validate(logits, anchor_logits, rewards, valid, temperature)
    logp, probability = masked_log_policy(logits, valid, temperature)
    with torch.no_grad():
        anchor_logp, anchor_probability = masked_log_policy(
            anchor_logits, valid, temperature
        )
        advantage, active = normalized_advantage(rewards, valid)
        quality = full_feedback_distribution(advantage, valid)
        weight, headroom, oracle_index, incumbent_index = opportunity(
            rewards,
            valid,
            anchor_probability,
            lower=opportunity_lower,
            upper=opportunity_upper,
        )
        target = (
            (1.0 - weight[..., None]) * anchor_probability
            + weight[..., None] * quality
        )

    if arm == "direct_grpo":
        per_set_policy = -(probability * advantage).sum(dim=-1)
    elif arm == "opportunity_grpo":
        per_set_policy = -weight * (probability * advantage).sum(dim=-1)
    elif arm == "full_feedback":
        per_set_policy = -(quality * logp.masked_fill(~valid, 0.0)).sum(dim=-1)
    else:
        per_set_policy = -(target * logp.masked_fill(~valid, 0.0)).sum(dim=-1)

    # Tied/singleton sets carry no scalar preference.  PAF keeps them exactly at
    # the anchor via a zero opportunity weight; other arms also skip them.
    if not bool(active.any()):
        zero = logits.sum() * 0.0
        return zero, zero, zero, {
            "active": active,
            "advantage": advantage,
            "quality_target": quality,
            "target": target,
            "opportunity_weight": weight,
            "headroom": headroom,
            "oracle_index": oracle_index,
            "incumbent_index": incumbent_index,
            "probability": probability,
            "anchor_probability": anchor_probability,
        }
    policy = per_set_policy[active].mean()
    per_set_kl = (
        probability
        * torch.where(valid, logp - anchor_logp, torch.zeros_like(logp))
    ).sum(dim=-1)
    kl = per_set_kl[active].mean()
    loss = policy + kl_weight * kl
    diagnostics = {
        "active": active,
        "advantage": advantage,
        "quality_target": quality,
        "target": target,
        "opportunity_weight": weight,
        "headroom": headroom,
        "oracle_index": oracle_index,
        "incumbent_index": incumbent_index,
        "probability": probability,
        "anchor_probability": anchor_probability,
    }
    return loss, policy, kl, diagnostics


def exact_grpo_logit_ascent(probability, advantage):
    """Analytic ascent direction for E_pi[A], used by the mechanism audit."""

    expected = (probability * advantage).sum(dim=-1, keepdim=True)
    return probability * (advantage - expected)
