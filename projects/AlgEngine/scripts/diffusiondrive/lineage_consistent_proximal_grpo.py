#!/usr/bin/env python3
"""Lineage-consistent proximal GRPO for diffusion candidate selectors.

DiffusionDrive creates ``D`` noisy realizations from the same ordered bank of
``K`` plan anchors.  Candidate index ``k`` is therefore a trajectory lineage,
not an unrelated action in each draw.  This module makes that structure an
explicit part of selector post-training while consuming only the official
scalar reward.

The four arms form a 2 x 2 causal ablation:

* ``direct``: current-draw utility and exact legacy GRPO;
* ``lineage_direct``: leave-one-draw lineage utility and exact GRPO;
* ``per_draw_proximal``: current-draw utility and a KL-budgeted target;
* ``lineage_proximal``: leave-one-draw lineage utility and a KL-budgeted target.

The implementation is independent of MMCV/NAVSIM so the objective can be
tested on CPU.  ``direct`` deliberately delegates to the already-audited
legacy implementation; this makes A0 an exact code-path control rather than a
reimplementation that is merely mathematically similar.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

import proposal_aware_full_feedback_grpo as legacy


ARMS = (
    "direct",
    "lineage_direct",
    "per_draw_proximal",
    "lineage_proximal",
)
LINEAGE_ARMS = frozenset(("lineage_direct", "lineage_proximal"))
PROXIMAL_ARMS = frozenset(("per_draw_proximal", "lineage_proximal"))
DIRECT_ARMS = frozenset(("direct", "lineage_direct"))


def arm_contract(arm: str) -> Dict[str, object]:
    """Return the locked causal factors for one attribution arm."""

    if arm not in ARMS:
        raise ValueError(f"unknown LC-PGRPO arm: {arm}")
    lineage = arm in LINEAGE_ARMS
    proximal = arm in PROXIMAL_ARMS
    return {
        "utility": (
            "leave_one_draw_same_candidate_lineage"
            if lineage
            else "current_draw_scalar_reward"
        ),
        "update": (
            "v3_anchored_kl_budgeted_exponential_tilt"
            if proximal
            else "exact_expected_advantage_grpo"
        ),
        "lineage_consistent": lineage,
        "proximal_target": proximal,
        "official_scalar_pdm_only": True,
    }


def _validate(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    rewards: torch.Tensor,
    valid: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("LC-PGRPO tensors must have shape [B, D, K]")
    if not (logits.shape == anchor_logits.shape == rewards.shape == valid.shape):
        raise ValueError("logits, anchor logits, rewards, and valid mask must align")
    if logits.shape[1] < 3:
        raise ValueError("lineage estimation requires at least three diffusion draws")
    if logits.shape[2] < 2:
        raise ValueError("each diffusion draw requires at least two candidates")
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    valid = valid.to(dtype=torch.bool) & torch.isfinite(rewards)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("every diffusion draw needs at least one finite reward")
    return valid


def leave_one_draw_lineage_utility(
    rewards: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate a lineage's utility from all *other* diffusion draws.

    The held-out draw's reward never enters its own target.  The returned
    ``complete`` mask marks draw-level sets for which every locally valid
    candidate has at least one valid observation in another draw.  Incomplete
    sets are retained at V3 instead of silently changing action support.
    """

    if rewards.ndim != 3 or valid.shape != rewards.shape:
        raise ValueError("lineage tensors must align with shape [B, D, K]")
    if rewards.shape[1] < 3:
        raise ValueError("lineage utility requires D >= 3")
    finite = valid.to(dtype=torch.bool) & torch.isfinite(rewards)
    safe = torch.where(finite, rewards, torch.zeros_like(rewards))
    total = safe.sum(dim=1, keepdim=True)
    count = finite.sum(dim=1, keepdim=True)
    other_count = count - finite.to(count.dtype)
    other_sum = total - safe
    utility = other_sum / other_count.clamp_min(1).to(rewards.dtype)
    utility_valid = finite & (other_count > 0)
    utility = torch.where(utility_valid, utility, torch.zeros_like(utility))
    complete = ((~finite) | utility_valid).all(dim=-1)
    return utility, utility_valid, complete


def _utility_for_arm(
    rewards: torch.Tensor,
    valid: torch.Tensor,
    arm: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if arm in LINEAGE_ARMS:
        return leave_one_draw_lineage_utility(rewards, valid)
    complete = torch.ones(rewards.shape[:-1], dtype=torch.bool, device=rewards.device)
    return rewards, valid, complete


def _preference_statistics(
    utility: torch.Tensor,
    utility_valid: torch.Tensor,
    complete: torch.Tensor,
    anchor_probability: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    advantage, active = legacy.normalized_advantage(
        utility, utility_valid, epsilon=epsilon
    )
    active = active & complete
    advantage = torch.where(active[..., None], advantage, torch.zeros_like(advantage))

    oracle_value, oracle_index = utility.masked_fill(
        ~utility_valid, -torch.inf
    ).max(dim=-1)
    incumbent_index = anchor_probability.masked_fill(
        ~utility_valid, -1.0
    ).argmax(dim=-1)
    incumbent_value = utility.gather(
        -1, incumbent_index[..., None]
    ).squeeze(-1)
    headroom = torch.where(
        complete,
        oracle_value - incumbent_value,
        torch.zeros_like(oracle_value),
    )
    return advantage, active, headroom, oracle_index, incumbent_index


def _tilted_distribution(
    anchor_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    valid: torch.Tensor,
    eta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tilted = anchor_log_probability + eta[..., None] * advantage
    tilted = tilted.masked_fill(~valid, -torch.inf)
    log_target = torch.log_softmax(tilted, dim=-1)
    target = torch.where(valid, log_target.exp(), torch.zeros_like(log_target))
    target_kl = (
        target
        * torch.where(
            valid,
            log_target - anchor_log_probability,
            torch.zeros_like(log_target),
        )
    ).sum(dim=-1)
    return target, target_kl


def kl_budgeted_target(
    anchor_logits: torch.Tensor,
    advantage: torch.Tensor,
    valid: torch.Tensor,
    active: torch.Tensor,
    *,
    target_kl: float,
    temperature: float = 1.0,
    iterations: int = 64,
    tolerance: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a preference direction onto a fixed ``KL(q || pi_V3)`` ball.

    ``q`` is the exponential tilt ``pi_V3 * exp(eta * advantage)``.  A
    per-set bisection solves eta.  Inactive sets return the V3 distribution
    exactly and have eta/target-KL equal to zero.
    """

    if anchor_logits.shape != advantage.shape or valid.shape != advantage.shape:
        raise ValueError("anchor logits, advantage, and valid mask must align")
    if active.shape != advantage.shape[:-1]:
        raise ValueError("active mask must cover all candidate sets")
    if not math.isfinite(float(target_kl)) or target_kl <= 0.0:
        raise ValueError("target KL must be finite and positive")
    if iterations <= 0 or tolerance <= 0.0:
        raise ValueError("bisection controls must be positive")

    # Solve in float64; selector logits remain in their original dtype outside
    # the detached target construction.
    work_logits = anchor_logits.detach().double()
    work_advantage = advantage.detach().double()
    work_valid = valid.to(dtype=torch.bool)
    anchor_logp, anchor_probability = legacy.masked_log_policy(
        work_logits, work_valid, temperature
    )
    solve = active.to(dtype=torch.bool)
    low = torch.zeros_like(active, dtype=torch.float64)
    high = torch.ones_like(low)
    requested = torch.full_like(low, float(target_kl))

    # Exponential tilting has monotone KL for eta >= 0.  Expand a separate
    # bracket for every set, then use a fixed-count bisection for reproducible
    # targets across CPU/GPU environments.
    for _ in range(32):
        _, high_kl = _tilted_distribution(
            anchor_logp, work_advantage, work_valid, high
        )
        needs_expansion = solve & (high_kl < requested - tolerance)
        if not bool(needs_expansion.any()):
            break
        high = torch.where(needs_expansion, high * 2.0, high)

    _, bracket_kl = _tilted_distribution(
        anchor_logp, work_advantage, work_valid, high
    )
    reachable = solve & (bracket_kl >= requested - tolerance)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        _, middle_kl = _tilted_distribution(
            anchor_logp, work_advantage, work_valid, middle
        )
        below = reachable & (middle_kl < requested)
        low = torch.where(below, middle, low)
        high = torch.where(reachable & ~below, middle, high)

    eta = torch.where(reachable, 0.5 * (low + high), high)
    tilted_target, achieved = _tilted_distribution(
        anchor_logp, work_advantage, work_valid, eta
    )
    target = torch.where(
        solve[..., None], tilted_target, anchor_probability
    )
    achieved = torch.where(solve, achieved, torch.zeros_like(achieved))
    eta = torch.where(solve, eta, torch.zeros_like(eta))
    reached = (~solve) | (torch.abs(achieved - requested) <= 5.0 * tolerance)
    return (
        target.to(dtype=anchor_logits.dtype),
        eta.to(dtype=anchor_logits.dtype),
        achieved.to(dtype=anchor_logits.dtype),
        reached,
    )


def independently_shuffle_lineages(
    rewards: torch.Tensor,
    valid: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Break candidate lineage while preserving each draw's reward multiset."""

    if rewards.ndim != 3 or valid.shape != rewards.shape:
        raise ValueError("shuffle tensors must align with shape [B, D, K]")
    batch, draws, candidates = rewards.shape
    permutations = torch.empty(
        (batch, draws, candidates), dtype=torch.long, device="cpu"
    )
    for batch_index in range(batch):
        for draw_index in range(draws):
            permutations[batch_index, draw_index] = torch.randperm(
                candidates, generator=generator
            )
    permutations = permutations.to(rewards.device)
    return (
        rewards.gather(-1, permutations),
        valid.gather(-1, permutations),
        permutations,
    )


def lineage_consistent_loss(
    logits: torch.Tensor,
    anchor_logits: torch.Tensor,
    rewards: torch.Tensor,
    valid: torch.Tensor,
    *,
    arm: str,
    temperature: float = 1.0,
    kl_weight: float = 1e-3,
    target_kl: Optional[float] = None,
    retention_headroom: float = 0.005,
    reward_epsilon: float = 1e-6,
):
    """Compute one locked LC-PGRPO attribution objective."""

    contract = arm_contract(arm)
    if not math.isfinite(float(kl_weight)) or kl_weight < 0.0:
        raise ValueError("KL weight must be finite and non-negative")
    if retention_headroom < 0.0 or reward_epsilon <= 0.0:
        raise ValueError("retention controls are outside their valid range")
    valid = _validate(logits, anchor_logits, rewards, valid, temperature)

    # A0 is intentionally the exact prior Direct implementation.  Keeping the
    # same function call also keeps reduction order and gradients identical.
    if arm == "direct":
        loss, policy, kl, diagnostics = legacy.proposal_aware_loss(
            logits,
            anchor_logits,
            rewards,
            valid,
            arm="direct_grpo",
            temperature=temperature,
            kl_weight=kl_weight,
        )
        diagnostics = {
            **diagnostics,
            "utility": rewards,
            "utility_valid": valid,
            "lineage_complete": torch.ones_like(diagnostics["active"]),
            "retained": ~diagnostics["active"],
            "target_kl": torch.zeros_like(diagnostics["headroom"]),
            "target_kl_reached": torch.ones_like(diagnostics["active"]),
            "eta": torch.zeros_like(diagnostics["headroom"]),
            "arm_contract": contract,
        }
        return loss, policy, kl, diagnostics

    logp, probability = legacy.masked_log_policy(logits, valid, temperature)
    with torch.no_grad():
        anchor_logp, anchor_probability = legacy.masked_log_policy(
            anchor_logits.detach(), valid, temperature
        )
        utility, utility_valid, complete = _utility_for_arm(rewards, valid, arm)
        advantage, preference_active, headroom, oracle_index, incumbent_index = (
            _preference_statistics(
                utility,
                utility_valid,
                complete,
                anchor_probability,
                reward_epsilon,
            )
        )

    if arm == "lineage_direct":
        if bool(preference_active.any()):
            per_set_policy = -(probability * advantage).sum(dim=-1)
            policy = per_set_policy[preference_active].mean()
            per_set_kl = (
                probability
                * torch.where(valid, logp - anchor_logp, torch.zeros_like(logp))
            ).sum(dim=-1)
            kl = per_set_kl[preference_active].mean()
            loss = policy + float(kl_weight) * kl
        else:
            loss = logits.sum() * 0.0
            policy = loss
            kl = loss
        target = anchor_probability
        eta = torch.zeros_like(headroom)
        achieved_target_kl = torch.zeros_like(headroom)
        target_kl_reached = torch.ones_like(preference_active)
        retained = ~preference_active
    else:
        if target_kl is None:
            raise ValueError("proximal arms require --target-kl")
        preference_update = preference_active & (headroom > retention_headroom)
        target, eta, achieved_target_kl, target_kl_reached = kl_budgeted_target(
            anchor_logits,
            advantage,
            valid,
            preference_update,
            target_kl=float(target_kl),
            temperature=temperature,
        )
        target = torch.where(
            preference_update[..., None], target, anchor_probability
        )
        retained = ~preference_update
        optimizable = valid.sum(dim=-1) >= 2
        if bool(optimizable.any()):
            safe_logp = torch.where(valid, logp, torch.zeros_like(logp))
            per_set_policy = -(target * safe_logp).sum(dim=-1)
            policy = per_set_policy[optimizable].mean()
            per_set_kl = (
                probability
                * torch.where(valid, logp - anchor_logp, torch.zeros_like(logp))
            ).sum(dim=-1)
            kl = per_set_kl[optimizable].mean()
            loss = policy + float(kl_weight) * kl
        else:
            loss = logits.sum() * 0.0
            policy = loss
            kl = loss

    diagnostics = {
        "active": preference_active,
        "advantage": advantage,
        "utility": utility,
        "utility_valid": utility_valid,
        "lineage_complete": complete,
        "headroom": headroom,
        "oracle_index": oracle_index,
        "incumbent_index": incumbent_index,
        "probability": probability,
        "anchor_probability": anchor_probability,
        "target": target,
        "retained": retained,
        "eta": eta,
        "target_kl": achieved_target_kl,
        "target_kl_reached": target_kl_reached,
        "arm_contract": contract,
    }
    return loss, policy, kl, diagnostics
