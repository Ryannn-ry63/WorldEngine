#!/usr/bin/env python3
"""Pure helpers shared by the R1.5 pre-action oracle experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


MODES = (
    "observe_only",
    "one_shot_oracle",
    "one_shot_matched",
    "persistent_oracle",
)
HEADROOM_THRESHOLD = 0.02
ARRAY_TOLERANCE = 1e-5
ACTION_TOLERANCE = 1e-4


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_rewards(rewards) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    if values.shape != (20,) or not np.isfinite(values).all():
        raise ValueError(f"candidate rewards must be finite (20,), got {values.shape}")
    return values


def stable_oracle_index(rewards, policy_index: int, atol: float = 1e-8) -> int:
    """Return the scalar-PDM oracle, preserving the policy action on a top tie."""

    values = checked_rewards(rewards)
    policy_index = int(policy_index)
    if not 0 <= policy_index < 20:
        raise ValueError(f"policy index out of range: {policy_index}")
    maximum = float(values.max())
    if maximum - float(values[policy_index]) <= float(atol):
        return policy_index
    return int(np.argmax(values))


def intervention_applies(
    mode: str,
    decision_step: int,
    target_step: int | None,
    one_shot_already_applied: bool,
) -> bool:
    if mode not in MODES:
        raise ValueError(f"unsupported R1.5 mode: {mode}")
    if mode == "observe_only" or target_step is None:
        return False
    if mode.startswith("one_shot_"):
        return int(decision_step) == int(target_step) and not one_shot_already_applied
    return int(decision_step) >= int(target_step)


def max_abs_error(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return float("inf")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def trajectory_action_error(left, right) -> float:
    fields = ("waypoints", "velocities", "headings", "angular_velocities")
    errors = []
    for field in fields:
        left_value = getattr(left, field, None)
        right_value = getattr(right, field, None)
        if left_value is None or right_value is None:
            if left_value is not right_value:
                return float("inf")
            continue
        errors.append(max_abs_error(left_value, right_value))
    return max(errors, default=0.0)


def target_context_errors(target: dict, candidates, logits) -> dict[str, float]:
    """Check treatment identity without assuming cross-run scorer invariance."""

    return {
        "candidate_trajectories_8": max_abs_error(
            target["candidate_trajectories_8"], candidates
        ),
        "current_logits": max_abs_error(target["current_logits"], logits),
    }
