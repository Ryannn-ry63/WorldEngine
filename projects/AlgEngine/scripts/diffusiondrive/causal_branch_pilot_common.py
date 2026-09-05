#!/usr/bin/env python3
"""Pure helpers for the train-only selector causal-branch pilot v2."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import oracle_r15_common as r15


SOURCE_AUDIT_METHOD = "diffusiondrive_v3_rare_rollout_bwm_scenario_contract_v3"
PREPARED_METHOD = "diffusiondrive_selector_causal_branch_pilot_source_v1"
TARGET_METHOD = "diffusiondrive_selector_causal_branch_pilot_targets_v1"
COLLECTION_AUDIT_METHOD = "diffusiondrive_selector_causal_branch_pilot_collection_audit_v1"
GATE_METHOD = "diffusiondrive_selector_causal_branch_pilot_gate_v2"
TARGET_SCHEMA_VERSION = 2
PHASES = (
    "pipeline_smoke8",
    "baseline_a",
    "baseline_b",
    "intervention_smoke8",
    "intervention_target",
)
MODES = ("observe_only", "one_shot_oracle", "one_shot_matched")
HEADROOM_THRESHOLD = 0.02
MATCHED_REWARD_CEILING = 0.005
MATCHED_ADE_ABSOLUTE_ERROR_MAX_M = 0.5
MATCHED_ADE_RELATIVE_ERROR_MAX = 0.5
MATCHED_ADE_RELATIVE_EPSILON_M = 1e-6
ARRAY_TOLERANCE = 1e-5


def stable_hash(value: str, salt: str = "causal-branch-pilot-v1") -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def scenario_metadata(scene: dict) -> dict:
    metadata = scene.get("metadata", {})
    return {
        "origin_log": str(
            metadata.get("rollout_log_name")
            or metadata.get("log_name")
            or scene.get("name")
            or scene["id"]
        ),
        "origin_token": str(
            metadata.get("rollout_origin_token")
            or scene.get("token")
            or scene["id"]
        ),
        "source_kind": str(metadata.get("rollout_source_kind") or "unknown"),
        "pairing_method": str(metadata.get("common_pairing_method") or "unknown"),
    }


def trajectory_distance(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != (8, 3) or right.shape != (8, 3):
        raise ValueError("candidate trajectories must be (8, 3)")
    return float(np.mean(np.linalg.norm(left[:, :2] - right[:, :2], axis=1)))


def trajectory_perturbation_diagnostics(candidate, policy) -> dict[str, float]:
    """Summarize treatment magnitude without claiming directional geometry match."""

    candidate = np.asarray(candidate, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    if candidate.shape != (8, 3) or policy.shape != (8, 3):
        raise ValueError("candidate trajectories must be (8, 3)")
    if not np.isfinite(candidate).all() or not np.isfinite(policy).all():
        raise ValueError("candidate trajectories must be finite")
    xy = np.linalg.norm(candidate[:, :2] - policy[:, :2], axis=1)
    yaw_delta = candidate[:, 2] - policy[:, 2]
    abs_yaw = np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))
    return {
        "xy_ade_m": float(np.mean(xy)),
        "xy_final_m": float(xy[-1]),
        "xy_max_m": float(np.max(xy)),
        "yaw_abs_mean_rad": float(np.mean(abs_yaw)),
        "yaw_abs_final_rad": float(abs_yaw[-1]),
        "yaw_abs_max_rad": float(np.max(abs_yaw)),
    }


def choose_matched_index(
    trajectories,
    rewards_a,
    rewards_b,
    policy_index: int,
    oracle_index: int,
    ceiling: float = MATCHED_REWARD_CEILING,
    max_absolute_error_m: float = MATCHED_ADE_ABSOLUTE_ERROR_MAX_M,
    max_relative_error: float = MATCHED_ADE_RELATIVE_ERROR_MAX,
) -> tuple[int, float, float, float, float]:
    """Choose the closest non-improving perturbation and enforce its caliper."""

    trajectories = np.asarray(trajectories, dtype=np.float64)
    if trajectories.shape != (20, 8, 3) or not np.isfinite(trajectories).all():
        raise ValueError("candidate trajectories must be finite (20, 8, 3)")
    rewards_a = r15.checked_rewards(rewards_a)
    rewards_b = r15.checked_rewards(rewards_b)
    if ceiling < 0 or max_absolute_error_m < 0 or max_relative_error < 0:
        raise ValueError("matched control thresholds must be non-negative")
    policy_index, oracle_index = int(policy_index), int(oracle_index)
    oracle_distance = trajectory_distance(
        trajectories[oracle_index], trajectories[policy_index]
    )
    eligible = []
    for index in range(20):
        if index in (policy_index, oracle_index):
            continue
        if (
            rewards_a[index] <= rewards_a[policy_index] + ceiling
            and rewards_b[index] <= rewards_b[policy_index] + ceiling
        ):
            distance = trajectory_distance(
                trajectories[index], trajectories[policy_index]
            )
            eligible.append((abs(distance - oracle_distance), index, distance))
    if not eligible:
        raise ValueError("no_reward_nonimproving_candidate")
    absolute_error, index, distance = min(eligible, key=lambda row: (row[0], row[1]))
    relative_error = absolute_error / max(
        abs(oracle_distance), MATCHED_ADE_RELATIVE_EPSILON_M
    )
    if absolute_error > max_absolute_error_m or relative_error > max_relative_error:
        raise ValueError("matched_control_outside_magnitude_caliper")
    return (
        int(index),
        float(distance),
        float(oracle_distance),
        float(absolute_error),
        float(relative_error),
    )


def _largest_remainder_quota(rows: list[dict], limit: int) -> dict[tuple, int]:
    counts = Counter(
        (row["source_kind"], row["pairing_method"]) for row in rows
    )
    if not counts or limit <= 0:
        return {}
    total = sum(counts.values())
    raw = {key: limit * value / total for key, value in counts.items()}
    quota = {key: min(counts[key], int(math.floor(value))) for key, value in raw.items()}
    remaining = limit - sum(quota.values())
    order = sorted(
        counts,
        key=lambda key: (-(raw[key] - math.floor(raw[key])), key),
    )
    for key in order:
        if remaining <= 0:
            break
        if quota[key] < counts[key]:
            quota[key] += 1
            remaining -= 1
    return quota


def stratified_scene_cap(
    rows: list[dict],
    limit: int = 24,
    max_per_origin_log: int = 2,
    salt: str = "formal",
) -> list[dict]:
    """Deterministic proportional selection with an origin-log exposure cap."""

    if limit <= 0 or max_per_origin_log <= 0:
        return []
    ordered = sorted(rows, key=lambda row: stable_hash(row["scene_id"], salt))
    target = min(limit, len(rows))
    quota = _largest_remainder_quota(rows, target)
    used_logs = Counter()
    selected = []
    selected_ids = set()
    for key in sorted(quota):
        candidates = [
            row
            for row in ordered
            if (row["source_kind"], row["pairing_method"]) == key
        ]
        for row in candidates:
            if len([x for x in selected if (x["source_kind"], x["pairing_method"]) == key]) >= quota[key]:
                break
            if used_logs[row["origin_log"]] >= max_per_origin_log:
                continue
            selected.append(row)
            selected_ids.add(row["scene_id"])
            used_logs[row["origin_log"]] += 1
    for row in ordered:
        if len(selected) >= target:
            break
        if row["scene_id"] in selected_ids:
            continue
        if used_logs[row["origin_log"]] >= max_per_origin_log:
            continue
        selected.append(row)
        selected_ids.add(row["scene_id"])
        used_logs[row["origin_log"]] += 1
    return sorted(selected, key=lambda row: stable_hash(row["scene_id"], salt))


def smoke_subset(rows: list[dict], count: int, salt: str) -> list[dict]:
    """Prefer distinct logs, then deterministically fill to the requested count."""

    ordered = sorted(rows, key=lambda row: stable_hash(row["scene_id"], salt))
    selected, seen_ids, seen_logs = [], set(), set()
    for row in ordered:
        if len(selected) >= count:
            break
        if row["origin_log"] in seen_logs:
            continue
        selected.append(row)
        seen_ids.add(row["scene_id"])
        seen_logs.add(row["origin_log"])
    for row in ordered:
        if len(selected) >= count:
            break
        if row["scene_id"] not in seen_ids:
            selected.append(row)
            seen_ids.add(row["scene_id"])
    return selected


def cluster_bootstrap(
    rows: list[dict],
    value_key: str,
    repetitions: int,
    seed: int,
) -> dict:
    """Percentile bootstrap over origin logs, preserving within-log dependence."""

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["origin_log"]].append(float(row[value_key]))
    logs = sorted(grouped)
    if not logs:
        return {"estimate": None, "lower_95": None, "upper_95": None, "num_logs": 0}
    estimate = float(np.mean([float(row[value_key]) for row in rows]))
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for repeat in range(repetitions):
        sampled = rng.choice(logs, size=len(logs), replace=True)
        values = [value for log in sampled for value in grouped[str(log)]]
        draws[repeat] = np.mean(values)
    return {
        "estimate": estimate,
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "num_logs": len(logs),
        "repetitions": repetitions,
        "seed": seed,
    }


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def write_pickle(path: Path, payload):
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

