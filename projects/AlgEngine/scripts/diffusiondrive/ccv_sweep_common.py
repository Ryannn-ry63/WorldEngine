#!/usr/bin/env python3
"""Pure contracts and statistics for the candidate causal-value sweep."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MASTER_METHOD = "diffusiondrive_selector_ccv_sweep_master_v1"
TREATMENT_METHOD = "diffusiondrive_selector_ccv_treatment_v1"
COLLECTION_AUDIT_METHOD = "diffusiondrive_selector_ccv_collection_audit_v1"
SENTINEL_METHOD = "diffusiondrive_selector_ccv_sentinel_gate_v1"
ANALYSIS_METHOD = "diffusiondrive_selector_ccv_sweep_gate_v1"
SCHEMA_VERSION = 1
NUM_CANDIDATES = 20
NUM_TARGETS = 33
NUM_FAILED_TARGETS = 9
NUM_SOLVED_TARGETS = 24
ARRAY_TOLERANCE = 1e-5
OUTCOME_TOLERANCE = 1e-6
ACTION_TOLERANCE = 1e-4
GAP_POINT_THRESHOLD = 0.05
MINIMUM_POSITIVE_LOGS = 3
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260903


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checked_array(value, shape, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise RuntimeError(f"invalid {label}: expected finite {shape}, got {result.shape}")
    return result


def load_master(path: Path | str) -> tuple[Path, dict]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != MASTER_METHOD
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("design_version") != "ccv_sweep_v1"
        or payload.get("training_data_consumed") is not False
        or int(payload.get("num_candidates", -1)) != NUM_CANDIDATES
        or int(payload.get("target_count", -1)) != NUM_TARGETS
        or len(payload.get("targets", [])) != NUM_TARGETS
    ):
        raise RuntimeError(f"invalid CCV master manifest: {path}")
    for key, sha_key in (
        ("protocol_file", "protocol_file_sha256"),
        ("source_target_manifest", "source_target_manifest_sha256"),
        ("source_causal_gate", "source_causal_gate_sha256"),
        ("formal_scenario_file", "formal_scenario_file_sha256"),
        ("smoke_scenario_file", "smoke_scenario_file_sha256"),
    ):
        if sha256_file(payload[key]) != payload[sha_key]:
            raise RuntimeError(f"CCV master provenance drifted: {key}")
    if sorted(map(int, payload.get("arm_indices", []))) != list(range(NUM_CANDIDATES)):
        raise RuntimeError("CCV master does not contain exactly arms 0..19")
    return path, payload


def load_treatment(path: Path | str, expected_sha: str | None = None) -> tuple[Path, dict]:
    path = Path(path).expanduser().resolve()
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RuntimeError("CCV treatment manifest SHA256 drifted")
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != TREATMENT_METHOD
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("design_version") != "ccv_sweep_v1"
        or payload.get("training_data_consumed") is not False
    ):
        raise RuntimeError(f"invalid CCV treatment manifest: {path}")
    if sha256_file(payload["source_target_manifest"]) != payload["source_target_manifest_sha256"]:
        raise RuntimeError("CCV treatment source-target provenance drifted")
    if sha256_file(payload["protocol_file"]) != payload["protocol_file_sha256"]:
        raise RuntimeError("CCV treatment protocol provenance drifted")
    rows = payload.get("targets", [])
    if int(payload.get("target_count", -1)) != len(rows) or not rows:
        raise RuntimeError("CCV treatment target coverage drifted")
    seen = set()
    for row in rows:
        scene = str(row["scene_id"])
        if scene in seen:
            raise RuntimeError(f"duplicate CCV treatment scene: {scene}")
        seen.add(scene)
        if int(row.get("train_seed", -1)) != 0:
            raise RuntimeError("CCV v1 requires scalar V3 train seed 0")
        treatment = int(row["treatment_index"])
        if not 0 <= treatment < NUM_CANDIDATES:
            raise RuntimeError(f"CCV treatment index out of range: {treatment}")
        checked_array(row["candidate_trajectories_8"], (20, 8, 3), "candidate trajectories")
        checked_array(row["current_logits"], (20,), "current logits")
        checked_array(row["candidate_rewards"], (20,), "candidate rewards")
    return path, payload


def trajectory_distance(left, right) -> float:
    left = checked_array(left, (8, 3), "left trajectory")
    right = checked_array(right, (8, 3), "right trajectory")
    return float(np.linalg.norm(left[:, :2] - right[:, :2], axis=-1).mean())


def pairwise_trajectory_distance(trajectories) -> np.ndarray:
    trajectories = checked_array(trajectories, (20, 8, 3), "candidate trajectories")
    xy = trajectories[:, :, :2]
    return np.linalg.norm(xy[:, None] - xy[None, :], axis=-1).mean(axis=-1)


def average_ranks(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def pearson(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left, right):
    return pearson(average_ranks(left), average_ranks(right))


def kendall_tau_b(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    concordant = discordant = tie_left = tie_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = np.sign(left[i] - left[j])
            dy = np.sign(right[i] - right[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_left += 1
            elif dy == 0:
                tie_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + tie_left)
        * (concordant + discordant + tie_right)
    )
    if denominator == 0:
        return None
    return float((concordant - discordant) / denominator)


def cluster_bootstrap(rows, value_key: str, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is not None and np.isfinite(float(value)):
            grouped[str(row["origin_log"])].append(float(value))
    logs = sorted(grouped)
    if not logs:
        return {"estimate": None, "lower_95": None, "upper_95": None, "num_logs": 0}
    all_values = [value for log in logs for value in grouped[log]]
    estimate = float(np.mean(all_values))
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for repeat in range(repetitions):
        sampled = rng.choice(logs, len(logs), replace=True)
        values = [value for log in sampled for value in grouped[str(log)]]
        draws[repeat] = np.mean(values)
    return {
        "estimate": estimate,
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "num_logs": len(logs),
        "repetitions": int(repetitions),
        "seed": int(seed),
    }


def gap_gate(rows, value_key: str, *, threshold=GAP_POINT_THRESHOLD, seed=BOOTSTRAP_SEED):
    bootstrap = cluster_bootstrap(rows, value_key, seed=seed)
    positive_logs = {
        str(row["origin_log"])
        for row in rows
        if float(row[value_key]) > OUTCOME_TOLERANCE
    }
    point_pass = bool(
        bootstrap["estimate"] is not None
        and bootstrap["estimate"] >= threshold
        and len(positive_logs) >= MINIMUM_POSITIVE_LOGS
    )
    bootstrap_pass = bool(
        bootstrap["lower_95"] is not None and bootstrap["lower_95"] > 0.0
    )
    return {
        "value_key": value_key,
        "minimum_mean_gap": threshold,
        "minimum_positive_origin_logs": MINIMUM_POSITIVE_LOGS,
        "positive_origin_log_count": len(positive_logs),
        "point_pass": point_pass,
        "bootstrap_pass": bootstrap_pass,
        "pass": bool(point_pass and bootstrap_pass),
        "cluster_bootstrap": bootstrap,
    }


def knn_geometry_prediction(trajectories, causal_values, neighbors=3):
    if neighbors <= 0 or neighbors >= NUM_CANDIDATES:
        raise ValueError("neighbors must be between 1 and 19")
    distances = pairwise_trajectory_distance(trajectories)
    values = checked_array(causal_values, (20,), "causal values")
    predictions = np.empty(NUM_CANDIDATES, dtype=np.float64)
    for index in range(NUM_CANDIDATES):
        order = np.argsort(distances[index], kind="mergesort")
        order = order[order != index][:neighbors]
        inverse = 1.0 / np.maximum(distances[index, order], 1e-6)
        predictions[index] = float(np.sum(inverse * values[order]) / np.sum(inverse))
    predicted_top3 = set(np.argsort(-predictions, kind="mergesort")[:3].tolist())
    maximum = float(values.max())
    best = {int(i) for i in np.flatnonzero(np.isclose(values, maximum, atol=OUTCOME_TOLERANCE, rtol=0.0))}
    return {
        "predictions": predictions,
        "spearman": spearman(predictions, values),
        "top3_recall_best": bool(predicted_top3.intersection(best)),
    }
