"""Frozen continuous-deployment protocol, separate from the one-shot pilot."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import cfpi_common as c

METHOD = "selector_cfpi_continuous_deployment_v1"
POLICIES = {
    "q_grpo_t1": dict(lr=1e-4, steps=500, score_mode="residual"),
    "q_grpo_t5": dict(lr=3e-5, steps=500, score_mode="residual"),
    "local_grpo_t1": dict(lr=1e-4, steps=500, score_mode="residual"),
    "q_mse": dict(lr=1e-4, steps=50, score_mode="direct_q"),
}
SEEDS = (0, 1, 2)
PHASE_LIMITS = {"phase_a": 24.0, "phase_b": 40.0}
TERMINAL_PUBLICATION = 12  # Existing planner may publish this after the simulator's final scored state.
ADVANCEMENT = dict(min_mean_pdm_gain=0.005, min_positive_seeds=2,
                   max_mean_sr_drop=0.01, max_mean_nc_drop=0.01, max_mean_dac_drop=0.01)
METRICS = ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
           "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "success")
GATE_CHECKPOINT_SHA = "ae38d354de16a62ca0fd43b8629b6c8bcc04bf174df6fa7f8fc03ae2b27eb766"
GATE_SELECTOR_SHA = "fbf71c1104dea82b7f153217ac234f6f6e00f8af2b92d90d4cf5db248f760002"


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=c.sha256_file(path))


def verify(entry):
    path = Path(entry["path"])
    if c.sha256_file(path) != entry["sha256"]:
        raise RuntimeError(f"Frozen deployment input changed: {path}")
    return path


def verified_read(entry):
    return read(verify(entry))


def metrics_csv(path, expected=None):
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            scene = row["token"]
            if scene == "overall_average":
                continue
            if scene in result:
                raise RuntimeError(f"Duplicate metric scene: {scene}")
            values = {key: float(row[key]) for key in METRICS[:-1]}
            if not all(np.isfinite(v) and 0 <= v <= 1 for v in values.values()):
                raise RuntimeError(f"Nonfinite/out-of-range metrics: {scene}")
            values["success"] = float(values["no_at_fault_collisions"] == 1 and
                                       values["drivable_area_compliance"] == 1)
            result[scene] = values
    if not result or (expected is not None and set(result) != set(expected)):
        raise RuntimeError("Metrics coverage differs from frozen scene membership")
    return result


def learned_id(method, seed):
    if method not in POLICIES or seed not in SEEDS:
        raise ValueError("Policy outside the four fixed configurations / three seeds")
    return f"{method}_seed{seed}"


def route_for(manifest, scene_prefix, decision, *, allow_terminal=False):
    terminal = allow_terminal and decision == TERMINAL_PUBLICATION and manifest.get("terminal_publication_decision") == decision
    if decision not in c.DECISION_STEPS and not terminal:
        raise RuntimeError(f"Unexpected deployment decision: {decision}")
    matches = [r for r in manifest["routes"] if scene_prefix in (r["scene_id"], r["origin_token"])]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown/ambiguous deployment scene: {scene_prefix}")
    route = matches[0]
    return route, decision >= route["start_decision"]


def advancement(seed_deltas):
    if len(seed_deltas) != 3:
        raise RuntimeError("Advancement requires all three seeds")
    average = {key: float(np.mean([s[key] for s in seed_deltas])) for key in METRICS}
    positives = sum(s["score"] > 0 for s in seed_deltas)
    passed = (average["score"] >= ADVANCEMENT["min_mean_pdm_gain"] and
              positives >= ADVANCEMENT["min_positive_seeds"] and
              average["success"] >= -ADVANCEMENT["max_mean_sr_drop"] and
              average["no_at_fault_collisions"] >= -ADVANCEMENT["max_mean_nc_drop"] and
              average["drivable_area_compliance"] >= -ADVANCEMENT["max_mean_dac_drop"])
    return dict(pass_screening=bool(passed), mean_delta=average, positive_seed_count=positives)


def paired_interval(scene_deltas, rows):
    """Seed-average first, equal-log weights; exploratory, not certification."""
    logs = sorted({r["origin_log"] for r in rows})
    values = np.array([np.mean([scene_deltas[r["scene_id"]] for r in rows if r["origin_log"] == log])
                       for log in logs])
    rng = np.random.default_rng(20260906)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return dict(scene_mean=float(np.mean(list(scene_deltas.values()))), equal_log_mean=float(values.mean()),
                equal_log_bootstrap_95=np.quantile(means, [.025, .975]).tolist(), logs=len(logs),
                exploratory=True)
