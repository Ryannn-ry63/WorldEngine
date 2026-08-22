#!/usr/bin/env python3
"""Audit a V3 experimental candidate against its seed-matched incumbent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODES = ("bwm_generalization", "gate_conditioned")


def parse_seed_path(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("expected SEED=SUMMARY.json") from error
    if seed not in (0, 1, 2):
        raise argparse.ArgumentTypeError("seed must be 0, 1, or 2")
    return seed, Path(path_text).expanduser().resolve()


def load_summary(path: Path, seed: int) -> dict:
    row = json.loads(path.read_text())
    if (
        row.get("schema_version") != 1
        or row.get("status") != "PASS"
        or int(row.get("eval_seed", -1)) != seed
    ):
        raise RuntimeError(f"formal summary contract did not pass: {path}")
    metrics = row.get("metrics", {})
    required = {
        "openloop_navtest": ("score",),
        "closedloop_reactive": (
            "score",
            "ego_progress",
            "no_at_fault_collisions",
            "drivable_area_compliance",
        ),
    }
    for section, keys in required.items():
        values = metrics.get(section)
        if not isinstance(values, dict):
            raise RuntimeError(f"missing metric section {section}: {path}")
        for key in keys:
            value = float(values.get(key, float("nan")))
            if not np.isfinite(value):
                raise RuntimeError(f"non-finite metric {section}.{key}: {path}")
    success = float(metrics.get("success_rate", float("nan")))
    if not np.isfinite(success):
        raise RuntimeError(f"non-finite success rate: {path}")
    return row


def compact_metrics(summary: dict) -> dict[str, float]:
    metrics = summary["metrics"]
    reactive = metrics["closedloop_reactive"]
    return {
        "openloop_navtest_pdm": float(metrics["openloop_navtest"]["score"]),
        "reactive_pdm": float(reactive["score"]),
        "reactive_ep": float(reactive["ego_progress"]),
        "reactive_nc": float(reactive["no_at_fault_collisions"]),
        "reactive_dac": float(reactive["drivable_area_compliance"]),
        "reactive_success_rate": float(metrics["success_rate"]),
    }


def thresholds(mode: str) -> dict[str, tuple[str, float]]:
    if mode == "bwm_generalization":
        return {
            "reactive_pdm": ("gt", 0.0),
            "reactive_success_rate": ("ge", 0.0),
            "openloop_navtest_pdm": ("ge", -0.01),
        }
    if mode == "gate_conditioned":
        return {
            "reactive_ep": ("ge", 0.01),
            "reactive_pdm": ("ge", -0.005),
            "reactive_success_rate": ("ge", -0.005),
            "reactive_nc": ("ge", -0.005),
            "reactive_dac": ("ge", -0.005),
        }
    raise ValueError(f"unsupported promotion mode: {mode}")


def audit(
    mode: str,
    baseline_paths: list[tuple[int, Path]],
    candidate_paths: list[tuple[int, Path]],
) -> dict:
    baselines = dict(baseline_paths)
    candidates = dict(candidate_paths)
    if len(baselines) != len(baseline_paths) or len(candidates) != len(candidate_paths):
        raise RuntimeError("a seed was specified more than once")
    if not baselines or set(baselines) != set(candidates):
        raise RuntimeError("baseline and candidate seed sets must be identical and nonempty")

    seed_rows = []
    delta_columns: dict[str, list[float]] = {}
    for seed in sorted(baselines):
        baseline_summary = load_summary(baselines[seed], seed)
        candidate_summary = load_summary(candidates[seed], seed)
        baseline = compact_metrics(baseline_summary)
        candidate = compact_metrics(candidate_summary)
        delta = {key: candidate[key] - baseline[key] for key in baseline}
        for key, value in delta.items():
            delta_columns.setdefault(key, []).append(value)
        seed_rows.append(
            {
                "seed": seed,
                "baseline_summary": str(baselines[seed]),
                "baseline_model": baseline_summary.get("model_name"),
                "candidate_summary": str(candidates[seed]),
                "candidate_model": candidate_summary.get("model_name"),
                "baseline": baseline,
                "candidate": candidate,
                "delta": delta,
            }
        )

    mean_delta = {
        key: float(np.mean(values)) for key, values in sorted(delta_columns.items())
    }
    checks = []
    for metric, (operator, threshold) in thresholds(mode).items():
        actual = mean_delta[metric]
        passed = actual > threshold if operator == "gt" else actual + 1e-12 >= threshold
        checks.append(
            {
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "actual_mean_delta": actual,
                "passed": bool(passed),
            }
        )
    promote = all(row["passed"] for row in checks)
    return {
        "schema_version": 1,
        "status": "PASS",
        "mode": mode,
        "decision": "PROMOTE" if promote else "HOLD_INCUMBENT",
        "promotion_gate_passed": promote,
        "seed_count": len(seed_rows),
        "seeds": sorted(baselines),
        "aggregation": "arithmetic_mean_of_seed_matched_metric_deltas",
        "mean_delta": mean_delta,
        "checks": checks,
        "per_seed": seed_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--baseline", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--candidate", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.mode, args.baseline, args.candidate)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
