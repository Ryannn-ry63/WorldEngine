#!/usr/bin/env python3
"""Apply the frozen closed-loop-first selector promotion and safety guardrails."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


REQUIRED_METRICS = (
    "cl_nonreactive_pdm",
    "cl_reactive_pdm",
    "success_rate",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "navtest_pdm",
)


def parse_labeled_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    return label, Path(path)


def load_seed_metrics(path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict) or not seeds:
        raise RuntimeError(f"missing per-seed metrics: {path}")
    normalized = {}
    for seed, metrics in seeds.items():
        missing = set(REQUIRED_METRICS) - set(metrics)
        if missing:
            raise RuntimeError(f"{path}: seed {seed} missing {sorted(missing)}")
        normalized[str(seed)] = {
            key: float(metrics[key]) for key in REQUIRED_METRICS
        }
    return path, normalized


def mean_metric(seeds, key):
    return statistics.fmean(row[key] for row in seeds.values())


def audit_candidate(label, path, candidate, baseline, args):
    if set(candidate) != set(baseline):
        raise RuntimeError(
            f"{label}: seed set {sorted(candidate)} != baseline {sorted(baseline)}"
        )
    per_seed = {}
    for seed in sorted(candidate):
        candidate_cl = statistics.fmean(
            (
                candidate[seed]["cl_nonreactive_pdm"],
                candidate[seed]["cl_reactive_pdm"],
            )
        )
        baseline_cl = statistics.fmean(
            (
                baseline[seed]["cl_nonreactive_pdm"],
                baseline[seed]["cl_reactive_pdm"],
            )
        )
        per_seed[seed] = {
            "mean_closed_loop_pdm": candidate_cl,
            "delta_mean_closed_loop_pdm": candidate_cl - baseline_cl,
            **{
                f"delta_{key}": candidate[seed][key] - baseline[seed][key]
                for key in REQUIRED_METRICS
            },
        }

    mean_deltas = {
        key: mean_metric(candidate, key) - mean_metric(baseline, key)
        for key in REQUIRED_METRICS
    }
    closed_loop_deltas = [
        row["delta_mean_closed_loop_pdm"] for row in per_seed.values()
    ]
    mean_closed_loop_delta = statistics.fmean(closed_loop_deltas)
    gates = {
        "mean_closed_loop_gain": mean_closed_loop_delta
        >= args.minimum_closed_loop_gain,
        "positive_closed_loop_seeds": sum(value > 0.0 for value in closed_loop_deltas)
        >= args.minimum_positive_seeds,
        "success_guardrail": mean_deltas["success_rate"]
        >= -args.maximum_success_drop,
        "collision_guardrail": mean_deltas["no_at_fault_collisions"]
        >= -args.maximum_component_drop,
        "dac_guardrail": mean_deltas["drivable_area_compliance"]
        >= -args.maximum_component_drop,
        "navtest_guardrail": mean_deltas["navtest_pdm"]
        >= -args.maximum_navtest_drop,
        "no_catastrophic_seed": min(closed_loop_deltas)
        >= -args.maximum_seed_closed_loop_drop,
    }
    return {
        "label": label,
        "metrics": str(path),
        "per_seed": per_seed,
        "mean_metric_deltas": mean_deltas,
        "mean_closed_loop_pdm": statistics.fmean(
            row["mean_closed_loop_pdm"] for row in per_seed.values()
        ),
        "mean_closed_loop_pdm_delta": mean_closed_loop_delta,
        "positive_closed_loop_seeds": sum(value > 0.0 for value in closed_loop_deltas),
        "gates": gates,
        "eligible": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate", type=parse_labeled_path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-closed-loop-gain", type=float, default=0.01)
    parser.add_argument("--minimum-positive-seeds", type=int, default=2)
    parser.add_argument("--maximum-success-drop", type=float, default=0.035)
    parser.add_argument("--maximum-component-drop", type=float, default=0.02)
    parser.add_argument("--maximum-navtest-drop", type=float, default=0.01)
    parser.add_argument("--maximum-seed-closed-loop-drop", type=float, default=0.03)
    args = parser.parse_args()

    baseline_path, baseline = load_seed_metrics(args.baseline)
    if set(baseline) != {"0", "1", "2"}:
        raise RuntimeError(
            f"promotion requires exactly paired seeds 0/1/2, got {sorted(baseline)}"
        )
    candidates = []
    labels = set()
    for label, candidate_argument in args.candidate:
        if label in labels:
            raise RuntimeError(f"duplicate candidate label: {label}")
        labels.add(label)
        candidate_path, candidate = load_seed_metrics(candidate_argument)
        candidates.append(
            audit_candidate(label, candidate_path, candidate, baseline, args)
        )
    ranked = sorted(
        candidates,
        key=lambda row: row["mean_closed_loop_pdm_delta"],
        reverse=True,
    )
    eligible = [row for row in ranked if row["eligible"]]
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "trajectory_set_reasoner_closed_loop_promotion_v1",
        "primary_metric": "paired_three_seed_mean_of_cl_nonreactive_and_cl_reactive_pdm",
        "baseline": str(baseline_path),
        "thresholds": {
            "minimum_closed_loop_gain": args.minimum_closed_loop_gain,
            "minimum_positive_seeds": args.minimum_positive_seeds,
            "maximum_success_drop": args.maximum_success_drop,
            "maximum_nc_and_dac_drop": args.maximum_component_drop,
            "maximum_navtest_drop": args.maximum_navtest_drop,
            "maximum_single_seed_closed_loop_drop": args.maximum_seed_closed_loop_drop,
        },
        "candidates": ranked,
        "mainline_selected": eligible[0] if eligible else None,
        "pareto_highest_closed_loop": ranked[0],
        "gate_conditioned_results_are_external_ablation": True,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
