#!/usr/bin/env python3
"""Apply the frozen 0.80 RAPG formal gate against the CQR diagnostic baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "navtest_pdm",
    "rare_pdm",
    "cl_nonreactive_pdm",
    "cl_reactive_pdm",
    "success_rate",
)


def load(path: Path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if set(payload.get("seeds", {})) != {"0", "1", "2"}:
        raise RuntimeError(f"formal metrics require paired seeds 0/1/2: {path}")
    seeds = {}
    for seed, row in payload["seeds"].items():
        if not set(METRICS).issubset(row):
            raise RuntimeError(f"formal metrics missing fields: {path} seed {seed}")
        seeds[seed] = {key: float(row[key]) for key in METRICS}
    return path, seeds


def average(values):
    return statistics.fmean(values)


def cl(row):
    return average((row["cl_nonreactive_pdm"], row["cl_reactive_pdm"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cqr-baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-cl-mean", type=float, default=0.80)
    parser.add_argument("--minimum-cl-gain-vs-cqr", type=float, default=0.005)
    parser.add_argument("--minimum-positive-seeds", type=int, default=2)
    parser.add_argument("--maximum-seed-cl-drop", type=float, default=0.02)
    parser.add_argument("--minimum-success", type=float, default=0.862)
    parser.add_argument("--minimum-navtest", type=float, default=0.846)
    parser.add_argument("--minimum-rare", type=float, default=0.607)
    args = parser.parse_args()
    baseline_path, baseline = load(args.cqr_baseline)
    candidate_path, candidate = load(args.candidate)
    paired = {
        seed: {
            "candidate_cl_mean": cl(candidate[seed]),
            "cqr_cl_mean": cl(baseline[seed]),
            "delta_cl_mean": cl(candidate[seed]) - cl(baseline[seed]),
        }
        for seed in sorted(candidate)
    }
    mean_metrics = {
        key: average([row[key] for row in candidate.values()]) for key in METRICS
    }
    candidate_cl = average([row["candidate_cl_mean"] for row in paired.values()])
    cqr_cl = average([row["cqr_cl_mean"] for row in paired.values()])
    deltas = [row["delta_cl_mean"] for row in paired.values()]
    gates = {
        "cl_mean_at_least_0_80": candidate_cl >= args.minimum_cl_mean,
        "cl_gain_vs_cqr": candidate_cl - cqr_cl >= args.minimum_cl_gain_vs_cqr,
        "paired_seed_support": sum(value >= 0.0 for value in deltas)
        >= args.minimum_positive_seeds,
        "no_catastrophic_seed": min(deltas) >= -args.maximum_seed_cl_drop,
        "success_guardrail": mean_metrics["success_rate"] >= args.minimum_success,
        "navtest_guardrail": mean_metrics["navtest_pdm"] >= args.minimum_navtest,
        "rare_guardrail": mean_metrics["rare_pdm"] >= args.minimum_rare,
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "rapg_formal_promotion_gate_v1",
        "main_objective_uses_official_scalar_pdm_only": True,
        "cqr_position": "diagnostic_pareto_baseline",
        "cqr_baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "paired_seeds": paired,
        "candidate_mean_metrics": mean_metrics,
        "candidate_cl_mean": candidate_cl,
        "cqr_cl_mean": cqr_cl,
        "cl_gain_vs_cqr": candidate_cl - cqr_cl,
        "thresholds": {
            "minimum_cl_mean": args.minimum_cl_mean,
            "minimum_cl_gain_vs_cqr": args.minimum_cl_gain_vs_cqr,
            "minimum_nonnegative_paired_seeds": args.minimum_positive_seeds,
            "maximum_single_seed_cl_drop": args.maximum_seed_cl_drop,
            "minimum_success": args.minimum_success,
            "minimum_navtest": args.minimum_navtest,
            "minimum_rare": args.minimum_rare,
        },
        "gates": gates,
        "promoted": all(gates.values()),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "promoted": report["promoted"]}, sort_keys=True))


if __name__ == "__main__":
    main()
