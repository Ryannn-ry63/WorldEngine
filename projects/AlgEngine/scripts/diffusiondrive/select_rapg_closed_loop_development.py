#!/usr/bin/env python3
"""Choose the locked RAPG variant on CL-dev, prioritizing reactive PDM."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def labeled_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_metrics(path: Path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if set(payload.get("seeds", {})) != {"0", "1", "2"}:
        raise RuntimeError(f"CL-dev requires paired seeds 0/1/2: {path}")
    required = {"cl_nonreactive_pdm", "cl_reactive_pdm", "success_rate"}
    seeds = {}
    for seed, row in payload["seeds"].items():
        if not required.issubset(row):
            raise RuntimeError(f"CL-dev metrics missing fields: {path} seed {seed}")
        seeds[seed] = {key: float(row[key]) for key in required}
    return path, seeds


def mean(seeds, key):
    return statistics.fmean(row[key] for row in seeds.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cqr-baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=labeled_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-success-drop", type=float, default=0.02)
    args = parser.parse_args()
    baseline_path, baseline = load_metrics(args.cqr_baseline)
    minimum_success = mean(baseline, "success_rate") - args.maximum_success_drop
    candidates = []
    labels = set()
    for label, path_arg in args.candidate:
        if label in labels:
            raise RuntimeError(f"duplicate candidate label: {label}")
        labels.add(label)
        path, seeds = load_metrics(path_arg)
        row = {
            "label": label,
            "metrics": str(path),
            "mean_cl_nonreactive_pdm": mean(seeds, "cl_nonreactive_pdm"),
            "mean_cl_reactive_pdm": mean(seeds, "cl_reactive_pdm"),
            "mean_success_rate": mean(seeds, "success_rate"),
        }
        row["success_guardrail"] = row["mean_success_rate"] >= minimum_success
        candidates.append(row)
    eligible = [row for row in candidates if row["success_guardrail"]]
    eligible.sort(key=lambda row: row["mean_cl_reactive_pdm"], reverse=True)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "rapg_closed_loop_development_selection_v1",
        "selection_metric": "three_seed_mean_cl_reactive_pdm",
        "cqr_is_diagnostic_baseline_not_training_objective": True,
        "cqr_baseline": str(baseline_path),
        "minimum_success_rate": minimum_success,
        "candidates": candidates,
        "selected": eligible[0] if eligible else None,
        "method_locked_after_selection": bool(eligible),
        "certification_consumed": False,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
