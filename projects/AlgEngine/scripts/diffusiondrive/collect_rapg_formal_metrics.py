#!/usr/bin/env python3
"""Collect three RAPG four-block summaries into the frozen promotion schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpo_selector_v3_cached_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = {}
    checkpoint_shas = {}
    inputs = []
    for argument in args.summary:
        path = argument.expanduser().resolve()
        row = json.loads(path.read_text())
        if row.get("status") != "PASS" or row.get("schema_version") != 1:
            raise RuntimeError(f"invalid formal summary: {path}")
        seed = str(int(row["eval_seed"]))
        if seed in seeds:
            raise RuntimeError(f"duplicate formal seed: {seed}")
        metrics = row["metrics"]
        nonreactive = metrics["closedloop_nonreactive"]
        reactive = metrics["closedloop_reactive"]
        seeds[seed] = {
            "navtest_pdm": float(metrics["openloop_navtest"]["score"]),
            "rare_pdm": float(metrics["openloop_failures"]["score"]),
            "cl_nonreactive_pdm": float(nonreactive["score"]),
            "cl_reactive_pdm": float(reactive["score"]),
            "success_rate": float(metrics["success_rate"]),
            "no_at_fault_collisions": float(
                reactive["no_at_fault_collisions"]
            ),
            "drivable_area_compliance": float(
                reactive["drivable_area_compliance"]
            ),
        }
        checkpoint_shas[seed] = str(row["checkpoint_sha256"])
        inputs.append(
            {
                "eval_seed": int(seed),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "path": str(path),
                "sha256": common.sha256_file(path),
            }
        )
    if set(seeds) != {"0", "1", "2"}:
        raise RuntimeError(f"expected seeds 0/1/2, got {sorted(seeds)}")
    if len(set(checkpoint_shas.values())) != 3:
        raise RuntimeError("formal metrics require independently trained checkpoints")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "rapg_three_seed_four_block_formal_metrics_v1",
        "rare_definition": "openloop_navtest_failures_official_pdm",
        "independent_training_seeds": [0, 1, 2],
        "checkpoint_sha256_by_seed": checkpoint_shas,
        "inputs": inputs,
        "seeds": seeds,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
