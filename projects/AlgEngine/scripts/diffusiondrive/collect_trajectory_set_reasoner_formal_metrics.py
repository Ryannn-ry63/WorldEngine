#!/usr/bin/env python3
"""Collect three audited four-block summaries into the promotion schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = {}
    checkpoint_sha = None
    inputs = []
    for argument in args.summary:
        path = argument.expanduser().resolve()
        row = json.loads(path.read_text())
        if row.get("status") != "PASS" or row.get("schema_version") != 1:
            raise RuntimeError(f"invalid formal summary: {path}")
        seed = str(int(row["eval_seed"]))
        if seed in seeds:
            raise RuntimeError(f"duplicate formal eval seed: {seed}")
        if checkpoint_sha is None:
            checkpoint_sha = row["checkpoint_sha256"]
        elif checkpoint_sha != row["checkpoint_sha256"]:
            raise RuntimeError("formal summaries use different checkpoints")
        metrics = row["metrics"]
        nonreactive = metrics["closedloop_nonreactive"]
        reactive = metrics["closedloop_reactive"]
        seeds[seed] = {
            "cl_nonreactive_pdm": float(nonreactive["score"]),
            "cl_reactive_pdm": float(reactive["score"]),
            "success_rate": float(metrics["success_rate"]),
            "no_at_fault_collisions": float(
                reactive["no_at_fault_collisions"]
            ),
            "drivable_area_compliance": float(
                reactive["drivable_area_compliance"]
            ),
            "navtest_pdm": float(metrics["openloop_navtest"]["score"]),
        }
        inputs.append({"path": str(path), "sha256": sha256_file(path)})
    if set(seeds) != {"0", "1", "2"}:
        raise RuntimeError(f"expected formal seeds 0/1/2, got {sorted(seeds)}")
    output_payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "trajectory_set_reasoner_three_seed_formal_metrics_v1",
        "checkpoint_sha256": checkpoint_sha,
        "inputs": inputs,
        "seeds": seeds,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
