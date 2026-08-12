#!/usr/bin/env python3
"""Prove a V3 checkpoint changes only the new DiffusionDrive scene selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import grpo_selector_v3_cached_common as common


PREFIX = "planning_head.scene_selector."


def extract(checkpoint):
    for key in ("state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_path = args.baseline.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    baseline = extract(torch.load(baseline_path, map_location="cpu"))
    checkpoint = extract(torch.load(checkpoint_path, map_location="cpu"))
    missing = sorted(set(baseline) - set(checkpoint))
    if missing:
        raise RuntimeError("V3 checkpoint dropped baseline tensors")
    changed = []
    for key, value in baseline.items():
        if not torch.equal(value, checkpoint[key]):
            changed.append(key)
    if changed:
        raise RuntimeError("V3 modified frozen baseline tensors: " + ", ".join(changed[:10]))
    added = sorted(set(checkpoint) - set(baseline))
    if not added or not all(key.startswith(PREFIX) for key in added):
        raise RuntimeError("V3 checkpoint contains forbidden added tensors")
    if not all(torch.isfinite(checkpoint[key]).all() for key in added):
        raise RuntimeError("V3 scene-selector checkpoint contains non-finite tensors")
    report = {
        "schema_version": 3,
        "status": "PASS",
        "baseline": str(baseline_path),
        "baseline_sha256": common.sha256_file(baseline_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": common.sha256_file(checkpoint_path),
        "baseline_tensor_count": len(baseline),
        "checkpoint_tensor_count": len(checkpoint),
        "changed_baseline_tensor_count": 0,
        "scene_selector_tensor_count": len(added),
        "scene_selector_tensors": added,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
