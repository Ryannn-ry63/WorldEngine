#!/usr/bin/env python3
"""Fail closed unless a V2 checkpoint changes exactly the plan-cls MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import grpo_selector_v2_rare_common as common
from materialize_grpo_selector_v2_rare import checkpoint_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = args.baseline.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if common.sha256_file(baseline_path) != common.BASELINE_SHA256:
        raise RuntimeError("baseline SHA256 mismatch")
    baseline, _ = checkpoint_state(torch.load(baseline_path, map_location="cpu"))
    checkpoint, _ = checkpoint_state(torch.load(checkpoint_path, map_location="cpu"))
    module_prefix = (
        "module." if any(key.startswith("module.") for key in baseline) else ""
    )
    relative = set(common.Selector().state_dict())
    allowed_changed = {module_prefix + common.SELECTOR_PREFIX + key for key in relative}
    allowed_added = {module_prefix + common.REFERENCE_PREFIX + key for key in relative}
    removed = set(baseline) - set(checkpoint)
    added = set(checkpoint) - set(baseline)
    changed = {
        key
        for key in set(baseline).intersection(checkpoint)
        if not torch.equal(baseline[key], checkpoint[key])
    }
    if removed:
        raise RuntimeError(f"V2 checkpoint removed tensors: {sorted(removed)}")
    if added != allowed_added:
        raise RuntimeError(
            "V2 checkpoint added unexpected tensors: "
            f"missing={sorted(allowed_added - added)} extra={sorted(added - allowed_added)}"
        )
    if changed != allowed_changed:
        raise RuntimeError(
            "V2 checkpoint changed tensors outside/excluding exact selector: "
            f"missing={sorted(allowed_changed - changed)} "
            f"extra={sorted(changed - allowed_changed)}"
        )
    for key in sorted(relative):
        baseline_key = module_prefix + common.SELECTOR_PREFIX + key
        reference_key = module_prefix + common.REFERENCE_PREFIX + key
        if not torch.equal(checkpoint[reference_key], baseline[baseline_key]):
            raise RuntimeError(f"frozen V2 reference selector drifted: {key}")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "baseline": str(baseline_path),
        "baseline_sha256": common.BASELINE_SHA256,
        "baseline_tensor_count": len(baseline),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": common.sha256_file(checkpoint_path),
        "checkpoint_tensor_count": len(checkpoint),
        "changed_tensor_count": len(changed),
        "changed_tensors": sorted(changed),
        "added_reference_tensor_count": len(added),
        "added_reference_tensors": sorted(added),
        "forbidden_changed_tensor_count": 0,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
