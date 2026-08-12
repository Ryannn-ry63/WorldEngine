#!/usr/bin/env python3
"""Summarize V3 input ablations without selecting a formal checkpoint."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for report_path in sorted(args.root.expanduser().resolve().glob("*/report.json")):
        report = json.loads(report_path.read_text())
        if report.get("status") != "PASS":
            raise RuntimeError(f"failed ablation report: {report_path}")
        final = report["checkpoints"][-1]
        rows[report["ablation"]] = {
            "report": str(report_path),
            "epoch": final["epoch"],
            "development_metrics": final["development_metrics"],
            "development_by_noise_seed": final["development_by_noise_seed"],
        }
    expected = {"full", "feature_only", "feature_geometry", "feature_geometry_route"}
    if set(rows) != expected:
        raise RuntimeError(f"V3 ablation coverage {sorted(rows)} != {sorted(expected)}")
    full_gain = rows["full"]["development_metrics"]["top1_reward_gain"]
    feature_gain = rows["feature_only"]["development_metrics"]["top1_reward_gain"]
    payload = {
        "schema_version": 3,
        "status": "PASS",
        "purpose": "representation_ablation_only_not_model_selection",
        "rows": rows,
        "full_minus_feature_only_top1_gain": full_gain - feature_gain,
        "scene_context_helped_at_probe_budget": full_gain > feature_gain,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
