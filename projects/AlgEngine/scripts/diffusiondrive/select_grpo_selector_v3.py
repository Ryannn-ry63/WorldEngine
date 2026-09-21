#!/usr/bin/env python3
"""Select one V3 checkpoint using only scene-disjoint development caches."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import grpo_selector_v3_cached_common as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-worst-seed-gain", type=float, default=-0.001)
    parser.add_argument(
        "--method-name", default="scene_conditioned_exact_group_grpo"
    )
    parser.add_argument("--minimum-delta-no-at-fault-collisions", type=float, default=float("-inf"))
    parser.add_argument("--minimum-delta-drivable-area-compliance", type=float, default=float("-inf"))
    parser.add_argument("--minimum-delta-time-to-collision", type=float, default=float("-inf"))
    args = parser.parse_args()
    trial_root = args.trial_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    reports = sorted(trial_root.glob("*/report.json"))
    if not reports:
        raise RuntimeError(f"no V3 trial reports found under {trial_root}")

    candidates = []
    for report_path in reports:
        report = json.loads(report_path.read_text())
        if report.get("status") != "PASS" or report.get("schema_version") != 3:
            raise RuntimeError(f"invalid V3 trial report: {report_path}")
        if report.get("method") != args.method_name:
            continue
        if report.get("ablation") != "full":
            continue
        for checkpoint in report["checkpoints"]:
            by_seed = checkpoint["development_by_noise_seed"]
            gains = [float(by_seed[str(seed)]["top1_reward_gain"]) for seed in (3, 4, 5)]
            mean_gain = statistics.fmean(gains)
            standard_deviation = statistics.pstdev(gains)
            candidates.append(
                {
                    "report": str(report_path),
                    "trial": report_path.parent.name,
                    "epoch": int(checkpoint["epoch"]),
                    "scene_selector_state": checkpoint["scene_selector_state"],
                    "scene_selector_state_sha256": checkpoint[
                        "scene_selector_state_sha256"
                    ],
                    "temperature": float(report["temperature"]),
                    "learning_rate": float(report["learning_rate"]),
                    "kl_weight": float(report["kl_weight"]),
                    "train_seed": int(report["train_seed"]),
                    "development_metrics": checkpoint["development_metrics"],
                    "development_seed_gains": gains,
                    "mean_gain": mean_gain,
                    "worst_seed_gain": min(gains),
                    "stability_score": mean_gain - 0.5 * standard_deviation,
                }
            )
    eligible = [
        row
        for row in candidates
        if row["mean_gain"] > 0.0
        and row["worst_seed_gain"] >= args.minimum_worst_seed_gain
        and row["development_metrics"]["delta_no_at_fault_collisions"]
        >= args.minimum_delta_no_at_fault_collisions
        and row["development_metrics"]["delta_drivable_area_compliance"]
        >= args.minimum_delta_drivable_area_compliance
        and row["development_metrics"]["delta_time_to_collision_within_bound"]
        >= args.minimum_delta_time_to_collision
    ]
    if not eligible:
        raise RuntimeError("no V3 checkpoint passed the development-only stability gate")
    eligible.sort(
        key=lambda row: (row["stability_score"], row["mean_gain"], -row["epoch"]),
        reverse=True,
    )
    selected = eligible[0]
    state_path = Path(selected["scene_selector_state"]).resolve()
    if common.sha256_file(state_path) != selected["scene_selector_state_sha256"]:
        raise RuntimeError("selected V3 scene-selector SHA256 drifted")
    payload = {
        "schema_version": 3,
        "status": "PASS",
        "method": args.method_name,
        "selection_data": "navtrain_scene_disjoint_development_only",
        "certification_consumed": False,
        "ranking_rule": "mean_gain_minus_half_noise_seed_standard_deviation",
        "minimum_worst_seed_gain": args.minimum_worst_seed_gain,
        "component_floors": {
            "delta_no_at_fault_collisions": args.minimum_delta_no_at_fault_collisions,
            "delta_drivable_area_compliance": args.minimum_delta_drivable_area_compliance,
            "delta_time_to_collision_within_bound": args.minimum_delta_time_to_collision,
        },
        "num_candidates": len(candidates),
        "num_eligible": len(eligible),
        "selected": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
