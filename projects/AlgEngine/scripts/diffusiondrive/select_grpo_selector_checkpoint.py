#!/usr/bin/env python3
"""Select the formal LR/epoch using only scene-held-out navtrain calibration."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REQUIRED_NOISE_SEEDS = (0, 1, 2)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--split-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nc-floor", type=float, default=-0.001)
    parser.add_argument("--dac-floor", type=float, default=-0.001)
    parser.add_argument("--tie-tolerance", type=float, default=0.0002)
    return parser.parse_args()


def main():
    args = parse_args()
    split_audit_path = args.split_audit.expanduser().resolve()
    if not split_audit_path.is_file():
        raise FileNotFoundError(split_audit_path)
    split_audit = json.loads(split_audit_path.read_text())
    if split_audit.get("status") != "PASS":
        raise RuntimeError("navtrain split audit did not pass")

    groups = defaultdict(list)
    for report_path_arg in args.reports:
        report_path = report_path_arg.expanduser().resolve()
        report = json.loads(report_path.read_text())
        if report.get("status") != "PASS":
            raise RuntimeError(f"calibration report failed: {report_path}")
        if int(report["train_seed"]) != 0:
            raise RuntimeError("LR/epoch selection may use train seed 0 only")
        key = (
            report["checkpoint_sha256"],
            float(report["learning_rate"]),
            int(report["epoch"]),
        )
        report["_path"] = str(report_path)
        groups[key].append(report)

    candidates = []
    metric_keys = (
        "current_reward",
        "reference_reward",
        "top1_reward_gain",
        "current_expected_reward",
        "reference_expected_reward",
        "expected_reward_gain",
        "current_oracle_match",
        "reference_oracle_match",
        "selection_disagreement",
        "delta_no_at_fault_collisions",
        "delta_drivable_area_compliance",
        "delta_ego_progress",
        "delta_time_to_collision_within_bound",
        "delta_comfort",
        "delta_driving_direction_compliance",
    )
    for (checkpoint_sha, learning_rate, epoch), reports in groups.items():
        noise_seeds = tuple(sorted(int(row["noise_seed"]) for row in reports))
        if noise_seeds != REQUIRED_NOISE_SEEDS:
            raise RuntimeError(
                f"checkpoint {checkpoint_sha} has noise seeds {noise_seeds}, "
                f"expected {REQUIRED_NOISE_SEEDS}"
            )
        if len({row["checkpoint"] for row in reports}) != 1:
            raise RuntimeError("checkpoint path drifted within a calibration group")
        metrics = {
            key: mean_std([row["metrics"][key] for row in reports])
            for key in metric_keys
        }
        eligible = (
            metrics["delta_no_at_fault_collisions"]["mean"] >= args.nc_floor
            and metrics["delta_drivable_area_compliance"]["mean"] >= args.dac_floor
        )
        candidates.append(
            {
                "checkpoint": reports[0]["checkpoint"],
                "checkpoint_sha256": checkpoint_sha,
                "learning_rate": learning_rate,
                "epoch": epoch,
                "train_seed": 0,
                "noise_seeds": list(noise_seeds),
                "eligible": eligible,
                "metrics": metrics,
                "reports": [row["_path"] for row in reports],
            }
        )
    if not candidates:
        raise RuntimeError("no complete calibration candidates")

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    pool = eligible or candidates
    best_gain = max(
        candidate["metrics"]["top1_reward_gain"]["mean"] for candidate in pool
    )
    tied = [
        candidate
        for candidate in pool
        if candidate["metrics"]["top1_reward_gain"]["mean"]
        >= best_gain - args.tie_tolerance
    ]
    selected = min(
        tied,
        key=lambda candidate: (
            candidate["epoch"],
            candidate["learning_rate"],
        ),
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "PASS" if eligible else "SAFETY_GATE_FAIL",
        "method": "e2e_diffusiondrive_grpo_selector",
        "selection_data": "navtrain_scene_disjoint_calibration_only",
        "selection_train_seed": 0,
        "required_noise_seeds": list(REQUIRED_NOISE_SEEDS),
        "selection_rule": {
            "primary": "mean_paired_top1_pdm_gain",
            "nc_delta_floor": args.nc_floor,
            "dac_delta_floor": args.dac_floor,
            "tie_tolerance": args.tie_tolerance,
            "tie_break": ["earlier_epoch", "lower_learning_rate"],
        },
        "split_audit": str(split_audit_path),
        "split_audit_sha256": sha256_file(split_audit_path),
        "selected": selected,
        "num_candidates": len(candidates),
        "num_eligible_candidates": len(eligible),
        "candidates": sorted(
            candidates,
            key=lambda candidate: (
                candidate["learning_rate"],
                candidate["epoch"],
            ),
        ),
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    if not eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
