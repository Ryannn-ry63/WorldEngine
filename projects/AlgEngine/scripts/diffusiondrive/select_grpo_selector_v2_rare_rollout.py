#!/usr/bin/env python3
"""Select V2 rollout hyperparameters on development logs only."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import grpo_selector_v2_rare_common as common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=56)
    parser.add_argument("--minimum-worst-seed-gain", type=float, default=-0.001)
    parser.add_argument("--require-positive-gate", action="store_true")
    args = parser.parse_args()
    evaluations = sorted(
        args.trial_root.expanduser().resolve().glob("*/development_evaluation.json")
    )
    if not evaluations:
        raise RuntimeError("no V2 rollout development evaluations found")
    candidates = []
    for path in evaluations:
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "PASS"
            or payload.get("method")
            != "diffusiondrive_v2_rare_rollout_checkpoint_evaluation_v1"
            or payload.get("split") != "development"
            or not payload.get("hard_only")
        ):
            raise RuntimeError(f"invalid V2 rollout evaluation: {path}")
        for checkpoint in payload["checkpoints"]:
            by_seed = checkpoint["hard_metrics_by_noise_seed"]
            seed_ids = sorted(by_seed, key=int)
            gains = [float(by_seed[seed]["top1_reward_gain"]) for seed in seed_ids]
            mean_gain = statistics.fmean(gains)
            deviation = statistics.pstdev(gains)
            candidates.append(
                {
                    "evaluation": str(path),
                    "evaluation_sha256": common.sha256_file(path),
                    "trial": path.parent.name,
                    "epoch": int(checkpoint["epoch"]),
                    "selector_state": checkpoint["selector_state"],
                    "selector_state_sha256": checkpoint["selector_state_sha256"],
                    "temperature": float(checkpoint["temperature"]),
                    "learning_rate": float(checkpoint["learning_rate"]),
                    "kl_weight": float(checkpoint["kl_weight"]),
                    "train_seed": int(checkpoint["train_seed"]),
                    "hard_development_metrics": checkpoint["hard_metrics"],
                    "hard_development_seed_gains": gains,
                    "hard_kind_metrics_by_noise_seed": checkpoint[
                        "hard_kind_metrics_by_noise_seed"
                    ],
                    "mean_gain": mean_gain,
                    "worst_seed_gain": min(gains),
                    "stability_score": mean_gain - 0.5 * deviation,
                }
            )
    if len(candidates) != args.expected_candidates:
        raise RuntimeError(
            f"V2 rollout candidates {len(candidates)} != {args.expected_candidates}"
        )
    eligible = [
        row
        for row in candidates
        if row["mean_gain"] > 0.0
        and row["worst_seed_gain"] >= args.minimum_worst_seed_gain
    ]
    if not eligible and args.require_positive_gate:
        raise RuntimeError("no V2 rollout candidate passed development gate")
    ranked = eligible if eligible else candidates
    ranked.sort(
        key=lambda row: (row["stability_score"], row["mean_gain"], -row["epoch"]),
        reverse=True,
    )
    selected = ranked[0]
    state_path = Path(selected["selector_state"]).resolve()
    if common.sha256_file(state_path) != selected["selector_state_sha256"]:
        raise RuntimeError("selected V2 rollout selector SHA256 drifted")
    gate_passed = bool(eligible)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v2_rare_rollout_development_selection_v1",
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "selection_data": "log_disjoint_rollout_hard_development_only",
        "certification_consumed": False,
        "search_budget": {
            "trials": len(evaluations),
            "checkpoints_per_trial": len(candidates) // len(evaluations),
            "candidates": len(candidates),
            "identical_to_v3_rare_original": True,
        },
        "ranking_rule": "mean_gain_minus_half_noise_seed_standard_deviation",
        "minimum_worst_seed_gain": args.minimum_worst_seed_gain,
        "num_candidates": len(candidates),
        "num_eligible": len(eligible),
        "development_gate_passed": gate_passed,
        "fallback_policy": (
            None
            if gate_passed
            else "best development stability score; formal results remain descriptive"
        ),
        "selected": selected,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
