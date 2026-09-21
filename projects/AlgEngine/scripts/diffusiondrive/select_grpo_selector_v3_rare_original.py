#!/usr/bin/env python3
"""Select a rare-original V3 checkpoint using only log-disjoint rare development tokens."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import grpo_selector_v3_cached_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=56)
    parser.add_argument("--minimum-worst-seed-gain", type=float, default=-0.001)
    parser.add_argument(
        "--require-positive-gate",
        action="store_true",
        help="fail instead of selecting the best audited fallback when the gate is empty",
    )
    args = parser.parse_args()

    evaluations = sorted(
        args.trial_root.expanduser().resolve().glob(
            "*/development_evaluation.json"
        )
    )
    if not evaluations:
        raise RuntimeError("no rare-original development evaluations found")
    candidates = []
    for path in evaluations:
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "PASS"
            or payload.get("split") != "development"
            or not payload.get("rare_only")
        ):
            raise RuntimeError(f"invalid development evaluation: {path}")
        for checkpoint in payload["checkpoints"]:
            by_seed = checkpoint["rare_metrics_by_noise_seed"]
            gains = [
                float(by_seed[str(seed)]["top1_reward_gain"])
                for seed in (3, 4, 5)
            ]
            mean_gain = statistics.fmean(gains)
            deviation = statistics.pstdev(gains)
            candidates.append(
                {
                    "evaluation": str(path),
                    "evaluation_sha256": common.sha256_file(path),
                    "trial": path.parent.name,
                    "epoch": int(checkpoint["epoch"]),
                    "scene_selector_state": checkpoint["scene_selector_state"],
                    "scene_selector_state_sha256": checkpoint[
                        "scene_selector_state_sha256"
                    ],
                    "temperature": float(checkpoint["temperature"]),
                    "learning_rate": float(checkpoint["learning_rate"]),
                    "kl_weight": float(checkpoint["kl_weight"]),
                    "train_seed": int(checkpoint["train_seed"]),
                    "rare_development_metrics": checkpoint["rare_metrics"],
                    "rare_development_seed_gains": gains,
                    "rare_vote_strata": checkpoint["rare_vote_strata"],
                    "mean_gain": mean_gain,
                    "worst_seed_gain": min(gains),
                    "stability_score": mean_gain - 0.5 * deviation,
                }
            )
    if len(candidates) != args.expected_candidates:
        raise RuntimeError(
            f"rare-original candidate count {len(candidates)} "
            f"!= expected {args.expected_candidates}"
        )
    eligible = [
        row
        for row in candidates
        if row["mean_gain"] > 0.0
        and row["worst_seed_gain"] >= args.minimum_worst_seed_gain
    ]
    development_gate_passed = bool(eligible)
    if not eligible and args.require_positive_gate:
        raise RuntimeError("no rare-original checkpoint passed development gate")
    ranked = eligible if eligible else candidates
    ranked.sort(
        key=lambda row: (row["stability_score"], row["mean_gain"], -row["epoch"]),
        reverse=True,
    )
    selected = ranked[0]
    state_path = Path(selected["scene_selector_state"]).resolve()
    if common.sha256_file(state_path) != selected["scene_selector_state_sha256"]:
        raise RuntimeError("selected rare-original selector SHA256 drifted")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_original_development_selection_v1",
        "selection_data": "log_disjoint_original_navtrain_rare_tokens_only",
        "certification_consumed": False,
        "ranking_rule": "mean_gain_minus_half_noise_seed_standard_deviation",
        "minimum_worst_seed_gain": args.minimum_worst_seed_gain,
        "num_candidates": len(candidates),
        "num_eligible": len(eligible),
        "development_gate_passed": development_gate_passed,
        "fallback_policy": (
            None
            if development_gate_passed
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
