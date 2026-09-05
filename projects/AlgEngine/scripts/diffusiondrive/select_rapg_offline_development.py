#!/usr/bin/env python3
"""Shortlist RAPG variants on development strata without reading certification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpo_selector_v3_cached_common as common


STRATA = ("common", "real_rare", "synthetic")


def labeled_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def load_evaluation(path: Path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("split") != "development"
        or payload.get("certification_consumed")
        or not payload.get("checkpoints")
    ):
        raise RuntimeError(f"invalid development evaluation: {path}")
    return path, payload["checkpoints"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-b00", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=labeled_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shortlist-size", type=int, default=3)
    parser.add_argument("--maximum-stratum-drop", type=float, default=0.005)
    parser.add_argument("--maximum-common-degraded-increase", type=float, default=0.02)
    args = parser.parse_args()
    baseline_path, baseline_rows = load_evaluation(args.baseline_b00)
    if len(baseline_rows) != 1:
        raise RuntimeError("B00 evaluation must contain exactly one checkpoint")
    baseline = baseline_rows[0]

    rows = []
    input_labels = set()
    labels = set()
    for label, candidate_path_arg in args.candidate:
        if label in input_labels:
            raise RuntimeError(f"duplicate candidate label: {label}")
        input_labels.add(label)
        candidate_path, candidates = load_evaluation(candidate_path_arg)
        for candidate in candidates:
            candidate_label = label
            if len(candidates) > 1:
                arm = Path(candidate["checkpoint"]).parent.name
                candidate_label = f"{label}:{arm}"
            if candidate_label in labels:
                raise RuntimeError(f"duplicate candidate label: {candidate_label}")
            labels.add(candidate_label)
            stratum_deltas = {
                name: candidate["strata"][name]["current_reward"]
                - baseline["strata"][name]["current_reward"]
                for name in STRATA
            }
            gates = {
                "positive_gain_in_every_stratum": all(
                    candidate["strata"][name]["top1_reward_gain"] > 0.0
                    for name in STRATA
                ),
                "no_stratum_drop_vs_b00": min(stratum_deltas.values())
                >= -args.maximum_stratum_drop,
                "common_degraded_fraction_guardrail": candidate["strata"]["common"][
                    "degraded_fraction"
                ]
                <= baseline["strata"]["common"]["degraded_fraction"]
                + args.maximum_common_degraded_increase,
            }
            rows.append(
                {
                    "label": candidate_label,
                    "evaluation": str(candidate_path),
                    "evaluation_sha256": common.sha256_file(candidate_path),
                    "checkpoint": candidate,
                    "stratum_selected_reward_deltas_vs_b00": stratum_deltas,
                    "equal_weight_selected_reward_delta_vs_b00": candidate[
                        "equal_weight_selected_reward"
                    ]
                    - baseline["equal_weight_selected_reward"],
                    "gates": gates,
                    "eligible": all(gates.values()),
                }
            )
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(
        key=lambda row: row["checkpoint"]["equal_weight_selected_reward"],
        reverse=True,
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "rapg_offline_development_shortlist_v1",
        "development_only": True,
        "certification_consumed": False,
        "baseline_b00": str(baseline_path),
        "baseline_b00_sha256": common.sha256_file(baseline_path),
        "thresholds": {
            "maximum_stratum_selected_reward_drop": args.maximum_stratum_drop,
            "maximum_common_degraded_fraction_increase": args.maximum_common_degraded_increase,
        },
        "candidates": rows,
        "shortlist": eligible[: args.shortlist_size],
        "development_gate_passed": bool(eligible),
        "closed_loop_development_required": True,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
