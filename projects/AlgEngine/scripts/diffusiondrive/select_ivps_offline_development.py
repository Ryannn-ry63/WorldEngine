#!/usr/bin/env python3
"""Apply the frozen IVPS development gate without reading certification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpo_selector_v3_cached_common as common


def labeled_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def load_development(path):
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
    parser.add_argument("--proposal-baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=labeled_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shortlist-size", type=int, default=2)
    args = parser.parse_args()

    baseline_path, baseline_rows = load_development(args.proposal_baseline)
    if len(baseline_rows) != 1:
        raise RuntimeError("proposal baseline must contain exactly one checkpoint")
    baseline = baseline_rows[0]
    rows = []
    seen = set()
    input_labels = set()
    for input_label, raw_path in args.candidate:
        if input_label in input_labels:
            raise RuntimeError(f"duplicate IVPS input label: {input_label}")
        input_labels.add(input_label)
        path, candidates = load_development(raw_path)
        for candidate in candidates:
            label = input_label
            if len(candidates) > 1:
                label = f"{input_label}:{Path(candidate['checkpoint']).parent.name}"
            if label in seen:
                raise RuntimeError(f"duplicate IVPS candidate label: {label}")
            seen.add(label)
            if candidate.get("selector_architecture") != (
                "incumbent_verified_preference"
            ):
                raise RuntimeError(f"candidate is not IVPS: {label}")
            common_row = candidate["strata"]["common"]
            coverage = float(common_row["override_coverage"])
            precision = float(common_row["override_precision"])
            gates = {
                "equal_weight_beats_proposal_by_005": candidate[
                    "equal_weight_selected_reward"
                ]
                >= baseline["equal_weight_selected_reward"] + 0.005,
                "common_preserves_reference_within_005": common_row[
                    "current_reward"
                ]
                >= common_row["reference_reward"] - 0.005,
                "rare_preserves_proposal_within_020": candidate["strata"][
                    "real_rare"
                ]["current_reward"]
                >= baseline["strata"]["real_rare"]["current_reward"] - 0.020,
                "synthetic_preserves_proposal_within_020": candidate["strata"][
                    "synthetic"
                ]["current_reward"]
                >= baseline["strata"]["synthetic"]["current_reward"] - 0.020,
                "common_degraded_fraction_at_most_010": common_row[
                    "degraded_fraction"
                ]
                <= 0.10,
                "common_override_precision_at_least_050": (
                    not coverage or precision >= 0.50
                ),
            }
            rows.append(
                {
                    "label": label,
                    "evaluation": str(path),
                    "evaluation_sha256": common.sha256_file(path),
                    "checkpoint": candidate,
                    "equal_weight_delta_vs_proposal": candidate[
                        "equal_weight_selected_reward"
                    ]
                    - baseline["equal_weight_selected_reward"],
                    "gates": gates,
                    "eligible": all(gates.values()),
                }
            )
    eligible = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: row["checkpoint"]["equal_weight_selected_reward"],
        reverse=True,
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "ivps_frozen_offline_development_gate_v1",
        "development_only": True,
        "certification_consumed": False,
        "proposal_baseline": str(baseline_path),
        "proposal_baseline_sha256": common.sha256_file(baseline_path),
        "thresholds": {
            "minimum_equal_weight_gain_vs_proposal": 0.005,
            "maximum_common_drop_vs_reference": 0.005,
            "maximum_rare_or_synthetic_drop_vs_proposal": 0.020,
            "maximum_common_degraded_fraction": 0.10,
            "minimum_common_override_precision": 0.50,
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
