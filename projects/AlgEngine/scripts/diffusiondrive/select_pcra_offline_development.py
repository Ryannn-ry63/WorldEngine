#!/usr/bin/env python3
"""Apply the frozen PCRA offline development gate without reading certification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpo_selector_v3_cached_common as common


PROPOSAL32_SHA256 = "562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6"
PCRA_ARCHITECTURE = "proposal_conditioned_regret_arbitration"


def labeled_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("input label must not be empty")
    return label, Path(raw_path)


def load_one(path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("split") != "development"
        or payload.get("certification_consumed")
        or len(payload.get("checkpoints", ())) != 1
    ):
        raise RuntimeError(f"invalid single-checkpoint development evaluation: {path}")
    return path, payload["checkpoints"][0]


def load_labeled(inputs):
    rows = []
    seen = set()
    for label, raw_path in inputs:
        if label in seen:
            raise RuntimeError(f"duplicate PCRA label: {label}")
        seen.add(label)
        path, checkpoint = load_one(raw_path)
        rows.append(
            {
                "label": label,
                "evaluation": str(path),
                "evaluation_sha256": common.sha256_file(path),
                "checkpoint": checkpoint,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-32", type=Path, required=True)
    parser.add_argument("--proposal-48", type=Path, required=True)
    parser.add_argument("--control", action="append", type=labeled_path, default=[])
    parser.add_argument(
        "--candidate", action="append", type=labeled_path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    proposal32_path, proposal32 = load_one(args.proposal_32)
    proposal48_path, proposal48 = load_one(args.proposal_48)
    proposals = (proposal32, proposal48)
    proposal_equal_floor = max(row["equal_weight_selected_reward"] for row in proposals)
    proposal_stratum_floor = {
        name: max(row["strata"][name]["current_reward"] for row in proposals)
        for name in ("real_rare", "synthetic")
    }

    controls = load_labeled(args.control)
    candidates = load_labeled(args.candidate)
    rows = []
    for row in candidates:
        candidate = row["checkpoint"]
        if candidate.get("selector_architecture") != PCRA_ARCHITECTURE:
            raise RuntimeError(f"candidate is not PCRA: {row['label']}")
        if candidate.get("arbiter_loss") != "regret":
            raise RuntimeError(
                f"only regret-weighted PCRA is promotable: {row['label']}"
            )
        if candidate.get("arbiter_target") != (
            "actual_top1_proposal_vs_reference_incumbent"
        ):
            raise RuntimeError(f"PCRA target drifted: {row['label']}")
        if float(candidate.get("override_threshold", float("nan"))) != 0.0:
            raise RuntimeError(f"PCRA threshold must be exactly zero: {row['label']}")
        if candidate.get("proposal_checkpoint_sha256") != PROPOSAL32_SHA256:
            raise RuntimeError(f"PCRA proposal32 SHA256 drifted: {row['label']}")

        common_row = candidate["strata"]["common"]
        gates = {
            "equal_weight_beats_best_proposal_by_005": candidate[
                "equal_weight_selected_reward"
            ]
            >= proposal_equal_floor + 0.005,
            "common_preserves_reference_within_005": common_row["current_reward"]
            >= common_row["reference_reward"] - 0.005,
            "real_rare_preserves_best_proposal_within_020": candidate["strata"][
                "real_rare"
            ]["current_reward"]
            >= proposal_stratum_floor["real_rare"] - 0.020,
            "synthetic_preserves_best_proposal_within_020": candidate["strata"][
                "synthetic"
            ]["current_reward"]
            >= proposal_stratum_floor["synthetic"] - 0.020,
            "common_degraded_fraction_at_most_010": common_row["degraded_fraction"]
            <= 0.10,
        }
        rows.append(
            {
                **row,
                "equal_weight_delta_vs_best_proposal": candidate[
                    "equal_weight_selected_reward"
                ]
                - proposal_equal_floor,
                "gates": gates,
                "eligible": all(gates.values()),
            }
        )

    eligible = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: row["checkpoint"]["equal_weight_selected_reward"],
        reverse=True,
    )
    selected = None
    selection_rule = None
    if eligible:
        selected = eligible[0]
        best_score = selected["checkpoint"]["equal_weight_selected_reward"]
        simpler = [
            row
            for row in eligible
            if not bool(row["checkpoint"].get("use_decision_context"))
            and best_score - row["checkpoint"]["equal_weight_selected_reward"] <= 0.001
        ]
        if simpler:
            selected = max(
                simpler,
                key=lambda row: row["checkpoint"]["equal_weight_selected_reward"],
            )
            selection_rule = "prefer_no_context_within_001"
        else:
            selection_rule = "maximum_equal_weight_selected_reward"

    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "pcra_frozen_offline_development_gate_v1",
        "development_only": True,
        "certification_consumed": False,
        "proposal32": {
            "evaluation": str(proposal32_path),
            "evaluation_sha256": common.sha256_file(proposal32_path),
            "checkpoint": proposal32,
        },
        "proposal48": {
            "evaluation": str(proposal48_path),
            "evaluation_sha256": common.sha256_file(proposal48_path),
            "checkpoint": proposal48,
        },
        "proposal32_required_sha256": PROPOSAL32_SHA256,
        "thresholds": {
            "minimum_equal_weight_gain_vs_best_proposal": 0.005,
            "maximum_common_drop_vs_reference": 0.005,
            "maximum_rare_or_synthetic_drop_vs_best_proposal": 0.020,
            "maximum_common_degraded_fraction": 0.10,
            "fixed_override_threshold": 0.0,
            "simplicity_tie_tolerance": 0.001,
        },
        "non_promotable_controls": controls,
        "candidates": rows,
        "selected": selected,
        "selection_rule": selection_rule,
        "development_gate_passed": selected is not None,
        "closed_loop_development_required": selected is not None,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
