#!/usr/bin/env python3
"""Apply the pre-registered CPV E2 common-coverage development decision."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cpv_e2_gate_contract as e2_contract
import grpo_selector_v3_cached_common as common
import select_cpv_offline_development as e1_gate


STRATA = ("common", "real_rare", "synthetic")
PROPOSAL32_SHA256 = "562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6"
ARMS = {
    "D1": "D1_diverse_matched",
    "D2": "D2_diverse_20k",
}


def load_one(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("split") != "development"
        or payload.get("certification_consumed") is not False
        or len(payload.get("checkpoints", [])) != 1
    ):
        raise RuntimeError(f"invalid development evaluation: {path}")
    return payload["checkpoints"][0]


def validate_arm(row: dict, arm: str) -> None:
    expected_method = f"cpv_e2_{arm.lower()}_source_sign_pair_regret"
    expected = {
        "method": expected_method,
        "selector_architecture": "proposal_conditioned_regret_arbitration",
        "objective": "official_pdm_proposal_regret_arbitration",
        "arbiter_loss": "regret",
        "arbiter_risk": "source_sign",
        "use_decision_context": False,
        "proposal_checkpoint_sha256": PROPOSAL32_SHA256,
        "reward_components_consumed": False,
        "training_data_split": "train",
        "training_epochs": 16,
        "training_examples_per_cache_epoch": 6339,
        "training_batch_size": 64,
        "formal_contract": True,
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "kl_weight": 1e-3,
        "sampling_mode": "cpv_decision_source_sign",
        "override_threshold": 0.0,
        "train_seed": 0,
        "epoch": 16,
        "common_arm": arm,
    }
    drift = {
        key: {"actual": row.get(key), "expected": value}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if drift:
        raise RuntimeError(f"{arm} E2 contract drifted: {drift}")


def proposal_floors(rows: list[dict]):
    return (
        max(row["equal_weight_selected_reward"] for row in rows),
        {
            name: max(row["strata"][name]["current_reward"] for row in rows)
            for name in ("real_rare", "synthetic")
        },
    )


def performance_gates(candidate: dict, proposal_equal: float, proposal_strata: dict):
    common_row = candidate["strata"]["common"]
    return {
        "equal_weight_beats_best_proposal_by_005": candidate[
            "equal_weight_selected_reward"
        ] >= proposal_equal + 0.005,
        "common_preserves_reference_within_005": common_row["current_reward"]
        >= common_row["reference_reward"] - 0.005,
        "real_rare_preserves_best_proposal_within_020": candidate["strata"]
        ["real_rare"]["current_reward"] >= proposal_strata["real_rare"] - 0.020,
        "synthetic_preserves_best_proposal_within_020": candidate["strata"]
        ["synthetic"]["current_reward"] >= proposal_strata["synthetic"] - 0.020,
        "common_degraded_fraction_at_most_010": common_row["degraded_fraction"] <= 0.10,
    }


def update_ledger(path: Path, report: dict, report_path: Path) -> None:
    ledger = json.loads(path.read_text())
    if ledger.get("certification_consumed"):
        raise RuntimeError("refusing to update a ledger that consumed certification")
    if (
        ledger.get("authorized_followup_stage") != "E2_common_coverage"
        or ledger.get("stage_result") != "D"
    ):
        raise RuntimeError("decision ledger does not authorize E2")
    ledger.update(
        {
            "current_stage": "E2_common_coverage",
            "current_stage_status": "COMPLETED",
            "stage_result": report["stage_result"],
            "authorized_followup_stage": report["authorized_followup_stage"],
            "next_action": report["next_action"],
            "protocol_status": "E2_DEVELOPMENT_RESULT_RECORDED",
            "updated_utc": datetime.now(timezone.utc).date().isoformat(),
            "e2_development_result": {
                "decision": report["decision"],
                "selected_arm": report["selected_arm"],
                "D1_common_auc": report["arms"]["D1"]["auc"]["common"],
                "D2_common_auc": report["arms"]["D2"]["auc"]["common"],
                "D1_equal_stratum_pdm": report["arms"]["D1"]["equal_weight_pdm"],
                "D2_equal_stratum_pdm": report["arms"]["D2"]["equal_weight_pdm"],
                "full_common_authorized": report["full_common_authorized"],
                "development_gate_passed": report["development_gate_passed"],
            },
        }
    )
    ledger.setdefault("decision_history", []).append(
        {
            "stage": "E2_common_coverage",
            "result": report["stage_result"],
            "authorized_followup_stage": report["authorized_followup_stage"],
            "gate_report": str(report_path),
            "gate_report_sha256": common.sha256_file(report_path),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-32", type=Path, required=True)
    parser.add_argument("--proposal-48", type=Path, required=True)
    parser.add_argument("--d0", type=Path, required=True)
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--d2", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-ledger", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data_manifest = json.loads(args.data_manifest.read_text())
    e2_contract.validate_data_manifest(data_manifest)
    data_manifest_sha256 = common.sha256_file(args.data_manifest)
    proposal32 = load_one(args.proposal_32)
    proposal48 = load_one(args.proposal_48)
    d0 = load_one(args.d0)
    d1 = load_one(args.d1)
    d2 = load_one(args.d2)
    e2_contract.validate_d0(d0)
    e2_contract.validate_arm(d1, ARMS["D1"], data_manifest_sha256)
    e2_contract.validate_arm(d2, ARMS["D2"], data_manifest_sha256)
    proposal_equal, proposal_strata = proposal_floors([proposal32, proposal48])
    gates = {
        "D1": performance_gates(d1, proposal_equal, proposal_strata),
        "D2": performance_gates(d2, proposal_equal, proposal_strata),
    }
    bootstraps = {
        "D1_minus_D0": e1_gate.paired_auc_bootstrap(
            d0, d1, args.bootstrap_repetitions, args.bootstrap_seed
        ),
        "D2_minus_D0": e1_gate.paired_auc_bootstrap(
            d0, d2, args.bootstrap_repetitions, args.bootstrap_seed + 1
        ),
        "D2_minus_D1": e1_gate.paired_auc_bootstrap(
            d1, d2, args.bootstrap_repetitions, args.bootstrap_seed + 2
        ),
    }
    passed = {arm: all(values.values()) for arm, values in gates.items()}
    d1_reliable = bootstraps["D1_minus_D0"]["strictly_positive"]
    d2_reliable = bootstraps["D2_minus_D0"]["strictly_positive"]
    d2_adds_reliably = bootstraps["D2_minus_D1"]["strictly_positive"]
    d1_equal_preserved = d1["equal_weight_selected_reward"] >= (
        d0["equal_weight_selected_reward"] - 0.005
    )
    d2_equal_improved = d2["equal_weight_selected_reward"] > d0[
        "equal_weight_selected_reward"
    ]

    (
        decision,
        selected,
        followup,
        next_action,
    ) = e2_contract.choose_decision(
        passed,
        d1_reliable,
        d1_equal_preserved,
        d2_reliable,
        d2_adds_reliably,
        d2_equal_improved,
    )

    candidate = {"D1": d1, "D2": d2}
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "cpv_e2_frozen_common_coverage_gate_v1",
        "development_only": True,
        "certification_consumed": False,
        "decision": decision,
        "stage_result": decision,
        "selected_arm": selected,
        "authorized_followup_stage": followup,
        "next_action": next_action,
        "development_gate_passed": any(passed.values()),
        "closed_loop_development_required": any(passed.values()),
        "full_common_authorized": followup == "E2_full_common_once",
        "data_manifest": str(args.data_manifest.resolve()),
        "data_manifest_sha256": data_manifest_sha256,
        "thresholds": {
            "minimum_equal_weight_gain_vs_best_proposal": 0.005,
            "maximum_common_drop_vs_reference": 0.005,
            "maximum_rare_or_synthetic_drop_vs_best_proposal": 0.020,
            "maximum_common_degraded_fraction": 0.10,
            "D1_equal_weight_preservation": 0.005,
        },
        "arms": {
            arm: {
                "checkpoint": candidate[arm],
                "auc": {
                    name: candidate[arm]["strata"][name]["proposal_benefit_roc_auc"]
                    for name in STRATA
                },
                "equal_weight_pdm": candidate[arm]["equal_weight_selected_reward"],
                "performance_gates": gates[arm],
                "all_performance_gates_passed": passed[arm],
            }
            for arm in ("D1", "D2")
        },
        "comparisons": bootstraps,
        "decision_flags": {
            "D1_reliable_common_auc_gain": d1_reliable,
            "D1_equal_weight_preserved_within_005": d1_equal_preserved,
            "D2_reliable_common_auc_gain": d2_reliable,
            "D2_adds_reliably_over_D1": d2_adds_reliably,
            "D2_equal_weight_improved_over_D0": d2_equal_improved,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.decision_ledger is not None:
        update_ledger(args.decision_ledger.resolve(), report, args.output.resolve())
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": decision,
                "authorized_followup_stage": followup,
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
