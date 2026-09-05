#!/usr/bin/env python3
"""Apply the frozen CPV E1 development gate and record outcome S/C/D."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common


PROPOSAL32_SHA256 = "562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6"
PCRA_ARCHITECTURE = "proposal_conditioned_regret_arbitration"
PCRA_OBJECTIVE = "official_pdm_proposal_regret_arbitration"
PCRA_TARGET = "actual_top1_proposal_vs_reference_incumbent"
STRATA = ("common", "real_rare", "synthetic")


def load_one(path: Path):
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


def evaluation_row(label: str, path: Path):
    resolved, checkpoint = load_one(path)
    return {
        "label": label,
        "evaluation": str(resolved),
        "evaluation_sha256": common.sha256_file(resolved),
        "checkpoint": checkpoint,
    }


def validate_pcra(row, expected_risk: str, promotable: bool):
    checkpoint = row["checkpoint"]
    expected_method = (
        "pcra_v1_pair_regret_proposal_conditioned_regret"
        if expected_risk == "global"
        else f"cpv_v1_{expected_risk}_pair_regret"
    )
    contract = {
        "selector_architecture": PCRA_ARCHITECTURE,
        "objective": PCRA_OBJECTIVE,
        "method": expected_method,
        "arbiter_loss": "regret",
        "arbiter_risk": expected_risk,
        "use_decision_context": False,
        "arbiter_target": PCRA_TARGET,
        "proposal_checkpoint_sha256": PROPOSAL32_SHA256,
        "override_threshold": 0.0,
        "reward_components_consumed": False,
        "training_data_split": "train",
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "kl_weight": 1e-3,
        "sampling_mode": (
            "common_hard_balanced"
            if expected_risk == "global"
            else f"cpv_decision_{expected_risk}"
        ),
        "train_seed": 0,
        "epoch": 16,
    }
    if expected_risk != "global":
        contract.update(
            training_epochs=16,
            training_examples_per_cache_epoch=6339,
            training_batch_size=64,
            formal_contract=True,
            sampling_mode=f"cpv_decision_{expected_risk}",
        )
    hydrated = []
    for key in ("sampling_mode",):
        if checkpoint.get(key) is not None:
            continue
        checkpoint_path = Path(checkpoint["checkpoint"]).expanduser().resolve()
        expected_sha = checkpoint.get("checkpoint_sha256")
        if common.sha256_file(checkpoint_path) != expected_sha:
            raise RuntimeError(
                f"{row['label']} checkpoint changed during metadata hydration"
            )
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint[key] = raw_checkpoint.get(key)
        hydrated.append(key)
    row["contract_metadata_hydrated_from_checkpoint"] = hydrated
    drift = {
        key: {"actual": checkpoint.get(key), "expected": expected}
        for key, expected in contract.items()
        if checkpoint.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"{row['label']} CPV contract drifted: {drift}")
    row["promotable"] = promotable


def roc_auc(scores, gains):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(gains, dtype=np.float64) > 0.0
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float(
        (ranks[targets].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def paired_auc_bootstrap(a0, a3, repetitions: int, seed: int):
    a0_records = a0["strata"]["common"].get("proposal_ranking_records")
    a3_records = a3["strata"]["common"].get("proposal_ranking_records")
    if not a0_records or not a3_records:
        raise RuntimeError("common paired ranking records are missing")
    a0_scores = np.asarray(a0_records["evidence"], dtype=np.float64)
    a3_scores = np.asarray(a3_records["evidence"], dtype=np.float64)
    a0_gain = np.asarray(a0_records["gain"], dtype=np.float64)
    a3_gain = np.asarray(a3_records["gain"], dtype=np.float64)
    if not (
        len(a0_scores) == len(a3_scores) == len(a0_gain) == len(a3_gain)
        and np.allclose(a0_gain, a3_gain, rtol=0.0, atol=1e-8)
    ):
        raise RuntimeError("A0/A3 common ranking examples are not paired")
    if not ((a0_gain > 0.0).any() and (a0_gain < 0.0).any()):
        raise RuntimeError("paired common ranking data lacks both decision signs")

    point_a0 = roc_auc(a0_scores, a0_gain)
    point_a3 = roc_auc(a3_scores, a3_gain)
    rng = np.random.default_rng(seed)
    differences = []
    maximum_attempts = max(10 * repetitions, repetitions + 100)
    for _ in range(maximum_attempts):
        indices = rng.integers(0, len(a0_gain), size=len(a0_gain))
        sampled_gain = a0_gain[indices]
        if not ((sampled_gain > 0.0).any() and (sampled_gain < 0.0).any()):
            continue
        differences.append(
            roc_auc(a3_scores[indices], sampled_gain)
            - roc_auc(a0_scores[indices], sampled_gain)
        )
        if len(differences) == repetitions:
            break
    if len(differences) != repetitions:
        raise RuntimeError("could not obtain the fixed number of valid bootstraps")
    lower, upper = np.quantile(differences, (0.025, 0.975))
    return {
        "method": "paired_example_bootstrap_percentile",
        "seed": seed,
        "repetitions": repetitions,
        "examples": len(a0_gain),
        "a0_auc": point_a0,
        "a3_auc": point_a3,
        "difference_a3_minus_a0": point_a3 - point_a0,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "strictly_positive": float(lower) > 0.0,
    }


def update_decision_ledger(path: Path, outcome: str, report_path: Path):
    path = path.expanduser().resolve()
    ledger = json.loads(path.read_text())
    if ledger.get("certification_consumed"):
        raise RuntimeError("refusing to update a ledger that consumed certification")
    if ledger.get("current_stage") != "E1_same_data_counterfactual_verifier":
        raise RuntimeError("decision ledger current stage is not E1")
    followup = {
        "S": "closed_loop_development",
        "C": "E1b_train_only_calibration",
        "D": "E2_common_coverage",
    }[outcome]
    next_action = {
        "S": "lock_A3_and_run_single_closed_loop_development",
        "C": "implement_only_E1b_train_log_split_affine_calibration",
        "D": "implement_only_E2_unique_common_coverage",
    }[outcome]
    now = datetime.now(timezone.utc)
    ledger.update(
        {
            "updated_utc": now.date().isoformat(),
            "current_stage_status": "COMPLETED",
            "stage_result": outcome,
            "authorized_followup_stage": followup,
            "next_action": next_action,
        }
    )
    history = list(ledger.get("decision_history", ()))
    history.append(
        {
            "timestamp_utc": now.replace(microsecond=0).isoformat(),
            "stage": "E1_same_data_counterfactual_verifier",
            "result": outcome,
            "gate_report": str(report_path),
            "gate_report_sha256": common.sha256_file(report_path),
            "authorized_followup_stage": followup,
        }
    )
    ledger["decision_history"] = history
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-32", type=Path, required=True)
    parser.add_argument("--proposal-48", type=Path, required=True)
    parser.add_argument("--a0", type=Path, required=True)
    parser.add_argument("--a1", type=Path, required=True)
    parser.add_argument("--a2", type=Path, required=True)
    parser.add_argument("--a3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-ledger", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")

    proposal32 = evaluation_row("proposal32", args.proposal_32)
    proposal48 = evaluation_row("proposal48", args.proposal_48)
    a0 = evaluation_row("A0_global_regret", args.a0)
    a1 = evaluation_row("A1_source_regret", args.a1)
    a2 = evaluation_row("A2_sign_regret", args.a2)
    a3 = evaluation_row("A3_source_sign_regret", args.a3)
    validate_pcra(a0, "global", False)
    validate_pcra(a1, "source", False)
    validate_pcra(a2, "sign", False)
    validate_pcra(a3, "source_sign", True)

    proposals = (proposal32["checkpoint"], proposal48["checkpoint"])
    proposal_equal_floor = max(row["equal_weight_selected_reward"] for row in proposals)
    proposal_stratum_floor = {
        name: max(row["strata"][name]["current_reward"] for row in proposals)
        for name in ("real_rare", "synthetic")
    }
    candidate = a3["checkpoint"]
    common_row = candidate["strata"]["common"]
    gates = {
        "equal_weight_beats_best_proposal_by_005": candidate[
            "equal_weight_selected_reward"
        ] >= proposal_equal_floor + 0.005,
        "common_preserves_reference_within_005": common_row["current_reward"]
        >= common_row["reference_reward"] - 0.005,
        "real_rare_preserves_best_proposal_within_020": candidate["strata"]
        ["real_rare"]["current_reward"] >= proposal_stratum_floor["real_rare"] - 0.020,
        "synthetic_preserves_best_proposal_within_020": candidate["strata"]
        ["synthetic"]["current_reward"] >= proposal_stratum_floor["synthetic"] - 0.020,
        "common_degraded_fraction_at_most_010": common_row["degraded_fraction"] <= 0.10,
    }
    bootstrap = paired_auc_bootstrap(
        a0["checkpoint"], candidate, args.bootstrap_repetitions, args.bootstrap_seed
    )
    a0_auc = {
        name: a0["checkpoint"]["strata"][name]["proposal_benefit_roc_auc"]
        for name in STRATA
    }
    a3_auc = {
        name: candidate["strata"][name]["proposal_benefit_roc_auc"]
        for name in STRATA
    }
    ranking_improved = {
        "common_auc_at_least_070": a3_auc["common"] >= 0.70,
        "common_auc_paired_ci_strictly_positive": bootstrap["strictly_positive"],
        "real_rare_auc_no_worse_than_a0_minus_002": a3_auc["real_rare"]
        >= a0_auc["real_rare"] - 0.02,
        "synthetic_auc_no_worse_than_a0_minus_002": a3_auc["synthetic"]
        >= a0_auc["synthetic"] - 0.02,
        "rare_and_synthetic_performance_preserved": gates[
            "real_rare_preserves_best_proposal_within_020"
        ] and gates["synthetic_preserves_best_proposal_within_020"],
        "failure_is_deployment_or_common_preservation": not all(
            gates[name]
            for name in (
                "equal_weight_beats_best_proposal_by_005",
                "common_preserves_reference_within_005",
                "common_degraded_fraction_at_most_010",
            )
        ),
    }
    if all(gates.values()):
        outcome = "S"
    elif all(ranking_improved.values()):
        outcome = "C"
    else:
        outcome = "D"
    followup = {
        "S": "closed_loop_development",
        "C": "E1b_train_only_calibration",
        "D": "E2_common_coverage",
    }[outcome]

    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "cpv_e1_frozen_offline_development_gate_v1",
        "development_only": True,
        "certification_consumed": False,
        "stage_outcome": outcome,
        "authorized_followup_stage": followup,
        "development_gate_passed": outcome == "S",
        "closed_loop_development_required": outcome == "S",
        "proposal32_required_sha256": PROPOSAL32_SHA256,
        "proposal32": proposal32,
        "proposal48": proposal48,
        "non_promotable_controls": [a0, a1, a2],
        "candidate": a3,
        "thresholds": {
            "minimum_equal_weight_gain_vs_best_proposal": 0.005,
            "maximum_common_drop_vs_reference": 0.005,
            "maximum_rare_or_synthetic_drop_vs_best_proposal": 0.020,
            "maximum_common_degraded_fraction": 0.10,
            "fixed_override_threshold": 0.0,
            "outcome_c_minimum_common_auc": 0.70,
            "outcome_c_maximum_other_auc_drop": 0.02,
        },
        "a3_equal_weight_delta_vs_best_proposal": candidate[
            "equal_weight_selected_reward"
        ] - proposal_equal_floor,
        "a3_gates": gates,
        "a0_auc": a0_auc,
        "a3_auc": a3_auc,
        "common_auc_paired_bootstrap": bootstrap,
        "outcome_c_requirements": ranking_improved,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.decision_ledger is not None:
        update_decision_ledger(args.decision_ledger, outcome, output)
    print(json.dumps({"status": "PASS", "output": str(output), "stage_outcome": outcome}, sort_keys=True))


if __name__ == "__main__":
    main()
