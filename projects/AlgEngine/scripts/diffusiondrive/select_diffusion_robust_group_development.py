#!/usr/bin/env python3
"""Apply pre-registered efficacy and attribution gates to the DRG search."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import grpo_selector_v3_cached_common as common


TRIALS = {
    "repeat_grpo": "repeat_grpo",
    "mean_soft": "mean_soft",
    "mean_top1": "mean_top1",
    "pure_softmin_top1": "pure_softmin_top1",
    "bounded_relative_top1": "bounded_relative_top1",
    "bounded_relative_soft": "bounded_relative_soft",
    "bounded_relative_top1_shuffled": "bounded_relative_top1_shuffled",
    "bounded_absolute_top1": "bounded_absolute_top1",
}
RISK_TEMPERATURE = 0.05
RISK_MIX = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def load_json(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text())


def load_trial(root, label):
    expected_arm = TRIALS[label]
    trial = root / label
    report_path, report = load_json(trial / "report.json")
    evaluation_path, evaluation = load_json(
        trial / "development_evaluation.json"
    )
    if (
        report.get("status") != "PASS"
        or evaluation.get("status") != "PASS"
        or report.get("method") != "diffusion_set_selector_posttraining_v1"
        or evaluation.get("method")
        != "diffusion_set_selector_evaluation_v1"
        or report.get("arm") != expected_arm
        or evaluation.get("arm") != expected_arm
        or float(report.get("risk_temperature")) != RISK_TEMPERATURE
        or float(report.get("risk_mix")) != RISK_MIX
        or evaluation.get("split") != "development"
    ):
        raise RuntimeError(f"development trial contract drifted: {label}")
    checkpoints = report.get("checkpoints", [])
    if len(checkpoints) != 1:
        raise RuntimeError(f"{label} must expose exactly one fixed final checkpoint")
    if (
        evaluation.get("checkpoint_sha256")
        != checkpoints[0].get("scene_selector_state_sha256")
    ):
        raise RuntimeError(f"{label} evaluation/checkpoint mismatch")
    records_path = Path(evaluation["records"]).resolve()
    if common.sha256_file(records_path) != evaluation["records_sha256"]:
        raise RuntimeError(f"{label} scalar record SHA256 drifted")
    return {
        "label": label,
        "report_path": str(report_path),
        "report_sha256": common.sha256_file(report_path),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": common.sha256_file(evaluation_path),
        "records_path": records_path,
        "report": report,
        "evaluation": evaluation,
    }


def load_records(path):
    records = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["token"]), int(row["noise_seed"]))
        if key in records:
            raise RuntimeError(f"duplicate scalar record at {path}:{line_number}")
        records[key] = row
    if not records:
        raise RuntimeError(f"empty scalar record file: {path}")
    return records


def paired_log_bootstrap(target, reference, stratum, repetitions, seed):
    target_records = load_records(target["records_path"])
    reference_records = load_records(reference["records_path"])
    if set(target_records) != set(reference_records):
        raise RuntimeError("paired selector arms do not cover identical token/draw keys")
    by_log = defaultdict(lambda: defaultdict(list))
    for key in sorted(target_records):
        left = target_records[key]
        right = reference_records[key]
        invariants = (
            "token",
            "scene",
            "log",
            "stratum",
            "noise_seed",
            "anchor_reward",
        )
        if any(left[name] != right[name] for name in invariants):
            raise RuntimeError(f"paired selector record drift at {key}")
        if stratum not in {"all", "equal"} and left["stratum"] != stratum:
            continue
        by_log[left["log"]][left["stratum"]].append(
            float(left["current_reward"]) - float(right["current_reward"])
        )
    if not by_log:
        raise RuntimeError(f"empty paired comparison stratum={stratum}")
    log_values = []
    for name in sorted(by_log):
        groups = by_log[name]
        if stratum == "equal":
            if not groups["rare"] or not groups["common"]:
                raise RuntimeError(f"log lacks one paired stratum: {name}")
            value = 0.5 * (
                np.mean(groups["rare"], dtype=np.float64)
                + np.mean(groups["common"], dtype=np.float64)
            )
        else:
            values = [value for rows in groups.values() for value in rows]
            value = np.mean(values, dtype=np.float64)
        log_values.append(value)
    log_values = np.asarray(log_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0, len(log_values), size=(repetitions, len(log_values))
    )
    estimates = log_values[samples].mean(axis=1)
    return {
        "mean": float(log_values.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "log",
        "num_logs": int(len(log_values)),
        "repetitions": int(repetitions),
    }


def compare(target, reference, repetitions, seed):
    paired = {
        stratum: paired_log_bootstrap(
            target, reference, stratum, repetitions, seed + offset
        )
        for offset, stratum in enumerate(("equal", "all", "rare", "common"))
    }
    target_min = target["evaluation"]["summary"]["mean_min_hard_gain"]
    reference_min = reference["evaluation"]["summary"]["mean_min_hard_gain"]
    target_equal = target["evaluation"]["summary"][
        "equal_stratum_current_pdm"
    ]
    reference_equal = reference["evaluation"]["summary"][
        "equal_stratum_current_pdm"
    ]
    return {
        "paired_current_pdm": paired,
        "equal_stratum_current_pdm_delta": target_equal - reference_equal,
        "mean_min_hard_gain_delta": target_min - reference_min,
    }


def efficacy_gates(trial):
    summary = trial["evaluation"]["summary"]
    all_stratum = trial["evaluation"]["strata"]["all"]
    per_seed = [
        row["hard_gain"] for row in all_stratum["by_noise_seed"].values()
    ]
    gates = {
        "equal_stratum_gain_at_least_0002": (
            summary["equal_stratum_hard_gain"] >= 0.002
        ),
        "rare_floor_minus_0001": summary["rare_hard_gain"] >= -0.001,
        "common_floor_minus_0001": summary["common_hard_gain"] >= -0.001,
        "log_bootstrap_lower_positive": (
            all_stratum["log_bootstrap_hard_gain"]["lower_95"] > 0.0
        ),
        "every_noise_seed_floor_minus_0001": min(per_seed) >= -0.001,
    }
    return gates, all(gates.values())


def paired_margin_pass(comparison, margin):
    paired = comparison["paired_current_pdm"]["equal"]
    return (
        comparison["equal_stratum_current_pdm_delta"] >= margin
        and paired["lower_95"] > 0.0
    )


def main():
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    root = args.trial_root.expanduser().resolve()
    trials = {label: load_trial(root, label) for label in TRIALS}
    anchors = {
        row["evaluation"]["v3_anchor_sha256"] for row in trials.values()
    }
    if len(anchors) != 1:
        raise RuntimeError("search arms do not share one frozen V3 anchor")
    budgets = {
        (
            row["report"]["epochs"],
            row["report"]["examples_per_epoch"],
            row["report"]["budget"]["view_examples"],
            row["report"]["outer_batch_size"],
            row["report"]["learning_rate"],
            row["report"]["kl_weight"],
            row["report"]["reward_scale"],
        )
        for row in trials.values()
    }
    if len(budgets) != 1:
        raise RuntimeError("search arms do not share a fixed training budget")

    primary_label = "bounded_relative_top1"
    primary = trials[primary_label]
    comparators = {
        "repeat_grpo": trials["repeat_grpo"],
        "mean_top1": trials["mean_top1"],
        "bounded_relative_soft": trials["bounded_relative_soft"],
        "shuffled_group": trials["bounded_relative_top1_shuffled"],
        "absolute_risk": trials["bounded_absolute_top1"],
        "pure_softmin": trials["pure_softmin_top1"],
    }
    comparisons = {
        name: compare(
            primary,
            reference,
            args.bootstrap_repetitions,
            args.seed + 100 * offset,
        )
        for offset, (name, reference) in enumerate(comparators.items())
    }
    pure_over_bounded = compare(
        trials["pure_softmin_top1"],
        primary,
        args.bootstrap_repetitions,
        args.seed + 900,
    )
    all_efficacy = {}
    efficacy_passes = {}
    for label, trial in trials.items():
        gates, passed = efficacy_gates(trial)
        all_efficacy[label] = gates
        efficacy_passes[label] = passed

    top1_over_soft = comparisons["bounded_relative_soft"]
    relative_over_absolute = comparisons["absolute_risk"]
    primary_weight_metrics = primary["report"]["training_means_per_step"]
    pure_weight_metrics = trials["pure_softmin_top1"]["report"][
        "training_means_per_step"
    ]
    attribution_gates = {
        "bounded_over_mean_top1": paired_margin_pass(
            comparisons["mean_top1"], 0.001
        ),
        "bounded_over_repeat_grpo": paired_margin_pass(
            comparisons["repeat_grpo"], 0.001
        ),
        "bounded_lower_tail_over_mean_top1": (
            comparisons["mean_top1"]["mean_min_hard_gain_delta"] >= 0.0005
        ),
        "true_group_over_shuffled": paired_margin_pass(
            comparisons["shuffled_group"], 0.001
        ),
        "top1_over_soft": paired_margin_pass(top1_over_soft, 0.001),
        "relative_over_absolute": paired_margin_pass(
            relative_over_absolute, 0.001
        ),
        "bounded_weight_upper_contract": (
            primary_weight_metrics["draw_weight_max"] <= 2.0 / 3.0 + 1e-5
        ),
        "bounded_never_collapsed_above_09": (
            primary_weight_metrics["draw_weight_gt_09_fraction"] == 0.0
        ),
        "pure_softmin_not_collapsed": (
            pure_weight_metrics["draw_weight_max"] <= 0.7
            and pure_weight_metrics["draw_weight_gt_09_fraction"] <= 0.1
        ),
        "pure_over_bounded": paired_margin_pass(pure_over_bounded, 0.001),
    }
    core_grouping = (
        efficacy_passes[primary_label]
        and attribution_gates["bounded_over_mean_top1"]
        and attribution_gates["bounded_over_repeat_grpo"]
        and attribution_gates["bounded_lower_tail_over_mean_top1"]
        and attribution_gates["true_group_over_shuffled"]
        and attribution_gates["bounded_weight_upper_contract"]
        and attribution_gates["bounded_never_collapsed_above_09"]
    )
    selected_label = None
    next_action = "stop_objective_only_and_audit_temporal_history"
    if (
        core_grouping
        and attribution_gates["top1_over_soft"]
        and attribution_gates["relative_over_absolute"]
        and not (
            attribution_gates["pure_over_bounded"]
            and attribution_gates["pure_softmin_not_collapsed"]
        )
    ):
        decision = "AUTHORIZE_BOUNDED_RELATIVE_TOP1_FORMAL_REPLICAS"
        selected_label = primary_label
        next_action = "train_three_seed_formal_replicas"
    elif (
        core_grouping
        and not attribution_gates["top1_over_soft"]
        and efficacy_passes["bounded_relative_soft"]
    ):
        decision = "AUTHORIZE_BOUNDED_SOFT_MATCHED_CONTROLS"
        next_action = "run_soft_shuffled_and_soft_absolute_controls_only"
    elif core_grouping and not attribution_gates["relative_over_absolute"]:
        decision = "EFFICACY_WITHOUT_REFERENCE_ATTRIBUTION"
        next_action = "run_absolute_matched_grouping_control_only"
    elif (
        efficacy_passes["pure_softmin_top1"]
        and attribution_gates["pure_softmin_not_collapsed"]
        and attribution_gates["pure_over_bounded"]
    ):
        decision = "AUTHORIZE_PURE_SOFTMIN_MATCHED_SHUFFLE_CONTROL"
        next_action = "run_pure_softmin_shuffled_control_only"
    elif efficacy_passes[primary_label]:
        decision = "EFFICACY_WITHOUT_DIFFUSION_GROUPING_ATTRIBUTION"
        next_action = "stop_formal_and_report_noncausal_efficacy"
    elif any(
        efficacy_passes[label]
        for label in ("repeat_grpo", "mean_soft", "mean_top1")
    ):
        decision = "NON_DIFFUSION_CONTROL_WINS"
        next_action = "stop_diffusion_claim_and_retain_simpler_control"
    else:
        decision = "STOP_DIFFUSION_SET_OBJECTIVE_AFTER_DEVELOPMENT"

    compact_trials = {}
    for label, row in trials.items():
        compact_trials[label] = {
            "arm": row["report"]["arm"],
            "risk_temperature": row["report"]["risk_temperature"],
            "risk_mix": row["report"]["risk_mix"],
            "training_means_per_step": row["report"][
                "training_means_per_step"
            ],
            "summary": row["evaluation"]["summary"],
            "log_bootstrap_hard_gain": row["evaluation"]["strata"]["all"][
                "log_bootstrap_hard_gain"
            ],
            "report": row["report_path"],
            "report_sha256": row["report_sha256"],
            "evaluation": row["evaluation_path"],
            "evaluation_sha256": row["evaluation_sha256"],
        }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusion_set_selector_development_gate_v1",
        "decision": decision,
        "next_action": next_action,
        "selected_trial": selected_label,
        "selected_checkpoint": (
            trials[selected_label]["evaluation"]["checkpoint"]
            if selected_label is not None
            else None
        ),
        "selected_checkpoint_sha256": (
            trials[selected_label]["evaluation"]["checkpoint_sha256"]
            if selected_label is not None
            else None
        ),
        "efficacy_gates": all_efficacy,
        "efficacy_passes": efficacy_passes,
        "attribution_gates": attribution_gates,
        "comparisons": comparisons,
        "pure_over_bounded": pure_over_bounded,
        "trials": compact_trials,
        "fixed_budget": list(next(iter(budgets))),
        "scientific_contract": {
            "selector_only": True,
            "official_scalar_pdm_only": True,
            "reward_components_consumed": False,
            "new_training_data_consumed": False,
            "development_consumed": True,
            "certification_consumed": False,
            "formal_runs_authorized": decision
            == "AUTHORIZE_BOUNDED_RELATIVE_TOP1_FORMAL_REPLICAS",
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": decision,
                "selected_trial": payload["selected_trial"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
