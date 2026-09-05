#!/usr/bin/env python3
"""Apply locked efficacy and causal-attribution gates to the four PAF arms."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import grpo_selector_v3_cached_common as common
import proposal_aware_full_feedback_grpo as paf
import train_proposal_aware_full_feedback_grpo as trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def load_json(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text())


def load_trial(root, arm):
    trial_root = root / arm
    report_path, report = load_json(trial_root / "report.json")
    evaluation_path, evaluation = load_json(trial_root / "fresh_noise_evaluation.json")
    if (
        report.get("status") != "PASS"
        or report.get("method") != trainer.METHOD
        or report.get("arm") != arm
        or evaluation.get("status") != "PASS"
        or evaluation.get("method") != "proposal_aware_full_feedback_evaluation_v1"
        or evaluation.get("arm") != arm
        or evaluation.get("split") != "train_fresh_noise"
        or evaluation.get("noise_seeds") != [9, 10, 11]
    ):
        raise RuntimeError(f"PAF trial contract drifted: {arm}")
    final = [row for row in report.get("checkpoints", []) if row.get("epoch") == 8]
    if len(final) != 1 or evaluation.get("checkpoint_sha256") != final[0].get("scene_selector_state_sha256"):
        raise RuntimeError(f"{arm} evaluation is not the fixed epoch-8 checkpoint")
    records_path = Path(evaluation["records"]).resolve()
    if common.sha256_file(records_path) != evaluation["records_sha256"]:
        raise RuntimeError(f"{arm} record SHA256 drifted")
    return {
        "arm": arm,
        "report": report,
        "evaluation": evaluation,
        "report_path": str(report_path),
        "report_sha256": common.sha256_file(report_path),
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": common.sha256_file(evaluation_path),
        "records_path": records_path,
    }


def load_records(path):
    records = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["token"]), int(row["noise_seed"]))
        if key in records:
            raise RuntimeError(f"duplicate record at {path}:{line_number}")
        records[key] = row
    if not records:
        raise RuntimeError(f"empty records: {path}")
    return records


def paired_log_bootstrap(target, reference, repetitions, seed):
    left = load_records(target["records_path"])
    right = load_records(reference["records_path"])
    if set(left) != set(right):
        raise RuntimeError("PAF arms do not cover identical token/noise keys")
    by_log = defaultdict(lambda: defaultdict(list))
    invariants = (
        "token", "scene", "log", "stratum", "noise_seed", "anchor_reward",
        "oracle_reward", "anchor_oracle_probability", "recoverable",
        "suppressed_recoverable", "solved",
    )
    for key in sorted(left):
        current, baseline = left[key], right[key]
        if any(current[name] != baseline[name] for name in invariants):
            raise RuntimeError(f"paired invariant drift at {key}")
        by_log[current["log"]][current["stratum"]].append(
            float(current["current_reward"]) - float(baseline["current_reward"])
        )
    values = []
    for name in sorted(by_log):
        groups = by_log[name]
        if not groups["common"] or not groups["rare"]:
            raise RuntimeError(f"paired log lacks common/rare members: {name}")
        values.append(0.5 * (
            np.mean(groups["common"], dtype=np.float64)
            + np.mean(groups["rare"], dtype=np.float64)
        ))
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        stop = min(start + 500, repetitions)
        sampled = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[sampled].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "paired_log",
        "num_logs": len(values),
        "repetitions": repetitions,
    }


def efficacy(trial):
    summary = trial["evaluation"]["summary"]
    per_seed = list(summary["equal_stratum_by_noise_seed_hard_gain"].values())
    gates = {
        "equal_stratum_gain_at_least_0002": summary["equal_stratum_hard_gain"] >= 0.002,
        "common_nonnegative": summary["common_hard_gain"] >= 0.0,
        "rare_nonnegative": summary["rare_hard_gain"] >= 0.0,
        "every_fresh_noise_seed_floor_minus_0001": min(per_seed) >= -0.001,
        "paired_log_bootstrap_lower_positive": summary["equal_log_bootstrap_hard_gain"]["lower_95"] > 0.0,
        "suppressed_recoverable_regret_reduction_at_least_0005": (
            summary["equal_suppressed_recoverable_oracle_regret_reduction"] >= 0.005
        ),
        "solved_subset_drop_at_most_0001": summary["equal_solved_subset_hard_gain"] >= -0.001,
    }
    return gates, all(gates.values())


def margin_pass(comparison, margin=0.001):
    return comparison["mean"] >= margin and comparison["lower_95"] > 0.0


def main():
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    root = args.trial_root.expanduser().resolve()
    trials = {arm: load_trial(root, arm) for arm in paf.ARMS}
    anchors = {row["evaluation"]["v3_anchor_sha256"] for row in trials.values()}
    gates = {row["evaluation"]["mechanism_gate_sha256"] for row in trials.values()}
    if len(anchors) != 1 or len(gates) != 1:
        raise RuntimeError("PAF arms do not share one V3 anchor and mechanism gate")
    budgets = {
        (
            row["report"]["epochs"], row["report"]["examples_per_epoch"],
            row["report"]["outer_batch_size"], row["report"]["learning_rate"],
            row["report"]["kl_weight"], row["report"]["opportunity_lower"],
            row["report"]["opportunity_upper"], row["report"]["sampling_mode"],
            row["report"]["budget"]["optimizer_steps"],
            row["report"]["budget"]["view_examples"],
        )
        for row in trials.values()
    }
    if len(budgets) != 1:
        raise RuntimeError("PAF arms do not share one fixed training budget")

    efficacy_gates, eligible = {}, {}
    for arm, trial in trials.items():
        efficacy_gates[arm], eligible[arm] = efficacy(trial)
    comparisons = {}
    comparison_pairs = (
        ("paf_over_direct", "paf_grpo", "direct_grpo"),
        ("paf_over_full_feedback", "paf_grpo", "full_feedback"),
        ("paf_over_opportunity", "paf_grpo", "opportunity_grpo"),
        ("full_feedback_over_direct", "full_feedback", "direct_grpo"),
        ("paf_over_full_feedback_no_extra_test", "paf_grpo", "full_feedback"),
    )
    for offset, (label, target, reference) in enumerate(comparison_pairs):
        comparisons[label] = paired_log_bootstrap(
            trials[target], trials[reference], args.bootstrap_repetitions, args.seed + offset
        )

    full_attribution = eligible["paf_grpo"] and all(
        margin_pass(comparisons[name])
        for name in ("paf_over_direct", "paf_over_full_feedback", "paf_over_opportunity")
    )
    simpler_attribution = (
        eligible["full_feedback"]
        and margin_pass(comparisons["full_feedback_over_direct"])
        and not margin_pass(comparisons["paf_over_full_feedback_no_extra_test"])
    )
    only_opportunity = (
        eligible["opportunity_grpo"]
        and not eligible["paf_grpo"]
        and not eligible["full_feedback"]
    )
    if full_attribution:
        decision = "AUTHORIZE_PAF_GRPO_REPLICAS"
        winner = "paf_grpo"
        authorized_followup_stage = "optimizer_seed_replicas_then_development"
    elif simpler_attribution:
        decision = "AUTHORIZE_FULL_FEEDBACK_REPLICAS"
        winner = "full_feedback"
        authorized_followup_stage = "optimizer_seed_replicas_then_development"
    elif only_opportunity:
        decision = "STOP_POLICY_MISMATCH_ONLY_OPPORTUNITY_WINS"
        winner = None
        authorized_followup_stage = "stop_retain_v3"
    else:
        decision = "STOP_PAF_OBJECTIVE_DIRECTION"
        winner = None
        authorized_followup_stage = "stop_retain_v3"

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "proposal_aware_full_feedback_search_gate_v1",
        "decision": decision,
        "winner": winner,
        "authorized_followup_stage": authorized_followup_stage,
        "efficacy_gates": efficacy_gates,
        "eligible": eligible,
        "comparisons": comparisons,
        "attribution_gates": {
            "full_paf_attribution": full_attribution,
            "simpler_full_feedback_attribution": simpler_attribution,
            "only_opportunity_weighting_wins": only_opportunity,
        },
        "pre_registered_thresholds": {
            "equal_stratum_gain": 0.002,
            "common_and_rare_floor": 0.0,
            "per_noise_seed_floor": -0.001,
            "suppressed_recoverable_regret_reduction": 0.005,
            "solved_subset_floor": -0.001,
            "paired_arm_margin": 0.001,
            "bootstrap_lower_bound": 0.0,
        },
        "trials": {
            arm: {
                key: value
                for key, value in row.items()
                if key not in {"report", "evaluation", "records_path"}
            }
            for arm, row in trials.items()
        },
        "scientific_contract": {
            "matched_training_budgets": True,
            "selector_only": True,
            "generator_frozen": True,
            "official_scalar_pdm_only": True,
            "reward_components_consumed": False,
            "fresh_noise_search_only": True,
            "development_consumed": False,
            "certification_consumed": False,
        },
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(paf.__file__).resolve()): common.sha256_file(Path(paf.__file__).resolve()),
            str(Path(trainer.__file__).resolve()): common.sha256_file(Path(trainer.__file__).resolve()),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "decision": decision, "winner": winner, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
