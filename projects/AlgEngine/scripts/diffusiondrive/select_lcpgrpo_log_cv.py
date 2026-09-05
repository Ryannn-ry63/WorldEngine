#!/usr/bin/env python3
"""Select at most one LC-PGRPO configuration from locked five-fold log-CV."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import lcpgrpo_protocol as protocol
import lineage_consistent_proximal_grpo as lcp
import grpo_selector_v3_cached_common as common


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def load_records(path):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "token" not in row or "noise_seed" not in row:
            raise RuntimeError(f"malformed record at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"empty records: {path}")
    protocol.rows_by_key(rows)
    return rows


def configuration_id(arm, target_kl, epoch):
    if target_kl is None:
        delta = "none"
    else:
        delta = f"{float(target_kl):.2f}".replace(".", "p")
    return f"{arm}__delta_{delta}__epoch_{int(epoch)}"


def expected_configuration_ids():
    expected = set()
    for epoch in protocol.CHECKPOINT_EPOCHS:
        expected.add(configuration_id("direct", None, epoch))
        expected.add(configuration_id("lineage_direct", None, epoch))
        for target_kl in protocol.TARGET_KL_GRID:
            expected.add(configuration_id("per_draw_proximal", target_kl, epoch))
            expected.add(configuration_id("lineage_proximal", target_kl, epoch))
    return expected


def validate_training_report(path, evaluation):
    report_path = path.parent / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text())
    expected = {
        "status": "PASS",
        "method": protocol.METHOD,
        "arm": evaluation["arm"],
        "lineage_control": "real",
        "target_kl": evaluation["target_kl"],
        "heldout_fold": evaluation["heldout_fold"],
        "epochs": 8,
        "examples_per_epoch": 6339,
        "outer_batch_size": 64,
        "learning_rate": 3e-5,
        "kl_weight": 1e-3,
        "temperature": 1.0,
        "retention_headroom": 0.005,
        "reward_epsilon": 1e-6,
        "sampling_mode": "rare_common_50_50_log_disjoint",
    }
    drift = {
        key: {"actual": report.get(key), "expected": value}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if drift:
        raise RuntimeError(f"training report contract drifted at {report_path}: {drift}")
    if report.get("checkpoint_epochs") != list(protocol.CHECKPOINT_EPOCHS):
        raise RuntimeError("checkpoint schedule drifted")
    checkpoint = [
        row
        for row in report.get("checkpoints", [])
        if row.get("epoch") == evaluation["epoch"]
    ]
    if len(checkpoint) != 1 or checkpoint[0]["scene_selector_state_sha256"] != evaluation["checkpoint_sha256"]:
        raise RuntimeError("evaluation does not match its training checkpoint")
    return report_path, report


def load_evaluations(root):
    paths = sorted(root.rglob("epoch_*_evaluation.json"))
    if not paths:
        raise RuntimeError(f"no CV evaluations under {root}")
    groups = defaultdict(list)
    mechanism_sha = set()
    pair_sha = set()
    anchor_sha = set()
    for path in paths:
        evaluation = json.loads(path.read_text())
        if (
            evaluation.get("status") != "PASS"
            or evaluation.get("method") != protocol.EVALUATION_METHOD
            or evaluation.get("split") != "cv"
            or evaluation.get("noise_seeds") != list(protocol.FRESH_NOISE_SEEDS)
            or evaluation.get("lineage_control") != "real"
            or evaluation.get("arm") not in lcp.ARMS
            or evaluation.get("epoch") not in protocol.CHECKPOINT_EPOCHS
            or evaluation.get("heldout_fold") not in range(protocol.NUM_FOLDS)
        ):
            raise RuntimeError(f"CV evaluation contract drifted: {path}")
        if evaluation["arm"] in lcp.PROXIMAL_ARMS:
            if evaluation.get("target_kl") not in protocol.TARGET_KL_GRID:
                raise RuntimeError(f"target KL grid drifted: {path}")
        elif evaluation.get("target_kl") is not None:
            raise RuntimeError(f"Direct arm unexpectedly has target KL: {path}")
        checkpoint_path = Path(evaluation["checkpoint"]).resolve()
        if common.sha256_file(checkpoint_path) != evaluation["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint changed after evaluation: {path}")
        report_path, report = validate_training_report(path, evaluation)
        records_path = Path(evaluation["records"]).resolve()
        if common.sha256_file(records_path) != evaluation["records_sha256"]:
            raise RuntimeError(f"evaluation records changed: {records_path}")
        rows = load_records(records_path)
        fold = int(evaluation["heldout_fold"])
        if any(protocol.log_fold(row["log"]) != fold for row in rows):
            raise RuntimeError(f"non-heldout log present in fold {fold}: {path}")
        identifier = configuration_id(
            evaluation["arm"], evaluation.get("target_kl"), evaluation["epoch"]
        )
        groups[identifier].append(
            {
                "path": path,
                "path_sha256": common.sha256_file(path),
                "evaluation": evaluation,
                "records_path": records_path,
                "records": rows,
                "report_path": report_path,
                "report_sha256": common.sha256_file(report_path),
                "report": report,
            }
        )
        mechanism_sha.add(evaluation["mechanism_gate_sha256"])
        pair_sha.add(evaluation["pair_manifest_sha256"])
        anchor_sha.add(evaluation["v3_anchor_sha256"])
    if set(groups) != expected_configuration_ids():
        missing = expected_configuration_ids() - set(groups)
        extra = set(groups) - expected_configuration_ids()
        raise RuntimeError(f"CV grid incomplete: missing={sorted(missing)} extra={sorted(extra)}")
    if len(mechanism_sha) != 1 or len(pair_sha) != 1 or len(anchor_sha) != 1:
        raise RuntimeError("CV trials do not share one mechanism gate/dataset/V3 anchor")
    return groups, mechanism_sha.pop(), pair_sha.pop(), anchor_sha.pop()


def summarize_configuration(identifier, trials, repetitions, seed):
    folds = [int(row["evaluation"]["heldout_fold"]) for row in trials]
    if sorted(folds) != list(range(protocol.NUM_FOLDS)):
        raise RuntimeError(f"{identifier} does not contain exactly five folds")
    rows = []
    fold_gains = {}
    keys = set()
    files = []
    for trial in sorted(trials, key=lambda row: row["evaluation"]["heldout_fold"]):
        fold = int(trial["evaluation"]["heldout_fold"])
        current_keys = set(protocol.rows_by_key(trial["records"]))
        if keys.intersection(current_keys):
            raise RuntimeError(f"token leakage across folds for {identifier}")
        keys.update(current_keys)
        rows.extend(trial["records"])
        fold_gains[fold] = float(
            trial["evaluation"]["summary"]["equal_stratum_hard_gain"]
        )
        files.append(
            {
                "fold": fold,
                "evaluation": str(trial["path"]),
                "evaluation_sha256": trial["path_sha256"],
                "records": str(trial["records_path"]),
                "records_sha256": trial["evaluation"]["records_sha256"],
                "report": str(trial["report_path"]),
                "report_sha256": trial["report_sha256"],
            }
        )
    aggregate = protocol.aggregate_records(
        rows,
        expected_noise_seeds=protocol.FRESH_NOISE_SEEDS,
        fold_gains=fold_gains,
        bootstrap_repetitions=repetitions,
        seed=seed,
    )
    gates = protocol.efficacy_gates(aggregate["summary"], require_folds=True)
    first = trials[0]["evaluation"]
    return {
        "id": identifier,
        "arm": first["arm"],
        "target_kl": first.get("target_kl"),
        "epoch": int(first["epoch"]),
        "summary": aggregate["summary"],
        "strata": aggregate["strata"],
        "efficacy_gates": gates,
        "eligible": all(gates.values()),
        "files": files,
        "_records": rows,
    }


def best_eligible(configurations, arm):
    candidates = [
        row for row in configurations.values() if row["arm"] == arm and row["eligible"]
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["summary"]["equal_stratum_hard_gain"]),
            float(row["target_kl"]) if row["target_kl"] is not None else 0.0,
            int(row["epoch"]),
        ),
    )[0]


def paired_comparison(target, reference, repetitions, seed):
    if target is None or reference is None:
        return None
    return protocol.paired_log_bootstrap(
        target["_records"],
        reference["_records"],
        repetitions=repetitions,
        seed=seed,
    )


def beats_with_margin(comparison):
    return (
        comparison is not None
        and comparison["mean"] >= protocol.THRESHOLDS["method_margin"]
        and comparison["lower_95"] > 0.0
    )


def public_configuration(row):
    if row is None:
        return None
    return {key: value for key, value in row.items() if key != "_records"}


def main():
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    root = args.cv_root.expanduser().resolve()
    groups, mechanism_sha, pair_sha, anchor_sha = load_evaluations(root)
    configurations = {
        identifier: summarize_configuration(
            identifier,
            trials,
            args.bootstrap_repetitions,
            args.seed + offset,
        )
        for offset, (identifier, trials) in enumerate(sorted(groups.items()))
    }

    a0 = best_eligible(configurations, "direct")
    a1 = best_eligible(configurations, "lineage_direct")
    a2 = best_eligible(configurations, "per_draw_proximal")
    a3 = best_eligible(configurations, "lineage_proximal")
    # Attribution baselines need not themselves clear every efficacy gate.  Use
    # the exact matched epoch A0 when the best eligible A0 is absent.
    comparisons = {}
    if a1 is not None:
        a1_a0 = configurations[configuration_id("direct", None, a1["epoch"])]
        comparisons["a1_over_matched_a0"] = paired_comparison(
            a1, a1_a0, args.bootstrap_repetitions, args.seed + 1000
        )
    if a3 is not None:
        a3_a0 = configurations[configuration_id("direct", None, a3["epoch"])]
        comparisons["a3_over_matched_a0"] = paired_comparison(
            a3, a3_a0, args.bootstrap_repetitions, args.seed + 1001
        )
    if a3 is not None and a1 is not None:
        comparisons["best_a3_over_best_a1"] = paired_comparison(
            a3, a1, args.bootstrap_repetitions, args.seed + 1002
        )

    a1_attributed = a1 is not None and beats_with_margin(
        comparisons.get("a1_over_matched_a0")
    )
    a3_over_a0 = a3 is not None and beats_with_margin(
        comparisons.get("a3_over_matched_a0")
    )
    a3_over_a1 = a1 is None or beats_with_margin(
        comparisons.get("best_a3_over_best_a1")
    )
    if a3 is not None and a3_over_a0 and a3_over_a1:
        decision = "AUTHORIZE_A3_LINEAGE_PROXIMAL_NEGATIVE_CONTROL"
        winner = a3
        followup = "five_fold_independent_lineage_shuffle_control"
    elif a1 is not None and a1_attributed and (
        a3 is None
        or comparisons.get("best_a3_over_best_a1", {}).get("mean", -float("inf"))
        < protocol.THRESHOLDS["method_margin"]
        or comparisons.get("best_a3_over_best_a1", {}).get("lower_95", -float("inf"))
        <= 0.0
    ):
        decision = "AUTHORIZE_A1_LINEAGE_DIRECT_NEGATIVE_CONTROL"
        winner = a1
        followup = "five_fold_independent_lineage_shuffle_control"
    elif a2 is not None and a1 is None and a3 is None:
        decision = "PROXIMAL_ONLY_ENGINEERING_BASELINE"
        winner = a2
        followup = "stop_lcpgrpo_paper_claim_and_run_temporal_information_audit"
    else:
        decision = "STOP_LCPGRPO_OBJECTIVE_AND_RUN_TEMPORAL_INFORMATION_AUDIT"
        winner = None
        followup = "proposal_history_zero_new_label_audit"

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable CV gate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "lcpgrpo_five_fold_log_cv_gate_v1",
        "decision": decision,
        "authorized_followup_stage": followup,
        "winner": public_configuration(winner),
        "best_by_arm": {
            "a0_direct": public_configuration(a0),
            "a1_lineage_direct": public_configuration(a1),
            "a2_per_draw_proximal": public_configuration(a2),
            "a3_lineage_proximal": public_configuration(a3),
        },
        "comparisons": comparisons,
        "attribution_gates": {
            "a1_beats_matched_a0_by_0001_with_positive_lower": a1_attributed,
            "a3_beats_matched_a0_by_0001_with_positive_lower": a3_over_a0,
            "a3_beats_a1_by_0001_with_positive_lower": a3_over_a1,
        },
        "configurations": {
            key: public_configuration(value)
            for key, value in sorted(configurations.items())
        },
        "pre_registered_thresholds": protocol.THRESHOLDS,
        "grid": {
            "arms": list(lcp.ARMS),
            "target_kl": list(protocol.TARGET_KL_GRID),
            "checkpoint_epochs": list(protocol.CHECKPOINT_EPOCHS),
            "folds": protocol.NUM_FOLDS,
            "fold_salt": protocol.FOLD_SALT,
        },
        "mechanism_gate_sha256": mechanism_sha,
        "pair_manifest_sha256": pair_sha,
        "v3_anchor_sha256": anchor_sha,
        "cv_root": str(root),
        "scientific_contract": {
            "five_fold_log_disjoint": True,
            "matched_training_budgets": True,
            "fresh_noise_same_train_scene_tokens": True,
            "selector_only": True,
            "generator_frozen": True,
            "official_scalar_pdm_only": True,
            "development_consumed": False,
            "certification_consumed": False,
            "single_winner_maximum_for_development": True,
        },
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(protocol.__file__).resolve()): common.sha256_file(
                Path(protocol.__file__).resolve()
            ),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": decision,
                "winner": winner["id"] if winner is not None else None,
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
