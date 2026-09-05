#!/usr/bin/env python3
"""Gate the real candidate-lineage signal against independent draw shuffles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import grpo_selector_v3_cached_common as common
import lcpgrpo_protocol as protocol
import lineage_consistent_proximal_grpo as lcp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-gate", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def read_json(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text())


def read_records(path, expected_sha):
    path = Path(path).expanduser().resolve()
    if common.sha256_file(path) != expected_sha:
        raise RuntimeError(f"record SHA256 drifted: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    protocol.rows_by_key(rows)
    return rows


def load_real(winner):
    rows = []
    folds = []
    for file_row in winner["files"]:
        fold = int(file_row["fold"])
        folds.append(fold)
        rows.extend(read_records(file_row["records"], file_row["records_sha256"]))
    if sorted(folds) != list(range(protocol.NUM_FOLDS)):
        raise RuntimeError("CV winner lacks five real-lineage folds")
    protocol.rows_by_key(rows)
    return rows


def load_controls(root, winner):
    paths = sorted(root.rglob("control_evaluation.json"))
    if len(paths) != protocol.NUM_FOLDS:
        raise RuntimeError("lineage control requires exactly five fold evaluations")
    rows, folds, fold_gains, files = [], [], {}, []
    for path in paths:
        evaluation = json.loads(path.read_text())
        expected = {
            "status": "PASS",
            "method": protocol.EVALUATION_METHOD,
            "split": "cv",
            "arm": winner["arm"],
            "target_kl": winner["target_kl"],
            "epoch": winner["epoch"],
            "lineage_control": "independent_shuffle",
            "noise_seeds": list(protocol.FRESH_NOISE_SEEDS),
        }
        drift = {
            key: {"actual": evaluation.get(key), "expected": value}
            for key, value in expected.items()
            if evaluation.get(key) != value
        }
        if drift:
            raise RuntimeError(f"lineage-control evaluation drifted at {path}: {drift}")
        fold = int(evaluation["heldout_fold"])
        folds.append(fold)
        record_rows = read_records(
            evaluation["records"], evaluation["records_sha256"]
        )
        if any(protocol.log_fold(row["log"]) != fold for row in record_rows):
            raise RuntimeError("control evaluation contains a training-fold log")
        rows.extend(record_rows)
        fold_gains[fold] = float(evaluation["summary"]["equal_stratum_hard_gain"])
        files.append(
            {
                "fold": fold,
                "evaluation": str(path.resolve()),
                "evaluation_sha256": common.sha256_file(path),
                "records": str(Path(evaluation["records"]).resolve()),
                "records_sha256": evaluation["records_sha256"],
            }
        )
    if sorted(folds) != list(range(protocol.NUM_FOLDS)):
        raise RuntimeError("control folds must be exactly 0..4")
    protocol.rows_by_key(rows)
    aggregate = protocol.aggregate_records(
        rows,
        expected_noise_seeds=protocol.FRESH_NOISE_SEEDS,
        fold_gains=fold_gains,
        bootstrap_repetitions=5000,
        seed=20260902,
    )
    return rows, aggregate, sorted(files, key=lambda row: row["fold"])


def main():
    args = parse_args()
    if args.bootstrap_repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    cv_path, cv_gate = read_json(args.cv_gate)
    if cv_gate.get("decision") not in (
        "AUTHORIZE_A1_LINEAGE_DIRECT_NEGATIVE_CONTROL",
        "AUTHORIZE_A3_LINEAGE_PROXIMAL_NEGATIVE_CONTROL",
    ):
        raise RuntimeError("CV gate did not authorize a lineage negative control")
    winner = cv_gate.get("winner")
    if winner is None or winner.get("arm") not in lcp.LINEAGE_ARMS:
        raise RuntimeError("CV winner is not a lineage method")
    real_rows = load_real(winner)
    control_rows, control_aggregate, control_files = load_controls(
        args.control_root.expanduser().resolve(), winner
    )
    comparison = protocol.paired_log_bootstrap(
        real_rows,
        control_rows,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    gates = {
        "real_over_shuffled_mean_at_least_0001": (
            comparison["mean"] >= protocol.THRESHOLDS["method_margin"]
        ),
        "real_over_shuffled_paired_log_lower_positive": comparison["lower_95"] > 0.0,
    }
    if all(gates.values()):
        decision = "AUTHORIZE_SINGLE_WINNER_DEVELOPMENT"
        followup = "train_once_on_all_train_logs_then_evaluate_development_seeds_3_4_5"
    else:
        decision = "STOP_LINEAGE_ATTRIBUTION_RUN_TEMPORAL_INFORMATION_AUDIT"
        followup = "proposal_history_zero_new_label_audit"

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable control gate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "lcpgrpo_independent_lineage_shuffle_gate_v1",
        "decision": decision,
        "authorized_followup_stage": followup,
        "winner": winner if all(gates.values()) else None,
        "gates": gates,
        "comparison_real_over_independent_shuffle": comparison,
        "control_summary": control_aggregate["summary"],
        "control_strata": control_aggregate["strata"],
        "control_files": control_files,
        "cv_gate": str(cv_path),
        "cv_gate_sha256": common.sha256_file(cv_path),
        "pre_registered_thresholds": {
            "real_over_shuffle_mean": protocol.THRESHOLDS["method_margin"],
            "paired_log_lower": 0.0,
        },
        "scientific_contract": {
            "negative_control_preserves_per_draw_reward_multisets": True,
            "negative_control_breaks_only_cross_draw_candidate_lineage": True,
            "five_fold_log_disjoint": True,
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
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
