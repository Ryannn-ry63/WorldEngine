#!/usr/bin/env python3
"""Fail-closed aggregation of three-seed V2 four-block results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BASELINE_SHA256 = "1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
SEEDS = (0, 1, 2)
PDM_KEYS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "score",
)
FORMAL_HPARAMS = {
    "temperature": 1.0,
    "learning_rate": 1e-3,
    "kl_weight": 1e-3,
    "epoch": 32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pass(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"report did not pass: {path}")
    return path, payload


def verify_hparams(payload, seed):
    for key, value in FORMAL_HPARAMS.items():
        if payload.get(key) != value:
            raise RuntimeError(f"seed {seed} {key} drifted")
    if int(payload.get("train_seed", -1)) != seed:
        raise RuntimeError(f"train seed drifted for seed {seed}")


def flatten(metrics):
    result = {
        "openloop_navtest.ade": float(metrics["openloop_navtest"]["ade"]),
        "openloop_navtest.fde": float(metrics["openloop_navtest"]["fde"]),
    }
    for block in (
        "openloop_navtest",
        "openloop_failures",
        "closedloop_nonreactive",
        "closedloop_reactive",
    ):
        for key in PDM_KEYS:
            result[f"{block}.{key}"] = float(metrics[block][key])
    result["success_rate"] = float(metrics["success_rate"])
    return result


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def result_row(model_name, note, metrics):
    values = [
        metrics["openloop_navtest.ade"],
        metrics["openloop_navtest.fde"],
    ]
    for block in (
        "openloop_navtest",
        "openloop_failures",
        "closedloop_nonreactive",
        "closedloop_reactive",
    ):
        values.extend(metrics[f"{block}.{key}"] for key in PDM_KEYS)
    values.append(metrics["success_rate"])
    return "\t".join(
        [model_name, note] + [f"{value:.6f}" for value in values]
    )


def checkpoint_for_seed(v2_root, selection, seed):
    if seed == 0:
        selected = selection["selected"]
        verify_hparams(selected, seed)
        checkpoint = Path(selection["selected_checkpoint"]).expanduser().resolve()
        expected_sha = selection["selected_checkpoint_sha256"]
        source_path = v2_root / "selection/selection.json"
        source = selection
        audit_path = v2_root / "selection/selected_checkpoint_audit.json"
        training_report = Path(selected["report"]).expanduser().resolve()
    else:
        source_path, source = load_pass(
            v2_root / f"formal_replicas/seed{seed}/checkpoint_manifest.json"
        )
        verify_hparams(source, seed)
        checkpoint = Path(source["checkpoint"]).expanduser().resolve()
        expected_sha = source["checkpoint_sha256"]
        audit_path = v2_root / f"formal_replicas/seed{seed}/checkpoint_audit.json"
        training_report = v2_root / f"formal_replicas/seed{seed}/train/report.json"
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_sha:
        raise RuntimeError(f"checkpoint SHA drift for seed {seed}")
    audit_path, audit = load_pass(audit_path)
    if audit["checkpoint_sha256"] != expected_sha:
        raise RuntimeError(f"checkpoint audit drift for seed {seed}")
    if audit["forbidden_changed_tensor_count"] != 0:
        raise RuntimeError(f"frozen tensor drift for seed {seed}")
    if audit["changed_tensor_count"] != 10:
        raise RuntimeError(f"selector tensor count drift for seed {seed}")
    training_path, training = load_pass(training_report)
    for key, value in (
        ("method", "exact_group_grpo"),
        ("temperature", FORMAL_HPARAMS["temperature"]),
        ("learning_rate", FORMAL_HPARAMS["learning_rate"]),
        ("kl_weight", FORMAL_HPARAMS["kl_weight"]),
        ("train_seed", seed),
    ):
        if training.get(key) != value:
            raise RuntimeError(f"training report {key} drifted for seed {seed}")
    epoch_rows = [
        row
        for row in training.get("checkpoints", [])
        if int(row.get("epoch", -1)) == FORMAL_HPARAMS["epoch"]
    ]
    if len(epoch_rows) != 1:
        raise RuntimeError(
            f"training report lacks exact epoch-32 state for seed {seed}"
        )
    return {
        "path": str(checkpoint),
        "sha256": expected_sha,
        "source_manifest": str(Path(source_path).resolve()),
        "source_manifest_sha256": sha256_file(source_path),
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "training_report": str(training_path),
        "training_report_sha256": sha256_file(training_path),
    }


def main():
    args = parse_args()
    v2_root = args.v2_root.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    selection_path, selection = load_pass(v2_root / "selection/selection.json")
    if selection.get("objective") != "exact_complete_action_expected_advantage":
        raise RuntimeError("V2 objective drifted")
    if selection.get("baseline_sha256") != BASELINE_SHA256:
        raise RuntimeError("V2 baseline SHA drifted")

    checkpoints = {}
    paired = {}
    all_reference = []
    all_current = []
    for seed in SEEDS:
        checkpoint = checkpoint_for_seed(v2_root, selection, seed)
        checkpoints[str(seed)] = checkpoint
        reference_path, reference = load_pass(
            reference_root
            / f"formal_eval/e2e_diffusiondrive_reference_paired_s{seed}/summary.json"
        )
        current_path, current = load_pass(
            v2_root
            / f"formal/formal_eval/e2e_diffusiondrive_grpo_selector_v2_s{seed}/summary.json"
        )
        if int(reference.get("eval_seed", -1)) != seed:
            raise RuntimeError(f"reference eval seed drift for seed {seed}")
        if int(current.get("eval_seed", -1)) != seed:
            raise RuntimeError(f"current eval seed drift for seed {seed}")
        if reference.get("checkpoint_sha256") != BASELINE_SHA256:
            raise RuntimeError(f"reference checkpoint drift for seed {seed}")
        if current.get("checkpoint_sha256") != checkpoint["sha256"]:
            raise RuntimeError(f"current checkpoint drift for seed {seed}")
        for count_key in (
            "openloop_navtest",
            "openloop_failures",
            "closedloop_nr",
            "closedloop_r",
        ):
            if reference["counts"][count_key] != current["counts"][count_key]:
                raise RuntimeError(
                    f"paired count mismatch seed={seed} block={count_key}"
                )
        reference_flat = flatten(reference["metrics"])
        current_flat = flatten(current["metrics"])
        delta = {
            key: current_flat[key] - reference_flat[key]
            for key in reference_flat
        }
        all_reference.append(reference_flat)
        all_current.append(current_flat)
        paired[str(seed)] = {
            "reference_summary": str(reference_path),
            "reference_summary_sha256": sha256_file(reference_path),
            "current_summary": str(current_path),
            "current_summary_sha256": sha256_file(current_path),
            "reference": reference_flat,
            "current": current_flat,
            "delta_current_minus_reference": delta,
        }

    metric_keys = tuple(all_reference[0])
    aggregate_reference = {
        key: stats([row[key] for row in all_reference]) for key in metric_keys
    }
    aggregate_current = {
        key: stats([row[key] for row in all_current]) for key in metric_keys
    }
    aggregate_delta = {
        key: stats(
            [
                all_current[index][key] - all_reference[index][key]
                for index in range(len(SEEDS))
            ]
        )
        for key in metric_keys
    }
    reference_means = {
        key: value["mean"] for key, value in aggregate_reference.items()
    }
    current_means = {
        key: value["mean"] for key, value in aggregate_current.items()
    }
    reference_row = result_row(
        "e2e_diffusiondrive_reference_paired_3seed",
        "epoch100; paired three-seed mean; exact episode success",
        reference_means,
    )
    current_row = result_row(
        "e2e_diffusiondrive_grpo_selector_v2",
        "exact-group selector GRPO; T=1; lr=1e-3; KL=1e-3; epoch=32; paired three-seed mean",
        current_means,
    )
    score_blocks = (
        "openloop_navtest.score",
        "openloop_failures.score",
        "closedloop_nonreactive.score",
        "closedloop_reactive.score",
    )
    performance_diagnostic = {
        "positive_pdm_blocks": sum(
            aggregate_delta[key]["mean"] > 0.0 for key in score_blocks
        ),
        "pdm_block_mean_deltas": {
            key: aggregate_delta[key]["mean"] for key in score_blocks
        },
        "closedloop_safety_mean_deltas": {
            key: aggregate_delta[key]["mean"]
            for key in (
                "closedloop_nonreactive.no_at_fault_collisions",
                "closedloop_nonreactive.time_to_collision_within_bound",
                "closedloop_reactive.no_at_fault_collisions",
                "closedloop_reactive.time_to_collision_within_bound",
            )
        },
        "note": "diagnostic only; status PASS means completeness, not performance significance",
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "e2e_diffusiondrive_grpo_selector_v2",
        "objective": "exact_complete_action_expected_advantage",
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "formal_hyperparameters": FORMAL_HPARAMS,
        "train_seeds": list(SEEDS),
        "evaluation_seeds": list(SEEDS),
        "reference_reuse_contract": (
            "same epoch100 SHA, formal_navtest_seed{seed} candidate noise, "
            "unchanged V1-to-V2 inference path, and paired block counts"
        ),
        "success_rate_definition": (
            "mean across seeds of each seed's fraction of reactive episodes "
            "with NC == 1 and DAC == 1"
        ),
        "checkpoints": checkpoints,
        "paired_results": paired,
        "aggregate_reference": aggregate_reference,
        "aggregate_current": aggregate_current,
        "aggregate_delta_current_minus_reference": aggregate_delta,
        "performance_diagnostic": performance_diagnostic,
        "reference_tsv_row": reference_row,
        "current_tsv_row": current_row,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.with_suffix(output.suffix + ".tsv").write_text(
        reference_row + "\n" + current_row + "\n"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
