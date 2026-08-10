#!/usr/bin/env python3
"""Fail-closed final acceptance for the three-seed selector-GRPO study.

This script never launches evaluation and never edits the human result table.  It
only accepts already completed, paired reference/current summaries and writes an
auditable aggregate plus two ready-to-paste TSV rows.
"""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"report did not pass: {path}")
    return path, payload


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    selection_path, selection = load_json(root / "selection/selected_seed0.json")
    selected = selection["selected"]
    learning_rate = float(selected["learning_rate"])
    epoch = int(selected["epoch"])
    if int(selected["train_seed"]) != 0:
        raise RuntimeError("selection was not made with train seed 0")
    if tuple(selected["noise_seeds"]) != SEEDS:
        raise RuntimeError("selection did not use calibration noise seeds 0/1/2")

    checkpoints = {}
    calibrations = {}
    pairs = {}
    all_reference = []
    all_current = []
    for seed in SEEDS:
        if seed == 0:
            checkpoint = Path(selected["checkpoint"]).expanduser().resolve()
            expected_checkpoint_sha = selected["checkpoint_sha256"]
            calibration_paths = [Path(path) for path in selected["reports"]]
        else:
            run_name = f"selected_lr{selected['learning_rate']}_seed{seed}"
            checkpoint = root / f"train/{run_name}/epoch_{epoch}.pth"
            audit_path, audit = load_json(
                root / f"train/{run_name}/checkpoint_audits/epoch_{epoch}.json"
            )
            expected_checkpoint_sha = audit["checkpoint_sha256"]
            calibration_paths = [
                root / f"calibration/{run_name}/epoch_{epoch}_noise_{noise}.json"
                for noise in SEEDS
            ]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_checkpoint_sha = sha256_file(checkpoint)
        if actual_checkpoint_sha != expected_checkpoint_sha:
            raise RuntimeError(f"checkpoint SHA drift for train seed {seed}")
        checkpoints[str(seed)] = {
            "path": str(checkpoint),
            "sha256": actual_checkpoint_sha,
        }

        calibration_reports = []
        for calibration_path in calibration_paths:
            path, report = load_json(calibration_path)
            if int(report["train_seed"]) != seed:
                raise RuntimeError(f"calibration train-seed drift: {path}")
            if int(report["epoch"]) != epoch:
                raise RuntimeError(f"calibration epoch drift: {path}")
            if float(report["learning_rate"]) != learning_rate:
                raise RuntimeError(f"calibration LR drift: {path}")
            if report["checkpoint_sha256"] != actual_checkpoint_sha:
                raise RuntimeError(f"calibration checkpoint drift: {path}")
            calibration_reports.append((path, report))
        noise_seeds = tuple(sorted(int(report["noise_seed"]) for _, report in calibration_reports))
        if noise_seeds != SEEDS:
            raise RuntimeError(f"seed {seed} lacks calibration noise seeds 0/1/2")
        calibrations[str(seed)] = [
            {"path": str(path), "sha256": sha256_file(path)}
            for path, _ in sorted(calibration_reports, key=lambda item: item[1]["noise_seed"])
        ]

        reference_path, reference = load_json(
            root / f"formal_eval/e2e_diffusiondrive_reference_paired_s{seed}/summary.json"
        )
        current_path, current = load_json(
            root / f"formal_eval/e2e_diffusiondrive_grpo_selector_s{seed}/summary.json"
        )
        if int(reference["eval_seed"]) != seed or int(current["eval_seed"]) != seed:
            raise RuntimeError(f"evaluation seed drift for pair {seed}")
        if reference["checkpoint_sha256"] != BASELINE_SHA256:
            raise RuntimeError(f"reference is not epoch-100 baseline for seed {seed}")
        if current["checkpoint_sha256"] != actual_checkpoint_sha:
            raise RuntimeError(f"current evaluation checkpoint drift for seed {seed}")
        for count_key in (
            "openloop_navtest",
            "openloop_failures",
            "closedloop_nr",
            "closedloop_r",
        ):
            if reference["counts"][count_key] != current["counts"][count_key]:
                raise RuntimeError(f"paired count mismatch seed={seed} block={count_key}")
        reference_flat = flatten(reference["metrics"])
        current_flat = flatten(current["metrics"])
        delta = {key: current_flat[key] - reference_flat[key] for key in reference_flat}
        all_reference.append(reference_flat)
        all_current.append(current_flat)
        pairs[str(seed)] = {
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
            [all_current[index][key] - all_reference[index][key] for index in range(3)]
        )
        for key in metric_keys
    }
    reference_means = {key: value["mean"] for key, value in aggregate_reference.items()}
    current_means = {key: value["mean"] for key, value in aggregate_current.items()}
    reference_row = result_row(
        "e2e_diffusiondrive_reference_paired_3seed",
        "epoch100; paired three-seed mean; exact episode success",
        reference_means,
    )
    current_row = result_row(
        "e2e_diffusiondrive_grpo_selector",
        f"selector-only GRPO; lr={selected['learning_rate']}; epoch={epoch}; paired three-seed mean",
        current_means,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "e2e_diffusiondrive_grpo_selector",
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selected_learning_rate": learning_rate,
        "selected_epoch": epoch,
        "train_seeds": list(SEEDS),
        "evaluation_seeds": list(SEEDS),
        "success_rate_definition": (
            "mean across seeds of each seed's fraction of reactive episodes "
            "with NC == 1 and DAC == 1"
        ),
        "checkpoints": checkpoints,
        "calibration_reports": calibrations,
        "paired_results": pairs,
        "aggregate_reference": aggregate_reference,
        "aggregate_current": aggregate_current,
        "aggregate_delta_current_minus_reference": aggregate_delta,
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
