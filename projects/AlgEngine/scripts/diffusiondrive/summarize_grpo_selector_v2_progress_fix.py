#!/usr/bin/env python3
"""Audit and compare corrected V2, historical V2, and epoch-100 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


BASELINE_SHA256 = (
    "1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
)
REWARD_CONTRACT = "navsim_pairwise_raw_progress_then_candidate_gate_v1"
SEEDS = (0, 1, 2)
EXPECTED_COUNTS = {
    "openloop_navtest": 12147,
    "openloop_failures": 289,
    "closedloop_nr": 289,
    "closedloop_r": 289,
}
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
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--old-v2-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
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
    result["closedloop_mean.score"] = 0.5 * (
        result["closedloop_nonreactive.score"]
        + result["closedloop_reactive.score"]
    )
    return result


def stats(rows):
    keys = tuple(rows[0])
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows], ddof=0)),
        }
        for key in keys
    }


def verify_counts(payload, label):
    for key, expected in EXPECTED_COUNTS.items():
        if payload.get("counts", {}).get(key) != expected:
            raise RuntimeError(f"{label} count drifted for {key}")


def verify_training(report, hparams, seed, reward_sha):
    expected = {
        "status": "PASS",
        "method": "exact_group_grpo",
        "reward_contract": REWARD_CONTRACT,
        "reward_implementation_sha256": reward_sha,
        "temperature": hparams["temperature"],
        "learning_rate": hparams["learning_rate"],
        "kl_weight": hparams["kl_weight"],
        "train_seed": seed,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"seed {seed} training {key} drifted")
    rows = [
        row
        for row in report.get("checkpoints", [])
        if int(row.get("epoch", -1)) == hparams["epoch"]
    ]
    if len(rows) != 1:
        raise RuntimeError(f"seed {seed} selected epoch is missing or ambiguous")


def fixed_checkpoint(fixed_root, selection, hparams, seed, reward_sha):
    if seed == 0:
        checkpoint = Path(selection["selected_checkpoint"]).resolve()
        expected_sha = selection["selected_checkpoint_sha256"]
        audit = fixed_root / "selection/selected_checkpoint_audit.json"
        training = Path(selection["selected"]["report"]).resolve()
        manifest = fixed_root / "selection/selection.json"
    else:
        manifest, materialized = load_pass(
            fixed_root / f"formal_replicas/seed{seed}/checkpoint_manifest.json"
        )
        for key, value in {
            "temperature": hparams["temperature"],
            "learning_rate": hparams["learning_rate"],
            "kl_weight": hparams["kl_weight"],
            "epoch": hparams["epoch"],
            "train_seed": seed,
            "reward_contract": REWARD_CONTRACT,
            "reward_implementation_sha256": reward_sha,
        }.items():
            if materialized.get(key) != value:
                raise RuntimeError(f"seed {seed} materialized {key} drifted")
        checkpoint = Path(materialized["checkpoint"]).resolve()
        expected_sha = materialized["checkpoint_sha256"]
        audit = fixed_root / f"formal_replicas/seed{seed}/checkpoint_audit.json"
        training = fixed_root / f"formal_replicas/seed{seed}/train/report.json"
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_sha:
        raise RuntimeError(f"seed {seed} checkpoint SHA drifted")
    audit_path, audit_payload = load_pass(audit)
    if audit_payload.get("checkpoint_sha256") != expected_sha:
        raise RuntimeError(f"seed {seed} checkpoint audit SHA drifted")
    if audit_payload.get("changed_tensor_count") != 10:
        raise RuntimeError(f"seed {seed} did not change exactly 10 selector tensors")
    if audit_payload.get("forbidden_changed_tensor_count") != 0:
        raise RuntimeError(f"seed {seed} changed frozen tensors")
    training_path, training_payload = load_pass(training)
    verify_training(training_payload, hparams, seed, reward_sha)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": expected_sha,
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "training_report": str(training_path),
        "training_report_sha256": sha256_file(training_path),
        "materialization_manifest": str(Path(manifest).resolve()),
    }


def delta_stats(current, reference):
    keys = tuple(current[0])
    rows = [
        {key: current[index][key] - reference[index][key] for key in keys}
        for index in range(len(current))
    ]
    return stats(rows)


def format_metric(value):
    return f"{value:.6f}"


def markdown_report(payload):
    selected = payload["selected_hyperparameters"]
    lines = [
        "# DiffusionDrive V2 progress-normalization fix results",
        "",
        f"- Selection mode: `{payload['selection_mode']}`",
        f"- Calibration gate: `{payload['gate_status']}`",
        (
            "- Selected: "
            f"T={selected['temperature']}, lr={selected['learning_rate']}, "
            f"KL={selected['kl_weight']}, epoch={selected['epoch']}"
        ),
        f"- Reward contract: `{REWARD_CONTRACT}`",
        "",
        "| seed | OpenLoop PDM | Rare PDM | CL-NR PDM | CL-R PDM | CL mean | ADE | FDE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        row = payload["seeds"][str(seed)]["fixed"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    format_metric(row["openloop_navtest.score"]),
                    format_metric(row["openloop_failures.score"]),
                    format_metric(row["closedloop_nonreactive.score"]),
                    format_metric(row["closedloop_reactive.score"]),
                    format_metric(row["closedloop_mean.score"]),
                    format_metric(row["openloop_navtest.ade"]),
                    format_metric(row["openloop_navtest.fde"]),
                ]
            )
            + " |"
        )
    best = payload["best_fixed_seed_by_closedloop_mean_pdms"]
    lines.extend(
        [
            "",
            f"Best fixed seed by mean(CL-NR, CL-R) PDM: seed `{best['seed']}` "
            f"with `{best['value']:.6f}`.",
            "",
            "The JSON report contains three-seed means/stds and paired deltas "
            "against both epoch-100 and historical pre-fix V2.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    fixed_root = args.fixed_root.expanduser().resolve()
    old_root = args.old_v2_root.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    selection_path, selection = load_pass(fixed_root / "selection/selection.json")
    if selection.get("baseline_sha256") != BASELINE_SHA256:
        raise RuntimeError("selection baseline SHA drifted")
    if selection.get("reward_contract") != REWARD_CONTRACT:
        raise RuntimeError("selection reward contract drifted")
    selected = selection["selected"]
    hparams = {
        key: selected[key]
        for key in ("temperature", "learning_rate", "kl_weight", "epoch")
    }
    reward_path = (
        Path(__file__).resolve().parents[2]
        / "mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
    )
    reward_sha = sha256_file(reward_path)

    fixed_rows = []
    old_rows = []
    reference_rows = []
    seed_payload = {}
    checkpoints = {}
    for seed in SEEDS:
        checkpoints[str(seed)] = fixed_checkpoint(
            fixed_root, selection, hparams, seed, reward_sha
        )
        reference_path, reference = load_pass(
            reference_root
            / f"formal_eval/e2e_diffusiondrive_reference_paired_s{seed}/summary.json"
        )
        old_path, old = load_pass(
            old_root
            / f"formal/formal_eval/e2e_diffusiondrive_grpo_selector_v2_s{seed}/summary.json"
        )
        fixed_path, fixed = load_pass(
            fixed_root
            / f"formal/formal_eval/e2e_diffusiondrive_grpo_selector_v2_progress_fix_v1_s{seed}/summary.json"
        )
        for label, report in (("reference", reference), ("old", old), ("fixed", fixed)):
            if int(report.get("eval_seed", -1)) != seed:
                raise RuntimeError(f"{label} eval seed drifted for seed {seed}")
            verify_counts(report, f"{label} seed {seed}")
        if reference.get("checkpoint_sha256") != BASELINE_SHA256:
            raise RuntimeError(f"reference checkpoint drifted for seed {seed}")
        if fixed.get("checkpoint_sha256") != checkpoints[str(seed)]["checkpoint_sha256"]:
            raise RuntimeError(f"fixed checkpoint drifted for seed {seed}")
        reference_flat = flatten(reference["metrics"])
        old_flat = flatten(old["metrics"])
        fixed_flat = flatten(fixed["metrics"])
        reference_rows.append(reference_flat)
        old_rows.append(old_flat)
        fixed_rows.append(fixed_flat)
        seed_payload[str(seed)] = {
            "reference_summary": str(reference_path),
            "old_v2_summary": str(old_path),
            "fixed_summary": str(fixed_path),
            "reference": reference_flat,
            "old_v2": old_flat,
            "fixed": fixed_flat,
            "delta_fixed_minus_reference": {
                key: fixed_flat[key] - reference_flat[key] for key in fixed_flat
            },
            "delta_fixed_minus_old_v2": {
                key: fixed_flat[key] - old_flat[key] for key in fixed_flat
            },
        }

    best_seed = max(
        SEEDS,
        key=lambda seed: seed_payload[str(seed)]["fixed"][
            "closedloop_mean.score"
        ],
    )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "e2e_diffusiondrive_grpo_selector_v2_progress_fix_v1",
        "objective": "exact_complete_action_expected_advantage",
        "reward_contract": REWARD_CONTRACT,
        "reward_implementation": str(reward_path),
        "reward_implementation_sha256": reward_sha,
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_mode": selection["selection_mode"],
        "gate_status": selection["gate_status"],
        "selected_hyperparameters": hparams,
        "old_hyperparameter_offline_comparison": selection[
            "old_hyperparameter_comparison_candidate"
        ],
        "expected_counts_per_seed": EXPECTED_COUNTS,
        "checkpoints": checkpoints,
        "seeds": seed_payload,
        "aggregate_reference": stats(reference_rows),
        "aggregate_old_v2": stats(old_rows),
        "aggregate_fixed": stats(fixed_rows),
        "aggregate_delta_fixed_minus_reference": delta_stats(
            fixed_rows, reference_rows
        ),
        "aggregate_delta_fixed_minus_old_v2": delta_stats(fixed_rows, old_rows),
        "best_fixed_seed_by_closedloop_mean_pdms": {
            "seed": best_seed,
            "value": seed_payload[str(best_seed)]["fixed"][
                "closedloop_mean.score"
            ],
        },
        "status_semantics": (
            "PASS certifies provenance, completeness, counts, and frozen-tensor "
            "audits; it does not assert a performance improvement"
        ),
    }
    output_json = args.output_json.expanduser().resolve()
    output_md = args.output_md.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output_md.write_text(markdown_report(payload))
    print(json.dumps({"status": "PASS", "output": str(output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
