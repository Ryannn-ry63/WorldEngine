#!/usr/bin/env python3
"""Apply the scene-held-out V2 gate and materialize one full checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


CURRENT_PREFIX = "planning_head.diff_decoder.layers.1.task_decoder.plan_cls_branch."
REFERENCE_PREFIX = "planning_head.reference_selector."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--gain-floor", type=float, default=0.003)
    parser.add_argument("--worst-seed-floor", type=float, default=-0.001)
    parser.add_argument("--disagreement-floor", type=float, default=0.02)
    parser.add_argument("--tie-tolerance", type=float, default=0.0002)
    parser.add_argument("--expected-reward-contract")
    parser.add_argument(
        "--allow-predeclared-fallback", action="store_true"
    )
    parser.add_argument("--fallback-temperature", type=float, default=1.0)
    parser.add_argument("--fallback-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fallback-kl-weight", type=float, default=1e-3)
    parser.add_argument("--fallback-epoch", type=int, default=32)
    parser.add_argument(
        "--experiment-method",
        default="e2e_diffusiondrive_grpo_selector_v2",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path, expected_sha):
    path = Path(path).expanduser().resolve()
    if sha256_file(path) != expected_sha:
        raise RuntimeError(f"calibration record SHA256 mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise RuntimeError(f"empty calibration records: {path}")
    return rows


def scene_bootstrap(rows, replicates, seed=20260810):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["scene"])].append(float(row["top1_reward_gain"]))
    names = sorted(grouped)
    arrays = [np.asarray(grouped[name], dtype=np.float64) for name in names]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        samples[index] = np.concatenate([arrays[item] for item in selected]).mean()
    values = np.asarray(
        [float(row["top1_reward_gain"]) for row in rows], dtype=np.float64
    )
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_unit": "scene/log",
        "replicates": replicates,
    }


def checkpoint_state(payload):
    if not isinstance(payload, dict):
        raise TypeError("baseline checkpoint must be a dict")
    for key in ("state_dict", "model"):
        if isinstance(payload.get(key), dict):
            return payload[key], key
    return payload, None


def materialize_checkpoint(baseline_path, selector_path, output, selection):
    baseline_payload = torch.load(str(baseline_path), map_location="cpu")
    payload = copy.deepcopy(baseline_payload)
    state, state_container = checkpoint_state(payload)
    selector_payload = torch.load(str(selector_path), map_location="cpu")
    selector_state = selector_payload.get("selector_state")
    if not isinstance(selector_state, dict) or len(selector_state) != 10:
        raise RuntimeError("selected state is not the 10-tensor final selector")

    has_module_prefix = any(key.startswith("module.") for key in state)
    current_prefix = ("module." if has_module_prefix else "") + CURRENT_PREFIX
    reference_prefix = ("module." if has_module_prefix else "") + REFERENCE_PREFIX
    missing = [key for key in selector_state if current_prefix + key not in state]
    if missing:
        raise RuntimeError("baseline selector keys missing: " + ", ".join(missing))
    for relative, value in selector_state.items():
        baseline_value = state[current_prefix + relative]
        if tuple(value.shape) != tuple(baseline_value.shape):
            raise RuntimeError(f"selector shape drift: {relative}")
        state[current_prefix + relative] = value.to(dtype=baseline_value.dtype)
        state[reference_prefix + relative] = baseline_value.clone()
    if state_container is not None:
        payload[state_container] = state

    meta = payload.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {"upstream_meta": str(meta)}
        payload["meta"] = meta
    meta["diffusiondrive_selector_grpo_v2"] = selection
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def main():
    args = parse_args()
    baseline = args.baseline.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_sha = sha256_file(baseline)
    if (
        args.expected_baseline_sha256 is not None
        and baseline_sha != args.expected_baseline_sha256
    ):
        raise RuntimeError("baseline checkpoint SHA256 mismatch")

    candidates = []
    for report_arg in args.reports:
        report_path = report_arg.expanduser().resolve()
        report = json.loads(report_path.read_text())
        if report.get("status") != "PASS" or report.get("method") != "exact_group_grpo":
            raise RuntimeError(f"invalid V2 report: {report_path}")
        if (
            args.expected_reward_contract is not None
            and report.get("reward_contract") != args.expected_reward_contract
        ):
            raise RuntimeError(f"stale V2 reward contract: {report_path}")
        for checkpoint in report["checkpoints"]:
            rows = read_records(
                checkpoint["calibration_records"],
                checkpoint["calibration_records_sha256"],
            )
            noise_values = defaultdict(list)
            for row in rows:
                noise_values[int(row["noise_seed"])].append(
                    float(row["top1_reward_gain"])
                )
            if sorted(noise_values) != [0, 1, 2]:
                raise RuntimeError("candidate omitted a calibration noise seed")
            noise_means = {
                str(seed): float(np.mean(values))
                for seed, values in sorted(noise_values.items())
            }
            bootstrap = scene_bootstrap(
                rows, args.bootstrap_replicates, seed=20260810 + int(checkpoint["epoch"])
            )
            disagreement = float(
                np.mean([float(row["selection_disagreement"]) for row in rows])
            )
            positive_seeds = sum(value > 0.0 for value in noise_means.values())
            gate_checks = {
                "mean_gain": bootstrap["mean"] >= args.gain_floor,
                "ci95_lower_positive": bootstrap["ci95_low"] > 0.0,
                "positive_noise_seeds": positive_seeds >= 2,
                "worst_noise_seed": min(noise_means.values())
                >= args.worst_seed_floor,
                "selection_disagreement": disagreement
                > args.disagreement_floor,
            }
            candidates.append(
                {
                    "report": str(report_path),
                    "selector_state": checkpoint["selector_state"],
                    "selector_state_sha256": checkpoint["selector_state_sha256"],
                    "temperature": float(report["temperature"]),
                    "learning_rate": float(report["learning_rate"]),
                    "kl_weight": float(report["kl_weight"]),
                    "train_seed": int(report["train_seed"]),
                    "epoch": int(checkpoint["epoch"]),
                    "top1_gain_bootstrap": bootstrap,
                    "noise_seed_top1_gain": noise_means,
                    "positive_noise_seed_count": positive_seeds,
                    "selection_disagreement": disagreement,
                    "gate_checks": gate_checks,
                    "eligible": all(gate_checks.values()),
                    "calibration_metrics": checkpoint["calibration_metrics"],
                }
            )
    if not candidates:
        raise RuntimeError("no V2 candidates found")

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selection_mode = "calibration_gate"
    if eligible:
        best_gain = max(
            row["top1_gain_bootstrap"]["mean"] for row in eligible
        )
        tied = [
            row
            for row in eligible
            if row["top1_gain_bootstrap"]["mean"]
            >= best_gain - args.tie_tolerance
        ]
        selected = min(
            tied,
            key=lambda row: (
                row["epoch"],
                row["kl_weight"],
                row["temperature"],
                row["learning_rate"],
            ),
        )
    elif args.allow_predeclared_fallback:
        fallback = [
            row
            for row in candidates
            if row["temperature"] == args.fallback_temperature
            and row["learning_rate"] == args.fallback_learning_rate
            and row["kl_weight"] == args.fallback_kl_weight
            and row["epoch"] == args.fallback_epoch
            and row["train_seed"] == 0
        ]
        if len(fallback) != 1:
            raise RuntimeError(
                "predeclared fallback did not identify exactly one candidate"
            )
        selected = fallback[0]
        selection_mode = "predeclared_old_hparams_fallback"
    else:
        pool = candidates
        best_gain = max(
            row["top1_gain_bootstrap"]["mean"] for row in pool
        )
        tied = [
            row
            for row in pool
            if row["top1_gain_bootstrap"]["mean"]
            >= best_gain - args.tie_tolerance
        ]
        selected = min(
            tied,
            key=lambda row: (
                row["epoch"],
                row["kl_weight"],
                row["temperature"],
                row["learning_rate"],
            ),
        )
    selector_path = Path(selected["selector_state"]).expanduser().resolve()
    if sha256_file(selector_path) != selected["selector_state_sha256"]:
        raise RuntimeError("selected selector-state SHA256 mismatch")

    gate_status = "PASS" if eligible else "CALIBRATION_GATE_FAIL"
    status = (
        "PASS"
        if eligible or args.allow_predeclared_fallback
        else "CALIBRATION_GATE_FAIL"
    )
    old_hparams = [
        row
        for row in candidates
        if row["temperature"] == args.fallback_temperature
        and row["learning_rate"] == args.fallback_learning_rate
        and row["kl_weight"] == args.fallback_kl_weight
        and row["epoch"] == args.fallback_epoch
        and row["train_seed"] == 0
    ]
    if len(old_hparams) != 1:
        raise RuntimeError("old-hyperparameter comparison candidate is ambiguous")
    manifest = {
        "schema_version": 3,
        "status": status,
        "gate_status": gate_status,
        "selection_mode": selection_mode,
        "method": args.experiment_method,
        "objective": "exact_complete_action_expected_advantage",
        "reward_contract": args.expected_reward_contract,
        "selection_data": "navtrain_scene_disjoint_calibration_only",
        "baseline": str(baseline),
        "baseline_sha256": baseline_sha,
        "gate": {
            "mean_top1_pdm_gain_floor": args.gain_floor,
            "scene_bootstrap_ci95_lower_floor": 0.0,
            "minimum_positive_noise_seeds": 2,
            "worst_noise_seed_floor": args.worst_seed_floor,
            "selection_disagreement_floor": args.disagreement_floor,
        },
        "selected": selected,
        "predeclared_fallback": {
            "temperature": args.fallback_temperature,
            "learning_rate": args.fallback_learning_rate,
            "kl_weight": args.fallback_kl_weight,
            "epoch": args.fallback_epoch,
        },
        "old_hyperparameter_comparison_candidate": old_hparams[0],
        "num_candidates": len(candidates),
        "num_eligible_candidates": len(eligible),
        "candidates": sorted(
            candidates,
            key=lambda row: (
                row["temperature"],
                row["learning_rate"],
                row["kl_weight"],
                row["epoch"],
            ),
        ),
    }
    selected_checkpoint = output_dir / "selected_checkpoint.pth"
    if status == "PASS":
        materialize_checkpoint(
            baseline,
            selector_path,
            selected_checkpoint,
            {
                **{
                    key: selected[key]
                    for key in (
                        "temperature",
                        "learning_rate",
                        "kl_weight",
                        "train_seed",
                        "epoch",
                        "top1_gain_bootstrap",
                    )
                },
                "method": args.experiment_method,
                "reward_contract": args.expected_reward_contract,
                "selection_mode": selection_mode,
                "gate_status": gate_status,
            },
        )
        manifest["selected_checkpoint"] = str(selected_checkpoint)
        manifest["selected_checkpoint_sha256"] = sha256_file(selected_checkpoint)
    manifest_path = output_dir / "selection.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "selection": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
