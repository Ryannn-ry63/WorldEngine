#!/usr/bin/env python3
"""Evaluate V2 rare-original checkpoints on log-disjoint rare tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import evaluate_grpo_selector_v3_rare_original as v3_evaluator
import grpo_selector_v2_rare_common as common


def evaluate_state(state_path, caches, rare_indices, temperature, batch_size, device):
    state_path = Path(state_path).expanduser().resolve()
    payload = torch.load(state_path, map_location="cpu")
    if (
        payload.get("schema_version") != 3
        or payload.get("architecture") != "diffusiondrive_plan_cls_branch_v2"
    ):
        raise RuntimeError(f"invalid V2 selector state: {state_path}")
    model = common.model_from_cache(caches[0], device)
    model.load_state_dict(payload["selector_state"], strict=True)
    values = [
        v3_evaluator.subset_values(
            common.evaluate(model, cache, device, temperature, batch_size),
            rare_indices,
        )
        for cache in caches
    ]
    return payload, values


def validate_training_report(path):
    path = Path(path).expanduser().resolve()
    report = json.loads(path.read_text())
    if (
        report.get("status") != "PASS"
        or report.get("schema_version") != 3
        or report.get("architecture") != "diffusiondrive_plan_cls_branch_v2"
        or report.get("sampling_mode") != "rare_balanced"
        or report.get("reward_contract") != common.REWARD_CONTRACT
    ):
        raise RuntimeError(f"invalid V2 rare-original training report: {path}")
    for implementation, expected_sha in report["implementation_files"].items():
        if common.sha256_file(Path(implementation)) != expected_sha:
            raise RuntimeError(f"training implementation drifted: {implementation}")
    return path, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--split", choices=tuple(v3_evaluator.EXPECTED_SEEDS), default="development"
    )
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    report_path, training = validate_training_report(args.training_report)
    (
        caches,
        manifests,
        pair_rows,
        rare_indices,
        pair_path,
        audit_path,
    ) = v3_evaluator.load_evaluation_contract(
        args.cache,
        args.split,
        args.pair_manifest,
        args.rare_data_audit,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint_rows = []
    for checkpoint in training["checkpoints"]:
        state_path = Path(checkpoint["selector_state"]).resolve()
        if common.sha256_file(state_path) != checkpoint["selector_state_sha256"]:
            raise RuntimeError(f"V2 selector state SHA256 drifted: {state_path}")
        state, values_by_seed = evaluate_state(
            state_path,
            caches,
            rare_indices,
            float(training["temperature"]),
            args.batch_size,
            device,
        )
        if state.get("method") != training["method"]:
            raise RuntimeError("selector state/training method drifted")
        checkpoint_rows.append(
            {
                **checkpoint,
                "temperature": float(training["temperature"]),
                "learning_rate": float(training["learning_rate"]),
                "kl_weight": float(training["kl_weight"]),
                "train_seed": int(training["train_seed"]),
                "rare_metrics": common.summarize(
                    v3_evaluator.pooled(values_by_seed)
                ),
                "rare_metrics_by_noise_seed": {
                    str(manifest["noise_seed"]): common.summarize(values)
                    for manifest, values in zip(manifests, values_by_seed)
                },
                "rare_vote_strata": v3_evaluator.vote_strata(
                    values_by_seed, pair_rows
                ),
            }
        )

    output_payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v2_rare_original_checkpoint_evaluation_v1",
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "split": args.split,
        "noise_seeds": list(v3_evaluator.EXPECTED_SEEDS[args.split]),
        "rare_only": True,
        "training_report": str(report_path),
        "training_report_sha256": common.sha256_file(report_path),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "manifests": list(manifests),
        "checkpoints": checkpoint_rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
