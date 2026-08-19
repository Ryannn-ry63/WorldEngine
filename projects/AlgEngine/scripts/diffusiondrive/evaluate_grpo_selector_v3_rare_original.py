#!/usr/bin/env python3
"""Evaluate rare-original V3 checkpoints on rare tokens from fixed-noise caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as trainer


EXPECTED_SEEDS = {
    "development": (3, 4, 5),
    "certification": (6, 7, 8),
}


def load_evaluation_contract(cache_paths, split, pair_manifest, rare_data_audit):
    pairs = [common.load_cache(path, split) for path in cache_paths]
    pairs.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*pairs)
    standard.assert_cache_group(caches, manifests, split, EXPECTED_SEEDS[split])
    pair_rows, audit, pair_path, audit_path = trainer.load_pair_contract(
        pair_manifest, rare_data_audit
    )
    rare_indices, _ = trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, audit
    )
    return (
        caches,
        manifests,
        pair_rows,
        torch.tensor(rare_indices, dtype=torch.long),
        pair_path,
        audit_path,
    )


def subset_values(values, indices):
    return {key: value[indices] for key, value in values.items()}


def evaluate_state(state_path, caches, rare_indices, temperature, batch_size, device):
    state_path = Path(state_path).expanduser().resolve()
    payload = torch.load(state_path, map_location="cpu")
    if payload.get("schema_version") != 3:
        raise RuntimeError(f"invalid selector state schema: {state_path}")
    model, config = common.model_from_cache(caches[0], payload["ablation"])
    if config != payload["scene_selector_config"]:
        raise RuntimeError("selector state/cache architecture drifted")
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    model = model.to(device)
    values = [
        subset_values(
            common.evaluate(model, cache, device, temperature, batch_size),
            rare_indices,
        )
        for cache in caches
    ]
    return payload, values


def pooled(values_by_seed):
    return {
        key: torch.cat([values[key] for values in values_by_seed])
        for key in values_by_seed[0]
    }


def vote_strata(values_by_seed, pair_rows):
    output = {}
    for vote in (1, 2, 3):
        positions = torch.tensor(
            [
                index
                for index, row in enumerate(pair_rows)
                if int(row["rare_votes"]) == vote
            ],
            dtype=torch.long,
        )
        if not len(positions):
            continue
        output[str(vote)] = {
            "rare_tokens": len(positions),
            "metrics": common.summarize(
                pooled(
                    [
                        subset_values(values, positions)
                        for values in values_by_seed
                    ]
                )
            ),
        }
    return output


def validate_training_report(path):
    path = Path(path).expanduser().resolve()
    report = json.loads(path.read_text())
    if (
        report.get("status") != "PASS"
        or report.get("schema_version") != 3
        or report.get("sampling_mode") != "rare_balanced"
    ):
        raise RuntimeError(f"invalid rare-original training report: {path}")
    for implementation, expected_sha in report.get(
        "implementation_files", {}
    ).items():
        if common.sha256_file(Path(implementation)) != expected_sha:
            raise RuntimeError(
                f"training implementation provenance drifted: {implementation}"
            )
    return path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--split", choices=tuple(EXPECTED_SEEDS), default="development"
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
    ) = load_evaluation_contract(
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
        state_path = Path(checkpoint["scene_selector_state"]).resolve()
        if common.sha256_file(state_path) != checkpoint[
            "scene_selector_state_sha256"
        ]:
            raise RuntimeError(f"selector state SHA256 drifted: {state_path}")
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
                "rare_metrics": common.summarize(pooled(values_by_seed)),
                "rare_metrics_by_noise_seed": {
                    str(manifest["noise_seed"]): common.summarize(values)
                    for manifest, values in zip(manifests, values_by_seed)
                },
                "rare_vote_strata": vote_strata(values_by_seed, pair_rows),
            }
        )

    output_payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_original_checkpoint_evaluation_v1",
        "split": args.split,
        "noise_seeds": list(EXPECTED_SEEDS[args.split]),
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
