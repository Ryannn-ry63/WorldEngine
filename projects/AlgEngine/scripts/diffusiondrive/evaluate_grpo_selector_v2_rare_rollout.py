#!/usr/bin/env python3
"""Evaluate V2 rollout checkpoints on a held-out, log-disjoint hard pool."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

import grpo_selector_v2_rare_common as common
import train_grpo_selector_v3_cached_rare_rollout as rollout
from build_grpo_selector_v3_rare_rollout_data import parse_seed_path


def gather(cache, indices, device):
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    return (
        cache["candidate_features"][tensor_indices].to(device, dtype=torch.float32),
        cache["reference_logits"][tensor_indices].to(device, dtype=torch.float32),
        cache["candidate_rewards"][tensor_indices].to(device, dtype=torch.float32),
        cache["candidate_reward_components"][tensor_indices].to(
            device, dtype=torch.float32
        ),
        cache["candidate_reward_valid_mask"][tensor_indices].to(device),
        [str(cache["scenes"][index]) for index in indices],
    )


def evaluate_rows(model, hard_rows, real_cache, token_map, synthetic, device, temperature, batch_size):
    output = defaultdict(list)
    scenes = []
    kinds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(hard_rows), batch_size):
            rows = hard_rows[start : start + batch_size]
            real_indices = [
                token_map[row["rare_token"]]
                for row in rows
                if row["hard_kind"] == "real_rare"
            ]
            synthetic_indices = [
                int(row["synthetic_index"])
                for row in rows
                if row["hard_kind"] == "synthetic_rollout"
            ]
            batches = []
            if real_indices:
                batches.append(("real_rare", gather(real_cache, real_indices, device)))
            if synthetic_indices:
                batches.append(
                    ("synthetic_rollout", gather(synthetic, synthetic_indices, device))
                )
            for kind, batch in batches:
                features, reference, rewards, components, valid, batch_scenes = batch
                values = common.selector_metrics(
                    reference + common.delta_logits(model, features),
                    reference,
                    rewards,
                    components,
                    valid,
                    temperature,
                )
                for key, value in values.items():
                    output[key].append(value.cpu())
                scenes.extend(batch_scenes)
                kinds.extend([kind] * len(batch_scenes))
    return {key: torch.cat(value) for key, value in output.items()}, scenes, kinds


def subset(values, positions):
    index = torch.tensor(positions, dtype=torch.long)
    return {key: value[index] for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--real-cache", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "certification"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    report_path = args.training_report.expanduser().resolve()
    training = json.loads(report_path.read_text())
    if (
        training.get("status") != "PASS"
        or training.get("architecture") != "diffusiondrive_plan_cls_branch_v2"
        or training.get("hard_pool_split") != "train"
    ):
        raise RuntimeError("rollout tuning report did not use train logs only")
    for implementation, expected_sha in training["implementation_files"].items():
        if common.sha256_file(Path(implementation)) != expected_sha:
            raise RuntimeError(f"training implementation drifted: {implementation}")

    (
        manifest_path,
        manifest,
        hard_pool_path,
        all_rows,
        real_caches,
        real_manifests,
        token_maps,
        synthetic_path,
        synthetic,
    ) = rollout.load_contract(args)
    if training["data_manifest_sha256"] != common.sha256_file(manifest_path):
        raise RuntimeError("training/evaluation data manifest drifted")
    if training["hard_pool_sha256"] != common.sha256_file(hard_pool_path):
        raise RuntimeError("training/evaluation hard pool drifted")
    if training["synthetic_cache_sha256"] != common.sha256_file(synthetic_path):
        raise RuntimeError("training/evaluation synthetic cache drifted")
    hard_rows = [row for row in all_rows if row["split"] == args.split]
    if not hard_rows:
        raise RuntimeError("held-out rollout hard pool is empty")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoints = []
    for checkpoint in training["checkpoints"]:
        state_path = Path(checkpoint["selector_state"]).resolve()
        if common.sha256_file(state_path) != checkpoint["selector_state_sha256"]:
            raise RuntimeError("V2 rollout selector-state SHA256 drifted")
        payload = torch.load(state_path, map_location="cpu")
        if (
            payload.get("architecture") != "diffusiondrive_plan_cls_branch_v2"
            or payload.get("method") != training["method"]
        ):
            raise RuntimeError("V2 rollout selector payload drifted")
        model = common.model_from_cache(real_caches[0], device)
        model.load_state_dict(payload["selector_state"], strict=True)
        values_by_seed = []
        kind_metrics_by_seed = {}
        scenes = None
        for cache, token_map, cache_manifest in zip(
            real_caches, token_maps, real_manifests
        ):
            values, current_scenes, kinds = evaluate_rows(
                model,
                hard_rows,
                cache,
                token_map,
                synthetic,
                device,
                float(training["temperature"]),
                args.batch_size,
            )
            values_by_seed.append(values)
            scenes = current_scenes
            kind_metrics_by_seed[str(cache_manifest["noise_seed"])] = {
                kind: common.summarize(
                    subset(values, [i for i, value in enumerate(kinds) if value == kind])
                )
                for kind in ("real_rare", "synthetic_rollout")
                if kind in kinds
            }
        pooled = {
            key: torch.cat([values[key] for values in values_by_seed])
            for key in values_by_seed[0]
        }
        checkpoints.append(
            {
                **checkpoint,
                "temperature": float(training["temperature"]),
                "learning_rate": float(training["learning_rate"]),
                "kl_weight": float(training["kl_weight"]),
                "train_seed": int(training["train_seed"]),
                "hard_metrics": common.summarize(pooled),
                "hard_metrics_by_noise_seed": {
                    str(row["noise_seed"]): common.summarize(values)
                    for row, values in zip(real_manifests, values_by_seed)
                },
                "hard_kind_metrics_by_noise_seed": kind_metrics_by_seed,
            }
        )

    output_payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v2_rare_rollout_checkpoint_evaluation_v1",
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "split": args.split,
        "log_disjoint": True,
        "hard_only": True,
        "hard_rows": len(hard_rows),
        "hard_logs": len({row["log_name"] for row in hard_rows}),
        "noise_seeds": [int(row["noise_seed"]) for row in real_manifests],
        "training_report": str(report_path),
        "training_report_sha256": common.sha256_file(report_path),
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(hard_pool_path),
        "hard_pool_sha256": common.sha256_file(hard_pool_path),
        "synthetic_cache": str(synthetic_path),
        "synthetic_cache_sha256": common.sha256_file(synthetic_path),
        "checkpoints": checkpoints,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
