#!/usr/bin/env python3
"""Train a context-conditioned selector with the exact-group GRPO objective."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--development-cache", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--kl-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--checkpoint-epochs", default="1,2,4,8,16,32,64")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ablation",
        choices=("full", "feature_only", "feature_geometry", "feature_geometry_route"),
        default="full",
    )
    return parser.parse_args()


def selector_states_equal(left, right):
    return (
        set(left) == set(right)
        and all(torch.equal(left[key], right[key]) for key in left)
    )


def assert_cache_group(caches, manifests, split, expected_seeds):
    seeds = [int(manifest["noise_seed"]) for manifest in manifests]
    if seeds != list(expected_seeds):
        raise RuntimeError(f"{split} cache seeds {seeds} != {list(expected_seeds)}")
    first_tokens = caches[0]["tokens"]
    first_scenes = caches[0]["scenes"]
    first_config = caches[0]["scene_selector_config"]
    first_selector_state = caches[0]["baseline_selector_state"]
    identity_keys = ("checkpoint_sha256", "config_sha256", "annotation_sha256")
    first_identity = tuple(manifests[0][key] for key in identity_keys)
    for cache, manifest in zip(caches[1:], manifests[1:]):
        if cache["tokens"] != first_tokens or cache["scenes"] != first_scenes:
            raise RuntimeError(f"{split} cache token/scene alignment drifted across noise")
        if cache["scene_selector_config"] != first_config:
            raise RuntimeError(f"{split} selector config drifted across noise")
        identity = tuple(manifest[key] for key in identity_keys)
        if identity != first_identity:
            raise RuntimeError(
                f"{split} baseline/config/annotation identity drifted across noise"
            )
        if not selector_states_equal(
            cache["baseline_selector_state"], first_selector_state
        ):
            raise RuntimeError(f"{split} frozen reference selector drifted across noise")


def write_records(path, caches, manifests, values_by_seed):
    with path.open("w") as stream:
        for cache, manifest, values in zip(caches, manifests, values_by_seed):
            for index, token in enumerate(cache["tokens"]):
                row = {
                    "token": str(token),
                    "scene": str(cache["scenes"][index]),
                    "noise_seed": int(manifest["noise_seed"]),
                    "top1_reward_gain": float(values["top1_reward_gain"][index]),
                    "selection_disagreement": float(values["selection_disagreement"][index]),
                    "oracle_match": float(values["oracle_match"][index]),
                    "oracle_regret": float(values["oracle_regret"][index]),
                    "component_delta": {
                        name: float(values["component_delta"][index, component_index])
                        for component_index, name in enumerate(common.COMPONENT_NAMES)
                    },
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    args = parse_args()
    if args.temperature <= 0 or args.learning_rate <= 0 or args.kl_weight < 0:
        raise ValueError("invalid temperature/learning-rate/KL weight")
    checkpoint_epochs = tuple(sorted({int(value) for value in args.checkpoint_epochs.split(",")}))
    if not checkpoint_epochs or checkpoint_epochs[-1] != args.epochs:
        raise ValueError("checkpoint epochs must end at --epochs")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_pairs = [common.load_cache(path, "train") for path in args.train_cache]
    train_pairs.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    development_pairs = [
        common.load_cache(path, "development") for path in args.development_cache
    ]
    development_pairs.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    train_caches, train_manifests = zip(*train_pairs)
    development_caches, development_manifests = zip(*development_pairs)
    assert_cache_group(train_caches, train_manifests, "train", (0, 1, 2))
    assert_cache_group(development_caches, development_manifests, "development", (3, 4, 5))
    train_identity = tuple(
        train_manifests[0][key]
        for key in ("checkpoint_sha256", "config_sha256", "annotation_sha256")
    )
    development_identity = tuple(
        development_manifests[0][key]
        for key in ("checkpoint_sha256", "config_sha256", "annotation_sha256")
    )
    if train_identity != development_identity:
        raise RuntimeError("train/development baseline/config identity drifted")
    if train_caches[0]["scene_selector_config"] != development_caches[0]["scene_selector_config"]:
        raise RuntimeError("train/development selector architecture drifted")
    if not selector_states_equal(
        train_caches[0]["baseline_selector_state"],
        development_caches[0]["baseline_selector_state"],
    ):
        raise RuntimeError("train/development frozen reference selector drifted")
    train_tokens = set(train_caches[0]["tokens"])
    development_tokens = set(development_caches[0]["tokens"])
    if train_tokens.intersection(development_tokens):
        raise RuntimeError("train/development token leakage")
    train_scenes = set(train_caches[0]["scenes"])
    development_scenes = set(development_caches[0]["scenes"])
    if train_scenes.intersection(development_scenes):
        raise RuntimeError("train/development scene leakage")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model, model_config = common.model_from_cache(train_caches[0], args.ablation)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        foreach=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    checkpoint_reports = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        cache_order = torch.randperm(len(train_caches), generator=generator).tolist()
        for cache_index in cache_order:
            cache = train_caches[cache_index]
            order = torch.randperm(len(cache["tokens"]), generator=generator)
            for start in range(0, len(order), args.batch_size):
                indices = order[start : start + args.batch_size]
                logits, reference = common.current_logits(model, cache, indices, device)
                rewards = cache["candidate_rewards"][indices].to(device)
                valid = cache["candidate_reward_valid_mask"][indices].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = common.exact_group_loss(
                    logits,
                    reference,
                    rewards,
                    valid,
                    args.temperature,
                    args.kl_weight,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite V3 loss at epoch {epoch}")
                loss.backward()
                common.clip_grad_norm_cpu_(model.parameters(), 10.0)
                optimizer.step()

        if epoch not in checkpoint_epochs:
            continue
        state_path = output_dir / f"epoch_{epoch}_scene_selector.pt"
        torch.save(
            {
                "schema_version": 3,
                "method": "scene_conditioned_exact_group_grpo",
                "scene_selector_state": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "scene_selector_config": model_config,
                "ablation": args.ablation,
                "temperature": args.temperature,
                "learning_rate": args.learning_rate,
                "kl_weight": args.kl_weight,
                "train_seed": args.seed,
                "epoch": epoch,
            },
            state_path,
        )
        development_values = [
            common.evaluate(model, cache, device, args.temperature, args.batch_size)
            for cache in development_caches
        ]
        pooled = {
            key: torch.cat([values[key] for values in development_values])
            for key in development_values[0]
        }
        records_path = output_dir / f"epoch_{epoch}_development.jsonl"
        write_records(
            records_path, development_caches, development_manifests, development_values
        )
        checkpoint_reports.append(
            {
                "epoch": epoch,
                "scene_selector_state": str(state_path),
                "scene_selector_state_sha256": common.sha256_file(state_path),
                "development_records": str(records_path),
                "development_records_sha256": common.sha256_file(records_path),
                "development_metrics": common.summarize(pooled),
                "development_by_noise_seed": {
                    str(manifest["noise_seed"]): common.summarize(values)
                    for manifest, values in zip(development_manifests, development_values)
                },
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "development_top1_gain": common.summarize(pooled)["top1_reward_gain"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report = {
        "schema_version": 3,
        "status": "PASS",
        "method": "scene_conditioned_exact_group_grpo",
        "ablation": args.ablation,
        "scene_selector_config": model_config,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_epochs": list(checkpoint_epochs),
        "early_stopping": False,
        "train_manifests": list(train_manifests),
        "development_manifests": list(development_manifests),
        "checkpoints": checkpoint_reports,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS V3 cached selector trial: {report_path}")


if __name__ == "__main__":
    main()
