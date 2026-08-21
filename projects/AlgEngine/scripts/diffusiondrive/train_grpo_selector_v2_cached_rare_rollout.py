#!/usr/bin/env python3
"""Train corrected V2 on the shared 50/50 real-common/rollout-hard pool."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v2_rare_common as common
import train_grpo_selector_v3_cached_rare_original as rare
import train_grpo_selector_v3_cached_rare_rollout as rollout
from build_grpo_selector_v3_rare_rollout_data import parse_seed_path


METHOD = "exact_group_grpo_v2_rare_rollout_v1"
FORMAL_EPOCHS = 16
FORMAL_EXAMPLES_PER_CACHE_EPOCH = 6339
FORMAL_BATCH_SIZE = 64
FORMAL_TOTAL_EXAMPLES = 304272
FORMAL_OPTIMIZER_STEPS = 4800


def implementation_provenance() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        Path(rare.__file__).resolve(),
        Path(rollout.__file__).resolve(),
        Path(__file__).with_name("build_grpo_selector_v3_rare_rollout_data.py"),
    )
    return {str(path): common.sha256_file(path) for path in paths}


def validate_formal_args(args) -> None:
    expected = {
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "kl_weight": 1e-3,
        "epochs": FORMAL_EPOCHS,
        "examples_per_cache_epoch": FORMAL_EXAMPLES_PER_CACHE_EPOCH,
        "batch_size": FORMAL_BATCH_SIZE,
        "method_name": METHOD,
        "hard_pool_split": "all",
    }
    drift = {
        key: {"actual": getattr(args, key), "expected": value}
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if drift:
        raise RuntimeError("formal V2 rare-rollout contract drifted: " + repr(drift))


def save_selector_state(path, model, args, epoch, manifest_path, pool_path) -> None:
    payload = {
        "schema_version": 3,
        "method": args.method_name,
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "source_kind": "full_navtrain_real_rare_plus_filtered_online_rollout_v1",
        "source_policy": "immutable_epoch100_diffusiondrive",
        "reward_contract": common.REWARD_CONTRACT,
        "implementation_files": implementation_provenance(),
        "selector_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "sampling_mode": "common_hard_balanced",
        "hard_pool_split": args.hard_pool_split,
        "train_seed": args.seed,
        "epoch": epoch,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(pool_path),
        "hard_pool_sha256": common.sha256_file(pool_path),
    }
    if len(payload["selector_state"]) != common.SELECTOR_TENSOR_COUNT:
        raise RuntimeError("V2 selector state tensor count drifted")
    torch.save(payload, path)


def train(args) -> dict:
    if args.method_name is None:
        args.method_name = METHOD
    if args.formal_contract:
        validate_formal_args(args)
    if args.temperature <= 0 or args.learning_rate <= 0 or args.kl_weight < 0:
        raise ValueError("invalid training hyperparameters")
    if min(args.epochs, args.examples_per_cache_epoch, args.batch_size) <= 0:
        raise ValueError("training budgets must be positive")
    (
        manifest_path,
        manifest,
        hard_pool_path,
        hard_rows,
        real_caches,
        real_manifests,
        token_maps,
        synthetic_path,
        synthetic,
    ) = rollout.load_contract(args)
    if args.hard_pool_split != "all":
        hard_rows = [
            row for row in hard_rows if row["split"] == args.hard_pool_split
        ]
    if not hard_rows:
        raise RuntimeError("selected hard-pool split is empty")
    if args.smoke_limit_hard_pool is not None:
        if args.formal_contract:
            raise RuntimeError("smoke limiting is forbidden in formal mode")
        if args.smoke_limit_hard_pool < 1:
            raise ValueError("--smoke-limit-hard-pool must be positive")
        real_rows = [row for row in hard_rows if row["hard_kind"] == "real_rare"]
        synthetic_rows = [
            row for row in hard_rows if row["hard_kind"] == "synthetic_rollout"
        ]
        hard_rows = real_rows[: args.smoke_limit_hard_pool]
        if synthetic_rows and args.smoke_limit_hard_pool > 1:
            hard_rows[-1] = synthetic_rows[0]
    minimum_hard_draws = args.epochs * (args.examples_per_cache_epoch // 2)
    if len(hard_rows) > minimum_hard_draws:
        raise RuntimeError(
            "fixed compute cannot cover every selected hard row before reuse"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = common.model_from_cache(real_caches[0], device)
    reference_errors = {
        f"real_seed{index}": common.assert_reference_parity(
            model, cache, device, batch_size=max(args.batch_size, 256)
        )
        for index, cache in enumerate(real_caches)
    }
    reference_errors["synthetic"] = common.assert_reference_parity(
        model, synthetic, device, batch_size=max(args.batch_size, 256)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        foreach=False,
    )
    schedule = torch.Generator().manual_seed(args.seed + 910000)
    samplers = {
        (cache_index, class_name): rare.CyclingPairSampler(
            len(hard_rows),
            args.seed * 100000
            + cache_index * 1000
            + (17 if class_name == "hard" else 37),
        )
        for cache_index in range(3)
        for class_name in ("hard", "common")
    }
    checkpoint_epochs = tuple(
        sorted({int(value) for value in args.checkpoint_epochs.split(",")})
    )
    if not checkpoint_epochs or checkpoint_epochs[-1] != args.epochs:
        raise ValueError("checkpoint epochs must end at --epochs")

    total_examples = 0
    optimizer_steps = 0
    class_examples = Counter()
    source_examples = Counter()
    accumulated = Counter()
    checkpoints = []
    global_block = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for cache_index in torch.randperm(3, generator=schedule).tolist():
            hard_count, common_count = rare.block_class_counts(
                args.examples_per_cache_epoch, global_block
            )
            selected = [
                ("hard", position)
                for position in samplers[(cache_index, "hard")].take(hard_count)
            ]
            selected.extend(
                ("common", position)
                for position in samplers[(cache_index, "common")].take(common_count)
            )
            permutation = torch.randperm(len(selected), generator=schedule).tolist()
            selected = [selected[index] for index in permutation]
            for start in range(0, len(selected), args.batch_size):
                inputs, reference, rewards, valid, kinds = rollout.mixed_batch(
                    selected[start : start + args.batch_size],
                    hard_rows,
                    real_caches[cache_index],
                    token_maps[cache_index],
                    synthetic,
                    device,
                )
                logits = reference + common.delta_logits(model, inputs["candidate_features"])
                optimizer.zero_grad(set_to_none=True)
                loss, policy, kl = common.exact_group_loss(
                    logits,
                    reference,
                    rewards,
                    valid,
                    args.temperature,
                    args.kl_weight,
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite V2 rollout loss at epoch {epoch}")
                loss.backward()
                common.clip_grad_norm_cpu_(model.parameters(), 10.0)
                optimizer.step()
                optimizer_steps += 1
                accumulated["loss"] += float(loss.detach().cpu())
                accumulated["policy"] += float(policy.detach().cpu())
                accumulated["kl"] += float(kl.detach().cpu())
                source_examples.update(kinds)
            class_examples["hard"] += hard_count
            class_examples["common"] += common_count
            total_examples += len(selected)
            global_block += 1

        if epoch in checkpoint_epochs:
            state_path = output_dir / f"epoch_{epoch}_selector.pt"
            save_selector_state(
                state_path,
                model,
                args,
                epoch,
                manifest_path,
                hard_pool_path,
            )
            checkpoints.append(
                {
                    "epoch": epoch,
                    "selector_state": str(state_path),
                    "selector_state_sha256": common.sha256_file(state_path),
                }
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "examples": total_examples,
                        "optimizer_steps": optimizer_steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if class_examples["hard"] != class_examples["common"]:
        raise RuntimeError("global 50/50 common/hard balance drifted")
    if args.formal_contract and (
        total_examples != FORMAL_TOTAL_EXAMPLES
        or optimizer_steps != FORMAL_OPTIMIZER_STEPS
    ):
        raise RuntimeError("formal V2 rollout fixed-compute budget drifted")
    coverage = {}
    for cache_index in range(3):
        coverage[str(cache_index)] = {}
        for class_name in ("hard", "common"):
            sampler = samplers[(cache_index, class_name)]
            if len(sampler.visited) != len(hard_rows):
                raise RuntimeError("V2 rollout sampler did not cover hard pool")
            coverage[str(cache_index)][class_name] = {
                "unique_hard_positions_visited": len(sampler.visited),
                "hard_pool_rows": len(hard_rows),
                "completed_cycles": sampler.completed_cycles,
            }

    report = {
        "schema_version": 3,
        "status": "PASS",
        "method": args.method_name,
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "source_kind": "full_navtrain_real_rare_plus_filtered_online_rollout_v1",
        "source_policy": "immutable_epoch100_diffusiondrive",
        "baseline_checkpoint_sha256": common.BASELINE_SHA256,
        "selector_initialization": "frozen_epoch100_plan_cls_branch_exact",
        "reward_contract": common.REWARD_CONTRACT,
        "sampling_mode": "common_hard_balanced",
        "hard_pool_split": args.hard_pool_split,
        "implementation_files": implementation_provenance(),
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_epochs": list(checkpoint_epochs),
        "fixed_budget_selection": "final_epoch",
        "early_stopping": False,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(hard_pool_path),
        "hard_pool_sha256": common.sha256_file(hard_pool_path),
        "synthetic_cache": str(synthetic_path),
        "synthetic_cache_sha256": common.sha256_file(synthetic_path),
        "real_cache_manifests": list(real_manifests),
        "initial_reference_max_abs_error": reference_errors,
        "sampling": {
            "contract": "50% common; 50% hard=(real rare + filtered synthetic)",
            "examples_per_cache_epoch": args.examples_per_cache_epoch,
            "total_examples": total_examples,
            "total_optimizer_steps": optimizer_steps,
            "class_examples": dict(class_examples),
            "source_examples": dict(source_examples),
            "coverage": coverage,
        },
        "mean_training_loss": accumulated["loss"] / optimizer_steps,
        "mean_training_policy": accumulated["policy"] / optimizer_steps,
        "mean_training_kl": accumulated["kl"] / optimizer_steps,
        "checkpoints": checkpoints,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS V2 rare-rollout cached selector training: {report_path}")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-cache", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument(
        "--examples-per-cache-epoch",
        type=int,
        default=FORMAL_EXAMPLES_PER_CACHE_EPOCH,
    )
    parser.add_argument("--checkpoint-epochs", default=str(FORMAL_EPOCHS))
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method-name")
    parser.add_argument("--hard-pool-split", choices=("all", "train"), default="all")
    parser.add_argument("--formal-contract", action="store_true")
    parser.add_argument("--smoke-limit-hard-pool", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
