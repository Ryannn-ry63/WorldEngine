#!/usr/bin/env python3
"""Train corrected selector V2 on audited rare/same-log-common pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v2_rare_common as common
import train_grpo_selector_v3_cached_rare_original as rare


FORMAL_RARE_METHOD = "exact_group_grpo_v2_rare_original_v1"
FORMAL_PAIRED_COMMON_METHOD = "exact_group_grpo_v2_paired_common_v1"
FORMAL_EPOCHS = 16
FORMAL_EXAMPLES_PER_CACHE_EPOCH = 6339
FORMAL_BATCH_SIZE = 64
FORMAL_TOTAL_EXAMPLES = 304272
FORMAL_OPTIMIZER_STEPS = 4800


def expected_method(sampling_mode: str) -> str:
    return (
        FORMAL_RARE_METHOD
        if sampling_mode == "rare_balanced"
        else FORMAL_PAIRED_COMMON_METHOD
    )


def implementation_provenance() -> dict[str, str]:
    algengine_root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        Path(rare.__file__).resolve(),
        algengine_root
        / "mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py",
    )
    return {str(path): common.sha256_file(path) for path in paths}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument(
        "--sampling-mode",
        choices=("rare_balanced", "paired_common"),
        default="rare_balanced",
    )
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
    parser.add_argument("--formal-contract", action="store_true")
    return parser.parse_args()


def validate_formal_contract(args) -> None:
    expected = {
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "kl_weight": 1e-3,
        "epochs": FORMAL_EPOCHS,
        "examples_per_cache_epoch": FORMAL_EXAMPLES_PER_CACHE_EPOCH,
        "batch_size": FORMAL_BATCH_SIZE,
        "method_name": expected_method(args.sampling_mode),
        "num_train_caches": 3,
    }
    actual = {
        key: getattr(args, key)
        for key in expected
        if key != "num_train_caches"
    }
    actual["num_train_caches"] = len(args.train_cache)
    drift = {
        key: {"actual": actual[key], "expected": value}
        for key, value in expected.items()
        if actual[key] != value
    }
    if drift:
        raise RuntimeError("formal V2 rare-original contract drifted: " + repr(drift))


def save_selector_state(path, model, args, epoch, pair_manifest, pair_audit) -> None:
    payload = {
        "schema_version": 3,
        "method": args.method_name,
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "source_kind": "full_navtrain_rare_original_v1",
        "reward_contract": common.REWARD_CONTRACT,
        "implementation_files": implementation_provenance(),
        "selector_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "sampling_mode": args.sampling_mode,
        "train_seed": args.seed,
        "epoch": epoch,
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": common.sha256_file(pair_manifest),
        "rare_data_audit": str(pair_audit),
        "rare_data_audit_sha256": common.sha256_file(pair_audit),
    }
    if len(payload["selector_state"]) != common.SELECTOR_TENSOR_COUNT:
        raise RuntimeError("V2 selector state tensor count drifted")
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if args.method_name is None:
        args.method_name = expected_method(args.sampling_mode)
    if args.temperature <= 0 or args.learning_rate <= 0 or args.kl_weight < 0:
        raise ValueError("invalid temperature/learning-rate/KL weight")
    if min(args.epochs, args.examples_per_cache_epoch, args.batch_size) <= 0:
        raise ValueError("training budgets must be positive")
    if args.formal_contract:
        validate_formal_contract(args)
    checkpoint_epochs = tuple(
        sorted({int(value) for value in args.checkpoint_epochs.split(",")})
    )
    if not checkpoint_epochs or checkpoint_epochs[-1] != args.epochs:
        raise ValueError("checkpoint epochs must end at --epochs")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    pair_rows, rare_audit, pair_manifest, pair_audit = rare.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    caches, manifests = rare.load_caches(args.train_cache)
    rare_indices, common_indices = rare.validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )
    minimum_per_class = args.epochs * (
        args.examples_per_cache_epoch // 2
        if args.sampling_mode == "rare_balanced"
        else args.examples_per_cache_epoch
    )
    if len(pair_rows) > minimum_per_class:
        raise RuntimeError("fixed budget cannot cover every rare/common pair")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = common.model_from_cache(caches[0], device)
    reference_errors = {
        str(index): common.assert_reference_parity(
            model, cache, device, batch_size=max(args.batch_size, 256)
        )
        for index, cache in enumerate(caches)
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        foreach=False,
    )
    schedule = torch.Generator().manual_seed(args.seed + 910000)
    sampled_classes = (
        ("rare", "common")
        if args.sampling_mode == "rare_balanced"
        else ("common",)
    )
    offsets = {"rare": 17, "common": 37}
    samplers = {
        (cache_index, class_name): rare.CyclingPairSampler(
            len(pair_rows),
            args.seed * 100000 + cache_index * 1000 + offsets[class_name],
        )
        for cache_index in range(3)
        for class_name in sampled_classes
    }

    total_examples = 0
    optimizer_steps = 0
    class_examples = {"rare": 0, "common": 0}
    class_examples_by_cache = {
        str(index): {"rare": 0, "common": 0} for index in range(3)
    }
    accumulated = {"loss": 0.0, "policy": 0.0, "kl": 0.0}
    checkpoints = []
    global_block = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for cache_index in torch.randperm(3, generator=schedule).tolist():
            if args.sampling_mode == "rare_balanced":
                rare_count, common_count = rare.block_class_counts(
                    args.examples_per_cache_epoch, global_block
                )
                selected = [
                    rare_indices[position]
                    for position in samplers[(cache_index, "rare")].take(rare_count)
                ]
            else:
                rare_count, common_count, selected = (
                    0,
                    args.examples_per_cache_epoch,
                    [],
                )
            selected.extend(
                common_indices[position]
                for position in samplers[(cache_index, "common")].take(common_count)
            )
            permutation = torch.randperm(len(selected), generator=schedule).tolist()
            order = torch.tensor(
                [selected[index] for index in permutation], dtype=torch.long
            )
            cache = caches[cache_index]
            for start in range(0, len(order), args.batch_size):
                indices = order[start : start + args.batch_size]
                logits, reference = common.current_logits(
                    model, cache, indices, device
                )
                rewards = cache["candidate_rewards"][indices].to(device)
                valid = cache["candidate_reward_valid_mask"][indices].to(device)
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
                    raise RuntimeError(f"non-finite V2 loss at epoch {epoch}")
                loss.backward()
                common.clip_grad_norm_cpu_(model.parameters(), 10.0)
                optimizer.step()
                optimizer_steps += 1
                accumulated["loss"] += float(loss.detach().cpu())
                accumulated["policy"] += float(policy.detach().cpu())
                accumulated["kl"] += float(kl.detach().cpu())
            class_examples["rare"] += rare_count
            class_examples["common"] += common_count
            class_examples_by_cache[str(cache_index)]["rare"] += rare_count
            class_examples_by_cache[str(cache_index)]["common"] += common_count
            total_examples += len(selected)
            global_block += 1

        if epoch in checkpoint_epochs:
            state_path = output_dir / f"epoch_{epoch}_selector.pt"
            save_selector_state(
                state_path,
                model,
                args,
                epoch,
                pair_manifest,
                pair_audit,
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

    if args.sampling_mode == "rare_balanced" and (
        class_examples["rare"] != class_examples["common"]
    ):
        raise RuntimeError("global rare/common balance drifted")
    if args.sampling_mode == "paired_common" and class_examples != {
        "rare": 0,
        "common": total_examples,
    }:
        raise RuntimeError("paired-common sampling contract drifted")
    if args.formal_contract and (
        total_examples != FORMAL_TOTAL_EXAMPLES
        or optimizer_steps != FORMAL_OPTIMIZER_STEPS
    ):
        raise RuntimeError("formal V2 fixed-compute budget drifted")

    coverage = {}
    for cache_index in range(3):
        coverage[str(cache_index)] = {}
        for class_name in sampled_classes:
            sampler = samplers[(cache_index, class_name)]
            if len(sampler.visited) != len(pair_rows):
                raise RuntimeError("V2 sampler did not cover every pair row")
            coverage[str(cache_index)][class_name] = {
                "unique_pair_rows_visited": len(sampler.visited),
                "pair_rows": len(pair_rows),
                "completed_cycles": sampler.completed_cycles,
            }

    report = {
        "schema_version": 3,
        "status": "PASS",
        "method": args.method_name,
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "source_kind": "full_navtrain_rare_original_v1",
        "reward_contract": common.REWARD_CONTRACT,
        "sampling_mode": args.sampling_mode,
        "implementation_files": implementation_provenance(),
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_epochs": list(checkpoint_epochs),
        "early_stopping": False,
        "fixed_budget_selection": "final_epoch",
        "pair_manifest": str(pair_manifest),
        "pair_manifest_sha256": common.sha256_file(pair_manifest),
        "rare_data_audit": str(pair_audit),
        "rare_data_audit_sha256": common.sha256_file(pair_audit),
        "rare_pair_rows": len(pair_rows),
        "unique_common_tokens": len({row["common_token"] for row in pair_rows}),
        "train_manifests": list(manifests),
        "initial_reference_max_abs_error_by_cache": reference_errors,
        "sampling": {
            "contract": "three shared fixed-noise caches; V2-only selector update",
            "examples_per_cache_epoch": args.examples_per_cache_epoch,
            "total_examples": total_examples,
            "total_optimizer_steps": optimizer_steps,
            "class_examples": class_examples,
            "class_examples_by_cache": class_examples_by_cache,
            "coverage": coverage,
        },
        "mean_training_loss": accumulated["loss"] / optimizer_steps,
        "mean_training_policy": accumulated["policy"] / optimizer_steps,
        "mean_training_kl": accumulated["kl"] / optimizer_steps,
        "checkpoints": checkpoints,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS V2 rare-original cached selector training: {report_path}")


if __name__ == "__main__":
    main()
