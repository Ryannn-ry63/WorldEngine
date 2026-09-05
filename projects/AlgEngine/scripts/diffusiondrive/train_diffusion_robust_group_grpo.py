#!/usr/bin/env python3
"""Fine-tune V3 with aligned multi-draw scalar selector objectives."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

import diffusion_robust_group_grpo as robust
import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


ARMS = (
    "repeat_grpo",
    "mean_soft",
    "mean_top1",
    "pure_softmin_top1",
    "bounded_relative_top1",
    "bounded_relative_soft",
    "bounded_relative_top1_shuffled",
    "bounded_absolute_top1",
)
EXPECTED_NOISE_SEEDS = (0, 1, 2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--risk-temperature", type=float, default=0.02)
    parser.add_argument("--risk-mix", type=float, default=0.5)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--examples-per-epoch", type=int, default=6339)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--anchor-logits-mode",
        choices=("on_demand", "precompute"),
        default="on_demand",
    )
    parser.add_argument("--checkpoint-epochs", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="reject budgets that cannot visit every rare/common pair",
    )
    return parser.parse_args()


def implementation_provenance():
    selector = (
        Path(__file__).resolve().parents[2]
        / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
    )
    paths = (
        Path(__file__).resolve(),
        Path(robust.__file__).resolve(),
        Path(common.__file__).resolve(),
        selector,
    )
    return {str(path): common.sha256_file(path) for path in paths}


def load_v3_payload(path, expected_sha):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = common.sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"V3 checkpoint SHA256 drifted: actual={actual_sha} expected={expected_sha}"
        )
    payload = torch.load(path, map_location="cpu")
    if "scene_selector_state" not in payload or "scene_selector_config" not in payload:
        raise RuntimeError("V3 checkpoint lacks selector state/config")
    architecture = payload.get("selector_architecture", "scene_conditioned_v3")
    if architecture not in {"scene_conditioned_v3", "v3"}:
        raise RuntimeError(f"anchor must be V3, got {architecture}")
    return payload, path, actual_sha


def build_aligned_models(payload, cache, device):
    config = dict(payload["scene_selector_config"])
    anchor = common.model_from_config(config)
    anchor.load_state_dict(payload["scene_selector_state"], strict=True)
    model = copy.deepcopy(anchor)
    anchor.to(device).eval()
    anchor.requires_grad_(False)
    model.to(device)
    cache_config = dict(cache["scene_selector_config"])
    for key, value in cache_config.items():
        if key in config and config[key] != value:
            raise RuntimeError(f"V3/cache selector config drifted at {key}")
    return model, anchor, config


def cache_anchor_logits(model, cache, device, batch_size):
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            logits, _ = common.current_logits(
                model, cache, slice(start, stop), device
            )
            parts.append(logits.cpu())
    return torch.cat(parts)


def stratified_shuffle(indices, labels, generator):
    """Build a same-stratum donor map with no fixed points when possible."""

    output = indices.clone()
    for label in (0, 1):
        positions = torch.nonzero(labels.eq(label), as_tuple=False).flatten()
        if len(positions) > 1:
            cycle = positions[torch.randperm(len(positions), generator=generator)]
            output[cycle] = indices[cycle.roll(1)]
    return output


def arm_contract(arm):
    if arm == "repeat_grpo":
        return {
            "objective": "original_per_draw_zscore_exact_group_grpo",
            "selection_mode": None,
            "aggregation": None,
            "same_token_grouping": True,
        }
    selection_mode = "straight_through_top1" if "top1" in arm else "soft"
    if arm.startswith("bounded_"):
        aggregation = "bounded_mean_risk"
    elif arm == "pure_softmin_top1":
        aggregation = "softmin"
    else:
        aggregation = "mean"
    objective_mode = (
        "absolute_value" if arm == "bounded_absolute_top1" else "relative_gain"
    )
    return {
        "objective": "raw_scalar_diffusion_set_policy_value",
        "selection_mode": selection_mode,
        "aggregation": aggregation,
        "objective_mode": objective_mode,
        "same_token_grouping": arm != "bounded_relative_top1_shuffled",
    }


def save_state(
    path,
    model,
    model_config,
    args,
    epoch,
    anchor_path,
    anchor_sha,
    pair_path,
    audit_path,
):
    contract = arm_contract(args.arm)
    payload = {
        "schema_version": 1,
        "method": "diffusion_set_selector_posttraining_v1",
        "selector_architecture": "scene_conditioned_v3",
        "arm": args.arm,
        "arm_contract": contract,
        "scene_selector_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scene_selector_config": model_config,
        "ablation": "full",
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "risk_temperature": args.risk_temperature,
        "risk_mix": args.risk_mix,
        "reward_scale": args.reward_scale,
        "train_seed": args.seed,
        "epoch": epoch,
        "v3_anchor": str(anchor_path),
        "v3_anchor_sha256": anchor_sha,
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "scientific_contract": {
            "selector_only": True,
            "generator_frozen": True,
            "official_scalar_pdm_only": True,
            "reward_components_consumed": False,
            "new_training_data_consumed": False,
            "v3_initialized_and_anchored": True,
        },
        "implementation_files": implementation_provenance(),
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    positive = (
        args.temperature,
        args.learning_rate,
        args.risk_temperature,
        args.reward_scale,
        args.epochs,
        args.examples_per_epoch,
        args.batch_size,
    )
    if any(float(value) <= 0.0 for value in positive) or args.kl_weight < 0.0:
        raise ValueError("training hyperparameters are outside their valid range")
    if not 0.0 <= args.risk_mix <= 1.0:
        raise ValueError("risk mix must lie in [0, 1]")
    checkpoint_epochs = tuple(
        sorted({int(value) for value in args.checkpoint_epochs.split(",")})
    )
    if (
        not checkpoint_epochs
        or checkpoint_epochs[0] <= 0
        or checkpoint_epochs[-1] != args.epochs
    ):
        raise ValueError("checkpoint epochs must be positive and end at --epochs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    loaded = [common.load_cache(path, "train") for path in args.train_cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    standard.assert_cache_group(caches, manifests, "train", EXPECTED_NOISE_SEEDS)
    pair_rows, rare_audit, pair_path, audit_path = rare_trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    rare_indices, common_indices = rare_trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )
    if (
        args.require_full_coverage
        and len(pair_rows) > args.epochs * (args.examples_per_epoch // 2)
    ):
        raise RuntimeError(
            "prototype budget cannot visit every rare/common pair at least once"
        )
    anchor_payload, anchor_path, anchor_sha = load_v3_payload(
        args.v3_checkpoint, args.v3_checkpoint_sha256
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, anchor_model, model_config = build_aligned_models(
        anchor_payload, caches[0], device
    )
    anchor_logits = None
    if args.anchor_logits_mode == "precompute":
        anchor_logits = [
            cache_anchor_logits(anchor_model, cache, device, args.batch_size)
            for cache in caches
        ]
        del anchor_model
        anchor_model = None
        if device.type == "cuda":
            torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        foreach=False,
    )
    samplers = {
        "rare": rare_trainer.CyclingPairSampler(len(pair_rows), args.seed + 17),
        "common": rare_trainer.CyclingPairSampler(len(pair_rows), args.seed + 37),
    }
    schedule = torch.Generator().manual_seed(args.seed + 910000)
    grouping_control = torch.Generator().manual_seed(args.seed + 920000)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    totals = {
        "optimizer_steps": 0,
        "outer_groups": 0,
        "view_examples": 0,
        "loss": 0.0,
        "policy": 0.0,
        "kl": 0.0,
        "hard_gain": 0.0,
        "worst_hard_gain": 0.0,
        "draw_weight_max": 0.0,
        "draw_weight_entropy": 0.0,
        "draw_weight_gt_09_fraction": 0.0,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        rare_count, common_count = rare_trainer.block_class_counts(
            args.examples_per_epoch, epoch - 1
        )
        rare_positions = samplers["rare"].take(rare_count)
        common_positions = samplers["common"].take(common_count)
        selected = torch.tensor(
            [rare_indices[position] for position in rare_positions]
            + [common_indices[position] for position in common_positions],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [1] * rare_count + [0] * common_count, dtype=torch.long
        )
        permutation = torch.randperm(len(selected), generator=schedule)
        selected = selected[permutation]
        labels = labels[permutation]

        for start in range(0, len(selected), args.batch_size):
            base_indices = selected[start : start + args.batch_size]
            batch_labels = labels[start : start + args.batch_size]
            current_parts = []
            anchor_parts = []
            reward_parts = []
            valid_parts = []
            for draw, cache in enumerate(caches):
                indices = base_indices
                if args.arm == "bounded_relative_top1_shuffled" and draw > 0:
                    indices = stratified_shuffle(
                        base_indices, batch_labels, grouping_control
                    )
                logits, _ = common.current_logits(model, cache, indices, device)
                current_parts.append(logits)
                if anchor_logits is not None:
                    frozen_logits = anchor_logits[draw][indices].to(
                        device=device, dtype=torch.float32
                    )
                else:
                    with torch.no_grad():
                        frozen_logits, _ = common.current_logits(
                            anchor_model, cache, indices, device
                        )
                anchor_parts.append(frozen_logits)
                reward_parts.append(
                    cache["candidate_rewards"][indices].to(
                        device=device, dtype=torch.float32
                    )
                )
                valid_parts.append(
                    cache["candidate_reward_valid_mask"][indices].to(
                        device=device, dtype=torch.bool
                    )
                )
            current = torch.stack(current_parts, dim=1)
            anchor = torch.stack(anchor_parts, dim=1)
            rewards = torch.stack(reward_parts, dim=1)
            valid = torch.stack(valid_parts, dim=1)
            optimizer.zero_grad(set_to_none=True)
            if args.arm == "repeat_grpo":
                loss, policy, kl = common.exact_group_loss(
                    current.flatten(0, 1),
                    anchor.flatten(0, 1),
                    rewards.flatten(0, 1),
                    valid.flatten(0, 1),
                    args.temperature,
                    args.kl_weight,
                )
            else:
                contract = arm_contract(args.arm)
                loss, policy, kl, objective_diagnostics = robust.diffusion_robust_group_loss(
                    current,
                    anchor,
                    rewards,
                    valid,
                    temperature=args.temperature,
                    kl_weight=args.kl_weight,
                    reward_scale=args.reward_scale,
                    selection_mode=contract["selection_mode"],
                    aggregation=contract["aggregation"],
                    risk_temperature=args.risk_temperature,
                    risk_mix=args.risk_mix,
                    objective_mode=contract["objective_mode"],
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"non-finite {args.arm} loss at epoch {epoch}"
                )
            loss.backward()
            common.clip_grad_norm_cpu_(model.parameters(), 10.0)
            optimizer.step()

            with torch.no_grad():
                hard_gain, _, _ = robust.per_draw_reference_gain(
                    current.detach(),
                    anchor,
                    rewards,
                    valid,
                    temperature=args.temperature,
                    selection_mode="straight_through_top1",
                )
            batch_size = len(base_indices)
            totals["optimizer_steps"] += 1
            totals["outer_groups"] += batch_size
            totals["view_examples"] += batch_size * len(caches)
            totals["loss"] += float(loss.detach().cpu())
            totals["policy"] += float(policy.detach().cpu())
            totals["kl"] += float(kl.detach().cpu())
            totals["hard_gain"] += float(hard_gain.mean().cpu())
            totals["worst_hard_gain"] += float(hard_gain.min(dim=1).values.mean().cpu())
            if args.arm == "repeat_grpo":
                draw_weights = torch.full_like(hard_gain, 1.0 / hard_gain.shape[1])
            else:
                draw_weights = objective_diagnostics["draw_weights"].detach()
            normalized_entropy = -(
                draw_weights
                * draw_weights.clamp_min(1e-12).log()
            ).sum(dim=1) / np.log(draw_weights.shape[1])
            totals["draw_weight_max"] += float(
                draw_weights.max(dim=1).values.mean().cpu()
            )
            totals["draw_weight_entropy"] += float(
                normalized_entropy.mean().cpu()
            )
            totals["draw_weight_gt_09_fraction"] += float(
                (draw_weights.max(dim=1).values > 0.9).float().mean().cpu()
            )

        if epoch in checkpoint_epochs:
            state_path = output_dir / f"epoch_{epoch}_scene_selector.pt"
            save_state(
                state_path,
                model,
                model_config,
                args,
                epoch,
                anchor_path,
                anchor_sha,
                pair_path,
                audit_path,
            )
            checkpoints.append(
                {
                    "epoch": epoch,
                    "scene_selector_state": str(state_path),
                    "scene_selector_state_sha256": common.sha256_file(state_path),
                }
            )
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "epoch": epoch,
                        "optimizer_steps": totals["optimizer_steps"],
                        "outer_groups": totals["outer_groups"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    steps = totals["optimizer_steps"]
    if not steps:
        raise RuntimeError("training performed no optimizer steps")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusion_set_selector_posttraining_v1",
        "arm": args.arm,
        "arm_contract": arm_contract(args.arm),
        "scene_selector_config": model_config,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "risk_temperature": args.risk_temperature,
        "risk_mix": args.risk_mix,
        "reward_scale": args.reward_scale,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "examples_per_epoch": args.examples_per_epoch,
        "outer_batch_size": args.batch_size,
        "noise_seeds": list(EXPECTED_NOISE_SEEDS),
        "v3_anchor": str(anchor_path),
        "anchor_logits_mode": args.anchor_logits_mode,
        "v3_anchor_sha256": anchor_sha,
        "train_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "scientific_contract": {
            "selector_only": True,
            "generator_frozen": True,
            "official_scalar_pdm_only": True,
            "reward_components_consumed": False,
            "new_training_data_consumed": False,
            "v3_initialized_and_anchored": True,
        },
        "budget": {
            "optimizer_steps": totals["optimizer_steps"],
            "outer_groups": totals["outer_groups"],
            "view_examples": totals["view_examples"],
        },
        "training_means_per_step": {
            key: totals[key] / steps
            for key in (
                "loss",
                "policy",
                "kl",
                "hard_gain",
                "worst_hard_gain",
                "draw_weight_max",
                "draw_weight_entropy",
                "draw_weight_gt_09_fraction",
            )
        },
        "checkpoints": checkpoints,
        "implementation_files": implementation_provenance(),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": "PASS", "arm": args.arm, "output": str(report_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
