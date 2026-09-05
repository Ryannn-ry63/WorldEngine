#!/usr/bin/env python3
"""Train one immutable LC-PGRPO arm on four log folds (or all train logs)."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common
import lcpgrpo_protocol as protocol
import lineage_consistent_proximal_grpo as lcp
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-sha256", required=True)
    parser.add_argument("--mechanism-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=lcp.ARMS, required=True)
    parser.add_argument("--heldout-fold", type=int, required=True)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument(
        "--lineage-control",
        choices=("real", "independent_shuffle"),
        default="real",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--retention-headroom", type=float, default=0.005)
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--examples-per-epoch", type=int, default=6339)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-epochs", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-full-coverage", action="store_true")
    return parser.parse_args()


def verify_mechanism_gate(path: Path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "PASS"
        or payload.get("method") != protocol.MECHANISM_METHOD
        or payload.get("decision") != "AUTHORIZE_LCPGRPO_LOG_CV"
        or payload.get("noise_seeds") != list(protocol.FRESH_NOISE_SEEDS)
        or payload.get("prediction_metric_contract", {})
        .get("gate", {})
        .get("metric")
        != "predicted_top1_is_in_heldout_top4_fraction"
        or payload.get("scientific_contract", {}).get(
            "development_consumed_for_new_method_selection"
        )
        is not False
        or payload.get("scientific_contract", {}).get("certification_consumed")
        is not False
    ):
        raise RuntimeError("mechanism gate did not authorize LC-PGRPO log-CV")
    return path, common.sha256_file(path)


def load_v3(path: Path, expected_sha: str):
    path = path.expanduser().resolve()
    actual_sha = common.sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError("V3 anchor SHA256 drifted")
    payload = torch.load(path, map_location="cpu")
    if "scene_selector_state" not in payload or "scene_selector_config" not in payload:
        raise RuntimeError("V3 anchor lacks selector state/config")
    return payload, path, actual_sha


def build_models(payload, cache, device):
    config = dict(payload["scene_selector_config"])
    anchor = common.model_from_config(config)
    anchor.load_state_dict(payload["scene_selector_state"], strict=True)
    model = copy.deepcopy(anchor)
    anchor.to(device).eval().requires_grad_(False)
    model.to(device)
    for key, value in cache["scene_selector_config"].items():
        if key in config and value != config[key]:
            raise RuntimeError(f"V3/cache selector config drifted at {key}")
    return model, anchor, config


def cache_logits(model, cache, device, batch_size):
    parts = []
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            logits, _ = common.current_logits(model, cache, slice(start, stop), device)
            parts.append(logits.cpu())
    return torch.cat(parts)


def scientific_contract(heldout_fold, lineage_control):
    return {
        "selector_only": True,
        "generator_frozen": True,
        "official_scalar_pdm_only": True,
        "new_training_scene_tokens_consumed": False,
        "v3_initialized_and_anchored": True,
        "candidate_index_is_plan_anchor_lineage": True,
        "log_disjoint_cross_validation": heldout_fold != -1,
        "heldout_fold_consumed_by_training": False,
        "negative_control": lineage_control == "independent_shuffle",
        "development_consumed": False,
        "certification_consumed": False,
    }


def provenance():
    selector = (
        Path(__file__).resolve().parents[2]
        / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
    )
    paths = (
        Path(__file__).resolve(),
        Path(lcp.__file__).resolve(),
        Path(protocol.__file__).resolve(),
        Path(common.__file__).resolve(),
        selector,
    )
    return {str(path): common.sha256_file(path) for path in paths}


def save_checkpoint(
    path,
    model,
    config,
    args,
    epoch,
    anchor_path,
    anchor_sha,
    pair_path,
    audit_path,
    gate_path,
    gate_sha,
    fold_path,
    fold_sha,
):
    payload = {
        "schema_version": 1,
        "method": protocol.METHOD,
        "selector_architecture": "scene_conditioned_v3",
        "arm": args.arm,
        "arm_contract": lcp.arm_contract(args.arm),
        "lineage_control": args.lineage_control,
        "scene_selector_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scene_selector_config": config,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "target_kl": args.target_kl,
        "retention_headroom": args.retention_headroom,
        "reward_epsilon": args.reward_epsilon,
        "train_seed": args.seed,
        "epoch": epoch,
        "heldout_fold": args.heldout_fold,
        "v3_anchor": str(anchor_path),
        "v3_anchor_sha256": anchor_sha,
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "mechanism_gate": str(gate_path),
        "mechanism_gate_sha256": gate_sha,
        "fold_contract": str(fold_path),
        "fold_contract_sha256": fold_sha,
        "scientific_contract": scientific_contract(
            args.heldout_fold, args.lineage_control
        ),
        "implementation_files": provenance(),
    }
    torch.save(payload, path)


def validate_args(args):
    numeric_positive = (
        args.temperature,
        args.learning_rate,
        args.epochs,
        args.examples_per_epoch,
        args.batch_size,
        args.reward_epsilon,
    )
    if any(float(value) <= 0.0 for value in numeric_positive):
        raise ValueError("training hyperparameters must be positive")
    if args.kl_weight < 0.0 or args.retention_headroom < 0.0:
        raise ValueError("KL/retention controls must be non-negative")
    if args.heldout_fold not in (-1, 0, 1, 2, 3, 4):
        raise ValueError("heldout fold must be -1 or 0..4")
    if args.arm in lcp.PROXIMAL_ARMS:
        if args.target_kl not in protocol.TARGET_KL_GRID:
            raise ValueError("proximal target KL must be one of 0.01, 0.03, 0.10")
    elif args.target_kl is not None:
        raise ValueError("Direct arms do not take a proximal target KL")
    if args.lineage_control == "independent_shuffle" and args.arm not in lcp.LINEAGE_ARMS:
        raise ValueError("lineage shuffle is defined only for lineage arms")
    checkpoint_epochs = tuple(
        sorted({int(value) for value in args.checkpoint_epochs.split(",")})
    )
    if (
        not checkpoint_epochs
        or checkpoint_epochs[0] <= 0
        or checkpoint_epochs[-1] != args.epochs
        or any(epoch > args.epochs for epoch in checkpoint_epochs)
    ):
        raise ValueError("checkpoint epochs must be positive and end at --epochs")
    return checkpoint_epochs


def main():
    args = parse_args()
    checkpoint_epochs = validate_args(args)
    gate_path, gate_sha = verify_mechanism_gate(args.mechanism_gate)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    loaded = [common.load_cache(path, "train") for path in args.train_cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    standard.assert_cache_group(
        caches, manifests, "train", protocol.TRAIN_NOISE_SEEDS
    )
    pair_rows, rare_audit, pair_path, audit_path = rare_trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    rare_indices, common_indices = rare_trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )
    fold_contract = protocol.build_fold_contract(pair_rows)
    training_positions = protocol.pair_positions_for_training(
        fold_contract, args.heldout_fold
    )
    heldout_positions = set(
        protocol.pair_positions_for_evaluation(fold_contract, args.heldout_fold)
    )
    if args.heldout_fold != -1 and set(training_positions).intersection(heldout_positions):
        raise RuntimeError("heldout fold leaked into training positions")
    train_rare_indices = [rare_indices[position] for position in training_positions]
    train_common_indices = [common_indices[position] for position in training_positions]
    if args.require_full_coverage:
        minimum_per_class = args.epochs * (args.examples_per_epoch // 2)
        if len(training_positions) > minimum_per_class:
            raise RuntimeError("fixed budget cannot cover every training pair")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "fold_contract.json"
    fold_path.write_text(json.dumps(fold_contract, indent=2, sort_keys=True) + "\n")
    fold_sha = common.sha256_file(fold_path)

    v3_payload, anchor_path, anchor_sha = load_v3(
        args.v3_checkpoint, args.v3_checkpoint_sha256
    )
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, anchor, model_config = build_models(v3_payload, caches[0], device)
    anchor_logits = [
        cache_logits(anchor, cache, device, args.batch_size) for cache in caches
    ]
    del anchor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        foreach=False,
    )
    samplers = {
        "rare": rare_trainer.CyclingPairSampler(
            len(training_positions), args.seed + 17
        ),
        "common": rare_trainer.CyclingPairSampler(
            len(training_positions), args.seed + 37
        ),
    }
    schedule = torch.Generator().manual_seed(args.seed + 910000)
    lineage_generator = torch.Generator().manual_seed(args.seed + 20260902)
    totals = {
        "optimizer_steps": 0,
        "outer_groups": 0,
        "view_examples": 0,
        "loss": 0.0,
        "policy": 0.0,
        "current_to_v3_kl": 0.0,
        "utility_headroom": 0.0,
        "active_fraction": 0.0,
        "retained_fraction": 0.0,
        "target_kl": 0.0,
        "target_kl_reached_fraction": 0.0,
        "target_anchor_l1": 0.0,
    }
    checkpoints = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        rare_count, common_count = rare_trainer.block_class_counts(
            args.examples_per_epoch, epoch - 1
        )
        rare_local = samplers["rare"].take(rare_count)
        common_local = samplers["common"].take(common_count)
        selected = torch.tensor(
            [train_rare_indices[position] for position in rare_local]
            + [train_common_indices[position] for position in common_local],
            dtype=torch.long,
        )
        selected = selected[
            torch.randperm(len(selected), generator=schedule)
        ]

        for start in range(0, len(selected), args.batch_size):
            indices = selected[start : start + args.batch_size]
            current_parts, anchor_parts, reward_parts, valid_parts = [], [], [], []
            for draw, cache in enumerate(caches):
                logits, _ = common.current_logits(model, cache, indices, device)
                current_parts.append(logits)
                anchor_parts.append(
                    anchor_logits[draw][indices].to(device=device, dtype=torch.float32)
                )
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
            frozen = torch.stack(anchor_parts, dim=1)
            rewards = torch.stack(reward_parts, dim=1)
            valid = torch.stack(valid_parts, dim=1)
            if args.lineage_control == "independent_shuffle":
                rewards, valid, _ = lcp.independently_shuffle_lineages(
                    rewards, valid, generator=lineage_generator
                )

            optimizer.zero_grad(set_to_none=True)
            loss, policy_loss, current_kl, diagnostics = (
                lcp.lineage_consistent_loss(
                    current,
                    frozen,
                    rewards,
                    valid,
                    arm=args.arm,
                    temperature=args.temperature,
                    kl_weight=args.kl_weight,
                    target_kl=args.target_kl,
                    retention_headroom=args.retention_headroom,
                    reward_epsilon=args.reward_epsilon,
                )
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite {args.arm} loss at epoch {epoch}")
            loss.backward()
            common.clip_grad_norm_cpu_(model.parameters(), 10.0)
            optimizer.step()

            batch_size = len(indices)
            totals["optimizer_steps"] += 1
            totals["outer_groups"] += batch_size
            totals["view_examples"] += batch_size * len(caches)
            totals["loss"] += float(loss.detach().cpu())
            totals["policy"] += float(policy_loss.detach().cpu())
            totals["current_to_v3_kl"] += float(current_kl.detach().cpu())
            totals["utility_headroom"] += float(
                diagnostics["headroom"].mean().cpu()
            )
            totals["active_fraction"] += float(
                diagnostics["active"].float().mean().cpu()
            )
            totals["retained_fraction"] += float(
                diagnostics["retained"].float().mean().cpu()
            )
            totals["target_kl"] += float(
                diagnostics["target_kl"].mean().cpu()
            )
            totals["target_kl_reached_fraction"] += float(
                diagnostics["target_kl_reached"].float().mean().cpu()
            )
            totals["target_anchor_l1"] += float(
                (
                    diagnostics["target"] - diagnostics["anchor_probability"]
                )
                .abs()
                .sum(dim=-1)
                .mean()
                .cpu()
            )

        if epoch in checkpoint_epochs:
            checkpoint_path = output_dir / f"epoch_{epoch}_scene_selector.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                model_config,
                args,
                epoch,
                anchor_path,
                anchor_sha,
                pair_path,
                audit_path,
                gate_path,
                gate_sha,
                fold_path,
                fold_sha,
            )
            checkpoints.append(
                {
                    "epoch": epoch,
                    "scene_selector_state": str(checkpoint_path),
                    "scene_selector_state_sha256": common.sha256_file(
                        checkpoint_path
                    ),
                    "selection_scope": (
                        "log_cv_candidate"
                        if args.heldout_fold != -1
                        else "single_winner_development_candidate"
                    ),
                }
            )
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "epoch": epoch,
                        "heldout_fold": args.heldout_fold,
                        "optimizer_steps": totals["optimizer_steps"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    steps = totals["optimizer_steps"]
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": protocol.METHOD,
        "arm": args.arm,
        "arm_contract": lcp.arm_contract(args.arm),
        "lineage_control": args.lineage_control,
        "target_kl": args.target_kl,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "retention_headroom": args.retention_headroom,
        "reward_epsilon": args.reward_epsilon,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_epochs": list(checkpoint_epochs),
        "examples_per_epoch": args.examples_per_epoch,
        "outer_batch_size": args.batch_size,
        "noise_seeds": list(protocol.TRAIN_NOISE_SEEDS),
        "heldout_fold": args.heldout_fold,
        "num_training_pairs": len(training_positions),
        "num_heldout_pairs": (
            len(heldout_positions) if args.heldout_fold != -1 else 0
        ),
        "v3_anchor": str(anchor_path),
        "v3_anchor_sha256": anchor_sha,
        "mechanism_gate": str(gate_path),
        "mechanism_gate_sha256": gate_sha,
        "train_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "fold_contract": str(fold_path),
        "fold_contract_sha256": fold_sha,
        "sampling_mode": "rare_common_50_50_log_disjoint",
        "scientific_contract": scientific_contract(
            args.heldout_fold, args.lineage_control
        ),
        "budget": {
            key: totals[key]
            for key in ("optimizer_steps", "outer_groups", "view_examples")
        },
        "training_means_per_step": {
            key: totals[key] / steps
            for key in (
                "loss",
                "policy",
                "current_to_v3_kl",
                "utility_headroom",
                "active_fraction",
                "retained_fraction",
                "target_kl",
                "target_kl_reached_fraction",
                "target_anchor_l1",
            )
        },
        "checkpoints": checkpoints,
        "implementation_files": provenance(),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": args.arm,
                "heldout_fold": args.heldout_fold,
                "output": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
