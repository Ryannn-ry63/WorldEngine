#!/usr/bin/env python3
"""Fixed-budget V3 selector post-training for the four PAF attribution arms."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common
import proposal_aware_full_feedback_grpo as paf
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


EXPECTED_NOISE_SEEDS = (0, 1, 2)
METHOD = "proposal_aware_full_feedback_selector_v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-sha256", required=True)
    parser.add_argument("--mechanism-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=paf.ARMS, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--opportunity-lower", type=float, default=0.005)
    parser.add_argument("--opportunity-upper", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--examples-per-epoch", type=int, default=6339)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-epochs", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-full-coverage", action="store_true")
    return parser.parse_args()


def verify_mechanism_gate(path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != "selector_policy_suppression_audit_v1"
        or payload.get("stage") != "fresh"
        or payload.get("decision") != "AUTHORIZE_PROPOSAL_AWARE_FULL_FEEDBACK_GRPO"
        or payload.get("noise_seeds") != [9, 10, 11]
        or payload.get("scientific_contract", {}).get("development_consumed") is not False
    ):
        raise RuntimeError("fresh-noise mechanism gate did not authorize PAF training")
    return path, common.sha256_file(path)


def load_v3(path, expected_sha):
    path = path.expanduser().resolve()
    actual_sha = common.sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError("V3 checkpoint SHA256 drifted")
    payload = torch.load(path, map_location="cpu")
    if "scene_selector_state" not in payload or "scene_selector_config" not in payload:
        raise RuntimeError("V3 checkpoint lacks selector state/config")
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


def provenance():
    selector = Path(__file__).resolve().parents[2] / "mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
    paths = (Path(__file__).resolve(), Path(paf.__file__).resolve(), Path(common.__file__).resolve(), selector)
    return {str(path): common.sha256_file(path) for path in paths}


def scientific_contract():
    return {
        "selector_only": True,
        "generator_frozen": True,
        "official_scalar_pdm_only": True,
        "reward_components_consumed": False,
        "new_training_scene_tokens_consumed": False,
        "v3_initialized_and_anchored": True,
        "development_consumed": False,
        "certification_consumed": False,
    }


def save_checkpoint(path, model, config, args, epoch, anchor_path, anchor_sha, pair_path, audit_path, gate_path, gate_sha):
    payload = {
        "schema_version": 1,
        "method": METHOD,
        "selector_architecture": "scene_conditioned_v3",
        "arm": args.arm,
        "arm_contract": paf.arm_contract(args.arm),
        "scene_selector_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "scene_selector_config": config,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "opportunity_lower": args.opportunity_lower,
        "opportunity_upper": args.opportunity_upper,
        "train_seed": args.seed,
        "epoch": epoch,
        "v3_anchor": str(anchor_path),
        "v3_anchor_sha256": anchor_sha,
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "mechanism_gate": str(gate_path),
        "mechanism_gate_sha256": gate_sha,
        "scientific_contract": scientific_contract(),
        "implementation_files": provenance(),
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    positives = (args.temperature, args.learning_rate, args.epochs, args.examples_per_epoch, args.batch_size)
    if any(float(value) <= 0.0 for value in positives) or args.kl_weight < 0.0:
        raise ValueError("training hyperparameters are outside their valid range")
    if not 0.0 <= args.opportunity_lower < args.opportunity_upper:
        raise ValueError("invalid opportunity bounds")
    checkpoint_epochs = tuple(sorted({int(value) for value in args.checkpoint_epochs.split(",")}))
    if not checkpoint_epochs or checkpoint_epochs[0] <= 0 or checkpoint_epochs[-1] != args.epochs:
        raise ValueError("checkpoint epochs must be positive and end at --epochs")
    gate_path, gate_sha = verify_mechanism_gate(args.mechanism_gate)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    loaded = [common.load_cache(path, "train") for path in args.train_cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    standard.assert_cache_group(caches, manifests, "train", EXPECTED_NOISE_SEEDS)
    pair_rows, rare_audit, pair_path, audit_path = rare_trainer.load_pair_contract(args.pair_manifest, args.rare_data_audit)
    rare_indices, common_indices = rare_trainer.validate_cache_pair_alignment(caches, manifests, pair_rows, rare_audit)
    if args.require_full_coverage and len(pair_rows) > args.epochs * (args.examples_per_epoch // 2):
        raise RuntimeError("budget cannot visit every rare/common pair")
    v3_payload, anchor_path, anchor_sha = load_v3(args.v3_checkpoint, args.v3_checkpoint_sha256)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, anchor, model_config = build_models(v3_payload, caches[0], device)
    anchor_logits = [cache_logits(anchor, cache, device, args.batch_size) for cache in caches]
    del anchor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4, foreach=False)
    samplers = {
        "rare": rare_trainer.CyclingPairSampler(len(pair_rows), args.seed + 17),
        "common": rare_trainer.CyclingPairSampler(len(pair_rows), args.seed + 37),
    }
    schedule = torch.Generator().manual_seed(args.seed + 910000)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    totals = defaultdict_float = {
        "optimizer_steps": 0,
        "outer_groups": 0,
        "view_examples": 0,
        "loss": 0.0,
        "policy": 0.0,
        "kl": 0.0,
        "headroom": 0.0,
        "opportunity_weight": 0.0,
        "target_anchor_l1": 0.0,
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        rare_count, common_count = rare_trainer.block_class_counts(args.examples_per_epoch, epoch - 1)
        rare_positions = samplers["rare"].take(rare_count)
        common_positions = samplers["common"].take(common_count)
        selected = torch.tensor(
            [rare_indices[position] for position in rare_positions]
            + [common_indices[position] for position in common_positions],
            dtype=torch.long,
        )
        labels = torch.tensor([1] * rare_count + [0] * common_count, dtype=torch.long)
        permutation = torch.randperm(len(selected), generator=schedule)
        selected = selected[permutation]
        labels = labels[permutation]
        del labels

        for start in range(0, len(selected), args.batch_size):
            indices = selected[start : start + args.batch_size]
            current_parts, anchor_parts, reward_parts, valid_parts = [], [], [], []
            for draw, cache in enumerate(caches):
                logits, _ = common.current_logits(model, cache, indices, device)
                current_parts.append(logits)
                anchor_parts.append(anchor_logits[draw][indices].to(device=device, dtype=torch.float32))
                reward_parts.append(cache["candidate_rewards"][indices].to(device=device, dtype=torch.float32))
                valid_parts.append(cache["candidate_reward_valid_mask"][indices].to(device=device, dtype=torch.bool))
            current = torch.stack(current_parts, dim=1)
            frozen = torch.stack(anchor_parts, dim=1)
            rewards = torch.stack(reward_parts, dim=1)
            valid = torch.stack(valid_parts, dim=1)
            optimizer.zero_grad(set_to_none=True)
            loss, policy, kl, diagnostics = paf.proposal_aware_loss(
                current,
                frozen,
                rewards,
                valid,
                arm=args.arm,
                temperature=args.temperature,
                kl_weight=args.kl_weight,
                opportunity_lower=args.opportunity_lower,
                opportunity_upper=args.opportunity_upper,
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
            totals["policy"] += float(policy.detach().cpu())
            totals["kl"] += float(kl.detach().cpu())
            totals["headroom"] += float(diagnostics["headroom"].mean().cpu())
            totals["opportunity_weight"] += float(diagnostics["opportunity_weight"].mean().cpu())
            totals["target_anchor_l1"] += float(
                (diagnostics["target"] - diagnostics["anchor_probability"]).abs().sum(dim=-1).mean().cpu()
            )

        if epoch in checkpoint_epochs:
            checkpoint_path = output_dir / f"epoch_{epoch}_scene_selector.pt"
            save_checkpoint(
                checkpoint_path, model, model_config, args, epoch,
                anchor_path, anchor_sha, pair_path, audit_path, gate_path, gate_sha,
            )
            checkpoints.append({
                "epoch": epoch,
                "scene_selector_state": str(checkpoint_path),
                "scene_selector_state_sha256": common.sha256_file(checkpoint_path),
                "selection_scope": "diagnostic_only" if epoch != args.epochs else "fixed_primary",
            })
            print(json.dumps({"arm": args.arm, "epoch": epoch, "optimizer_steps": totals["optimizer_steps"]}, sort_keys=True), flush=True)

    steps = totals["optimizer_steps"]
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": METHOD,
        "arm": args.arm,
        "arm_contract": paf.arm_contract(args.arm),
        "scene_selector_config": model_config,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "opportunity_lower": args.opportunity_lower,
        "opportunity_upper": args.opportunity_upper,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "examples_per_epoch": args.examples_per_epoch,
        "outer_batch_size": args.batch_size,
        "noise_seeds": list(EXPECTED_NOISE_SEEDS),
        "v3_anchor": str(anchor_path),
        "v3_anchor_sha256": anchor_sha,
        "mechanism_gate": str(gate_path),
        "mechanism_gate_sha256": gate_sha,
        "train_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "sampling_mode": "rare_common_50_50",
        "scientific_contract": scientific_contract(),
        "budget": {key: totals[key] for key in ("optimizer_steps", "outer_groups", "view_examples")},
        "training_means_per_step": {
            key: totals[key] / steps
            for key in ("loss", "policy", "kl", "headroom", "opportunity_weight", "target_anchor_l1")
        },
        "checkpoints": checkpoints,
        "implementation_files": provenance(),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "arm": args.arm, "output": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
