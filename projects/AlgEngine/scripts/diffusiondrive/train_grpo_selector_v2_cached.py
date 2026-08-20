#!/usr/bin/env python3
"""Train one exact-group selector trial on immutable frozen-candidate caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


COMPONENT_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)
REWARD_CONTRACT = "navsim_pairwise_raw_progress_then_candidate_gate_v1"
CACHE_METHOD = "frozen_diffusiondrive_selector_v2_progress_fixed_cache"
REWARD_IMPLEMENTATION = (
    Path(__file__).resolve().parents[2]
    / "mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
)


class Selector(nn.Sequential):
    """Exact final DiffusionDrive plan_cls_branch architecture."""

    def __init__(self):
        super().__init__(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument(
        "--calibration-cache", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--kl-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--checkpoint-epochs", default="1,2,4,8,16,32,64,128,200"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path, expected_split=None):
    path = path.expanduser().resolve()
    manifest_path = path.parent / "manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(path if not path.is_file() else manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PASS":
        raise RuntimeError(f"cache manifest did not pass: {manifest_path}")
    if manifest.get("schema_version") != 2:
        raise RuntimeError(f"cache manifest is not corrected schema v2: {manifest_path}")
    if manifest.get("method") != CACHE_METHOD:
        raise RuntimeError(f"cache method is not corrected V2 rescore: {manifest_path}")
    if manifest.get("reward_contract") != REWARD_CONTRACT:
        raise RuntimeError(f"cache reward contract drifted: {manifest_path}")
    reward_sha = sha256_file(REWARD_IMPLEMENTATION)
    if manifest.get("reward_implementation_sha256") != reward_sha:
        raise RuntimeError(f"cache was scored by stale reward code: {manifest_path}")
    if manifest.get("cache_sha256") != sha256_file(path):
        raise RuntimeError(f"cache SHA256 mismatch: {path}")
    if expected_split is not None and manifest.get("split") != expected_split:
        raise RuntimeError(f"cache split mismatch: {path}")
    cache = torch.load(path, map_location="cpu")
    if cache.get("schema_version") != 2:
        raise RuntimeError(f"cache payload is not corrected schema v2: {path}")
    if cache.get("reward_contract") != REWARD_CONTRACT:
        raise RuntimeError(f"cache payload reward contract drifted: {path}")
    if cache.get("reward_implementation_sha256") != reward_sha:
        raise RuntimeError(f"cache payload reward implementation drifted: {path}")
    if cache.get("source_cache_sha256") != manifest.get("source_cache_sha256"):
        raise RuntimeError(f"cache source provenance drifted: {path}")
    count = len(cache["tokens"])
    required = {
        "candidate_features": (count, 20, 256),
        "candidate_rewards": (count, 20),
        "candidate_reward_components": (count, 20, 6),
        "candidate_reward_valid_mask": (count, 20),
        "reference_logits": (count, 20),
    }
    for key, shape in required.items():
        if tuple(cache[key].shape) != shape:
            raise RuntimeError(f"{path}: {key} shape drifted")
    if len(cache["scenes"]) != count or len(set(cache["tokens"])) != count:
        raise RuntimeError(f"{path}: token/scene provenance drifted")
    if not bool(cache["candidate_reward_valid_mask"].all()):
        raise RuntimeError(f"{path}: corrected reward contains invalid candidates")
    for key in (
        "candidate_features",
        "candidate_rewards",
        "candidate_reward_components",
        "reference_logits",
    ):
        if not bool(torch.isfinite(cache[key]).all()):
            raise RuntimeError(f"{path}: {key} contains non-finite values")
    return cache, manifest


def normalized_advantage(rewards, valid):
    count = valid.sum(dim=-1).clamp_min(1).to(rewards.dtype)
    safe = torch.where(valid, rewards, torch.zeros_like(rewards))
    mean = safe.sum(dim=-1) / count
    centered = torch.where(valid, rewards - mean[:, None], torch.zeros_like(rewards))
    std = (centered.square().sum(dim=-1) / count).sqrt()
    active = (valid.sum(dim=-1) >= 2) & (std > 1e-6)
    advantage = centered / std.clamp_min(1e-6)[:, None]
    return torch.where(valid & active[:, None], advantage, torch.zeros_like(advantage)), active


def exact_group_loss(logits, reference_logits, rewards, valid, temperature, kl_weight):
    current_masked = (logits / temperature).masked_fill(~valid, -1e4)
    reference_masked = (reference_logits / temperature).masked_fill(~valid, -1e4)
    current_logp = F.log_softmax(current_masked, dim=-1)
    reference_logp = F.log_softmax(reference_masked, dim=-1)
    probability = current_logp.exp()
    advantage, active = normalized_advantage(rewards, valid)
    if not active.any():
        zero = logits.sum() * 0.0
        return zero, zero, zero
    policy = -(probability * advantage).sum(dim=-1)[active].mean()
    kl = (
        probability * (current_logp - reference_logp) * valid.to(logits.dtype)
    ).sum(dim=-1)[active].mean()
    return policy + kl_weight * kl, policy, kl


def selector_metrics(logits, reference_logits, rewards, components, valid, temperature):
    current_masked = (logits / temperature).masked_fill(~valid, -1e4)
    reference_masked = (reference_logits / temperature).masked_fill(~valid, -1e4)
    current_logp = F.log_softmax(current_masked, dim=-1)
    reference_logp = F.log_softmax(reference_masked, dim=-1)
    probability = current_logp.exp()
    reference_probability = reference_logp.exp()
    current_index = current_masked.argmax(dim=-1)
    reference_index = reference_masked.argmax(dim=-1)
    oracle_index = rewards.masked_fill(~valid, -torch.inf).argmax(dim=-1)

    def gather(values, index):
        suffix = (1,) * (values.ndim - 2)
        gather_index = index.reshape(-1, 1, *suffix).expand(
            -1, 1, *values.shape[2:]
        )
        return values.gather(1, gather_index).squeeze(1)

    safe_rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
    current_reward = gather(rewards, current_index)
    reference_reward = gather(rewards, reference_index)
    oracle_reward = gather(rewards, oracle_index)
    return {
        "current_reward": current_reward,
        "reference_reward": reference_reward,
        "top1_reward_gain": current_reward - reference_reward,
        "current_expected_reward": (probability * safe_rewards).sum(dim=-1),
        "reference_expected_reward": (
            reference_probability * safe_rewards
        ).sum(dim=-1),
        "oracle_match": current_index.eq(oracle_index).float(),
        "reference_oracle_match": reference_index.eq(oracle_index).float(),
        "oracle_regret": oracle_reward - current_reward,
        "selection_disagreement": current_index.ne(reference_index).float(),
        "kl": (
            probability * (current_logp - reference_logp) * valid.to(logits.dtype)
        ).sum(dim=-1),
        "component_delta": gather(components, current_index)
        - gather(components, reference_index),
    }


def evaluate(model, cache, device, temperature, batch_size):
    output = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            feature = cache["candidate_features"][start:stop].to(
                device=device, dtype=torch.float32
            )
            values = selector_metrics(
                model(feature).squeeze(-1),
                cache["reference_logits"][start:stop].to(device),
                cache["candidate_rewards"][start:stop].to(device),
                cache["candidate_reward_components"][start:stop].to(device),
                cache["candidate_reward_valid_mask"][start:stop].to(device),
                temperature,
            )
            for key, value in values.items():
                output.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(value) for key, value in output.items()}


def summarize(values):
    summary = {
        key: float(value.mean())
        for key, value in values.items()
        if key != "component_delta"
    }
    for index, name in enumerate(COMPONENT_NAMES):
        summary[f"delta_{name}"] = float(values["component_delta"][:, index].mean())
    return summary


def write_calibration_records(path, caches, manifests, values_by_seed):
    with path.open("w") as stream:
        for cache, manifest, values in zip(caches, manifests, values_by_seed):
            for index, token in enumerate(cache["tokens"]):
                row = {
                    "token": str(token),
                    "scene": str(cache["scenes"][index]),
                    "noise_seed": int(manifest["noise_seed"]),
                    "top1_reward_gain": float(values["top1_reward_gain"][index]),
                    "selection_disagreement": float(
                        values["selection_disagreement"][index]
                    ),
                    "oracle_match": float(values["oracle_match"][index]),
                    "oracle_regret": float(values["oracle_regret"][index]),
                    "component_delta": {
                        name: float(values["component_delta"][index, component_index])
                        for component_index, name in enumerate(COMPONENT_NAMES)
                    },
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    args = parse_args()
    if args.temperature <= 0 or args.learning_rate <= 0 or args.kl_weight < 0:
        raise ValueError("invalid temperature/learning-rate/KL weight")
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

    train_cache, train_manifest = load_cache(args.train_cache, "train")
    calibration_pairs = [
        load_cache(path, "calibration") for path in args.calibration_cache
    ]
    calibration_pairs.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    calibration_caches = [pair[0] for pair in calibration_pairs]
    calibration_manifests = [pair[1] for pair in calibration_pairs]
    if [int(row["noise_seed"]) for row in calibration_manifests] != [0, 1, 2]:
        raise RuntimeError("calibration caches must contain noise seeds 0,1,2")
    all_manifests = [train_manifest, *calibration_manifests]
    reward_shas = {
        row.get("reward_implementation_sha256") for row in all_manifests
    }
    if reward_shas != {sha256_file(REWARD_IMPLEMENTATION)}:
        raise RuntimeError("corrected cache reward provenance is inconsistent")
    if {row.get("reward_contract") for row in all_manifests} != {REWARD_CONTRACT}:
        raise RuntimeError("corrected cache reward contract is inconsistent")
    train_tokens = set(train_cache["tokens"])
    if any(train_tokens.intersection(cache["tokens"]) for cache in calibration_caches):
        raise RuntimeError("train/calibration token leakage")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = Selector().to(device)
    model.load_state_dict(train_cache["baseline_selector_state"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    train_indices = torch.arange(len(train_cache["tokens"]))
    generator = torch.Generator().manual_seed(args.seed)
    checkpoint_reports = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_indices[
            torch.randperm(len(train_indices), generator=generator)
        ]
        for start in range(0, len(order), args.batch_size):
            local = order[start : start + args.batch_size]
            feature = train_cache["candidate_features"][local].to(
                device=device, dtype=torch.float32
            )
            reward = train_cache["candidate_rewards"][local].to(device)
            reference = train_cache["reference_logits"][local].to(device)
            valid = train_cache["candidate_reward_valid_mask"][local].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = exact_group_loss(
                model(feature).squeeze(-1),
                reference,
                reward,
                valid,
                args.temperature,
                args.kl_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite V2 loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        if epoch not in checkpoint_epochs:
            continue
        state_path = output_dir / f"epoch_{epoch}_selector.pt"
        state_payload = {
            "schema_version": 2,
            "method": "exact_group_grpo",
            "reward_contract": REWARD_CONTRACT,
            "reward_implementation_sha256": sha256_file(REWARD_IMPLEMENTATION),
            "selector_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "kl_weight": args.kl_weight,
            "train_seed": args.seed,
            "epoch": epoch,
        }
        torch.save(state_payload, state_path)
        train_values = evaluate(
            model, train_cache, device, args.temperature, args.batch_size
        )
        calibration_values = [
            evaluate(model, cache, device, args.temperature, args.batch_size)
            for cache in calibration_caches
        ]
        records_path = output_dir / f"epoch_{epoch}_calibration.jsonl"
        write_calibration_records(
            records_path,
            calibration_caches,
            calibration_manifests,
            calibration_values,
        )
        pooled = {
            key: torch.cat([values[key] for values in calibration_values])
            for key in calibration_values[0]
        }
        checkpoint_reports.append(
            {
                "epoch": epoch,
                "selector_state": str(state_path),
                "selector_state_sha256": sha256_file(state_path),
                "calibration_records": str(records_path),
                "calibration_records_sha256": sha256_file(records_path),
                "train_metrics": summarize(train_values),
                "calibration_metrics": summarize(pooled),
                "calibration_by_noise_seed": {
                    str(manifest["noise_seed"]): summarize(values)
                    for manifest, values in zip(
                        calibration_manifests, calibration_values
                    )
                },
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "calibration_top1_gain": summarize(pooled)[
                        "top1_reward_gain"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report = {
        "schema_version": 2,
        "status": "PASS",
        "method": "exact_group_grpo",
        "reward_contract": REWARD_CONTRACT,
        "reward_implementation": str(REWARD_IMPLEMENTATION),
        "reward_implementation_sha256": sha256_file(REWARD_IMPLEMENTATION),
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "train_seed": args.seed,
        "epochs": args.epochs,
        "checkpoint_epochs": list(checkpoint_epochs),
        "early_stopping": False,
        "train_manifest": train_manifest,
        "calibration_manifests": calibration_manifests,
        "checkpoints": checkpoint_reports,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS V2 cached selector trial: {report_path}")


if __name__ == "__main__":
    main()
