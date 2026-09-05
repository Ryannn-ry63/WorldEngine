#!/usr/bin/env python3
"""Train-only, log-disjoint probe for explicit candidate-agent information.

This probe is intentionally upstream of selector training.  It asks whether
frozen track interactions add linearly recoverable pairwise PDM ordering beyond
the information already available to V3.  Rewards only define probe labels and
are never included in either feature set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import grpo_selector_v3_cached_common as common
import train_trajectory_set_reasoner_grpo as trainer


NUM_FOLDS = 5


def compact_channel_statistics(values: torch.Tensor, blocks: int = 16) -> torch.Tensor:
    if values.shape[-1] % blocks:
        raise ValueError("feature channels must divide into locked probe blocks")
    shape = (*values.shape[:-1], blocks, values.shape[-1] // blocks)
    blocked = values.reshape(shape)
    return torch.cat((blocked.mean(dim=-1), blocked.std(dim=-1, unbiased=False)), dim=-1)


def base_candidate_features(cache: dict, indices: torch.Tensor) -> torch.Tensor:
    module = common.SELECTOR_MODULE
    candidate = cache["candidate_features"][indices].float()
    trajectory = cache["candidate_trajectories_8"][indices].float()
    route = cache["route_bev_features"][indices].float().mean(dim=-2)
    reference = cache["reference_logits"][indices].float().unsqueeze(-1)
    return torch.cat(
        (
            compact_channel_statistics(candidate),
            module.candidate_trajectory_geometry(trajectory),
            compact_channel_statistics(route),
            reference,
        ),
        dim=-1,
    )


def augmented_candidate_features(cache: dict, indices: torch.Tensor) -> torch.Tensor:
    base = base_candidate_features(cache, indices)
    interaction = common.SELECTOR_MODULE.interaction_probe_features(
        cache["candidate_trajectories_8"][indices].float(),
        cache["frozen_track_states"][indices].float(),
        cache["frozen_track_mask"][indices].bool(),
    )
    return torch.cat((base, interaction), dim=-1)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def sampled_pair_indices(rewards: torch.Tensor, count: int, seed: int):
    left, right = torch.triu_indices(rewards.numel(), rewards.numel(), offset=1)
    informative = (rewards[left] - rewards[right]).abs().gt(1e-6)
    left, right = left[informative], right[informative]
    if not left.numel():
        return left, right
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(left.numel(), generator=generator)[:count]
    left, right = left[order], right[order]
    swap = torch.rand(left.numel(), generator=generator).gt(0.5)
    return torch.where(swap, right, left), torch.where(swap, left, right)


def build_examples(caches, token_logs, pairs_per_token, batch_size):
    base_parts, extra_parts, target_parts, log_parts = [], [], [], []
    log_names = sorted(set(token_logs.values()))
    log_ids = {name: index for index, name in enumerate(log_names)}
    for cache in caches:
        noise_seed = int(cache["probe_noise_seed"])
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            indices = torch.arange(start, stop, dtype=torch.long)
            base = base_candidate_features(cache, indices)
            extra = common.SELECTOR_MODULE.interaction_probe_features(
                cache["candidate_trajectories_8"][indices].float(),
                cache["frozen_track_states"][indices].float(),
                cache["frozen_track_mask"][indices].bool(),
            )
            for local, token_index in enumerate(range(start, stop)):
                token = str(cache["tokens"][token_index])
                if token not in token_logs:
                    raise RuntimeError(f"probe token has no audited log: {token}")
                rewards = cache["candidate_rewards"][token_index].float()
                left, right = sampled_pair_indices(
                    rewards,
                    pairs_per_token,
                    stable_seed(token, noise_seed, "pairwise_probe"),
                )
                if not left.numel():
                    continue
                base_parts.append(base[local, left] - base[local, right])
                extra_parts.append(extra[local, left] - extra[local, right])
                target_parts.append(rewards[left].gt(rewards[right]).float())
                log_parts.append(
                    torch.full(
                        (left.numel(),), log_ids[token_logs[token]], dtype=torch.long
                    )
                )
    if not base_parts:
        raise RuntimeError("probe produced no informative PDM pairs")
    return (
        torch.cat(base_parts),
        torch.cat(extra_parts),
        torch.cat(target_parts),
        torch.cat(log_parts),
        log_names,
    )


def assign_log_folds(log_ids: torch.Tensor, log_names: list[str]):
    counts = Counter(int(value) for value in log_ids.tolist())
    fold_load = [0] * NUM_FOLDS
    assignment = {}
    ordered = sorted(range(len(log_names)), key=lambda index: (-counts[index], log_names[index]))
    for log_id in ordered:
        fold = min(range(NUM_FOLDS), key=lambda value: (fold_load[value], value))
        assignment[log_id] = fold
        fold_load[fold] += counts[log_id]
    if any(value == 0 for value in fold_load):
        raise RuntimeError("fewer than five non-empty log folds")
    return torch.tensor([assignment[int(value)] for value in log_ids]), assignment


def binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.double().cpu()
    targets = targets.bool().cpu()
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        raise RuntimeError("AUC requires both pair orientations")
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    unique, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    del unique
    starts = counts.cumsum(0) - counts
    average = starts.double() + (counts.double() + 1.0) / 2.0
    tied_ranks = average[inverse]
    positive_rank_sum = tied_ranks[targets[order]].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def fit_probe(features, targets, train_mask, seed, epochs, batch_size, device):
    train = features[train_mask]
    labels = targets[train_mask]
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-4)
    model = torch.nn.Linear(features.shape[-1], 1).to(device)
    torch.manual_seed(seed)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(epochs):
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            x = ((train[index] - mean) / scale).to(device)
            y = labels[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        standardized = (features - mean) / scale
        outputs = []
        for start in range(0, len(features), 8192):
            outputs.append(model(standardized[start : start + 8192].to(device)).squeeze(-1).cpu())
    return torch.cat(outputs)


def bootstrap_log_delta(records, repetitions, seed):
    names = sorted(records)
    deltas = np.asarray(
        [records[name]["augmented_auc"] - records[name]["base_auc"] for name in names],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(names), size=(repetitions, len(names)))
    estimates = deltas[samples].mean(axis=1)
    return {
        "mean": float(deltas.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "log",
        "repetitions": repetitions,
        "num_logs": len(names),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-per-token", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(
        args.pairs_per_token,
        args.epochs,
        args.batch_size,
        args.feature_batch_size,
        args.bootstrap_repetitions,
    ) <= 0:
        raise ValueError("probe budgets must be positive")
    pairs = [common.load_cache(path, "train") for path in args.train_cache]
    pairs.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*pairs)
    if [int(row["noise_seed"]) for row in manifests] != [0, 1, 2]:
        raise RuntimeError("probe requires train noise seeds 0,1,2")
    if any(cache.get("schema_version") != 4 for cache in caches):
        raise RuntimeError("probe requires frozen-track schema-v4 caches")
    pair_rows, audit, pair_path, audit_path = trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    trainer.validate_cache_pair_alignment(caches, manifests, pair_rows, audit)
    token_logs = {}
    for row in pair_rows:
        for key in ("rare_token", "common_token"):
            token = str(row[key])
            previous = token_logs.setdefault(token, str(row["log_name"]))
            if previous != str(row["log_name"]):
                raise RuntimeError(f"token mapped to multiple logs: {token}")

    probe_caches = []
    for cache, manifest in zip(caches, manifests):
        copied = dict(cache)
        copied["probe_noise_seed"] = int(manifest["noise_seed"])
        probe_caches.append(copied)
    base, extra, targets, log_ids, log_names = build_examples(
        probe_caches, token_logs, args.pairs_per_token, args.feature_batch_size
    )
    folds, assignment = assign_log_folds(log_ids, log_names)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    fold_rows = []
    log_records = {}
    for fold in range(NUM_FOLDS):
        validation = folds.eq(fold)
        training = ~validation
        base_scores = fit_probe(
            base, targets, training, args.seed + fold * 10, args.epochs, args.batch_size, device
        )
        augmented_scores = fit_probe(
            torch.cat((base, extra), dim=-1),
            targets,
            training,
            args.seed + fold * 10 + 1,
            args.epochs,
            args.batch_size,
            device,
        )
        base_auc = binary_auc(base_scores[validation], targets[validation])
        augmented_auc = binary_auc(augmented_scores[validation], targets[validation])
        fold_logs = []
        for log_id, name in enumerate(log_names):
            if assignment[log_id] != fold:
                continue
            mask = validation & log_ids.eq(log_id)
            log_records[name] = {
                "base_auc": binary_auc(base_scores[mask], targets[mask]),
                "augmented_auc": binary_auc(augmented_scores[mask], targets[mask]),
                "examples": int(mask.sum()),
                "fold": fold,
            }
            fold_logs.append(name)
        fold_rows.append(
            {
                "fold": fold,
                "logs": fold_logs,
                "examples": int(validation.sum()),
                "base_auc": base_auc,
                "augmented_auc": augmented_auc,
                "delta_auc": augmented_auc - base_auc,
            }
        )

    bootstrap = bootstrap_log_delta(log_records, args.bootstrap_repetitions, args.seed + 999)
    mean_base = float(np.mean([row["base_auc"] for row in fold_rows]))
    mean_augmented = float(np.mean([row["augmented_auc"] for row in fold_rows]))
    mean_delta = mean_augmented - mean_base
    gates = {
        "mean_auc_gain_at_least_001": mean_delta >= 0.01,
        "paired_log_bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
        "no_fold_worse_than_minus_001": min(row["delta_auc"] for row in fold_rows) >= -0.01,
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "interaction_representation_train_only_log_cv_probe_v1",
        "development_or_certification_consumed": False,
        "decision": "AUTHORIZE_A0_A1_A2" if all(gates.values()) else "STOP_INTERACTION_AND_TEST_HISTORY",
        "probe_features": {
            "base": "V3 candidate/trajectory/route/reference compact statistics",
            "augmentation": "frozen-track constant-velocity candidate-frame interactions",
            "reward_or_components_as_input": False,
            "base_dimension": base.shape[-1],
            "interaction_dimension": extra.shape[-1],
        },
        "budgets": {
            "folds": NUM_FOLDS,
            "pairs_per_token_per_noise": args.pairs_per_token,
            "probe_epochs": args.epochs,
            "bootstrap_repetitions": args.bootstrap_repetitions,
        },
        "mean_base_auc": mean_base,
        "mean_augmented_auc": mean_augmented,
        "mean_delta_auc": mean_delta,
        "folds": fold_rows,
        "paired_log_bootstrap": bootstrap,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "per_log": log_records,
        "train_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "decision": report["decision"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
