#!/usr/bin/env python3
"""Train-only audit of whether proposal history adds selector information.

This is a backup-path information audit, not a selector architecture.  It uses
only already-cached current/preceding proposals, frozen V3 decisions, ego poses,
and the current token's scalar PDM as a probe label.  No preceding outcome is
read.  Current-only, real-history, and same-stratum cross-log shuffled-history
probes have identical input dimension and optimization budget.

The primary task is deliberately aligned with the observed selector failure:
among V3's locked top-k, can observable history tell whether a challenger
should replace the V3 incumbent?  A reward-independent random-pair ranking task
is retained only as a secondary diagnostic.  Authorization depends on the
incumbent-boundary task, not generic pairwise AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import grpo_selector_v3_cached_common as common
import lcpgrpo_protocol as protocol
import proposal_history_features as history
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


EXPECTED_PREDECESSOR_PAIRS = 2424


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-per-token", type=int, default=16)
    parser.add_argument("--incumbent-top-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--expected-predecessor-pairs", type=int, default=EXPECTED_PREDECESSOR_PAIRS)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class DigestingReader:
    """Minimal sequential file wrapper accepted by pickle.Unpickler."""

    def __init__(self, path):
        self.stream = Path(path).open("rb")
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        return value

    def readline(self, size=-1):
        value = self.stream.readline(size)
        self.digest.update(value)
        return value

    def close(self):
        self.stream.close()


def load_annotation_metadata(path, target_tokens, expected_sha):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = DigestingReader(path)
    try:
        payload = pickle.load(reader)
        actual_sha = reader.digest.hexdigest()
    finally:
        reader.close()
    if actual_sha != expected_sha:
        raise RuntimeError("annotation SHA256 differs from locked cache manifest")
    infos = payload.get("infos") if isinstance(payload, dict) else payload
    if not isinstance(infos, list):
        raise TypeError("unsupported annotation payload")
    metadata = {}
    for info in infos:
        token = str(info["token"])
        if token not in target_tokens:
            continue
        pose = np.asarray(info["ego2global"], dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise RuntimeError(f"invalid ego2global pose for {token}")
        if token in metadata:
            raise RuntimeError(f"annotation repeats target token: {token}")
        metadata[token] = {
            "sample_prev": str(info.get("sample_prev") or ""),
            "sample_next": str(info.get("sample_next") or ""),
            "log_name": str(info["log_name"]),
            "scene_token": str(info.get("scene_token") or ""),
            "timestamp": int(info["timestamp"]),
            "frame_idx": int(info["frame_idx"]),
            "ego2global": pose,
        }
    if set(metadata) != target_tokens:
        raise RuntimeError(
            f"annotation coverage drift: missing={len(target_tokens - set(metadata))}"
        )
    return metadata, path, actual_sha


def immediate_predecessor_pairs(tokens, metadata):
    token_to_index = {str(token): index for index, token in enumerate(tokens)}
    if len(token_to_index) != len(tokens):
        raise RuntimeError("cache contains duplicate tokens")
    token_set = set(token_to_index)
    pairs = []
    for current_index, current_token in enumerate(tokens):
        current_token = str(current_token)
        current = metadata[current_token]
        previous_token = current["sample_prev"]
        if not previous_token or previous_token not in token_set:
            continue
        previous = metadata[previous_token]
        if previous["sample_next"] != current_token:
            raise RuntimeError("sample_prev/sample_next annotations are not reciprocal")
        if previous["log_name"] != current["log_name"]:
            raise RuntimeError("immediate predecessor crosses logs")
        if previous["scene_token"] != current["scene_token"]:
            raise RuntimeError("immediate predecessor crosses scenes")
        if previous["timestamp"] >= current["timestamp"]:
            raise RuntimeError("immediate predecessor is not earlier")
        pairs.append((current_index, token_to_index[previous_token]))
    return pairs


def compact_channel_statistics(values, blocks=16):
    if values.shape[-1] % blocks:
        raise ValueError("feature channels must divide into probe blocks")
    blocked = values.reshape(
        *values.shape[:-1], blocks, values.shape[-1] // blocks
    )
    return torch.cat(
        (blocked.mean(dim=-1), blocked.std(dim=-1, unbiased=False)), dim=-1
    )


def load_v3(path, expected_sha, cache, device):
    path = path.expanduser().resolve()
    actual_sha = common.sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError("V3 checkpoint SHA256 drifted")
    payload = torch.load(path, map_location="cpu")
    model = common.model_from_config(payload["scene_selector_config"])
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    for key, value in cache["scene_selector_config"].items():
        if key in payload["scene_selector_config"] and payload["scene_selector_config"][key] != value:
            raise RuntimeError(f"V3/cache config drifted at {key}")
    model.to(device).eval()
    return model, payload, path, actual_sha


def infer_v3_logits(model, cache, device, batch_size):
    parts = []
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            logits, _ = common.current_logits(model, cache, slice(start, stop), device)
            parts.append(logits.cpu())
    return torch.cat(parts)


def current_candidate_features(cache, indices, logits, probability):
    candidate = cache["candidate_features"][indices].float()
    trajectory = cache["candidate_trajectories_8"][indices].float()
    route = cache["route_bev_features"][indices].float().mean(dim=-2)
    geometry = common.SELECTOR_MODULE.candidate_trajectory_geometry(trajectory)
    return torch.cat(
        (
            compact_channel_statistics(candidate),
            geometry,
            compact_channel_statistics(route),
            logits[indices].float().unsqueeze(-1),
            probability[indices].float().unsqueeze(-1),
        ),
        dim=-1,
    )


def build_feature_banks(
    caches,
    current_indices,
    previous_indices,
    current_poses,
    previous_poses,
    v3_logits,
    batch_size,
):
    base_by_draw, history_by_draw = [], []
    for draw, cache in enumerate(caches):
        # Reward validity is an evaluation artifact and is unavailable to a
        # deployed selector. The audit therefore forms V3 probabilities from
        # logits alone; reward tensors are consumed later only as probe labels.
        if not torch.isfinite(v3_logits[draw]).all():
            raise RuntimeError("non-finite frozen-V3 logits")
        probability = torch.softmax(v3_logits[draw].float(), dim=-1)
        base_parts, history_parts = [], []
        for start in range(0, len(current_indices), batch_size):
            stop = min(start + batch_size, len(current_indices))
            current = current_indices[start:stop]
            previous = previous_indices[start:stop]
            base_parts.append(
                current_candidate_features(
                    cache, current, v3_logits[draw], probability
                )
            )
            current_trajectory = cache["candidate_trajectories_8"][current].float()
            previous_trajectory = cache["candidate_trajectories_8"][previous].float()
            transformed = history.transform_trajectories_between_ego_frames(
                previous_trajectory,
                previous_poses[start:stop],
                current_poses[start:stop],
            )
            history_parts.append(
                history.proposal_history_candidate_features(
                    current_trajectory,
                    transformed,
                    probability[previous].float(),
                )
            )
        base_by_draw.append(torch.cat(base_parts))
        history_by_draw.append(torch.cat(history_parts))
    return base_by_draw, history_by_draw


def sampled_pair_indices(rewards, valid, count, seed):
    left, right = torch.triu_indices(rewards.numel(), rewards.numel(), offset=1)
    informative = (
        valid[left]
        & valid[right]
        & torch.isfinite(rewards[left])
        & torch.isfinite(rewards[right])
        & (rewards[left] - rewards[right]).abs().gt(1e-6)
    )
    left, right = left[informative], right[informative]
    if not left.numel():
        return left, right
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(left.numel(), generator=generator)[:count]
    left, right = left[order], right[order]
    swap = torch.rand(left.numel(), generator=generator).gt(0.5)
    return torch.where(swap, right, left), torch.where(swap, left, right)


def build_pairwise_probe_examples(
    caches,
    base_by_draw,
    history_by_draw,
    shuffle,
    current_indices,
    tokens,
    logs,
    strata,
    pairs_per_token,
):
    current_parts, real_parts, shuffled_parts = [], [], []
    target_parts, fold_parts, token_parts, log_parts, stratum_parts = [], [], [], [], []
    for draw, cache in enumerate(caches):
        base = base_by_draw[draw]
        real_history = history_by_draw[draw]
        shuffled_history = real_history[shuffle]
        for local, cache_index in enumerate(current_indices):
            reward = cache["candidate_rewards"][cache_index].float()
            valid = cache["candidate_reward_valid_mask"][cache_index].bool()
            left, right = sampled_pair_indices(
                reward,
                valid,
                pairs_per_token,
                history.stable_seed(tokens[local], draw, "proposal_history_probe"),
            )
            if not left.numel():
                continue
            base_delta = base[local, left] - base[local, right]
            real_delta = real_history[local, left] - real_history[local, right]
            shuffled_delta = (
                shuffled_history[local, left] - shuffled_history[local, right]
            )
            zero_history = torch.zeros_like(real_delta)
            current_parts.append(torch.cat((base_delta, zero_history), dim=-1))
            real_parts.append(torch.cat((base_delta, real_delta), dim=-1))
            shuffled_parts.append(torch.cat((base_delta, shuffled_delta), dim=-1))
            target_parts.append(reward[left].gt(reward[right]).float())
            fold_parts.append(
                torch.full(
                    (left.numel(),), protocol.log_fold(logs[local]), dtype=torch.long
                )
            )
            token_parts.extend([tokens[local]] * left.numel())
            log_parts.extend([logs[local]] * left.numel())
            stratum_parts.extend([strata[local]] * left.numel())
    if not current_parts:
        raise RuntimeError("proposal-history audit produced no informative pairs")
    return {
        "current": torch.cat(current_parts),
        "real": torch.cat(real_parts),
        "shuffled": torch.cat(shuffled_parts),
        "target": torch.cat(target_parts),
        "fold": torch.cat(fold_parts),
        "tokens": token_parts,
        "logs": log_parts,
        "strata": stratum_parts,
    }


def build_incumbent_probe_examples(
    caches,
    base_by_draw,
    history_by_draw,
    v3_logits,
    shuffle,
    current_indices,
    tokens,
    logs,
    strata,
    top_k,
):
    """Build challenger-versus-V3 examples without reward-based mining.

    V3 logits alone lock the incumbent and shortlist.  Scalar PDM is read only
    after that decision and supplies the binary probe label.  Reward ties and
    invalid comparisons are omitted rather than assigned an arbitrary target.
    """

    candidate_count = int(base_by_draw[0].shape[1])
    if not 2 <= top_k <= candidate_count:
        raise ValueError(
            f"incumbent top-k must be in [2,{candidate_count}], got {top_k}"
        )
    current_parts, real_parts, shuffled_parts = [], [], []
    target_parts, fold_parts, token_parts, log_parts, stratum_parts = [], [], [], [], []
    for draw, cache in enumerate(caches):
        base = base_by_draw[draw]
        real_history = history_by_draw[draw]
        shuffled_history = real_history[shuffle]
        shortlist = torch.topk(
            v3_logits[draw][current_indices].float(), k=top_k, dim=-1
        ).indices
        incumbent = shortlist[:, 0]
        for local, cache_index in enumerate(current_indices):
            reward = cache["candidate_rewards"][cache_index].float()
            valid = cache["candidate_reward_valid_mask"][cache_index].bool()
            anchor = incumbent[local]
            challenger = shortlist[local, 1:]
            informative = (
                valid[anchor]
                & valid[challenger]
                & torch.isfinite(reward[anchor])
                & torch.isfinite(reward[challenger])
                & (reward[challenger] - reward[anchor]).abs().gt(1e-6)
            )
            challenger = challenger[informative]
            if not challenger.numel():
                continue
            anchor_indices = anchor.expand_as(challenger)
            base_delta = base[local, challenger] - base[local, anchor_indices]
            real_delta = (
                real_history[local, challenger]
                - real_history[local, anchor_indices]
            )
            shuffled_delta = (
                shuffled_history[local, challenger]
                - shuffled_history[local, anchor_indices]
            )
            zero_history = torch.zeros_like(real_delta)
            current_parts.append(torch.cat((base_delta, zero_history), dim=-1))
            real_parts.append(torch.cat((base_delta, real_delta), dim=-1))
            shuffled_parts.append(
                torch.cat((base_delta, shuffled_delta), dim=-1)
            )
            target_parts.append(
                reward[challenger].gt(reward[anchor]).float()
            )
            fold_parts.append(
                torch.full(
                    (challenger.numel(),),
                    protocol.log_fold(logs[local]),
                    dtype=torch.long,
                )
            )
            token_parts.extend([tokens[local]] * challenger.numel())
            log_parts.extend([logs[local]] * challenger.numel())
            stratum_parts.extend([strata[local]] * challenger.numel())
    if not current_parts:
        raise RuntimeError(
            "proposal-history audit produced no informative incumbent comparisons"
        )
    return {
        "current": torch.cat(current_parts),
        "real": torch.cat(real_parts),
        "shuffled": torch.cat(shuffled_parts),
        "target": torch.cat(target_parts),
        "fold": torch.cat(fold_parts),
        "tokens": token_parts,
        "logs": log_parts,
        "strata": stratum_parts,
    }


def fit_probe(features, targets, training, seed, epochs, batch_size, device):
    train = features[training]
    labels = targets[training]
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-4)
    torch.manual_seed(seed)
    model = torch.nn.Linear(features.shape[-1], 1).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-3, foreach=False
    )
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(epochs):
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            x = ((train[indices] - mean) / scale).to(device)
            y = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(x).squeeze(-1), y
            )
            loss.backward()
            optimizer.step()
    outputs = []
    with torch.no_grad():
        standardized = (features - mean) / scale
        for start in range(0, len(features), 8192):
            outputs.append(
                model(standardized[start : start + 8192].to(device))
                .squeeze(-1)
                .cpu()
            )
    return torch.cat(outputs)


def binary_auc(scores, targets):
    scores = scores.double().cpu()
    targets = targets.bool().cpu()
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    starts = counts.cumsum(0) - counts
    average = starts.double() + (counts.double() + 1.0) / 2.0
    ranks = average[inverse]
    positive_rank_sum = ranks[targets[order]].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def bootstrap_log_delta(per_log, left, right, repetitions, seed):
    names = sorted(
        name
        for name, values in per_log.items()
        if values[left] is not None and values[right] is not None
    )
    if not names:
        raise RuntimeError("no logs support paired history bootstrap")
    deltas = np.asarray(
        [per_log[name][left] - per_log[name][right] for name in names],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        stop = min(start + 500, repetitions)
        sampled = rng.integers(0, len(names), size=(stop - start, len(names)))
        estimates[start:stop] = deltas[sampled].mean(axis=1)
    return {
        "mean": float(deltas.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "num_logs": len(names),
        "bootstrap_unit": "paired_log",
        "repetitions": repetitions,
    }


def evaluate_probe_task(examples, args, seed_offset):
    """Fit the three capacity-matched arms under one log-disjoint CV task."""

    fold_rows, per_log = [], {}
    names = ("current", "real", "shuffled")
    all_scores = {
        name: torch.empty(len(examples["target"])) for name in names
    }
    for fold in range(protocol.NUM_FOLDS):
        validation = examples["fold"].eq(fold)
        if not bool(validation.any()) or not bool((~validation).any()):
            raise RuntimeError(f"empty train/validation partition at fold {fold}")
        scores = {}
        for name in names:
            scores[name] = fit_probe(
                examples[name],
                examples["target"],
                ~validation,
                args.seed + seed_offset + fold * 10,
                args.epochs,
                args.batch_size,
                torch.device(args.device),
            )
            all_scores[name][validation] = scores[name][validation]
        fold_auc = {
            name: binary_auc(
                scores[name][validation], examples["target"][validation]
            )
            for name in names
        }
        if any(value is None for value in fold_auc.values()):
            raise RuntimeError(f"single-class validation fold {fold}")
        fold_rows.append(
            {
                "fold": fold,
                "num_examples": int(validation.sum()),
                "positive_fraction": float(
                    examples["target"][validation].mean()
                ),
                **{f"{name}_auc": fold_auc[name] for name in names},
                "real_minus_current_auc": (
                    fold_auc["real"] - fold_auc["current"]
                ),
                "real_minus_shuffled_auc": (
                    fold_auc["real"] - fold_auc["shuffled"]
                ),
            }
        )

    log_to_indices = defaultdict(list)
    for index, log_name in enumerate(examples["logs"]):
        log_to_indices[log_name].append(index)
    for log_name, indices in sorted(log_to_indices.items()):
        index = torch.tensor(indices, dtype=torch.long)
        values = {
            f"{name}_auc": binary_auc(
                all_scores[name][index], examples["target"][index]
            )
            for name in names
        }
        per_log[log_name] = {
            **values,
            "fold": protocol.log_fold(log_name),
            "num_examples": len(indices),
        }
    real_current = bootstrap_log_delta(
        per_log,
        "real_auc",
        "current_auc",
        args.bootstrap_repetitions,
        args.seed + seed_offset + 1000,
    )
    real_shuffled = bootstrap_log_delta(
        per_log,
        "real_auc",
        "shuffled_auc",
        args.bootstrap_repetitions,
        args.seed + seed_offset + 1001,
    )
    auc_by_stratum = {}
    for stratum in ("common", "rare"):
        selected = torch.tensor(
            [value == stratum for value in examples["strata"]],
            dtype=torch.bool,
        )
        values = {
            f"{name}_auc": binary_auc(
                all_scores[name][selected], examples["target"][selected]
            )
            for name in names
        }
        if any(value is None for value in values.values()):
            raise RuntimeError(f"single-class stratum {stratum}")
        auc_by_stratum[stratum] = {
            **values,
            "num_examples": int(selected.sum()),
            "positive_fraction": float(examples["target"][selected].mean()),
            "real_minus_current_auc": (
                values["real_auc"] - values["current_auc"]
            ),
            "real_minus_shuffled_auc": (
                values["real_auc"] - values["shuffled_auc"]
            ),
        }
    return {
        "num_examples": len(examples["target"]),
        "positive_fraction": float(examples["target"].mean()),
        "input_dimension": int(examples["current"].shape[-1]),
        "auc_by_stratum": auc_by_stratum,
        "folds": fold_rows,
        "per_log": per_log,
        "real_over_current_paired_log_bootstrap": real_current,
        "real_over_shuffled_paired_log_bootstrap": real_shuffled,
    }


def main():
    args = parse_args()
    if min(
        args.pairs_per_token,
        args.incumbent_top_k,
        args.epochs,
        args.batch_size,
        args.feature_batch_size,
        args.bootstrap_repetitions,
        args.expected_predecessor_pairs,
    ) <= 0:
        raise ValueError("audit budgets must be positive")
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
    token_to_index = {
        str(token): index for index, token in enumerate(caches[0]["tokens"])
    }
    token_to_log, token_to_stratum = {}, {}
    for row in pair_rows:
        log_name = str(row["log_name"])
        for key, label in (("rare_token", "rare"), ("common_token", "common")):
            token = str(row[key])
            previous_log = token_to_log.setdefault(token, log_name)
            previous_stratum = token_to_stratum.setdefault(token, label)
            if previous_log != log_name:
                raise RuntimeError("one cache token maps to multiple logs")
            if previous_stratum != label:
                raise RuntimeError("one cache token maps to multiple strata")
    expected_tokens = set(token_to_index)
    if set(token_to_log) != expected_tokens:
        raise RuntimeError("pair manifest does not exactly cover cache tokens")
    del rare_indices, common_indices
    annotation_shas = {manifest["annotation_sha256"] for manifest in manifests}
    if len(annotation_shas) != 1:
        raise RuntimeError("cache manifests disagree on annotation SHA256")
    metadata, annotation_path, annotation_sha = load_annotation_metadata(
        args.annotation, set(token_to_index), annotation_shas.pop()
    )
    pairs = immediate_predecessor_pairs(caches[0]["tokens"], metadata)
    current_indices = torch.tensor([row[0] for row in pairs], dtype=torch.long)
    previous_indices = torch.tensor([row[1] for row in pairs], dtype=torch.long)
    tokens = [str(caches[0]["tokens"][index]) for index in current_indices.tolist()]
    logs = [token_to_log[token] for token in tokens]
    strata = [token_to_stratum[token] for token in tokens]
    current_poses = torch.from_numpy(
        np.stack([metadata[token]["ego2global"] for token in tokens])
    ).float()
    previous_tokens = [str(caches[0]["tokens"][index]) for index in previous_indices.tolist()]
    previous_poses = torch.from_numpy(
        np.stack([metadata[token]["ego2global"] for token in previous_tokens])
    ).float()
    time_deltas = np.asarray(
        [
            (metadata[current]["timestamp"] - metadata[previous]["timestamp"])
            / 1e6
            for current, previous in zip(tokens, previous_tokens)
        ],
        dtype=np.float64,
    )
    shuffle = history.same_stratum_cross_log_derangement(
        strata, logs, tokens, seed=args.seed
    )

    v3, v3_payload, v3_path, v3_sha = load_v3(
        args.v3_checkpoint, args.v3_checkpoint_sha256, caches[0], device
    )
    logits = [
        infer_v3_logits(v3, cache, device, args.feature_batch_size)
        for cache in caches
    ]
    base, real_history = build_feature_banks(
        caches,
        current_indices,
        previous_indices,
        current_poses,
        previous_poses,
        logits,
        args.feature_batch_size,
    )
    pairwise_examples = build_pairwise_probe_examples(
        caches,
        base,
        real_history,
        shuffle,
        current_indices,
        tokens,
        logs,
        strata,
        args.pairs_per_token,
    )
    incumbent_examples = build_incumbent_probe_examples(
        caches,
        base,
        real_history,
        logits,
        shuffle,
        current_indices,
        tokens,
        logs,
        strata,
        args.incumbent_top_k,
    )
    pairwise_task = evaluate_probe_task(pairwise_examples, args, 0)
    incumbent_task = evaluate_probe_task(incumbent_examples, args, 10000)
    real_current = incumbent_task[
        "real_over_current_paired_log_bootstrap"
    ]
    real_shuffled = incumbent_task[
        "real_over_shuffled_paired_log_bootstrap"
    ]
    gates = {
        "locked_predecessor_coverage_reproduced": len(pairs) == args.expected_predecessor_pairs,
        "incumbent_real_over_current_mean_auc_at_least_001": real_current["mean"] >= 0.01,
        "incumbent_real_over_current_paired_log_lower_positive": real_current["lower_95"] > 0.0,
        "incumbent_real_over_shuffled_mean_auc_at_least_0001": real_shuffled["mean"] >= 0.001,
        "incumbent_real_over_shuffled_paired_log_lower_positive": real_shuffled["lower_95"] > 0.0,
        "no_fold_incumbent_real_history_drop_below_minus_001": min(
            row["real_minus_current_auc"] for row in incumbent_task["folds"]
        )
        >= -0.01,
        "no_stratum_incumbent_real_history_drop_below_minus_001": min(
            row["real_minus_current_auc"]
            for row in incumbent_task["auc_by_stratum"].values()
        )
        >= -0.01,
    }
    decision = (
        "AUTHORIZE_CAUSAL_PROPOSAL_HISTORY_SELECTOR"
        if all(gates.values())
        else "STOP_PROPOSAL_HISTORY_WITH_CURRENT_DATA"
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable history audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2,
        "status": "PASS",
        "method": "proposal_history_zero_new_label_log_cv_audit_v1",
        "decision": decision,
        "authorized_followup_stage": (
            "extract_full_predecessor_context_and_build_causal_memory_selector"
            if all(gates.values())
            else "retain_v3_and_reconsider_information_source"
        ),
        "gates": gates,
        "pre_registered_thresholds": {
            "expected_predecessor_pairs": args.expected_predecessor_pairs,
            "real_over_current_mean_auc": 0.01,
            "real_over_shuffled_mean_auc": 0.001,
            "paired_log_lower": 0.0,
            "per_fold_real_over_current_floor": -0.01,
            "per_stratum_real_over_current_floor": -0.01,
        },
        "coverage": {
            "cache_tokens": len(caches[0]["tokens"]),
            "tokens_with_immediate_cached_predecessor": len(pairs),
            "fraction": len(pairs) / len(caches[0]["tokens"]),
            "common": sum(value == "common" for value in strata),
            "rare": sum(value == "rare" for value in strata),
            "time_delta_seconds_mean": float(time_deltas.mean()),
            "time_delta_seconds_min": float(time_deltas.min()),
            "time_delta_seconds_max": float(time_deltas.max()),
        },
        "probe": {
            "history_dimension": int(real_history[0].shape[-1]),
            "pairs_per_token_per_noise": args.pairs_per_token,
            "incumbent_top_k": args.incumbent_top_k,
            "epochs": args.epochs,
            "folds": protocol.NUM_FOLDS,
            "fold_assignment": "sha256('20260902:' + log_name) mod 5",
            "arms": {
                "current": "current features plus zero-padded history channels",
                "real": "current features plus real immediate-predecessor proposal history",
                "shuffled": "current features plus same-stratum cross-log shuffled history",
            },
            "primary_task": "challenger_vs_frozen_v3_incumbent_within_v3_top_k",
            "secondary_task": "reward_independent_random_candidate_pair_ranking",
        },
        "incumbent_boundary_task": incumbent_task,
        "random_pairwise_task": pairwise_task,
        "annotation": str(annotation_path),
        "annotation_sha256": annotation_sha,
        "v3_checkpoint": str(v3_path),
        "v3_checkpoint_sha256": v3_sha,
        "v3_method": v3_payload.get("method"),
        "train_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "scientific_contract": {
            "information_audit_only": True,
            "selector_architecture_changed": False,
            "new_annotations_or_training_tokens": False,
            "preceding_outcome_as_input": False,
            "current_official_scalar_pdm_used_only_as_probe_label": True,
            "same_capacity_three_arm_probe": True,
            "same_stratum_cross_log_shuffle_control": True,
            "v3_shortlist_locked_before_reading_reward": True,
            "authorization_uses_incumbent_boundary_not_generic_pair_auc": True,
            "development_consumed": False,
            "certification_consumed": False,
        },
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(history.__file__).resolve()): common.sha256_file(
                Path(history.__file__).resolve()
            ),
            str(Path(protocol.__file__).resolve()): common.sha256_file(
                Path(protocol.__file__).resolve()
            ),
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": "PASS", "decision": decision, "output": str(output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
