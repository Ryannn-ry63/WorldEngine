#!/usr/bin/env python3
"""Probe whether frozen inference features predict common versus real-rare."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from build_grpo_selector_v3_rare_rollout_data import parse_seed_path
import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached_rare_rollout as trainer


def average_ranks(values):
    values = values.detach().cpu().to(torch.float64)
    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def roc_auc(scores, targets):
    targets = targets.detach().cpu().bool()
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        return None
    ranks = average_ranks(scores)
    return float(
        (
            ranks[targets].sum() - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def average_precision(scores, targets):
    targets = targets.detach().cpu().bool()
    positives = int(targets.sum())
    if not positives:
        return None
    order = torch.argsort(scores.detach().cpu(), descending=True, stable=True)
    ordered = targets[order].to(torch.float64)
    precision = ordered.cumsum(0) / torch.arange(
        1, len(ordered) + 1, dtype=torch.float64
    )
    return float(precision[ordered.bool()].mean())


def probe_metrics(logits, targets):
    logits = logits.detach().cpu()
    targets = targets.detach().cpu().float()
    probability = logits.sigmoid()
    prediction = logits.gt(0.0)
    positive = targets.bool()
    negative = ~positive
    true_positive_rate = (
        float(prediction[positive].float().mean()) if positive.any() else 0.0
    )
    true_negative_rate = (
        float((~prediction[negative]).float().mean()) if negative.any() else 0.0
    )
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        mask = probability.ge(lower) & (
            probability.le(upper) if index == 9 else probability.lt(upper)
        )
        if mask.any():
            ece += float(mask.float().mean()) * abs(
                float(probability[mask].mean()) - float(targets[mask].mean())
            )
    return {
        "examples": len(targets),
        "positive_fraction": float(targets.mean()),
        "roc_auc": roc_auc(logits, targets),
        "pr_auc": average_precision(logits, targets),
        "balanced_accuracy": 0.5 * (true_positive_rate + true_negative_rate),
        "brier": float((probability - targets).square().mean()),
        "ece_10": ece,
    }


def pair_feature_rows(split_rows):
    real_rows = [
        (index, row)
        for index, row in enumerate(split_rows)
        if row["hard_kind"] == "real_rare"
    ]
    specs = []
    labels = []
    for index, _ in real_rows:
        specs.extend((("common", index), ("hard", index)))
        labels.extend((0.0, 1.0))
    return specs, torch.tensor(labels, dtype=torch.float32)


@torch.no_grad()
def extract_features(
    model,
    row_specs,
    hard_rows,
    real_cache,
    token_map,
    synthetic,
    device,
    batch_size,
):
    features = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(row_specs), batch_size):
            batch_rows = row_specs[start : start + batch_size]
            inputs, reference, _, _, valid, _ = trainer.mixed_batch(
                batch_rows,
                hard_rows,
                real_cache,
                token_map,
                common_cache=None,
                common_token_map=None,
                common_tokens=None,
                synthetic=synthetic,
                device=device,
            )
            tokens = model.encode_tokens(**inputs)
            (
                proposal_delta,
                _,
                mask,
                incumbent,
                proposal,
            ) = model.proposal_and_arbitration(
                tokens,
                inputs["candidate_trajectories"],
                reference,
                valid,
            )
            batch_index = torch.arange(len(tokens), device=device)
            incumbent_token = tokens[batch_index, incumbent]
            proposal_token = tokens[batch_index, proposal]
            mask_float = mask[:, :, None].to(tokens.dtype)
            mean_token = (tokens * mask_float).sum(1) / mask_float.sum(1).clamp_min(1.0)
            max_token = tokens.masked_fill(~mask[:, :, None], -torch.inf).max(1).values
            reference_margin, reference_entropy = model._score_context(
                reference, mask
            )
            proposal_logits = reference + proposal_delta
            proposal_margin, proposal_entropy = model._score_context(
                proposal_logits, mask
            )
            reference_gap = (
                reference.gather(1, proposal[:, None])
                - reference.gather(1, incumbent[:, None])
            ).squeeze(1)
            proposal_gap = (
                proposal_logits.gather(1, proposal[:, None])
                - proposal_logits.gather(1, incumbent[:, None])
            ).squeeze(1)
            scalars = torch.stack(
                (
                    reference_margin,
                    proposal_margin,
                    reference_entropy,
                    proposal_entropy,
                    reference_gap,
                    proposal_gap,
                ),
                dim=-1,
            )
            features.append(
                torch.cat(
                    (
                        mean_token,
                        max_token,
                        incumbent_token,
                        proposal_token,
                        proposal_token - incumbent_token,
                        scalars,
                    ),
                    dim=-1,
                ).cpu()
            )
    finally:
        model.train(was_training)
    return torch.cat(features)


class RegimeProbe(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        if hidden_dim:
            self.network = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.network = nn.Linear(feature_dim, 1)

    def forward(self, features):
        return self.network(features).squeeze(-1)


def train_probe(
    name,
    train_features,
    train_targets,
    development_features,
    development_targets,
    hidden_dim,
    device,
    epochs,
    seed,
):
    torch.manual_seed(seed)
    model = RegimeProbe(train_features.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4, foreach=False
    )
    generator = torch.Generator().manual_seed(seed + 17000)
    batch_size = 256
    for _ in range(epochs):
        order = torch.randperm(len(train_targets), generator=generator)
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            features = train_features[indices].to(device)
            targets = train_targets[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(features), targets)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        train_logits = []
        for start in range(0, len(train_targets), batch_size):
            train_logits.append(
                model(train_features[start : start + batch_size].to(device)).cpu()
            )
        development_logits = []
        for start in range(0, len(development_targets), batch_size):
            development_logits.append(
                model(
                    development_features[start : start + batch_size].to(device)
                ).cpu()
            )
    return {
        "name": name,
        "hidden_dim": hidden_dim,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "train_metrics": probe_metrics(torch.cat(train_logits), train_targets),
        "development_metrics": probe_metrics(
            torch.cat(development_logits), development_targets
        ),
        "model": model,
    }


def synthetic_score_summary(model, features, device):
    if not len(features):
        return None
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 256):
            scores.append(model(features[start : start + 256].to(device)).sigmoid().cpu())
    scores = torch.cat(scores)
    return {
        "examples": len(scores),
        "mean_rare_probability": float(scores.mean()),
        "std_rare_probability": float(scores.std(unbiased=False)),
        "q10": float(scores.quantile(0.10)),
        "q50": float(scores.quantile(0.50)),
        "q90": float(scores.quantile(0.90)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-cache", action="append", type=parse_seed_path, required=True
    )
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-limit-pairs", type=int)
    args = parser.parse_args()
    if min(args.batch_size, args.probe_epochs) <= 0:
        raise ValueError("invalid E0 budget")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    (
        manifest_path,
        _,
        hard_pool_path,
        all_rows,
        real_caches,
        real_manifests,
        token_maps,
        synthetic_path,
        synthetic,
    ) = trainer.load_contract(args)
    rows_by_split = {
        split: [row for row in all_rows if row["split"] == split]
        for split in ("train", "development")
    }
    if args.smoke_limit_pairs is not None:
        if args.smoke_limit_pairs < 2:
            raise ValueError("--smoke-limit-pairs must be at least two")
        for split in rows_by_split:
            real_rows = [
                row for row in rows_by_split[split]
                if row["hard_kind"] == "real_rare"
            ][: args.smoke_limit_pairs]
            synthetic_rows = [
                row for row in rows_by_split[split]
                if row["hard_kind"] == "synthetic_rollout"
            ][: args.smoke_limit_pairs]
            rows_by_split[split] = real_rows + synthetic_rows

    model, model_config = common.model_from_cache(
        real_caches[0],
        "relational_only",
        "proposal_conditioned_regret_arbitration",
        {
            "num_temporal_layers": 2,
            "num_relation_layers": 2,
            "arbiter_hidden_dim": 128,
            "use_decision_context": False,
            "override_threshold": 0.0,
        },
    )
    proposal_path, proposal_sha, proposal_method, _ = trainer.load_and_freeze_proposal(
        model, args.proposal_checkpoint
    )
    model = model.to(device).eval()

    split_features = {}
    split_targets = {}
    for split in ("train", "development"):
        specs, targets = pair_feature_rows(rows_by_split[split])
        split_features[split] = extract_features(
            model,
            specs,
            rows_by_split[split],
            real_caches[0],
            token_maps[0],
            synthetic,
            device,
            args.batch_size,
        )
        split_targets[split] = targets

    mean = split_features["train"].mean(0)
    std = split_features["train"].std(0, unbiased=False).clamp_min(1e-6)
    for split in split_features:
        split_features[split] = (split_features[split] - mean) / std

    probe_rows = []
    for name, hidden_dim in (("linear", 0), ("mlp", 128)):
        probe_rows.append(
            train_probe(
                name,
                split_features["train"],
                split_targets["train"],
                split_features["development"],
                split_targets["development"],
                hidden_dim,
                device,
                args.probe_epochs,
                args.seed + hidden_dim,
            )
        )

    synthetic_specs = [
        ("hard", index)
        for index, row in enumerate(rows_by_split["development"])
        if row["hard_kind"] == "synthetic_rollout"
    ]
    synthetic_features = (
        extract_features(
            model,
            synthetic_specs,
            rows_by_split["development"],
            real_caches[0],
            token_maps[0],
            synthetic,
            device,
            args.batch_size,
        )
        if synthetic_specs
        else torch.empty((0, len(mean)))
    )
    if len(synthetic_features):
        synthetic_features = (synthetic_features - mean) / std

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "regime_probes.pt"
    torch.save(
        {
            "schema_version": 1,
            "method": "cpv_e0_frozen_feature_regime_probe",
            "feature_mean": mean,
            "feature_std": std,
            "scene_selector_config": model_config,
            "proposal_checkpoint": str(proposal_path),
            "proposal_checkpoint_sha256": proposal_sha,
            "probes": {
                row["name"]: {
                    "hidden_dim": row["hidden_dim"],
                    "state_dict": row["state_dict"],
                }
                for row in probe_rows
            },
        },
        checkpoint,
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "cpv_e0_frozen_feature_regime_probe",
        "promotable": False,
        "labels_are_model_inputs": False,
        "reward_consumed": False,
        "feature_noise_seed": 0,
        "train_split": "train",
        "evaluation_split": "development",
        "proposal_checkpoint": str(proposal_path),
        "proposal_checkpoint_sha256": proposal_sha,
        "proposal_method": proposal_method,
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(hard_pool_path),
        "hard_pool_sha256": common.sha256_file(hard_pool_path),
        "synthetic_cache": str(synthetic_path),
        "synthetic_cache_sha256": common.sha256_file(synthetic_path),
        "real_cache_manifests": list(real_manifests),
        "feature_dim": int(len(mean)),
        "probe_epochs": args.probe_epochs,
        "train_pairs": int(len(split_targets["train"]) // 2),
        "development_pairs": int(len(split_targets["development"]) // 2),
        "probes": [
            {
                "name": row["name"],
                "hidden_dim": row["hidden_dim"],
                "train_metrics": row["train_metrics"],
                "development_metrics": row["development_metrics"],
                "synthetic_development_scores": synthetic_score_summary(
                    row["model"], synthetic_features, device
                ),
            }
            for row in probe_rows
        ],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": common.sha256_file(checkpoint),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()

