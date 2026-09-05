#!/usr/bin/env python3
"""Audit whether same-scene diffusion draws justify hierarchical selector GRPO.

The audit consumes only frozen selector logits, validity masks, and the official
scalar PDM reward.  Reward components are deliberately never indexed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import grpo_selector_v3_cached_common as common
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--expected-noise-seeds", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sign-epsilon", type=float, default=1e-6)
    parser.add_argument("--headroom-margin", type=float, default=0.01)
    parser.add_argument("--permutation-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def correlation_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    return [
        [safe_correlation(values[:, row], values[:, column]) for column in range(values.shape[1])]
        for row in range(values.shape[1])
    ]


def binary_auc(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = float(ranks[targets].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def grouping_control(values, repetitions, seed):
    """Compare true same-token draw dispersion with shuffled outer groups."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("grouping control requires a token-by-draw matrix")
    true_std = float(values.std(axis=1).mean())
    rng = np.random.default_rng(seed)
    shuffled = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        permuted = values.copy()
        for draw in range(1, values.shape[1]):
            permuted[:, draw] = values[rng.permutation(len(values)), draw]
        shuffled[repetition] = permuted.std(axis=1).mean()
    return {
        "true_same_token_mean_std": true_std,
        "shuffled_mean_std": float(shuffled.mean()),
        "shuffled_lower_95": float(np.quantile(shuffled, 0.025)),
        "shuffled_upper_95": float(np.quantile(shuffled, 0.975)),
        "shuffled_minus_true": float(shuffled.mean() - true_std),
        "one_sided_permutation_p": float(
            (1 + int((shuffled <= true_std).sum())) / (repetitions + 1)
        ),
        "repetitions": int(repetitions),
    }


def load_checkpoint(path, expected_sha, cache, device):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = common.sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"V3 checkpoint SHA256 drifted: actual={actual_sha} expected={expected_sha}"
        )
    payload = torch.load(path, map_location="cpu")
    config = payload.get("scene_selector_config")
    if config:
        model = common.model_from_config(config)
    else:
        model, config = common.model_from_cache(cache, payload.get("ablation", "full"))
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    model.to(device).eval()
    return model, payload, path, actual_sha


def evaluate_scalar_only(model, cache, device, temperature, batch_size):
    output = {
        "hard_gain": [],
        "expected_gain": [],
        "current_reward": [],
        "reference_reward": [],
        "oracle_reward": [],
        "oracle_headroom": [],
        "reference_oracle_headroom": [],
        "selection_changed": [],
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            indices = slice(start, stop)
            logits, reference = common.current_logits(model, cache, indices, device)
            rewards = cache["candidate_rewards"][indices].to(
                device=device, dtype=torch.float32
            )
            valid = cache["candidate_reward_valid_mask"][indices].to(
                device=device, dtype=torch.bool
            )
            valid = valid & torch.isfinite(rewards)
            if not bool(valid.any(dim=-1).all()):
                raise RuntimeError("every audited draw needs a valid scalar reward")
            current_masked = (logits / temperature).masked_fill(~valid, -1e4)
            reference_masked = (reference / temperature).masked_fill(~valid, -1e4)
            current_probability = F.softmax(current_masked, dim=-1)
            reference_probability = F.softmax(reference_masked, dim=-1)
            safe_rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
            current_index = current_masked.argmax(dim=-1)
            reference_index = reference_masked.argmax(dim=-1)
            oracle_index = rewards.masked_fill(~valid, -torch.inf).argmax(dim=-1)

            def gather(index):
                return rewards.gather(1, index[:, None]).squeeze(1)

            current_reward = gather(current_index)
            reference_reward = gather(reference_index)
            oracle_reward = gather(oracle_index)
            values = {
                "hard_gain": current_reward - reference_reward,
                "expected_gain": (
                    (current_probability - reference_probability) * safe_rewards
                ).sum(dim=-1),
                "current_reward": current_reward,
                "reference_reward": reference_reward,
                "oracle_reward": oracle_reward,
                "oracle_headroom": oracle_reward - current_reward,
                "reference_oracle_headroom": oracle_reward - reference_reward,
                "selection_changed": current_index.ne(reference_index).float(),
            }
            for key, value in values.items():
                output[key].append(value.cpu())
    return {key: torch.cat(parts) for key, parts in output.items()}


def stack_draws(values_by_draw):
    return {
        key: torch.stack([values[key] for values in values_by_draw], dim=1)
        for key in values_by_draw[0]
    }


def summarize_subset(stacked, indices, epsilon, headroom_margin):
    arrays = {
        key: value[indices].double().numpy() for key, value in stacked.items()
    }
    hard = arrays["hard_gain"]
    expected = arrays["expected_gain"]
    headroom = arrays["oracle_headroom"]
    positive = hard > epsilon
    negative = hard < -epsilon
    mixed = positive.any(axis=1) & negative.any(axis=1)
    any_negative = negative.any(axis=1)
    hard_expected_opposite = (
        ((hard > epsilon) & (expected < -epsilon))
        | ((hard < -epsilon) & (expected > epsilon))
    )
    per_token_std = hard.std(axis=1)
    low_threshold, high_threshold = np.quantile(per_token_std, [0.25, 0.75])
    low = per_token_std <= low_threshold
    high = per_token_std >= high_threshold
    negative_views = negative
    nonnegative_views = ~negative

    def conditional_mean(values, mask):
        return float(values[mask].mean()) if mask.any() else None

    result = {
        "num_tokens": int(len(indices)),
        "num_draws": int(hard.shape[1]),
        "mean_hard_gain": float(hard.mean()),
        "mean_expected_gain": float(expected.mean()),
        "mean_min_hard_gain": float(hard.min(axis=1).mean()),
        "mean_max_hard_gain": float(hard.max(axis=1).mean()),
        "mean_cross_draw_hard_gain_std": float(per_token_std.mean()),
        "negative_view_fraction": float(negative.mean()),
        "positive_view_fraction": float(positive.mean()),
        "mixed_sign_token_fraction": float(mixed.mean()),
        "any_negative_token_fraction": float(any_negative.mean()),
        "all_nonnegative_token_fraction": float((~negative.any(axis=1)).mean()),
        "hard_expected_opposite_sign_fraction": float(hard_expected_opposite.mean()),
        "hard_expected_gain_correlation": safe_correlation(hard.ravel(), expected.ravel()),
        "mean_absolute_hard_expected_gap": float(np.abs(hard - expected).mean()),
        "selection_change_fraction": float(arrays["selection_changed"].mean()),
        "mean_current_reward": float(arrays["current_reward"].mean()),
        "mean_reference_reward": float(arrays["reference_reward"].mean()),
        "mean_oracle_reward": float(arrays["oracle_reward"].mean()),
        "mean_oracle_headroom": float(headroom.mean()),
        "mean_min_oracle_headroom": float(headroom.min(axis=1).mean()),
        "all_draws_oracle_headroom_above_margin_fraction": float(
            (headroom > headroom_margin).all(axis=1).mean()
        ),
        "negative_views_mean_oracle_headroom": conditional_mean(
            headroom, negative_views
        ),
        "nonnegative_views_mean_oracle_headroom": conditional_mean(
            headroom, nonnegative_views
        ),
        "mixed_tokens_mean_oracle_headroom": conditional_mean(
            headroom, np.broadcast_to(mixed[:, None], headroom.shape)
        ),
        "stable_tokens_mean_oracle_headroom": conditional_mean(
            headroom, np.broadcast_to((~mixed)[:, None], headroom.shape)
        ),
        "cross_draw_hard_gain_correlation": correlation_matrix(hard),
        "cross_draw_reference_reward_correlation": correlation_matrix(
            arrays["reference_reward"]
        ),
        "cross_draw_oracle_headroom_correlation": correlation_matrix(headroom),
        "hard_gain_std_auc_for_any_negative": binary_auc(
            per_token_std, any_negative
        ),
        "high_variability_negative_token_fraction": float(any_negative[high].mean()),
        "low_variability_negative_token_fraction": float(any_negative[low].mean()),
        "variability_quartile_thresholds": {
            "low_25": float(low_threshold),
            "high_75": float(high_threshold),
        },
    }
    return result


def main():
    args = parse_args()
    if args.temperature <= 0.0 or args.batch_size <= 0:
        raise ValueError("temperature and batch size must be positive")
    if args.sign_epsilon < 0.0 or args.headroom_margin < 0.0:
        raise ValueError("audit margins must be non-negative")
    if args.permutation_repetitions <= 0:
        raise ValueError("permutation repetitions must be positive")
    expected_seeds = tuple(
        int(value) for value in args.expected_noise_seeds.split(",")
    )
    if len(expected_seeds) < 2 or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected noise seeds must contain distinct draws")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    loaded = [common.load_cache(path, args.split) for path in args.cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    standard.assert_cache_group(caches, manifests, args.split, expected_seeds)
    pair_rows, rare_audit, pair_path, audit_path = rare_trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    rare_indices, common_indices = rare_trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )
    rare_indices = np.asarray(sorted(set(rare_indices)), dtype=np.int64)
    common_indices = np.asarray(sorted(set(common_indices)), dtype=np.int64)
    if np.intersect1d(rare_indices, common_indices).size:
        raise RuntimeError("rare/common audit strata overlap")
    all_indices = np.asarray(
        sorted(set(rare_indices.tolist()) | set(common_indices.tolist())),
        dtype=np.int64,
    )
    if len(all_indices) != len(caches[0]["tokens"]):
        raise RuntimeError("rare/common strata do not cover the aligned cache")

    model, checkpoint, checkpoint_path, checkpoint_sha = load_checkpoint(
        args.v3_checkpoint,
        args.v3_checkpoint_sha256,
        caches[0],
        device,
    )
    values_by_draw = [
        evaluate_scalar_only(
            model, cache, device, args.temperature, args.batch_size
        )
        for cache in caches
    ]
    stacked = stack_draws(values_by_draw)
    strata = {
        "all": summarize_subset(
            stacked,
            all_indices,
            args.sign_epsilon,
            args.headroom_margin,
        ),
        "rare": summarize_subset(
            stacked,
            rare_indices,
            args.sign_epsilon,
            args.headroom_margin,
        ),
        "common": summarize_subset(
            stacked,
            common_indices,
            args.sign_epsilon,
            args.headroom_margin,
        ),
    }

    controls = {}
    for offset, key in enumerate(
        ("hard_gain", "expected_gain", "current_reward", "reference_reward", "oracle_headroom")
    ):
        controls[key] = grouping_control(
            stacked[key][all_indices].double().numpy(),
            args.permutation_repetitions,
            args.seed + offset,
        )

    overall = strata["all"]
    gates = {
        "cross_draw_instability_present": (
            overall["mixed_sign_token_fraction"] >= 0.10
            or overall["any_negative_token_fraction"] >= 0.25
        ),
        "hard_soft_train_deploy_gap_present": (
            overall["hard_expected_opposite_sign_fraction"] >= 0.05
            or (
                overall["hard_expected_gain_correlation"] is not None
                and overall["hard_expected_gain_correlation"] < 0.80
            )
        ),
        "robust_per_draw_oracle_headroom_present": (
            overall["mean_min_oracle_headroom"] >= args.headroom_margin
            and overall[
                "all_draws_oracle_headroom_above_margin_fraction"
            ]
            >= 0.25
        ),
        "same_token_alignment_positive_control": (
            controls["reference_reward"]["shuffled_minus_true"] > 0.0
            and controls["reference_reward"]["one_sided_permutation_p"] < 0.05
        ),
    }
    core = (
        gates["cross_draw_instability_present"]
        and gates["robust_per_draw_oracle_headroom_present"]
        and gates["same_token_alignment_positive_control"]
    )
    if core and gates["hard_soft_train_deploy_gap_present"]:
        decision = "AUTHORIZE_TOP1_DIFFUSION_ROBUST_GROUP_PROTOTYPE"
    elif core:
        decision = "AUTHORIZE_SOFT_DIFFUSION_ROBUST_GROUP_PROTOTYPE"
    else:
        decision = "STOP_DIFFUSION_ROBUST_GROUP_HYPOTHESIS"

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusion_robust_group_mechanism_audit_v1",
        "decision": decision,
        "gates": gates,
        "gate_scope": (
            "mechanism authorization only; this audit cannot establish method efficacy"
        ),
        "scientific_contract": {
            "selector_only": True,
            "generator_frozen": True,
            "official_scalar_pdm_only": True,
            "reward_components_consumed": False,
            "new_training_data_consumed": False,
            "same_scene_outer_group": True,
            "development_or_certification_consumed": args.split != "train",
            "certification_consumed": False,
        },
        "split": args.split,
        "noise_seeds": list(expected_seeds),
        "temperature": args.temperature,
        "sign_epsilon": args.sign_epsilon,
        "headroom_margin": args.headroom_margin,
        "strata": strata,
        "same_token_vs_shuffled_controls": controls,
        "v3_checkpoint": str(checkpoint_path),
        "v3_checkpoint_sha256": checkpoint_sha,
        "v3_checkpoint_method": checkpoint.get("method"),
        "cache_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(common.__file__).resolve()): common.sha256_file(
                Path(common.__file__).resolve()
            ),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": "PASS", "decision": decision, "output": str(output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
