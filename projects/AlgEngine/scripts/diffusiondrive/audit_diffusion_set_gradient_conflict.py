#!/usr/bin/env python3
"""Audit the causal cache contract and V3 gradient conflict across draws."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import audit_diffusion_robust_group_mechanism as mechanism
import grpo_selector_v3_cached_common as common
import train_diffusion_robust_group_grpo as trainer
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


INVARIANT_TENSORS = (
    "status_tokens",
    "ego_queries",
    "agents_queries",
    "candidate_reward_valid_mask",
)
STOCHASTIC_TENSORS = (
    "candidate_features",
    "candidate_trajectories_8",
    "route_bev_features",
    "candidate_rewards",
    "reference_logits",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--expected-noise-seeds", default="0,1,2")
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-per-stratum", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def tensor_contract(caches):
    invariant = {}
    stochastic = {}
    for key in INVARIANT_TENSORS:
        comparisons = []
        for draw in range(1, len(caches)):
            comparisons.append(bool(torch.equal(caches[0][key], caches[draw][key])))
        invariant[key] = {
            "exactly_equal_across_draws": all(comparisons),
            "pairwise_equal": comparisons,
        }
    sample_count = min(512, len(caches[0]["tokens"]))
    for key in STOCHASTIC_TENSORS:
        comparisons = []
        left = caches[0][key][:sample_count]
        for draw in range(1, len(caches)):
            right = caches[draw][key][:sample_count]
            if left.dtype == torch.bool:
                fraction = float(left.ne(right).float().mean())
                mean_abs = None
                max_abs = None
            else:
                difference = left.float().sub(right.float()).abs()
                fraction = float(difference.gt(0).float().mean())
                mean_abs = float(difference.mean())
                max_abs = float(difference.max())
            comparisons.append(
                {
                    "draw": draw,
                    "different_fraction": fraction,
                    "mean_abs_difference": mean_abs,
                    "max_abs_difference": max_abs,
                }
            )
        stochastic[key] = {
            "sampled_tokens": sample_count,
            "comparisons": comparisons,
        }
    invariant_pass = all(
        row["exactly_equal_across_draws"] for row in invariant.values()
    )
    trajectories = stochastic["candidate_trajectories_8"]["comparisons"]
    stochastic_pass = all(row["different_fraction"] > 0.99 for row in trajectories)
    return {
        "invariant": invariant,
        "stochastic": stochastic,
        "scene_context_exactly_invariant": invariant_pass,
        "candidate_trajectories_change": stochastic_pass,
    }


def model_gradient(model, parameters, cache, indices, device):
    logits, _ = common.current_logits(model, cache, indices, device)
    rewards = cache["candidate_rewards"][indices].to(
        device=device, dtype=torch.float32
    )
    valid = cache["candidate_reward_valid_mask"][indices].to(
        device=device, dtype=torch.bool
    )
    loss, _, _ = common.exact_group_loss(
        logits,
        logits.detach(),
        rewards,
        valid,
        temperature=1.0,
        kl_weight=0.0,
    )
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return tuple(
        None if gradient is None else gradient.detach()
        for gradient in gradients
    )


def gradient_cosine(left, right):
    dot = None
    left_square = None
    right_square = None
    for left_part, right_part in zip(left, right):
        if left_part is None or right_part is None:
            continue
        part_dot = left_part.float().mul(right_part.float()).sum()
        part_left = left_part.float().square().sum()
        part_right = right_part.float().square().sum()
        dot = part_dot if dot is None else dot + part_dot
        left_square = part_left if left_square is None else left_square + part_left
        right_square = part_right if right_square is None else right_square + part_right
    if dot is None:
        raise RuntimeError("selector produced no trainable gradients")
    denominator = left_square.sqrt().mul(right_square.sqrt()).clamp_min(1e-12)
    return float(dot.div(denominator).cpu())


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError("gradient audit produced invalid cosine values")
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "negative_fraction": float((values < 0.0).mean()),
    }


def sample_indices(pool, count, generator):
    if count > len(pool):
        raise ValueError("gradient audit requests more tokens than available")
    choices = torch.randperm(len(pool), generator=generator)[:count]
    return pool[choices]


def audit_gradients(
    model,
    caches,
    rare_indices,
    common_indices,
    tokens_per_stratum,
    seed,
    device,
):
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("V3 selector exposes no trainable parameters")
    rare_pool = torch.as_tensor(rare_indices, dtype=torch.long)
    common_pool = torch.as_tensor(common_indices, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.cat(
        (
            sample_indices(rare_pool, tokens_per_stratum, generator),
            sample_indices(common_pool, tokens_per_stratum, generator),
        )
    )
    labels = torch.cat(
        (
            torch.ones(tokens_per_stratum, dtype=torch.long),
            torch.zeros(tokens_per_stratum, dtype=torch.long),
        )
    )
    order = torch.randperm(len(indices), generator=generator)
    indices = indices[order]
    labels = labels[order]
    shuffled_indices = [indices]
    for _ in range(1, len(caches)):
        shuffled_indices.append(
            trainer.stratified_shuffle(indices, labels, generator)
        )

    true_cosines = []
    shuffled_cosines = []
    for position, index in enumerate(indices):
        token_index = index.reshape(1)
        true_gradients = [
            model_gradient(model, parameters, cache, token_index, device)
            for cache in caches
        ]
        true_cosines.extend(
            gradient_cosine(true_gradients[left], true_gradients[right])
            for left, right in combinations(range(len(caches)), 2)
        )

        shuffled_gradients = [true_gradients[0]]
        for draw in range(1, len(caches)):
            donor_index = shuffled_indices[draw][position].reshape(1)
            shuffled_gradients.append(
                model_gradient(
                    model,
                    parameters,
                    caches[draw],
                    donor_index,
                    device,
                )
            )
        shuffled_cosines.extend(
            gradient_cosine(shuffled_gradients[left], shuffled_gradients[right])
            for left, right in combinations(range(len(caches)), 2)
        )
    true_summary = summarize(true_cosines)
    shuffled_summary = summarize(shuffled_cosines)
    return {
        "true_same_token": true_summary,
        "same_stratum_shuffled": shuffled_summary,
        "true_minus_shuffled_mean": (
            true_summary["mean"] - shuffled_summary["mean"]
        ),
        "tokens_per_stratum": tokens_per_stratum,
        "num_tokens": int(len(indices)),
        "pairwise_cosines_per_condition": len(true_cosines),
    }


def main():
    args = parse_args()
    if args.tokens_per_stratum < 2:
        raise ValueError("gradient audit budgets are invalid")
    expected_seeds = tuple(
        int(value) for value in args.expected_noise_seeds.split(",")
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    loaded = [common.load_cache(path, "train") for path in args.cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    standard.assert_cache_group(caches, manifests, "train", expected_seeds)
    pair_rows, rare_audit, pair_path, audit_path = rare_trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    rare_indices, common_indices = rare_trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )
    cache_contract = tensor_contract(caches)
    model, _, checkpoint_path, checkpoint_sha = mechanism.load_checkpoint(
        args.v3_checkpoint,
        args.v3_checkpoint_sha256,
        caches[0],
        device,
    )
    model.eval()
    gradient_contract = audit_gradients(
        model,
        caches,
        rare_indices,
        common_indices,
        args.tokens_per_stratum,
        args.seed,
        device,
    )
    passed = (
        cache_contract["scene_context_exactly_invariant"]
        and cache_contract["candidate_trajectories_change"]
    )
    if not passed:
        raise RuntimeError("diffusion-set causal cache contract failed")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "decision": "PASS_DIFFUSION_SET_TRAINING_MECHANISM_AUDIT",
        "method": "diffusion_set_gradient_conflict_audit_v1",
        "cache_contract": cache_contract,
        "gradient_contract": gradient_contract,
        "noise_seeds": list(expected_seeds),
        "cache_manifests": list(manifests),
        "v3_checkpoint": str(checkpoint_path),
        "v3_checkpoint_sha256": checkpoint_sha,
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "scientific_contract": {
            "selector_only": True,
            "official_scalar_pdm_only": True,
            "reward_components_consumed": False,
            "development_consumed": False,
            "certification_consumed": False,
        },
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(trainer.__file__).resolve()): common.sha256_file(
                Path(trainer.__file__).resolve()
            ),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "decision": payload["decision"],
                "true_gradient_cosine": gradient_contract["true_same_token"]["mean"],
                "shuffled_gradient_cosine": gradient_contract[
                    "same_stratum_shuffled"
                ]["mean"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
