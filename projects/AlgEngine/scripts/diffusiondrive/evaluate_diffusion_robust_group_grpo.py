#!/usr/bin/env python3
"""Evaluate a diffusion-robust selector against its frozen V3 anchor."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import audit_diffusion_robust_group_mechanism as mechanism
import grpo_selector_v3_cached_common as common
import train_diffusion_robust_group_grpo as trainer
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


EXPECTED_SEEDS = {"train": (0, 1, 2), "development": (3, 4, 5)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED_SEEDS), required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sign-epsilon", type=float, default=1e-6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_candidate(path, expected_sha, cache, device):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = common.sha256_file(path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise RuntimeError("candidate checkpoint SHA256 drifted")
    payload = torch.load(path, map_location="cpu")
    if (
        payload.get("method") != "diffusion_set_selector_posttraining_v1"
        or "scene_selector_state" not in payload
        or "scene_selector_config" not in payload
    ):
        raise RuntimeError("invalid diffusion-robust selector checkpoint")
    model = common.model_from_config(payload["scene_selector_config"])
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    model.to(device).eval()
    anchor, anchor_payload, anchor_path, anchor_sha = mechanism.load_checkpoint(
        Path(payload["v3_anchor"]),
        payload["v3_anchor_sha256"],
        cache,
        device,
    )
    return (
        model,
        payload,
        path,
        actual_sha,
        anchor,
        anchor_payload,
        anchor_path,
        anchor_sha,
    )


def evaluate_pair(candidate, anchor, cache, device, temperature, batch_size):
    output = defaultdict(list)
    candidate.eval()
    anchor.eval()
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            indices = slice(start, stop)
            current_logits, _ = common.current_logits(
                candidate, cache, indices, device
            )
            anchor_logits, _ = common.current_logits(anchor, cache, indices, device)
            rewards = cache["candidate_rewards"][indices].to(
                device=device, dtype=torch.float32
            )
            valid = cache["candidate_reward_valid_mask"][indices].to(
                device=device, dtype=torch.bool
            )
            valid = valid & torch.isfinite(rewards)
            current_masked = (current_logits / temperature).masked_fill(
                ~valid, -1e4
            )
            anchor_masked = (anchor_logits / temperature).masked_fill(
                ~valid, -1e4
            )
            current_probability = F.softmax(current_masked, dim=-1)
            anchor_probability = F.softmax(anchor_masked, dim=-1)
            safe_rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
            current_index = current_masked.argmax(dim=-1)
            anchor_index = anchor_masked.argmax(dim=-1)
            oracle_index = rewards.masked_fill(~valid, -torch.inf).argmax(dim=-1)

            def gather(index):
                return rewards.gather(1, index[:, None]).squeeze(1)

            current_reward = gather(current_index)
            anchor_reward = gather(anchor_index)
            oracle_reward = gather(oracle_index)
            values = {
                "hard_gain": current_reward - anchor_reward,
                "expected_gain": (
                    (current_probability - anchor_probability) * safe_rewards
                ).sum(dim=-1),
                "current_reward": current_reward,
                "anchor_reward": anchor_reward,
                "oracle_reward": oracle_reward,
                "oracle_headroom": oracle_reward - current_reward,
                "anchor_oracle_headroom": oracle_reward - anchor_reward,
                "selection_changed": current_index.ne(anchor_index).float(),
            }
            for key, value in values.items():
                output[key].append(value.cpu())
    return {key: torch.cat(parts) for key, parts in output.items()}


def bootstrap_log_gain(gains, logs, repetitions, seed):
    by_log = defaultdict(list)
    for index, log_name in enumerate(logs):
        by_log[str(log_name)].extend(gains[index].tolist())
    names = sorted(by_log)
    values = np.asarray(
        [np.mean(by_log[name], dtype=np.float64) for name in names],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(values), size=(repetitions, len(values)))
    estimates = values[samples].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "log",
        "num_logs": len(names),
        "repetitions": repetitions,
    }


def summarize(stacked, indices, logs, seeds, epsilon, repetitions, seed):
    arrays = {
        key: value[indices].double().numpy() for key, value in stacked.items()
    }
    hard = arrays["hard_gain"]
    expected = arrays["expected_gain"]
    positive = hard > epsilon
    negative = hard < -epsilon
    mixed = positive.any(axis=1) & negative.any(axis=1)
    per_seed = {}
    for draw, noise_seed in enumerate(seeds):
        per_seed[str(noise_seed)] = {
            "hard_gain": float(hard[:, draw].mean()),
            "expected_gain": float(expected[:, draw].mean()),
            "current_reward": float(arrays["current_reward"][:, draw].mean()),
            "anchor_reward": float(arrays["anchor_reward"][:, draw].mean()),
            "negative_fraction": float(negative[:, draw].mean()),
        }
    return {
        "num_tokens": int(len(indices)),
        "mean_current_reward": float(arrays["current_reward"].mean()),
        "mean_anchor_reward": float(arrays["anchor_reward"].mean()),
        "mean_hard_gain": float(hard.mean()),
        "mean_expected_gain": float(expected.mean()),
        "mean_min_hard_gain": float(hard.min(axis=1).mean()),
        "mean_max_hard_gain": float(hard.max(axis=1).mean()),
        "mean_cross_draw_gain_std": float(hard.std(axis=1).mean()),
        "negative_view_fraction": float(negative.mean()),
        "any_negative_token_fraction": float(negative.any(axis=1).mean()),
        "mixed_sign_token_fraction": float(mixed.mean()),
        "selection_change_fraction": float(arrays["selection_changed"].mean()),
        "mean_oracle_headroom": float(arrays["oracle_headroom"].mean()),
        "mean_anchor_oracle_headroom": float(
            arrays["anchor_oracle_headroom"].mean()
        ),
        "hard_expected_opposite_sign_fraction": float(
            (
                ((hard > epsilon) & (expected < -epsilon))
                | ((hard < -epsilon) & (expected > epsilon))
            ).mean()
        ),
        "hard_expected_gain_correlation": mechanism.safe_correlation(
            hard.ravel(), expected.ravel()
        ),
        "by_noise_seed": per_seed,
        "log_bootstrap_hard_gain": bootstrap_log_gain(
            hard,
            [logs[index] for index in indices],
            repetitions,
            seed,
        ),
    }


def write_records(path, stacked, cache, logs, seeds, rare_indices):
    rare_indices = set(int(value) for value in rare_indices)
    keys = (
        "hard_gain",
        "expected_gain",
        "current_reward",
        "anchor_reward",
        "oracle_reward",
        "oracle_headroom",
        "selection_changed",
    )
    with path.open("w") as stream:
        for index, (token, scene) in enumerate(
            zip(cache["tokens"], cache["scenes"])
        ):
            stratum = "rare" if index in rare_indices else "common"
            for draw, noise_seed in enumerate(seeds):
                row = {
                    "token": str(token),
                    "scene": str(scene),
                    "log": str(logs[index]),
                    "stratum": stratum,
                    "noise_seed": int(noise_seed),
                }
                for key in keys:
                    row[key] = float(stacked[key][index, draw])
                stream.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_repetitions <= 0:
        raise ValueError("evaluation budgets must be positive")
    if args.sign_epsilon < 0.0:
        raise ValueError("sign epsilon must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    loaded = [common.load_cache(path, args.split) for path in args.cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    seeds = EXPECTED_SEEDS[args.split]
    standard.assert_cache_group(caches, manifests, args.split, seeds)
    pair_rows, rare_audit, pair_path, audit_path = rare_trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    rare_indices, common_indices = rare_trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, rare_audit
    )
    rare_indices = np.asarray(sorted(set(rare_indices)), dtype=np.int64)
    common_indices = np.asarray(sorted(set(common_indices)), dtype=np.int64)
    all_indices = np.asarray(
        sorted(set(rare_indices.tolist()) | set(common_indices.tolist())),
        dtype=np.int64,
    )
    (
        candidate,
        checkpoint,
        checkpoint_path,
        checkpoint_sha,
        anchor,
        anchor_payload,
        anchor_path,
        anchor_sha,
    ) = load_candidate(args.checkpoint, args.checkpoint_sha256, caches[0], device)
    temperature = float(checkpoint["temperature"])
    values_by_draw = [
        evaluate_pair(
            candidate, anchor, cache, device, temperature, args.batch_size
        )
        for cache in caches
    ]
    stacked = {
        key: torch.stack([values[key] for values in values_by_draw], dim=1)
        for key in values_by_draw[0]
    }
    token_to_log = {}
    for row in pair_rows:
        log_name = str(row["log_name"])
        for key in ("rare_token", "common_token"):
            token = str(row[key])
            previous = token_to_log.setdefault(token, log_name)
            if previous != log_name:
                raise RuntimeError(f"token maps to multiple logs: {token}")
    try:
        logs = [token_to_log[str(token)] for token in caches[0]["tokens"]]
    except KeyError as error:
        raise RuntimeError(f"cache token lacks paired log provenance: {error}") from error
    strata = {
        "all": summarize(
            stacked,
            all_indices,
            logs,
            seeds,
            args.sign_epsilon,
            args.bootstrap_repetitions,
            args.seed,
        ),
        "rare": summarize(
            stacked,
            rare_indices,
            logs,
            seeds,
            args.sign_epsilon,
            args.bootstrap_repetitions,
            args.seed + 1,
        ),
        "common": summarize(
            stacked,
            common_indices,
            logs,
            seeds,
            args.sign_epsilon,
            args.bootstrap_repetitions,
            args.seed + 2,
        ),
    }
    equal_stratum_current = 0.5 * (
        strata["rare"]["mean_current_reward"]
        + strata["common"]["mean_current_reward"]
    )
    equal_stratum_anchor = 0.5 * (
        strata["rare"]["mean_anchor_reward"]
        + strata["common"]["mean_anchor_reward"]
    )
    summary = {
        "equal_stratum_current_pdm": equal_stratum_current,
        "equal_stratum_anchor_pdm": equal_stratum_anchor,
        "equal_stratum_hard_gain": equal_stratum_current - equal_stratum_anchor,
        "rare_hard_gain": strata["rare"]["mean_hard_gain"],
        "common_hard_gain": strata["common"]["mean_hard_gain"],
        "mean_min_hard_gain": strata["all"]["mean_min_hard_gain"],
        "negative_view_fraction": strata["all"]["negative_view_fraction"],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = output.with_name(output.stem + "_records.jsonl")
    write_records(
        records,
        stacked,
        caches[0],
        logs,
        seeds,
        rare_indices,
    )
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusion_set_selector_evaluation_v1",
        "arm": checkpoint["arm"],
        "arm_contract": checkpoint["arm_contract"],
        "split": args.split,
        "noise_seeds": list(seeds),
        "summary": summary,
        "strata": strata,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "v3_anchor": str(anchor_path),
        "v3_anchor_sha256": anchor_sha,
        "v3_anchor_method": anchor_payload.get("method"),
        "cache_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "records": str(records),
        "records_sha256": common.sha256_file(records),
        "scientific_contract": {
            "selector_only": True,
            "generator_frozen": True,
            "official_scalar_pdm_only": True,
            "reward_components_consumed": False,
            "new_training_data_consumed": False,
            "v3_initialized_and_anchored": True,
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
                "arm": checkpoint["arm"],
                "output": str(output),
                **summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
