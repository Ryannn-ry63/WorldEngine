#!/usr/bin/env python3
"""Train-only audit of probability suppression in exact selector GRPO."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common
import proposal_aware_full_feedback_grpo as paf
import train_grpo_selector_v3_cached as standard
import train_grpo_selector_v3_cached_rare_original as rare_trainer


PRIMARY_HEADROOM = 0.005
SENSITIVITY_HEADROOMS = (0.0, 0.005, 0.01)
SUPPRESSION_PROBABILITY = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--expected-noise-seeds", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-sha256", required=True)
    parser.add_argument("--stage", choices=("existing", "fresh"), required=True)
    parser.add_argument("--required-prior-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_v3(path, expected_sha, cache, device):
    path = path.expanduser().resolve()
    actual_sha = common.sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError("V3 checkpoint SHA256 drifted")
    payload = torch.load(path, map_location="cpu")
    model = common.model_from_config(payload["scene_selector_config"])
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, payload, path, actual_sha


def verify_prior(path):
    if path is None:
        raise RuntimeError("fresh audit requires --required-prior-gate")
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    if (
        payload.get("status") != "PASS"
        or payload.get("method") != "selector_policy_suppression_audit_v1"
        or payload.get("stage") != "existing"
        or payload.get("decision") != "AUTHORIZE_FRESH_NOISE_CONFIRMATION"
        or payload.get("scientific_contract", {}).get("development_consumed") is not False
    ):
        raise RuntimeError("existing-cache mechanism gate did not authorize fresh audit")
    return path, common.sha256_file(path)


def token_logs(pair_rows, tokens):
    mapping = {}
    for row in pair_rows:
        for key in ("rare_token", "common_token"):
            token = str(row[key])
            previous = mapping.setdefault(token, str(row["log_name"]))
            if previous != str(row["log_name"]):
                raise RuntimeError(f"token maps to multiple logs: {token}")
    try:
        return np.asarray([mapping[str(token)] for token in tokens], dtype=object)
    except KeyError as error:
        raise RuntimeError(f"cache token lacks log provenance: {error}") from error


def evaluate_draw(model, cache, device, temperature, batch_size):
    parts = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            logits, _ = common.current_logits(model, cache, slice(start, stop), device)
            rewards = cache["candidate_rewards"][start:stop].to(device=device, dtype=torch.float32)
            valid = cache["candidate_reward_valid_mask"][start:stop].to(device=device, dtype=torch.bool)
            valid = valid & torch.isfinite(rewards)
            logp, probability = paf.masked_log_policy(logits, valid, temperature)
            advantage, _ = paf.normalized_advantage(rewards, valid)
            ascent = paf.exact_grpo_logit_ascent(probability, advantage)
            oracle_reward, oracle_index = rewards.masked_fill(~valid, -torch.inf).max(dim=-1)
            incumbent_index = probability.masked_fill(~valid, -1.0).argmax(dim=-1)
            incumbent_reward = rewards.gather(-1, incumbent_index[:, None]).squeeze(-1)
            oracle_probability = probability.gather(-1, oracle_index[:, None]).squeeze(-1)
            incumbent_probability = probability.gather(-1, incumbent_index[:, None]).squeeze(-1)
            oracle_logit = logits.gather(-1, oracle_index[:, None]).squeeze(-1)
            incumbent_logit = logits.gather(-1, incumbent_index[:, None]).squeeze(-1)
            oracle_update = ascent.gather(-1, oracle_index[:, None]).squeeze(-1)
            incumbent_update = ascent.gather(-1, incumbent_index[:, None]).squeeze(-1)
            rank = 1 + (
                probability > oracle_probability[:, None]
            ).logical_and(valid).sum(dim=-1)
            values = {
                "headroom": oracle_reward - incumbent_reward,
                "oracle_probability": oracle_probability,
                "incumbent_probability": incumbent_probability,
                "oracle_incumbent_logit_margin": oracle_logit - incumbent_logit,
                "oracle_rank": rank.float(),
                "grpo_oracle_minus_incumbent_update": oracle_update - incumbent_update,
                "oracle_reward": oracle_reward,
                "incumbent_reward": incumbent_reward,
            }
            for key, value in values.items():
                parts[key].append(value.cpu())
    return {key: torch.cat(value).double().numpy() for key, value in parts.items()}


def bootstrap_ratio(numerator, denominator, logs, repetitions, seed):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    logs = np.asarray(logs, dtype=object)
    if numerator.shape != denominator.shape or numerator.shape != logs.shape:
        raise ValueError("bootstrap arrays must align")
    names = sorted(set(logs.tolist()))
    by_name = {name: index for index, name in enumerate(names)}
    num = np.zeros(len(names), dtype=np.float64)
    den = np.zeros(len(names), dtype=np.float64)
    for value, base, name in zip(numerator, denominator, logs):
        index = by_name[name]
        num[index] += value
        den[index] += base
    if den.sum() <= 0.0:
        raise RuntimeError("ratio metric has no eligible observations")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    chunk = 500
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        sampled = rng.integers(0, len(names), size=(stop - start, len(names)))
        sampled_num = num[sampled].sum(axis=1)
        sampled_den = den[sampled].sum(axis=1)
        estimates[start:stop] = sampled_num / np.maximum(sampled_den, 1.0)
    return {
        "estimate": float(num.sum() / den.sum()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "log",
        "num_logs": len(names),
        "repetitions": repetitions,
    }


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
    }


def summarize_stratum(stacked, indices, logs, threshold, repetitions, seed):
    h = stacked["headroom"][indices]
    oracle_probability = stacked["oracle_probability"][indices]
    boundary = stacked["grpo_oracle_minus_incumbent_update"][indices]
    recoverable = h > threshold
    suppressed = recoverable & (oracle_probability < SUPPRESSION_PROBABILITY)
    boundary_positive = recoverable & (boundary > 0.0)
    solved = ~recoverable
    mixed = recoverable.any(axis=1) & solved.any(axis=1)
    view_logs = np.repeat(logs[indices, None], h.shape[1], axis=1)

    metrics = {
        "recoverable_fraction": bootstrap_ratio(
            recoverable.ravel(), np.ones(recoverable.size), view_logs.ravel(), repetitions, seed
        ),
        "suppressed_oracle_fraction_among_recoverable": bootstrap_ratio(
            suppressed.ravel(), recoverable.ravel(), view_logs.ravel(), repetitions, seed + 1
        ),
        "grpo_boundary_positive_fraction_among_recoverable": bootstrap_ratio(
            boundary_positive.ravel(), recoverable.ravel(), view_logs.ravel(), repetitions, seed + 2
        ),
        "mixed_solved_missed_token_fraction": bootstrap_ratio(
            mixed, np.ones(len(mixed)), logs[indices], repetitions, seed + 3
        ),
    }
    metrics.update(
        {
            "num_tokens": int(len(indices)),
            "num_views": int(h.size),
            "headroom": percentile_summary(h.ravel()),
            "oracle_probability_on_recoverable": percentile_summary(oracle_probability[recoverable]),
            "incumbent_probability_on_recoverable": percentile_summary(
                stacked["incumbent_probability"][indices][recoverable]
            ),
            "oracle_incumbent_logit_margin_on_recoverable": percentile_summary(
                stacked["oracle_incumbent_logit_margin"][indices][recoverable]
            ),
            "oracle_rank_on_recoverable": percentile_summary(
                stacked["oracle_rank"][indices][recoverable]
            ),
            "grpo_boundary_on_recoverable": percentile_summary(boundary[recoverable]),
        }
    )
    return metrics


def mechanism_gates(strata):
    gates = {}
    for label in ("common", "rare"):
        row = strata[label]
        gates[f"{label}_recoverable_lower95_gt_025"] = (
            row["recoverable_fraction"]["lower_95"] > 0.25
        )
        gates[f"{label}_suppressed_lower95_gt_075"] = (
            row["suppressed_oracle_fraction_among_recoverable"]["lower_95"] > 0.75
        )
        gates[f"{label}_grpo_boundary_positive_upper95_lt_075"] = (
            row["grpo_boundary_positive_fraction_among_recoverable"]["upper_95"] < 0.75
        )
        gates[f"{label}_mixed_token_lower95_gt_010"] = (
            row["mixed_solved_missed_token_fraction"]["lower_95"] > 0.10
        )
    return gates


def main():
    args = parse_args()
    if args.temperature <= 0.0 or args.batch_size <= 0 or args.bootstrap_repetitions <= 0:
        raise ValueError("audit budgets and temperature must be positive")
    expected_seeds = tuple(int(value) for value in args.expected_noise_seeds.split(","))
    if len(expected_seeds) != 3 or len(set(expected_seeds)) != 3:
        raise ValueError("audit requires exactly three distinct noise seeds")
    if args.stage == "existing" and expected_seeds != (0, 1, 2):
        raise RuntimeError("existing audit is locked to seeds 0,1,2")
    if args.stage == "fresh" and set(expected_seeds).intersection((0, 1, 2, 3, 4, 5, 6, 7, 8)):
        raise RuntimeError("fresh audit seeds must be unseen by prior train/dev/certification work")
    prior = verify_prior(args.required_prior_gate) if args.stage == "fresh" else None
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
    rare_indices = np.asarray(sorted(set(rare_indices)), dtype=np.int64)
    common_indices = np.asarray(sorted(set(common_indices)), dtype=np.int64)
    all_indices = np.asarray(sorted(set(rare_indices.tolist()) | set(common_indices.tolist())), dtype=np.int64)
    if len(all_indices) != len(caches[0]["tokens"]):
        raise RuntimeError("rare/common strata do not cover the cache")
    logs = token_logs(pair_rows, caches[0]["tokens"])
    model, checkpoint, checkpoint_path, checkpoint_sha = load_v3(
        args.v3_checkpoint, args.v3_checkpoint_sha256, caches[0], device
    )
    draws = [evaluate_draw(model, cache, device, args.temperature, args.batch_size) for cache in caches]
    stacked = {key: np.stack([draw[key] for draw in draws], axis=1) for key in draws[0]}

    sensitivities = {}
    primary = None
    for offset, threshold in enumerate(SENSITIVITY_HEADROOMS):
        strata = {
            "all": summarize_stratum(stacked, all_indices, logs, threshold, args.bootstrap_repetitions, args.seed + 100 * offset),
            "common": summarize_stratum(stacked, common_indices, logs, threshold, args.bootstrap_repetitions, args.seed + 100 * offset + 10),
            "rare": summarize_stratum(stacked, rare_indices, logs, threshold, args.bootstrap_repetitions, args.seed + 100 * offset + 20),
        }
        sensitivities[str(threshold)] = strata
        if threshold == PRIMARY_HEADROOM:
            primary = strata
    gates = mechanism_gates(primary)
    passed = all(gates.values())
    if args.stage == "existing":
        decision = "AUTHORIZE_FRESH_NOISE_CONFIRMATION" if passed else "STOP_OBJECTIVE_DIRECTION"
    else:
        decision = "AUTHORIZE_PROPOSAL_AWARE_FULL_FEEDBACK_GRPO" if passed else "STOP_OBJECTIVE_DIRECTION"

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "selector_policy_suppression_audit_v1",
        "stage": args.stage,
        "decision": decision,
        "gates": gates,
        "primary_headroom_threshold": PRIMARY_HEADROOM,
        "suppression_probability_threshold": SUPPRESSION_PROBABILITY,
        "strata": primary,
        "sensitivity_by_headroom_threshold": sensitivities,
        "split": "train",
        "noise_seeds": list(expected_seeds),
        "v3_checkpoint": str(checkpoint_path),
        "v3_checkpoint_sha256": checkpoint_sha,
        "v3_checkpoint_method": checkpoint.get("method"),
        "cache_manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "required_prior_gate": None if prior is None else str(prior[0]),
        "required_prior_gate_sha256": None if prior is None else prior[1],
        "gate_scope": "mechanism existence only; this audit cannot establish selector efficacy",
        "scientific_contract": {
            "selector_only": True,
            "generator_frozen": True,
            "official_scalar_pdm_only": True,
            "reward_components_consumed": False,
            "new_scene_tokens_consumed": False,
            "fresh_diffusion_noise_only": args.stage == "fresh",
            "development_consumed": False,
            "certification_consumed": False,
        },
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(paf.__file__).resolve()): common.sha256_file(Path(paf.__file__).resolve()),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "stage": args.stage, "decision": decision, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
