#!/usr/bin/env python3
"""Evaluate one PAF selector arm on locked fresh-noise train-scene caches."""

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
import train_proposal_aware_full_feedback_grpo as trainer


EXPECTED_NOISE_SEEDS = (9, 10, 11)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--mechanism-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_candidate(path, expected_sha, cache, device):
    path = path.expanduser().resolve()
    actual_sha = common.sha256_file(path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise RuntimeError("candidate checkpoint SHA256 drifted")
    payload = torch.load(path, map_location="cpu")
    if payload.get("method") != trainer.METHOD or payload.get("arm") not in paf.ARMS:
        raise RuntimeError("invalid PAF selector checkpoint")
    model = common.model_from_config(payload["scene_selector_config"])
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    model.to(device).eval()
    anchor_path = Path(payload["v3_anchor"]).resolve()
    anchor_sha = common.sha256_file(anchor_path)
    if anchor_sha != payload["v3_anchor_sha256"]:
        raise RuntimeError("V3 anchor SHA256 drifted")
    anchor_payload = torch.load(anchor_path, map_location="cpu")
    anchor = common.model_from_config(anchor_payload["scene_selector_config"])
    anchor.load_state_dict(anchor_payload["scene_selector_state"], strict=True)
    anchor.to(device).eval()
    return model, payload, path, actual_sha, anchor, anchor_payload, anchor_path, anchor_sha


def evaluate_draw(candidate, anchor, cache, device, temperature, batch_size):
    output = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(cache["tokens"]), batch_size):
            stop = min(start + batch_size, len(cache["tokens"]))
            indices = slice(start, stop)
            current_logits, _ = common.current_logits(candidate, cache, indices, device)
            anchor_logits, _ = common.current_logits(anchor, cache, indices, device)
            rewards = cache["candidate_rewards"][indices].to(device=device, dtype=torch.float32)
            valid = cache["candidate_reward_valid_mask"][indices].to(device=device, dtype=torch.bool)
            valid = valid & torch.isfinite(rewards)
            _, current_probability = paf.masked_log_policy(current_logits, valid, temperature)
            _, anchor_probability = paf.masked_log_policy(anchor_logits, valid, temperature)
            current_index = current_probability.masked_fill(~valid, -1.0).argmax(dim=-1)
            anchor_index = anchor_probability.masked_fill(~valid, -1.0).argmax(dim=-1)
            oracle_reward, oracle_index = rewards.masked_fill(~valid, -torch.inf).max(dim=-1)

            def gather(values, index):
                return values.gather(-1, index[:, None]).squeeze(-1)

            current_reward = gather(rewards, current_index)
            anchor_reward = gather(rewards, anchor_index)
            oracle_probability = gather(anchor_probability, oracle_index)
            headroom = oracle_reward - anchor_reward
            values = {
                "hard_gain": current_reward - anchor_reward,
                "current_reward": current_reward,
                "anchor_reward": anchor_reward,
                "oracle_reward": oracle_reward,
                "anchor_oracle_headroom": headroom,
                "current_oracle_headroom": oracle_reward - current_reward,
                "anchor_oracle_probability": oracle_probability,
                "recoverable": (headroom > 0.005).float(),
                "suppressed_recoverable": ((headroom > 0.005) & (oracle_probability < 1e-4)).float(),
                "solved": (headroom <= 0.005).float(),
                "selection_changed": current_index.ne(anchor_index).float(),
            }
            for key, value in values.items():
                output[key].append(value.cpu())
    return {key: torch.cat(parts) for key, parts in output.items()}


def token_logs(pair_rows, tokens):
    mapping = {}
    for row in pair_rows:
        for key in ("rare_token", "common_token"):
            mapping.setdefault(str(row[key]), str(row["log_name"]))
    return [mapping[str(token)] for token in tokens]


def bootstrap_log_gain(hard, indices, logs, strata, repetitions, seed, equal=False):
    by_log = defaultdict(lambda: defaultdict(list))
    for index in indices:
        by_log[str(logs[index])][strata[index]].extend(hard[index].tolist())
    values = []
    for name in sorted(by_log):
        groups = by_log[name]
        if equal:
            if not groups["rare"] or not groups["common"]:
                raise RuntimeError(f"log lacks one stratum: {name}")
            value = 0.5 * (np.mean(groups["rare"], dtype=np.float64) + np.mean(groups["common"], dtype=np.float64))
        else:
            flattened = [value for rows in groups.values() for value in rows]
            value = np.mean(flattened, dtype=np.float64)
        values.append(value)
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 500):
        stop = min(start + 500, repetitions)
        sampled = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[sampled].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "log",
        "num_logs": len(values),
        "repetitions": repetitions,
    }


def conditional_mean(values, mask):
    selected = values[mask]
    return float(selected.mean()) if selected.size else None


def summarize(stacked, indices, logs, labels, seeds, repetitions, seed):
    arrays = {key: value[indices].double().numpy() for key, value in stacked.items()}
    hard = arrays["hard_gain"]
    suppressed = arrays["suppressed_recoverable"].astype(bool)
    solved = arrays["solved"].astype(bool)
    by_seed = {}
    for draw, noise_seed in enumerate(seeds):
        by_seed[str(noise_seed)] = {
            "hard_gain": float(hard[:, draw].mean()),
            "current_reward": float(arrays["current_reward"][:, draw].mean()),
            "anchor_reward": float(arrays["anchor_reward"][:, draw].mean()),
        }
    return {
        "num_tokens": int(len(indices)),
        "mean_current_reward": float(arrays["current_reward"].mean()),
        "mean_anchor_reward": float(arrays["anchor_reward"].mean()),
        "mean_hard_gain": float(hard.mean()),
        "selection_change_fraction": float(arrays["selection_changed"].mean()),
        "recoverable_fraction": float(arrays["recoverable"].mean()),
        "suppressed_recoverable_fraction": float(arrays["suppressed_recoverable"].mean()),
        "suppressed_recoverable_oracle_regret_reduction": conditional_mean(hard, suppressed),
        "solved_subset_hard_gain": conditional_mean(hard, solved),
        "by_noise_seed": by_seed,
        "log_bootstrap_hard_gain": bootstrap_log_gain(
            stacked["hard_gain"].double().numpy(), indices, logs, labels, repetitions, seed
        ),
    }


def write_records(path, stacked, cache, logs, labels, seeds):
    keys = (
        "hard_gain", "current_reward", "anchor_reward", "oracle_reward",
        "anchor_oracle_headroom", "current_oracle_headroom",
        "anchor_oracle_probability", "recoverable", "suppressed_recoverable",
        "solved", "selection_changed",
    )
    with path.open("w") as stream:
        for index, (token, scene) in enumerate(zip(cache["tokens"], cache["scenes"])):
            for draw, noise_seed in enumerate(seeds):
                row = {
                    "token": str(token),
                    "scene": str(scene),
                    "log": str(logs[index]),
                    "stratum": str(labels[index]),
                    "noise_seed": int(noise_seed),
                }
                for key in keys:
                    row[key] = float(stacked[key][index, draw])
                stream.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_repetitions <= 0:
        raise ValueError("evaluation budgets must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    gate_path = args.mechanism_gate.expanduser().resolve()
    gate = json.loads(gate_path.read_text())
    if gate.get("decision") != "AUTHORIZE_PROPOSAL_AWARE_FULL_FEEDBACK_GRPO" or gate.get("noise_seeds") != list(EXPECTED_NOISE_SEEDS):
        raise RuntimeError("evaluation requires the locked fresh-noise mechanism gate")

    loaded = [common.load_cache(path, "train") for path in args.cache]
    loaded.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*loaded)
    standard.assert_cache_group(caches, manifests, "train", EXPECTED_NOISE_SEEDS)
    pair_rows, rare_audit, pair_path, audit_path = rare_trainer.load_pair_contract(args.pair_manifest, args.rare_data_audit)
    rare_indices, common_indices = rare_trainer.validate_cache_pair_alignment(caches, manifests, pair_rows, rare_audit)
    rare_indices = np.asarray(sorted(set(rare_indices)), dtype=np.int64)
    common_indices = np.asarray(sorted(set(common_indices)), dtype=np.int64)
    all_indices = np.asarray(sorted(set(rare_indices.tolist()) | set(common_indices.tolist())), dtype=np.int64)
    labels = np.asarray(["common"] * len(caches[0]["tokens"]), dtype=object)
    labels[rare_indices] = "rare"
    logs = token_logs(pair_rows, caches[0]["tokens"])
    candidate, checkpoint, checkpoint_path, checkpoint_sha, anchor, anchor_payload, anchor_path, anchor_sha = load_candidate(
        args.checkpoint, args.checkpoint_sha256, caches[0], device
    )
    if checkpoint.get("epoch") != 8 or checkpoint.get("mechanism_gate_sha256") != common.sha256_file(gate_path):
        raise RuntimeError("evaluation is locked to epoch 8 and the supplied mechanism gate")
    draws = [evaluate_draw(candidate, anchor, cache, device, float(checkpoint["temperature"]), args.batch_size) for cache in caches]
    stacked = {key: torch.stack([draw[key] for draw in draws], dim=1) for key in draws[0]}
    strata = {
        "all": summarize(stacked, all_indices, logs, labels, EXPECTED_NOISE_SEEDS, args.bootstrap_repetitions, args.seed),
        "common": summarize(stacked, common_indices, logs, labels, EXPECTED_NOISE_SEEDS, args.bootstrap_repetitions, args.seed + 1),
        "rare": summarize(stacked, rare_indices, logs, labels, EXPECTED_NOISE_SEEDS, args.bootstrap_repetitions, args.seed + 2),
    }
    equal_seed_gains = {
        str(noise_seed): 0.5 * (
            strata["common"]["by_noise_seed"][str(noise_seed)]["hard_gain"]
            + strata["rare"]["by_noise_seed"][str(noise_seed)]["hard_gain"]
        )
        for noise_seed in EXPECTED_NOISE_SEEDS
    }
    equal_bootstrap = bootstrap_log_gain(
        stacked["hard_gain"].double().numpy(), all_indices, logs, labels,
        args.bootstrap_repetitions, args.seed + 10, equal=True,
    )
    summary = {
        "equal_stratum_current_pdm": 0.5 * (strata["common"]["mean_current_reward"] + strata["rare"]["mean_current_reward"]),
        "equal_stratum_anchor_pdm": 0.5 * (strata["common"]["mean_anchor_reward"] + strata["rare"]["mean_anchor_reward"]),
        "equal_stratum_hard_gain": 0.5 * (strata["common"]["mean_hard_gain"] + strata["rare"]["mean_hard_gain"]),
        "common_hard_gain": strata["common"]["mean_hard_gain"],
        "rare_hard_gain": strata["rare"]["mean_hard_gain"],
        "equal_stratum_by_noise_seed_hard_gain": equal_seed_gains,
        "equal_log_bootstrap_hard_gain": equal_bootstrap,
        "equal_suppressed_recoverable_oracle_regret_reduction": 0.5 * (
            strata["common"]["suppressed_recoverable_oracle_regret_reduction"]
            + strata["rare"]["suppressed_recoverable_oracle_regret_reduction"]
        ),
        "equal_solved_subset_hard_gain": 0.5 * (
            strata["common"]["solved_subset_hard_gain"] + strata["rare"]["solved_subset_hard_gain"]
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = output.with_name(output.stem + "_records.jsonl")
    write_records(records, stacked, caches[0], logs, labels, EXPECTED_NOISE_SEEDS)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "proposal_aware_full_feedback_evaluation_v1",
        "arm": checkpoint["arm"],
        "arm_contract": checkpoint["arm_contract"],
        "split": "train_fresh_noise",
        "noise_seeds": list(EXPECTED_NOISE_SEEDS),
        "summary": summary,
        "strata": strata,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "v3_anchor": str(anchor_path),
        "v3_anchor_sha256": anchor_sha,
        "v3_anchor_method": anchor_payload.get("method"),
        "mechanism_gate": str(gate_path),
        "mechanism_gate_sha256": common.sha256_file(gate_path),
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
            "same_training_scene_tokens": True,
            "fresh_diffusion_noise_only": True,
            "development_consumed": False,
            "certification_consumed": False,
        },
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve()),
            str(Path(paf.__file__).resolve()): common.sha256_file(Path(paf.__file__).resolve()),
            str(Path(trainer.__file__).resolve()): common.sha256_file(Path(trainer.__file__).resolve()),
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "arm": checkpoint["arm"], "output": str(output), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
