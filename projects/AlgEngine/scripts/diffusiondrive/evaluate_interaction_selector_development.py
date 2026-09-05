#!/usr/bin/env python3
"""Evaluate and gate the fixed A0/A1/A2 interaction-selector experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common
import train_trajectory_set_reasoner_grpo as trainer


ARMS = {
    "A0": "scene_conditioned_v3",
    "A1": "interaction_generic",
    "A2": "interaction_relation",
}


def parse_labeled_path(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("state must be A0=PATH, A1=PATH or A2=PATH")
    label, path = value.split("=", 1)
    if label not in ARMS or not path:
        raise argparse.ArgumentTypeError("state must use exactly A0, A1 or A2")
    return label, Path(path)


def load_state(path: Path, label: str, device: torch.device):
    path = path.expanduser().resolve()
    payload = torch.load(path, map_location="cpu")
    expected = {
        "schema_version": 4,
        "selector_architecture": ARMS[label],
        "sampling_mode": "rare_balanced",
        "temperature": 1.0,
        "learning_rate": 1e-4,
        "kl_weight": 1e-3,
        "train_seed": 0,
        "epoch": 16,
    }
    drift = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if drift:
        raise RuntimeError(f"{label} fixed contract drifted: {drift}")
    model = common.model_from_config(payload["scene_selector_config"])
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    return model.to(device), payload, path


def gather(values: torch.Tensor, indices: torch.Tensor):
    suffix = (1,) * (values.ndim - 2)
    expanded = indices.reshape(-1, 1, *suffix).expand(-1, 1, *values.shape[2:])
    return values.gather(1, expanded).squeeze(1)


def pairwise_ordering_accuracy(logits, rewards, valid):
    left, right = torch.triu_indices(logits.shape[1], logits.shape[1], offset=1, device=logits.device)
    valid_pair = valid[:, left] & valid[:, right]
    reward_delta = rewards[:, left] - rewards[:, right]
    informative = valid_pair & reward_delta.abs().gt(1e-6)
    logit_delta = logits[:, left] - logits[:, right]
    correct = logit_delta.mul(reward_delta).gt(0.0).float()
    correct = correct + 0.5 * logit_delta.eq(0.0).float()
    count = informative.sum(dim=-1)
    score = (correct * informative).sum(dim=-1) / count.clamp_min(1)
    return torch.where(count.gt(0), score, torch.full_like(score, 0.5))


def evaluate_subset(model, cache, indices, device, batch_size):
    output = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            logits, reference = common.current_logits(model, cache, batch_indices, device)
            rewards = cache["candidate_rewards"][batch_indices].to(device)
            components = cache["candidate_reward_components"][batch_indices].to(device)
            valid = cache["candidate_reward_valid_mask"][batch_indices].to(device)
            metrics = common.selector_metrics(
                logits, reference, rewards, components, valid, 1.0
            )
            metrics["pairwise_ordering_auc"] = pairwise_ordering_accuracy(
                logits, rewards, valid
            )
            metrics["current_index"] = logits.masked_fill(~valid, -1e4).argmax(dim=-1)
            metrics["degraded"] = metrics["top1_reward_gain"].lt(-1e-8).float()
            for key, value in metrics.items():
                output[key].append(value.cpu())
    return {key: torch.cat(parts) for key, parts in output.items()}


def pool(groups):
    return {key: torch.cat([group[key] for group in groups]) for key in groups[0]}


def summarize(groups):
    values = pool(groups)
    public = {
        key: value
        for key, value in values.items()
        if key not in {"current_index", "degraded"}
    }
    output = common.summarize(public)
    output.update(
        degraded_fraction=float(values["degraded"].mean()),
        pairwise_ordering_auc=float(values["pairwise_ordering_auc"].mean()),
        examples=len(values["current_reward"]),
    )
    if len(groups) < 2:
        output["noise_view_agreement"] = None
    else:
        pairwise = []
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                pairwise.append(
                    groups[left]["current_index"].eq(groups[right]["current_index"]).float().mean()
                )
        output["noise_view_agreement"] = float(torch.stack(pairwise).mean())
    return output, values


def scene_bootstrap_delta(candidate, baseline, scenes, repetitions, seed):
    grouped = defaultdict(list)
    delta = (candidate - baseline).double().numpy()
    for value, scene in zip(delta, scenes):
        grouped[str(scene)].append(float(value))
    names = sorted(grouped)
    means = np.asarray([np.mean(grouped[name]) for name in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(names), size=(repetitions, len(names)))
    estimates = means[sampled].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "scene",
        "num_scenes": len(names),
        "repetitions": repetitions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--state", type=parse_labeled_path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    states = dict(args.state)
    if set(states) != set(ARMS) or len(args.state) != 3:
        raise RuntimeError("development gate requires exactly one A0, A1 and A2 state")
    if min(args.batch_size, args.bootstrap_repetitions) <= 0:
        raise ValueError("evaluation budgets must be positive")

    pairs = [common.load_cache(path, "development") for path in args.cache]
    pairs.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*pairs)
    if [int(row["noise_seed"]) for row in manifests] != [3, 4, 5]:
        raise RuntimeError("development gate requires noise seeds 3,4,5")
    if any(cache.get("schema_version") != 4 for cache in caches):
        raise RuntimeError("development gate requires frozen-track schema-v4 caches")
    pair_rows, audit, pair_path, audit_path = trainer.load_pair_contract(
        args.pair_manifest, args.rare_data_audit
    )
    rare_indices, common_indices = trainer.validate_cache_pair_alignment(
        caches, manifests, pair_rows, audit
    )
    rare_indices = torch.tensor(rare_indices, dtype=torch.long)
    common_indices = torch.tensor(common_indices, dtype=torch.long)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    arm_outputs = {}
    raw = {}
    for label in ARMS:
        model, payload, path = load_state(states[label], label, device)
        populations = {}
        population_raw = {}
        for name, indices in (("common", common_indices), ("real_rare", rare_indices)):
            groups = [
                evaluate_subset(model, cache, indices, device, args.batch_size)
                for cache in caches
            ]
            populations[name], population_raw[name] = summarize(groups)
        equal_weight_reward = 0.5 * (
            populations["common"]["current_reward"]
            + populations["real_rare"]["current_reward"]
        )
        arm_outputs[label] = {
            "selector_architecture": ARMS[label],
            "state": str(path),
            "state_sha256": common.sha256_file(path),
            "equal_weight_selected_reward": equal_weight_reward,
            "strata": populations,
        }
        raw[label] = population_raw

    pooled_scenes = {
        "common": [
            cache["scenes"][int(index)]
            for cache in caches
            for index in common_indices
        ],
        "real_rare": [
            cache["scenes"][int(index)]
            for cache in caches
            for index in rare_indices
        ],
    }
    comparisons = {}
    gates = {}
    for offset, label in enumerate(("A1", "A2")):
        comparison = {}
        for population in ("common", "real_rare"):
            comparison[population] = scene_bootstrap_delta(
                raw[label][population]["pairwise_ordering_auc"],
                raw["A0"][population]["pairwise_ordering_auc"],
                pooled_scenes[population],
                args.bootstrap_repetitions,
                args.bootstrap_seed + offset * 10 + (population == "real_rare"),
            )
        comparisons[f"{label}_minus_A0_pairwise_auc"] = comparison
        current = arm_outputs[label]
        baseline = arm_outputs["A0"]
        candidate_gates = {
            "equal_weight_reward_gain_at_least_005": current["equal_weight_selected_reward"]
            >= baseline["equal_weight_selected_reward"] + 0.005,
            "common_reward_no_worse_than_minus_005": current["strata"]["common"]["current_reward"]
            >= baseline["strata"]["common"]["current_reward"] - 0.005,
            "rare_reward_nonnegative_vs_A0": current["strata"]["real_rare"]["current_reward"]
            >= baseline["strata"]["real_rare"]["current_reward"],
            "common_degraded_fraction_not_worse": current["strata"]["common"]["degraded_fraction"]
            <= baseline["strata"]["common"]["degraded_fraction"] + 1e-12,
            "common_pairwise_auc_ci_positive": comparison["common"]["lower_95"] > 0.0,
            "rare_pairwise_auc_ci_positive": comparison["real_rare"]["lower_95"] > 0.0,
            "common_noise_agreement_no_worse_than_minus_002": current["strata"]["common"]["noise_view_agreement"]
            >= baseline["strata"]["common"]["noise_view_agreement"] - 0.02,
            "rare_noise_agreement_no_worse_than_minus_002": current["strata"]["real_rare"]["noise_view_agreement"]
            >= baseline["strata"]["real_rare"]["noise_view_agreement"] - 0.02,
        }
        gates[label] = {**candidate_gates, "all_passed": all(candidate_gates.values())}

    if gates["A2"]["all_passed"] and (
        arm_outputs["A2"]["equal_weight_selected_reward"]
        >= arm_outputs["A1"]["equal_weight_selected_reward"] + 0.002
    ):
        decision, selected = "AUTHORIZE_A2_CLOSED_LOOP_DEVELOPMENT", "A2"
    elif gates["A1"]["all_passed"]:
        decision, selected = "AUTHORIZE_A1_CLOSED_LOOP_DEVELOPMENT", "A1"
    else:
        decision, selected = "STOP_INTERACTION_AND_RETAIN_V3", None
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "interaction_selector_A0_A1_A2_development_gate_v1",
        "development_only": True,
        "certification_consumed": False,
        "decision": decision,
        "selected_arm": selected,
        "arms": arm_outputs,
        "comparisons": comparisons,
        "gates": gates,
        "A2_selection_margin_over_A1": 0.002,
        "thresholds": {
            "minimum_equal_weight_gain": 0.005,
            "maximum_common_reward_drop": 0.005,
            "maximum_noise_agreement_drop": 0.02,
            "pairwise_auc_lower_95": 0.0,
        },
        "manifests": list(manifests),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "decision": decision, "selected_arm": selected, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
