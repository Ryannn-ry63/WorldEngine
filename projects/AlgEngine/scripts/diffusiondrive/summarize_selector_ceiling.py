#!/usr/bin/env python3
"""Measure selector realization against the fixed 20-candidate reward ceiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import evaluate_trajectory_set_reasoner_grpo as evaluator
import grpo_selector_v3_cached_common as common


def parse_labeled_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("state must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("state must be LABEL=PATH")
    return label, Path(path)


def gather(values, index):
    suffix = (1,) * (values.ndim - 2)
    gather_index = index.reshape(-1, 1, *suffix).expand(-1, 1, *values.shape[2:])
    return values.gather(1, gather_index).squeeze(1)


def evaluate_one(model, cache, indices, temperature, batch_size, device):
    output = {}
    model.eval() if model is not None else None
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            reference = cache["reference_logits"][batch_indices].to(
                device=device, dtype=torch.float32
            )
            if model is None:
                logits = reference
            else:
                logits, reference = common.current_logits(
                    model, cache, batch_indices, device
                )
            rewards = cache["candidate_rewards"][batch_indices].to(device)
            components = cache["candidate_reward_components"][batch_indices].to(device)
            valid = cache["candidate_reward_valid_mask"][batch_indices].to(device)
            metrics = common.selector_metrics(
                logits, reference, rewards, components, valid, temperature
            )

            oracle_index = rewards.masked_fill(~valid, -torch.inf).argmax(dim=-1)
            oracle_logit = gather(reference, oracle_index)
            oracle_rank = (
                (reference > oracle_logit[:, None]) & valid
            ).sum(dim=-1).to(torch.float32) + 1.0
            metrics["oracle_reference_rank"] = oracle_rank
            metrics["valid_candidate_fraction"] = valid.float().mean(dim=-1)
            for key, value in metrics.items():
                output.setdefault(key, []).append(value.cpu())
    return {key: torch.cat(parts) for key, parts in output.items()}


def pool(per_seed):
    return {
        key: torch.cat([values[key] for values in per_seed])
        for key in per_seed[0]
    }


def summarize_ceiling(values):
    current = values["current_reward"]
    reference = values["reference_reward"]
    oracle = current + values["oracle_regret"]
    headroom = oracle - reference
    gain = current - reference
    positive_headroom = headroom > 1e-6
    realization = gain[positive_headroom] / headroom[positive_headroom]
    ranks = values["oracle_reference_rank"]
    return {
        **common.summarize(values),
        "reference_reward": float(reference.mean()),
        "selected_reward": float(current.mean()),
        "oracle_reward": float(oracle.mean()),
        "reference_to_oracle_headroom": float(headroom.mean()),
        "realized_headroom": float(gain.mean()),
        "headroom_realization_ratio": (
            float(realization.mean()) if len(realization) else 0.0
        ),
        "improved_fraction": float((gain > 1e-6).float().mean()),
        "degraded_fraction": float((gain < -1e-6).float().mean()),
        "oracle_in_reference_top1": float((ranks <= 1).float().mean()),
        "oracle_in_reference_top3": float((ranks <= 3).float().mean()),
        "oracle_in_reference_top5": float((ranks <= 5).float().mean()),
        "valid_candidate_fraction": float(
            values["valid_candidate_fraction"].mean()
        ),
        "reward_quantiles": {
            str(quantile): float(torch.quantile(current, quantile))
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
    }


def load_model(state_path, device):
    state_path = state_path.expanduser().resolve()
    payload = torch.load(state_path, map_location="cpu")
    if payload.get("schema_version") not in (3, 4):
        raise RuntimeError(f"unknown selector state schema: {state_path}")
    model = common.model_from_config(payload["scene_selector_config"])
    model.load_state_dict(payload["scene_selector_state"], strict=True)
    return model.to(device), payload, state_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--split", choices=tuple(evaluator.EXPECTED_SEEDS), default="development"
    )
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--state", type=parse_labeled_path, action="append", default=[])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.temperature <= 0 or args.batch_size <= 0:
        raise ValueError("temperature and batch size must be positive")

    (
        caches,
        manifests,
        _,
        rare_indices,
        pair_path,
        audit_path,
    ) = evaluator.load_evaluation_contract(
        args.cache,
        args.split,
        args.pair_manifest,
        args.rare_data_audit,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    baseline_by_seed = [
        evaluate_one(
            None, cache, rare_indices, args.temperature, args.batch_size, device
        )
        for cache in caches
    ]
    methods = {
        "frozen_v3_reference": {
            "state": None,
            "metrics": summarize_ceiling(pool(baseline_by_seed)),
            "metrics_by_noise_seed": {
                str(manifest["noise_seed"]): summarize_ceiling(values)
                for manifest, values in zip(manifests, baseline_by_seed)
            },
        }
    }
    seen_labels = {"frozen_v3_reference"}
    for label, state_argument in args.state:
        if label in seen_labels:
            raise RuntimeError(f"duplicate selector label: {label}")
        seen_labels.add(label)
        model, payload, state_path = load_model(state_argument, device)
        values_by_seed = [
            evaluate_one(
                model, cache, rare_indices, args.temperature, args.batch_size, device
            )
            for cache in caches
        ]
        methods[label] = {
            "state": str(state_path),
            "state_sha256": common.sha256_file(state_path),
            "state_schema_version": payload["schema_version"],
            "state_method": payload["method"],
            "scene_selector_config": payload["scene_selector_config"],
            "metrics": summarize_ceiling(pool(values_by_seed)),
            "metrics_by_noise_seed": {
                str(manifest["noise_seed"]): summarize_ceiling(values)
                for manifest, values in zip(manifests, values_by_seed)
            },
        }

    output_payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_fixed_candidate_selector_ceiling_v1",
        "scope": "analysis_only_rewards_are_never_selector_inputs",
        "split": args.split,
        "rare_only": True,
        "noise_seeds": list(evaluator.EXPECTED_SEEDS[args.split]),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
        "manifests": list(manifests),
        "methods": methods,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
