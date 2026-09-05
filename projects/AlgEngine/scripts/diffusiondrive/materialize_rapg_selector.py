#!/usr/bin/env python3
"""Materialize an audited RAPG selector state into the epoch-100 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import grpo_selector_v3_cached_common as common


BASELINE_SHA256 = "1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
PREFIX = "planning_head.scene_selector."


def state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("baseline checkpoint must be a dictionary")
    for key in ("state_dict", "model"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scene-selector-state", type=Path, required=True)
    parser.add_argument("--expected-selector-sha256", required=True)
    parser.add_argument("--expected-method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release-name", default="diffusiondrive_selector_rapg_v1")
    args = parser.parse_args()

    baseline = args.baseline.expanduser().resolve()
    selector_path = args.scene_selector_state.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for path in (baseline, selector_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_sha = common.sha256_file(baseline)
    if baseline_sha != BASELINE_SHA256:
        raise RuntimeError("epoch-100 baseline SHA256 mismatch")
    selector_sha = common.sha256_file(selector_path)
    if selector_sha != args.expected_selector_sha256:
        raise RuntimeError("RAPG selector SHA256 mismatch")
    selector_payload = torch.load(selector_path, map_location="cpu")
    if selector_payload.get("schema_version") not in (3, 4, 5, 6):
        raise RuntimeError("RAPG selector payload schema drifted")
    if selector_payload.get("method") != args.expected_method:
        raise RuntimeError("RAPG selector method drifted")
    if selector_payload.get("selector_architecture") not in (
        "reference_anchored_preference_graph",
        "trajectory_set_reasoner",
        "capacity_matched_unary",
        "incumbent_verified_preference",
        "proposal_conditioned_regret_arbitration",
        "proposal_conditioned_counterfactual_evaluator",
    ):
        raise RuntimeError("unsupported selector architecture in RAPG payload")
    selector_state = selector_payload.get("scene_selector_state")
    if not isinstance(selector_state, dict) or not selector_state:
        raise RuntimeError("RAPG selector state is empty")

    checkpoint = torch.load(baseline, map_location="cpu")
    target = state_dict(checkpoint)
    if any(key.startswith(PREFIX) for key in target):
        raise RuntimeError("baseline unexpectedly already contains a scene selector")
    for key, value in selector_state.items():
        target[PREFIX + key] = value.detach().cpu()
    metadata = {
        key: selector_payload[key]
        for key in (
            "method",
            "selector_architecture",
            "ablation",
            "objective",
            "preference_weight",
            "temperature",
            "learning_rate",
            "kl_weight",
            "train_seed",
            "epoch",
            "scene_selector_config",
        )
    }
    metadata["selector_state_sha256"] = selector_sha
    for optional_key in (
        "proposal_checkpoint",
        "proposal_checkpoint_sha256",
        "verifier_reward_margin",
        "verifier_reward_temperature",
        "override_threshold",
        "arbiter_loss",
        "arbiter_risk",
        "use_decision_context",
        "arbiter_target",
        "counterfactual_loss",
        "train_evaluator_encoder",
        "source_risk",
        "evaluator_initialization_max_abs_delta",
    ):
        if optional_key in selector_payload:
            metadata[optional_key] = selector_payload[optional_key]
    checkpoint.setdefault("meta", {})[
        "reference_anchored_preference_graph_v1"
    ] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": args.release_name,
        "baseline": str(baseline),
        "baseline_sha256": baseline_sha,
        "scene_selector_state": str(selector_path),
        "scene_selector_state_sha256": selector_sha,
        "scene_selector_tensor_count": len(selector_state),
        "checkpoint": str(output),
        "checkpoint_sha256": common.sha256_file(output),
        "selector_payload": metadata,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
