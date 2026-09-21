#!/usr/bin/env python3
"""Materialize a cached V3 scene selector into the immutable baseline."""

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


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scene-selector-state", type=Path, required=True)
    parser.add_argument("--expected-selector-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-method", default="scene_conditioned_exact_group_grpo"
    )
    parser.add_argument(
        "--release-name", default="e2e_diffusiondrive_grpo_selector_v3"
    )
    return parser


def main():
    args = build_parser().parse_args()

    baseline = args.baseline.expanduser().resolve()
    selector_path = args.scene_selector_state.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    for path in (baseline, selector_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_sha = common.sha256_file(baseline)
    if baseline_sha != BASELINE_SHA256:
        raise RuntimeError("V3 baseline SHA256 mismatch")
    selector_sha = common.sha256_file(selector_path)
    if selector_sha != args.expected_selector_sha256:
        raise RuntimeError("V3 scene-selector SHA256 mismatch")
    selector_payload = torch.load(selector_path, map_location="cpu")
    if selector_payload.get("schema_version") != 3:
        raise RuntimeError("V3 selector payload schema drifted")
    if selector_payload.get("method") != args.expected_method:
        raise RuntimeError("V3 selector method drifted")
    selector_state = selector_payload.get("scene_selector_state")
    if not isinstance(selector_state, dict) or not selector_state:
        raise RuntimeError("V3 selector state is empty")

    checkpoint = torch.load(baseline, map_location="cpu")
    target = state_dict(checkpoint)
    if any(key.startswith(PREFIX) for key in target):
        raise RuntimeError("baseline unexpectedly already contains a V3 scene selector")
    for key, value in selector_state.items():
        target[PREFIX + key] = value.detach().cpu()
    checkpoint.setdefault("meta", {})["diffusiondrive_grpo_selector_v3"] = {
        "method": selector_payload["method"],
        "ablation": selector_payload["ablation"],
        "temperature": selector_payload["temperature"],
        "learning_rate": selector_payload["learning_rate"],
        "kl_weight": selector_payload["kl_weight"],
        "train_seed": selector_payload["train_seed"],
        "epoch": selector_payload["epoch"],
        "selector_state_sha256": selector_sha,
        "scene_selector_config": selector_payload["scene_selector_config"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    report = {
        "schema_version": 3,
        "status": "PASS",
        "method": args.release_name,
        "implementation_files": {
            str(Path(__file__).resolve()): common.sha256_file(Path(__file__).resolve())
        },
        "baseline": str(baseline),
        "baseline_sha256": baseline_sha,
        "scene_selector_state": str(selector_path),
        "scene_selector_state_sha256": selector_sha,
        "scene_selector_tensor_count": len(selector_state),
        "checkpoint": str(output),
        "checkpoint_sha256": common.sha256_file(output),
        "selector_payload": {
            key: selector_payload[key]
            for key in (
                "ablation",
                "temperature",
                "learning_rate",
                "kl_weight",
                "train_seed",
                "epoch",
                "scene_selector_config",
            )
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
