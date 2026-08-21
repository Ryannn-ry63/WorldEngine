#!/usr/bin/env python3
"""Materialize an audited 10-tensor V2 selector into epoch-100 DiffusionDrive."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

import grpo_selector_v2_rare_common as common


def checkpoint_state(payload):
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must be a dictionary")
    for key in ("state_dict", "model"):
        if isinstance(payload.get(key), dict):
            return payload[key], key
    return payload, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--selector-state", type=Path, required=True)
    parser.add_argument("--expected-selector-sha256", required=True)
    parser.add_argument("--expected-method", required=True)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    baseline = args.baseline.expanduser().resolve()
    selector_path = args.selector_state.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if common.sha256_file(baseline) != common.BASELINE_SHA256:
        raise RuntimeError("epoch-100 baseline SHA256 mismatch")
    if common.sha256_file(selector_path) != args.expected_selector_sha256:
        raise RuntimeError("V2 selector-state SHA256 mismatch")
    selector_payload = torch.load(selector_path, map_location="cpu")
    if (
        selector_payload.get("schema_version") != 3
        or selector_payload.get("architecture")
        != "diffusiondrive_plan_cls_branch_v2"
        or selector_payload.get("method") != args.expected_method
        or selector_payload.get("reward_contract") != common.REWARD_CONTRACT
    ):
        raise RuntimeError("V2 selector-state provenance drifted")
    selector_state = selector_payload.get("selector_state")
    expected_relative = set(common.Selector().state_dict())
    if not isinstance(selector_state, dict) or set(selector_state) != expected_relative:
        raise RuntimeError("V2 selector-state is not the exact 10-tensor plan-cls MLP")

    baseline_payload = torch.load(baseline, map_location="cpu")
    payload = copy.deepcopy(baseline_payload)
    state, container = checkpoint_state(payload)
    module_prefix = "module." if any(key.startswith("module.") for key in state) else ""
    current_prefix = module_prefix + common.SELECTOR_PREFIX
    reference_prefix = module_prefix + common.REFERENCE_PREFIX
    for relative in sorted(expected_relative):
        current_key = current_prefix + relative
        reference_key = reference_prefix + relative
        if current_key not in state:
            raise RuntimeError(f"baseline selector key missing: {current_key}")
        baseline_value = state[current_key]
        selected_value = selector_state[relative]
        if tuple(selected_value.shape) != tuple(baseline_value.shape):
            raise RuntimeError(f"V2 selector shape drifted: {relative}")
        state[current_key] = selected_value.to(dtype=baseline_value.dtype).clone()
        state[reference_key] = baseline_value.clone()
    if container is not None:
        payload[container] = state

    metadata = {
        "schema_version": 3,
        "status": "PASS",
        "method": args.expected_method,
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "release_name": args.release_name,
        "objective": "exact_complete_action_expected_advantage",
        "reward_contract": common.REWARD_CONTRACT,
        "baseline": str(baseline),
        "baseline_sha256": common.BASELINE_SHA256,
        "selector_state": str(selector_path),
        "selector_state_sha256": args.expected_selector_sha256,
        "selector_tensor_count": common.SELECTOR_TENSOR_COUNT,
        "selector_payload": {
            key: selector_payload[key]
            for key in (
                "temperature",
                "learning_rate",
                "kl_weight",
                "train_seed",
                "epoch",
                "sampling_mode",
            )
        },
    }
    meta = payload.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {"upstream_meta": str(meta)}
        payload["meta"] = meta
    meta["diffusiondrive_selector_grpo_v2"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    metadata.update(
        checkpoint=str(output), checkpoint_sha256=common.sha256_file(output)
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
