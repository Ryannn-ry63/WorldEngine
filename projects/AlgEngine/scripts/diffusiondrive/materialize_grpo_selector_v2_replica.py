#!/usr/bin/env python3
"""Materialize one fixed-hyperparameter V2 selector replica fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from select_grpo_selector_v2 import materialize_checkpoint


BASELINE_SHA256 = "1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
FORMAL_TEMPERATURE = 1.0
FORMAL_LEARNING_RATE = 1e-3
FORMAL_KL_WEIGHT = 1e-3
FORMAL_EPOCH = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--selector-state", type=Path, required=True)
    parser.add_argument("--expected-selector-sha256", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selector_payload(path, expected_sha, train_seed):
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError("selector-state SHA256 mismatch")
    payload = torch.load(str(path), map_location="cpu")
    expected = {
        "schema_version": 2,
        "method": "exact_group_grpo",
        "temperature": FORMAL_TEMPERATURE,
        "learning_rate": FORMAL_LEARNING_RATE,
        "kl_weight": FORMAL_KL_WEIGHT,
        "train_seed": train_seed,
        "epoch": FORMAL_EPOCH,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"selector-state {key} drifted: expected {value!r}, "
                f"got {payload.get(key)!r}"
            )
    selector = payload.get("selector_state")
    if not isinstance(selector, dict) or len(selector) != 10:
        raise RuntimeError("selector-state must contain exactly 10 tensors")
    return actual_sha


def main():
    args = parse_args()
    baseline = args.baseline.expanduser().resolve()
    selector_state = args.selector_state.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if args.train_seed not in (1, 2):
        raise ValueError("formal replica train seed must be 1 or 2")
    for path in (baseline, selector_state):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_sha = sha256_file(baseline)
    if baseline_sha != BASELINE_SHA256:
        raise RuntimeError("baseline SHA256 mismatch")
    selector_sha = validate_selector_payload(
        selector_state, args.expected_selector_sha256, args.train_seed
    )

    selection = {
        "schema_version": 1,
        "method": "e2e_diffusiondrive_grpo_selector_v2_replica",
        "objective": "exact_complete_action_expected_advantage",
        "temperature": FORMAL_TEMPERATURE,
        "learning_rate": FORMAL_LEARNING_RATE,
        "kl_weight": FORMAL_KL_WEIGHT,
        "train_seed": args.train_seed,
        "epoch": FORMAL_EPOCH,
        "selector_state": str(selector_state),
        "selector_state_sha256": selector_sha,
    }
    materialize_checkpoint(baseline, selector_state, output, selection)
    report = {
        **selection,
        "status": "PASS",
        "baseline": str(baseline),
        "baseline_sha256": baseline_sha,
        "checkpoint": str(output),
        "checkpoint_sha256": sha256_file(output),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
