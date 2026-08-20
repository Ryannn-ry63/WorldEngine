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
    parser.add_argument("--temperature", type=float, default=FORMAL_TEMPERATURE)
    parser.add_argument(
        "--learning-rate", type=float, default=FORMAL_LEARNING_RATE
    )
    parser.add_argument("--kl-weight", type=float, default=FORMAL_KL_WEIGHT)
    parser.add_argument("--epoch", type=int, default=FORMAL_EPOCH)
    parser.add_argument("--expected-reward-contract")
    parser.add_argument("--expected-reward-sha256")
    parser.add_argument(
        "--experiment-method",
        default="e2e_diffusiondrive_grpo_selector_v2_replica",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selector_payload(
    path,
    expected_sha,
    train_seed,
    temperature=FORMAL_TEMPERATURE,
    learning_rate=FORMAL_LEARNING_RATE,
    kl_weight=FORMAL_KL_WEIGHT,
    epoch=FORMAL_EPOCH,
    expected_reward_contract=None,
    expected_reward_sha256=None,
):
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise RuntimeError("selector-state SHA256 mismatch")
    payload = torch.load(str(path), map_location="cpu")
    expected = {
        "schema_version": 2,
        "method": "exact_group_grpo",
        "temperature": temperature,
        "learning_rate": learning_rate,
        "kl_weight": kl_weight,
        "train_seed": train_seed,
        "epoch": epoch,
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
    if (
        expected_reward_contract is not None
        and payload.get("reward_contract") != expected_reward_contract
    ):
        raise RuntimeError("selector-state reward contract drifted")
    if (
        expected_reward_sha256 is not None
        and payload.get("reward_implementation_sha256")
        != expected_reward_sha256
    ):
        raise RuntimeError("selector-state reward implementation drifted")
    return actual_sha


def main():
    args = parse_args()
    baseline = args.baseline.expanduser().resolve()
    selector_state = args.selector_state.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if args.train_seed not in (0, 1, 2):
        raise ValueError("train seed must be 0, 1, or 2")
    for path in (baseline, selector_state):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_sha = sha256_file(baseline)
    if baseline_sha != BASELINE_SHA256:
        raise RuntimeError("baseline SHA256 mismatch")
    selector_sha = validate_selector_payload(
        selector_state,
        args.expected_selector_sha256,
        args.train_seed,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        epoch=args.epoch,
        expected_reward_contract=args.expected_reward_contract,
        expected_reward_sha256=args.expected_reward_sha256,
    )

    selection = {
        "schema_version": 1,
        "method": args.experiment_method,
        "objective": "exact_complete_action_expected_advantage",
        "reward_contract": args.expected_reward_contract,
        "reward_implementation_sha256": args.expected_reward_sha256,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "kl_weight": args.kl_weight,
        "train_seed": args.train_seed,
        "epoch": args.epoch,
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
