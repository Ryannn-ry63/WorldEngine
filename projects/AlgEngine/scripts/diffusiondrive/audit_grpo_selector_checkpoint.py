#!/usr/bin/env python3
"""Fail if a formal checkpoint changes anything except the final selector."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


CURRENT_PREFIX = (
    "planning_head.diff_decoder.layers.1.task_decoder.plan_cls_branch."
)
REFERENCE_PREFIX = "planning_head.reference_selector."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError("checkpoint must be a dict")
    for key in ("state_dict", "model"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def normalize(state):
    normalized = {}
    for key, value in state.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    baseline_path = args.baseline.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for required in (baseline_path, checkpoint_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    baseline = normalize(
        state_dict(torch.load(str(baseline_path), map_location="cpu"))
    )
    trained = normalize(
        state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    )
    missing = sorted(set(baseline) - set(trained))
    unexpected = sorted(
        key
        for key in set(trained) - set(baseline)
        if not key.startswith(REFERENCE_PREFIX)
    )
    if missing:
        raise RuntimeError(f"trained checkpoint omitted {len(missing)} baseline tensors")
    if unexpected:
        raise RuntimeError(
            "trained checkpoint introduced non-reference tensors: "
            + ", ".join(unexpected[:10])
        )

    changed = []
    unchanged_selector = []
    max_abs_delta = {}
    for key, baseline_value in baseline.items():
        trained_value = trained[key]
        if baseline_value.shape != trained_value.shape:
            raise RuntimeError(f"shape drift for {key}")
        if torch.equal(baseline_value, trained_value):
            if key.startswith(CURRENT_PREFIX):
                unchanged_selector.append(key)
            continue
        changed.append(key)
        max_abs_delta[key] = float(
            (trained_value.float() - baseline_value.float()).abs().max()
        )

    forbidden_changed = [
        key for key in changed if not key.startswith(CURRENT_PREFIX)
    ]
    if forbidden_changed:
        raise RuntimeError(
            "frozen tensors changed: " + ", ".join(forbidden_changed[:10])
        )
    selector_keys = sorted(
        key for key in baseline if key.startswith(CURRENT_PREFIX)
    )
    if len(selector_keys) != 10:
        raise RuntimeError(
            f"baseline has {len(selector_keys)} final-selector tensors, expected 10"
        )
    if not changed:
        raise RuntimeError("formal checkpoint made no selector update")

    reference_keys = sorted(
        key for key in trained if key.startswith(REFERENCE_PREFIX)
    )
    if len(reference_keys) != 10:
        raise RuntimeError(
            f"checkpoint has {len(reference_keys)} reference tensors, expected 10"
        )
    reference_errors = {}
    for reference_key in reference_keys:
        relative = reference_key[len(REFERENCE_PREFIX) :]
        source_key = CURRENT_PREFIX + relative
        if source_key not in baseline:
            raise RuntimeError(f"reference tensor has no baseline source: {reference_key}")
        error = float(
            (
                trained[reference_key].float()
                - baseline[source_key].float()
            )
            .abs()
            .max()
        )
        reference_errors[reference_key] = error
    max_reference_error = max(reference_errors.values())
    if max_reference_error != 0.0:
        raise RuntimeError(
            f"frozen pi_ref drifted from epoch-100 baseline: {max_reference_error}"
        )

    report = {
        "schema_version": 1,
        "status": "PASS",
        "baseline": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "baseline_tensor_count": len(baseline),
        "checkpoint_tensor_count": len(trained),
        "changed_tensor_count": len(changed),
        "changed_tensors": sorted(changed),
        "unchanged_selector_tensors": sorted(unchanged_selector),
        "max_abs_delta": max_abs_delta,
        "reference_tensor_count": len(reference_keys),
        "max_reference_baseline_error": max_reference_error,
        "forbidden_changed_tensor_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
