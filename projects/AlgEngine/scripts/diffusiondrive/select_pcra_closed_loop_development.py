#!/usr/bin/env python3
"""Gate the offline-locked PCRA candidate on three-seed closed-loop development."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import grpo_selector_v3_cached_common as common


REQUIRED_SEEDS = ("0", "1", "2")


def labeled_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def load_metrics(path):
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    seeds = payload.get("seeds")
    if not isinstance(seeds, dict) or tuple(sorted(seeds)) != REQUIRED_SEEDS:
        raise RuntimeError(f"closed-loop metrics must contain seeds 0/1/2: {path}")
    required = ("cl_nonreactive_pdm", "cl_reactive_pdm", "success_rate")
    for seed, row in seeds.items():
        if any(
            key not in row or not math.isfinite(float(row[key])) for key in required
        ):
            raise RuntimeError(f"incomplete closed-loop seed {seed}: {path}")
    return path, payload


def seed_closed_loop(row):
    return 0.5 * (float(row["cl_nonreactive_pdm"]) + float(row["cl_reactive_pdm"]))


def summarize(payload):
    closed_loop = {
        seed: seed_closed_loop(payload["seeds"][seed]) for seed in REQUIRED_SEEDS
    }
    success = {
        seed: float(payload["seeds"][seed]["success_rate"]) for seed in REQUIRED_SEEDS
    }
    return {
        "per_seed_closed_loop_mean": closed_loop,
        "mean_closed_loop": sum(closed_loop.values()) / len(closed_loop),
        "mean_success_rate": sum(success.values()) / len(success),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-gate", type=Path, required=True)
    parser.add_argument("--proposal48-baseline", type=Path, required=True)
    parser.add_argument("--cqr-baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=labeled_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    offline_path = args.offline_gate.expanduser().resolve()
    offline = json.loads(offline_path.read_text())
    selected = offline.get("selected")
    if (
        offline.get("status") != "PASS"
        or not offline.get("development_gate_passed")
        or offline.get("certification_consumed")
        or not isinstance(selected, dict)
    ):
        raise RuntimeError("PCRA closed-loop evaluation requires a passed offline gate")
    candidate_label, candidate_raw_path = args.candidate
    if candidate_label != selected.get("label"):
        raise RuntimeError(
            "closed-loop candidate does not match the offline-locked arm"
        )

    proposal_path, proposal_payload = load_metrics(args.proposal48_baseline)
    cqr_path, cqr_payload = load_metrics(args.cqr_baseline)
    candidate_path, candidate_payload = load_metrics(candidate_raw_path)
    proposal = summarize(proposal_payload)
    cqr = summarize(cqr_payload)
    candidate = summarize(candidate_payload)

    strongest_seed = {
        seed: max(
            proposal["per_seed_closed_loop_mean"][seed],
            cqr["per_seed_closed_loop_mean"][seed],
        )
        for seed in REQUIRED_SEEDS
    }
    per_seed_delta = {
        seed: candidate["per_seed_closed_loop_mean"][seed] - strongest_seed[seed]
        for seed in REQUIRED_SEEDS
    }
    strongest_mean = max(proposal["mean_closed_loop"], cqr["mean_closed_loop"])
    gates = {
        "mean_closed_loop_beats_strongest_baseline_by_005": candidate[
            "mean_closed_loop"
        ]
        >= strongest_mean + 0.005,
        "nonnegative_on_at_least_two_seeds": sum(
            delta >= 0.0 for delta in per_seed_delta.values()
        )
        >= 2,
        "worst_seed_delta_at_least_minus_020": min(per_seed_delta.values()) >= -0.020,
        "success_preserves_cqr_within_020": candidate["mean_success_rate"]
        >= cqr["mean_success_rate"] - 0.020,
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "pcra_frozen_closed_loop_development_gate_v1",
        "development_only": True,
        "certification_consumed": False,
        "offline_gate": str(offline_path),
        "offline_gate_sha256": common.sha256_file(offline_path),
        "offline_locked_label": selected["label"],
        "proposal48_baseline": {
            "path": str(proposal_path),
            "sha256": common.sha256_file(proposal_path),
            **proposal,
        },
        "cqr_baseline": {
            "path": str(cqr_path),
            "sha256": common.sha256_file(cqr_path),
            **cqr,
        },
        "candidate": {
            "label": candidate_label,
            "path": str(candidate_path),
            "sha256": common.sha256_file(candidate_path),
            **candidate,
            "per_seed_delta_vs_strongest_baseline": per_seed_delta,
            "delta_vs_strongest_baseline_mean": candidate["mean_closed_loop"]
            - strongest_mean,
        },
        "thresholds": {
            "minimum_mean_closed_loop_gain": 0.005,
            "minimum_nonnegative_seeds": 2,
            "minimum_worst_seed_delta": -0.020,
            "maximum_success_drop_vs_cqr": 0.020,
        },
        "gates": gates,
        "closed_loop_development_passed": passed,
        "certification_allowed": passed,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
