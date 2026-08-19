#!/usr/bin/env python3
"""One-shot certification of the development-selected rare-original V3 policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import certify_grpo_selector_v3 as base
import evaluate_grpo_selector_v3_rare_original as evaluator
import grpo_selector_v3_cached_common as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--rare-data-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    args = parser.parse_args()

    selection_path = args.selection.expanduser().resolve()
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("status") != "PASS"
        or selection.get("method")
        != "diffusiondrive_v3_rare_original_development_selection_v1"
        or selection.get("certification_consumed")
    ):
        raise RuntimeError("invalid or already-consumed rare-original selection")
    selected = selection["selected"]
    state_path = Path(selected["scene_selector_state"]).resolve()
    if common.sha256_file(state_path) != selected["scene_selector_state_sha256"]:
        raise RuntimeError("selected rare-original selector SHA256 drifted")

    (
        caches,
        manifests,
        pair_rows,
        rare_indices,
        pair_path,
        audit_path,
    ) = evaluator.load_evaluation_contract(
        args.cache,
        "certification",
        args.pair_manifest,
        args.rare_data_audit,
    )
    device = torch.device(args.device)
    state, values_by_seed = evaluator.evaluate_state(
        state_path,
        caches,
        rare_indices,
        float(selected["temperature"]),
        args.batch_size,
        device,
    )
    pooled = evaluator.pooled(values_by_seed)
    rare_scenes = [
        str(caches[0]["scenes"][index]) for index in rare_indices.tolist()
    ]
    pooled_scenes = rare_scenes * len(caches)
    confidence_intervals = {
        "top1_reward_gain": base.bootstrap_scene_ci(
            pooled["top1_reward_gain"].numpy(),
            pooled_scenes,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
    }
    for component_index, name in enumerate(common.COMPONENT_NAMES):
        confidence_intervals[f"delta_{name}"] = base.bootstrap_scene_ci(
            pooled["component_delta"][:, component_index].numpy(),
            pooled_scenes,
            args.bootstrap_replicates,
            args.bootstrap_seed + component_index + 1,
        )

    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v3_rare_original_certification_v1",
        "selection": str(selection_path),
        "selection_sha256": common.sha256_file(selection_path),
        "certification_consumed": True,
        "rare_only": True,
        "certification_seeds": [6, 7, 8],
        "rare_tokens": len(pair_rows),
        "rare_scenes": len(set(rare_scenes)),
        "metrics": common.summarize(pooled),
        "metrics_by_noise_seed": {
            str(manifest["noise_seed"]): common.summarize(values)
            for manifest, values in zip(manifests, values_by_seed)
        },
        "rare_vote_strata": evaluator.vote_strata(values_by_seed, pair_rows),
        "confidence_intervals": confidence_intervals,
        "scientific_gate": {
            "mean_top1_gain_positive": (
                confidence_intervals["top1_reward_gain"]["mean"] > 0.0
            ),
            "lower_95_top1_gain_positive": (
                confidence_intervals["top1_reward_gain"]["lower_95"] > 0.0
            ),
        },
        "scene_selector_state": str(state_path),
        "scene_selector_state_sha256": common.sha256_file(state_path),
        "selector_payload_method": state["method"],
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": common.sha256_file(pair_path),
        "rare_data_audit": str(audit_path),
        "rare_data_audit_sha256": common.sha256_file(audit_path),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
