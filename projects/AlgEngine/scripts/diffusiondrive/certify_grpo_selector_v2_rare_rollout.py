#!/usr/bin/env python3
"""One-shot certification of development-selected V2 rollout policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import certify_grpo_selector_v3 as bootstrap
import evaluate_grpo_selector_v2_rare_rollout as evaluator
import grpo_selector_v2_rare_common as common
import train_grpo_selector_v3_cached_rare_rollout as rollout
from build_grpo_selector_v3_rare_rollout_data import parse_seed_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--real-cache", action="append", type=parse_seed_path, required=True)
    parser.add_argument("--synthetic-cache", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--hard-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    args = parser.parse_args()

    selection_path = args.selection.expanduser().resolve()
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("status") != "PASS"
        or selection.get("method")
        != "diffusiondrive_v2_rare_rollout_development_selection_v1"
        or selection.get("certification_consumed")
    ):
        raise RuntimeError("invalid or already-consumed V2 rollout selection")
    selected = selection["selected"]
    state_path = Path(selected["selector_state"]).resolve()
    if common.sha256_file(state_path) != selected["selector_state_sha256"]:
        raise RuntimeError("selected V2 rollout selector SHA256 drifted")
    payload = torch.load(state_path, map_location="cpu")
    if payload.get("architecture") != "diffusiondrive_plan_cls_branch_v2":
        raise RuntimeError("selected rollout state is not V2")

    (
        manifest_path,
        manifest,
        hard_pool_path,
        all_rows,
        real_caches,
        real_manifests,
        token_maps,
        synthetic_path,
        synthetic,
    ) = rollout.load_contract(args)
    if payload["data_manifest_sha256"] != common.sha256_file(manifest_path):
        raise RuntimeError("selected state/certification data manifest drifted")
    if payload["hard_pool_sha256"] != common.sha256_file(hard_pool_path):
        raise RuntimeError("selected state/certification hard pool drifted")
    hard_rows = [row for row in all_rows if row["split"] == "certification"]
    if not hard_rows:
        raise RuntimeError("certification hard pool is empty")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = common.model_from_cache(real_caches[0], device)
    model.load_state_dict(payload["selector_state"], strict=True)

    values_by_seed = []
    scenes_by_seed = []
    metrics_by_kind = {}
    for cache, token_map, cache_manifest in zip(
        real_caches, token_maps, real_manifests
    ):
        values, scenes, kinds = evaluator.evaluate_rows(
            model,
            hard_rows,
            cache,
            token_map,
            synthetic,
            device,
            float(selected["temperature"]),
            args.batch_size,
        )
        values_by_seed.append(values)
        scenes_by_seed.append(scenes)
        metrics_by_kind[str(cache_manifest["noise_seed"])] = {
            kind: common.summarize(
                evaluator.subset(
                    values, [index for index, value in enumerate(kinds) if value == kind]
                )
            )
            for kind in ("real_rare", "synthetic_rollout")
            if kind in kinds
        }
    pooled = {
        key: torch.cat([values[key] for values in values_by_seed])
        for key in values_by_seed[0]
    }
    pooled_scenes = [scene for scenes in scenes_by_seed for scene in scenes]
    confidence_intervals = {
        "top1_reward_gain": bootstrap.bootstrap_scene_ci(
            pooled["top1_reward_gain"].numpy(),
            pooled_scenes,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
    }
    for component_index, name in enumerate(common.COMPONENT_NAMES):
        confidence_intervals[f"delta_{name}"] = bootstrap.bootstrap_scene_ci(
            pooled["component_delta"][:, component_index].numpy(),
            pooled_scenes,
            args.bootstrap_replicates,
            args.bootstrap_seed + component_index + 1,
        )

    report = {
        "schema_version": 1,
        "status": "PASS",
        "method": "diffusiondrive_v2_rare_rollout_certification_v1",
        "architecture": "diffusiondrive_plan_cls_branch_v2",
        "selection": str(selection_path),
        "selection_sha256": common.sha256_file(selection_path),
        "certification_consumed": True,
        "hard_only": True,
        "hard_rows": len(hard_rows),
        "hard_logs": len({row["log_name"] for row in hard_rows}),
        "metrics": common.summarize(pooled),
        "metrics_by_noise_seed": {
            str(row["noise_seed"]): common.summarize(values)
            for row, values in zip(real_manifests, values_by_seed)
        },
        "hard_kind_metrics_by_noise_seed": metrics_by_kind,
        "confidence_intervals": confidence_intervals,
        "scientific_gate": {
            "mean_top1_gain_positive": (
                confidence_intervals["top1_reward_gain"]["mean"] > 0.0
            ),
            "lower_95_top1_gain_positive": (
                confidence_intervals["top1_reward_gain"]["lower_95"] > 0.0
            ),
        },
        "selector_state": str(state_path),
        "selector_state_sha256": common.sha256_file(state_path),
        "data_manifest": str(manifest_path),
        "data_manifest_sha256": common.sha256_file(manifest_path),
        "hard_pool": str(hard_pool_path),
        "hard_pool_sha256": common.sha256_file(hard_pool_path),
        "synthetic_cache": str(synthetic_path),
        "synthetic_cache_sha256": common.sha256_file(synthetic_path),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
