#!/usr/bin/env python3
"""One-shot scene-block certification for the development-selected V3 policy."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import grpo_selector_v3_cached_common as common


def bootstrap_scene_ci(values, scenes, replicates, seed):
    grouped_sum = defaultdict(float)
    grouped_count = defaultdict(int)
    for value, scene in zip(values, scenes):
        grouped_sum[str(scene)] += float(value)
        grouped_count[str(scene)] += 1
    scene_names = sorted(grouped_sum)
    sums = np.asarray([grouped_sum[name] for name in scene_names], dtype=np.float64)
    counts = np.asarray([grouped_count[name] for name in scene_names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        stop = min(start + 1000, replicates)
        sampled = rng.integers(0, len(scene_names), size=(stop - start, len(scene_names)))
        estimates[start:stop] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "mean": float(np.asarray(values, dtype=np.float64).mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "bootstrap_unit": "scene",
        "num_scenes": len(scene_names),
        "replicates": replicates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--certification-cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    args = parser.parse_args()

    selection_path = args.selection.expanduser().resolve()
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "PASS" or selection.get("schema_version") != 3:
        raise RuntimeError("V3 selection did not pass")
    if selection.get("certification_consumed"):
        raise RuntimeError("selection already records certification consumption")
    selected = selection["selected"]
    state_path = Path(selected["scene_selector_state"]).resolve()
    if common.sha256_file(state_path) != selected["scene_selector_state_sha256"]:
        raise RuntimeError("selected V3 state SHA256 mismatch")
    state_payload = torch.load(state_path, map_location="cpu")

    pairs = [common.load_cache(path, "certification") for path in args.certification_cache]
    pairs.sort(key=lambda pair: int(pair[1]["noise_seed"]))
    caches, manifests = zip(*pairs)
    if [int(manifest["noise_seed"]) for manifest in manifests] != [6, 7, 8]:
        raise RuntimeError("certification caches must contain noise seeds 6,7,8")
    if any(cache["tokens"] != caches[0]["tokens"] for cache in caches[1:]):
        raise RuntimeError("certification tokens drifted across noise seeds")
    if any(cache["scenes"] != caches[0]["scenes"] for cache in caches[1:]):
        raise RuntimeError("certification scenes drifted across noise seeds")

    model, config = common.model_from_cache(caches[0], state_payload["ablation"])
    if config != state_payload["scene_selector_config"]:
        raise RuntimeError("selected/certification scene-selector config drifted")
    model.load_state_dict(state_payload["scene_selector_state"], strict=True)
    device = torch.device(args.device)
    model = model.to(device)
    values_by_seed = [
        common.evaluate(
            model, cache, device, float(selected["temperature"]), args.batch_size
        )
        for cache in caches
    ]
    pooled = {
        key: torch.cat([values[key] for values in values_by_seed])
        for key in values_by_seed[0]
    }
    pooled_scenes = []
    for cache in caches:
        pooled_scenes.extend(cache["scenes"])
    confidence_intervals = {
        "top1_reward_gain": bootstrap_scene_ci(
            pooled["top1_reward_gain"].numpy(),
            pooled_scenes,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
    }
    for index, name in enumerate(common.COMPONENT_NAMES):
        confidence_intervals[f"delta_{name}"] = bootstrap_scene_ci(
            pooled["component_delta"][:, index].numpy(),
            pooled_scenes,
            args.bootstrap_replicates,
            args.bootstrap_seed + index + 1,
        )
    report = {
        "schema_version": 3,
        "status": "PASS",
        "method": selection.get("method", "scene_conditioned_exact_group_grpo"),
        "selection": str(selection_path),
        "selection_sha256": common.sha256_file(selection_path),
        "certification_consumed": True,
        "certification_seeds": [6, 7, 8],
        "certification_tokens": len(caches[0]["tokens"]),
        "certification_scenes": len(set(caches[0]["scenes"])),
        "metrics": common.summarize(pooled),
        "metrics_by_noise_seed": {
            str(manifest["noise_seed"]): common.summarize(values)
            for manifest, values in zip(manifests, values_by_seed)
        },
        "confidence_intervals": confidence_intervals,
        "scientific_gate": {
            "mean_top1_gain_positive": confidence_intervals["top1_reward_gain"]["mean"] > 0.0,
            "lower_95_top1_gain_positive": confidence_intervals["top1_reward_gain"]["lower_95"] > 0.0,
        },
        "scene_selector_state": str(state_path),
        "scene_selector_state_sha256": common.sha256_file(state_path),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
