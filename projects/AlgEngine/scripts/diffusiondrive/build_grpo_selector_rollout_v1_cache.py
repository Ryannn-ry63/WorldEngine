#!/usr/bin/env python3
"""Build an immutable selector cache from closed-loop candidate rewards."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from rollout_v1_provenance import validate_rollout_provenance


COMPONENT_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selector_template(config_path, checkpoint_path):
    """Rebuild V3's selector contract without depending on an old cache."""
    from mmcv import Config

    cfg = Config.fromfile(str(config_path))
    head = cfg.model.planning_head
    selector_config = dict(head.scene_selector)
    num_layers = int(head.num_diff_decoder_layers)
    selector_layer = int(head.get("selector_layer", -1))
    if selector_layer < 0:
        selector_layer += num_layers
    if not 0 <= selector_layer < num_layers:
        raise RuntimeError("selector layer is invalid in rollout config")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint
    for key in ("state_dict", "model"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            source = checkpoint[key]
            break
    prefixes = (
        f"planning_head.diff_decoder.layers.{selector_layer}.task_decoder.plan_cls_branch.",
        f"module.planning_head.diff_decoder.layers.{selector_layer}.task_decoder.plan_cls_branch.",
    )
    selector_state = {}
    for key, value in source.items():
        prefix = next(
            (candidate for candidate in prefixes if key.startswith(candidate)), None
        )
        if prefix is not None:
            selector_state[key[len(prefix):]] = value.detach().cpu()
    if len(selector_state) != 10:
        raise RuntimeError(
            f"expected 10 immutable baseline selector tensors, got {len(selector_state)}"
        )
    return selector_state, selector_config


def record_paths(root):
    candidates = (
        root,
        root / "diffusiondrive_rollout_records",
        root / "WE_output/openscene_format/diffusiondrive_rollout_records",
    )
    for candidate in candidates:
        paths = sorted(candidate.glob("*_reward.pkl")) if candidate.is_dir() else []
        if paths:
            return paths
    raise RuntimeError(f"no DiffusionDrive rollout records found under {root}")


def checked_array(row, key, shape, finite=True):
    value = np.asarray(row[key])
    if value.shape != shape:
        raise RuntimeError(f"{key} shape {value.shape} != {shape}")
    if finite and not np.isfinite(value).all():
        raise RuntimeError(f"non-finite {key}")
    return value


def load_record(path, expected_sha, expected_namespace):
    with path.open("rb") as stream:
        row = pickle.load(stream)
    if row.get("schema_version") != 2 or row.get("record_type") != (
        "diffusiondrive_closed_loop_candidate_reward"
    ):
        raise RuntimeError(f"invalid rollout record: {path}")
    if row.get("checkpoint_sha256") != expected_sha:
        raise RuntimeError(f"base checkpoint drifted: {path}")
    namespace = str(row.get("candidate_noise_namespace"))
    if expected_namespace is not None and namespace != expected_namespace:
        raise RuntimeError(f"noise namespace drifted: {path}")
    values = {
        "feature": checked_array(row, "candidate_features", (20, 256)),
        "candidates": checked_array(row, "candidate_trajectories_8", (20, 8, 3)),
        "route_bev": checked_array(row, "route_bev_features", (20, 8, 256)),
        "status": checked_array(row, "status_tokens", (1, 256)),
        "ego": checked_array(row, "ego_queries", (1, 256)),
        "agents": checked_array(row, "agents_queries", (30, 256)),
        "rewards": checked_array(row, "candidate_rewards", (20,)),
        "components": checked_array(row, "candidate_reward_components", (20, 6)),
        "valid": checked_array(
            row, "candidate_reward_valid_mask", (20,), finite=False
        ).astype(np.bool_),
        "reference_logits": checked_array(row, "reference_logits", (20,)),
        "current_logits": checked_array(row, "current_logits", (20,)),
    }
    if not values["valid"].all():
        raise RuntimeError(f"invalid candidates: {path}")
    if float(np.max(np.abs(values["current_logits"] - values["reference_logits"]))) > 1e-5:
        raise RuntimeError(f"base rollout selector is not deployment-equivalent: {path}")
    if tuple(row.get("reward_component_names", ())) != COMPONENT_NAMES:
        raise RuntimeError(f"reward component order drifted: {path}")
    selected = int(row["selected_index"])
    if selected != int(np.argmax(values["reference_logits"])):
        raise RuntimeError(f"deployed selection/reference logits drifted: {path}")
    if float(row.get("deployed_candidate_parity_max_abs_error", np.inf)) > 1e-4:
        raise RuntimeError(f"deployed candidate parity failed: {path}")
    scene = str(row["rollout_scene_id"])
    step = int(row["worldengine_step"])
    return {
        "token": f"{scene}:{step:04d}",
        "scene": scene,
        "rollout_sample_id": str(row["source_sample_token"]),
        "planner_step": int(row["planner_step"]),
        "namespace": namespace,
        "config_sha256": str(row["config_sha256"]),
        "resolved_config_sha256": str(row["resolved_config_sha256"]),
        "code_sha": str(row["code_sha"]),
        "sidecar_path": str(row["sidecar_path"]),
        **values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development", "certification"), required=True)
    parser.add_argument("--rollout-seed", type=int, required=True)
    parser.add_argument("--noise-namespace", required=True)
    parser.add_argument("--selector-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-num-records", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.baseline_checkpoint.expanduser().resolve()
    if sha256_file(checkpoint) != args.expected_baseline_sha256:
        raise RuntimeError("baseline checkpoint SHA256 mismatch")
    split_path = args.split_manifest.expanduser().resolve()
    split_payload = json.loads(split_path.read_text())
    if split_payload.get("status") != "PASS" or split_payload.get("schema_version") != 1:
        raise RuntimeError("rollout split manifest did not pass")
    if split_payload.get("base_checkpoint_sha256") != args.expected_baseline_sha256:
        raise RuntimeError("split/checkpoint provenance drifted")
    assignments = split_payload["scene_assignments"]

    selector_config_path = args.selector_config.expanduser().resolve()
    if not selector_config_path.is_file():
        raise FileNotFoundError(selector_config_path)
    baseline_selector_state, scene_selector_config = load_selector_template(
        selector_config_path, checkpoint
    )

    rows = {}
    source_digest = hashlib.sha256()
    for path in record_paths(args.record_root.expanduser().resolve()):
        row = load_record(path, args.expected_baseline_sha256, args.noise_namespace)
        if assignments.get(row["scene"]) != args.split:
            continue
        if row["token"] in rows:
            raise RuntimeError(f"duplicate logical rollout token: {row['token']}")
        rows[row["token"]] = row
        source_digest.update(path.name.encode("utf-8"))
        source_digest.update(sha256_file(path).encode("ascii"))
    ordered = [rows[token] for token in sorted(rows)]
    if args.expected_num_records is not None and len(ordered) != args.expected_num_records:
        raise RuntimeError(
            f"rollout cache coverage {len(ordered)} != {args.expected_num_records}"
        )
    if not ordered:
        raise RuntimeError("rollout split produced no cache rows")
    provenance = validate_rollout_provenance(ordered)

    cache = {
        "schema_version": 3,
        "source_kind": "base_policy_rollout",
        "tokens": [row["token"] for row in ordered],
        "scenes": [row["scene"] for row in ordered],
        "rollout_sample_ids": [row["rollout_sample_id"] for row in ordered],
        "planner_steps": [row["planner_step"] for row in ordered],
        "candidate_features": torch.from_numpy(
            np.stack([row["feature"] for row in ordered])
        ).half(),
        "candidate_trajectories_8": torch.from_numpy(
            np.stack([row["candidates"] for row in ordered])
        ).float(),
        "route_bev_features": torch.from_numpy(
            np.stack([row["route_bev"] for row in ordered])
        ).half(),
        "status_tokens": torch.from_numpy(np.stack([row["status"] for row in ordered])).half(),
        "ego_queries": torch.from_numpy(np.stack([row["ego"] for row in ordered])).half(),
        "agents_queries": torch.from_numpy(np.stack([row["agents"] for row in ordered])).half(),
        "candidate_rewards": torch.from_numpy(np.stack([row["rewards"] for row in ordered])).float(),
        "candidate_reward_components": torch.from_numpy(
            np.stack([row["components"] for row in ordered])
        ).float(),
        "candidate_reward_valid_mask": torch.from_numpy(
            np.stack([row["valid"] for row in ordered])
        ).bool(),
        "reference_logits": torch.from_numpy(
            np.stack([row["reference_logits"] for row in ordered])
        ).float(),
        "baseline_selector_state": baseline_selector_state,
        "scene_selector_config": scene_selector_config,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "cache.pt"
    torch.save(cache, cache_path)
    manifest = {
        "schema_version": 3,
        "status": "PASS",
        "method": "closed_loop_diffusiondrive_context_cache",
        "source_kind": "base_policy_rollout",
        "split": args.split,
        "noise_seed": args.rollout_seed,
        "noise_namespace": args.noise_namespace,
        "num_tokens": len(ordered),
        "num_scenes": len(set(cache["scenes"])),
        "num_candidates": 20,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.expected_baseline_sha256,
        **provenance,
        "annotation_sha256": sha256_file(split_path),
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "selector_config": str(selector_config_path),
        "selector_config_sha256": sha256_file(selector_config_path),
        "record_root": str(args.record_root.expanduser().resolve()),
        "source_records_sha256": source_digest.hexdigest(),
        "cache": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "tensor_shapes": {
            key: list(value.shape) for key, value in cache.items() if torch.is_tensor(value)
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
