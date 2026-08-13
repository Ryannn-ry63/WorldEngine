#!/usr/bin/env python3
"""Freeze scene-disjoint splits for DiffusionDrive base-policy rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


def parse_seed_root(value):
    try:
        seed_text, root_text = value.split("=", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("record root must be SEED=PATH") from error
    return seed, Path(root_text).expanduser().resolve()


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


def load_identity(path):
    with path.open("rb") as stream:
        row = pickle.load(stream)
    if row.get("schema_version") != 2 or row.get("record_type") != (
        "diffusiondrive_closed_loop_candidate_reward"
    ):
        raise RuntimeError(f"invalid rollout record: {path}")
    scene = str(row["rollout_scene_id"])
    step = int(row["worldengine_step"])
    return scene, step, str(row["checkpoint_sha256"])


def scene_split(scene):
    digest = hashlib.sha256(
        ("diffusiondrive-rollout-v1:" + scene).encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    if bucket < 8500:
        return "train"
    if bucket < 9400:
        return "development"
    return "certification"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-root", action="append", type=parse_seed_root, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-scenes", type=int, default=1)
    args = parser.parse_args()
    roots = dict(args.record_root)
    if len(roots) != len(args.record_root):
        raise RuntimeError("duplicate rollout seed")

    identities_by_seed = {}
    checkpoint_shas = set()
    for seed, root in sorted(roots.items()):
        identities = {}
        for path in record_paths(root):
            scene, step, checkpoint_sha = load_identity(path)
            key = (scene, step)
            if key in identities:
                raise RuntimeError(f"duplicate rollout identity seed={seed}: {key}")
            identities[key] = str(path.resolve())
            checkpoint_shas.add(checkpoint_sha)
        identities_by_seed[seed] = identities
    if len(checkpoint_shas) != 1:
        raise RuntimeError("rollout seeds used different base checkpoints")
    first_seed = min(identities_by_seed)
    reference_keys = set(identities_by_seed[first_seed])
    for seed, identities in identities_by_seed.items():
        if set(identities) != reference_keys:
            missing = len(reference_keys - set(identities))
            extra = len(set(identities) - reference_keys)
            raise RuntimeError(
                f"rollout coverage drifted seed={seed}: missing={missing} extra={extra}"
            )

    scenes = sorted({scene for scene, _ in reference_keys})
    if len(scenes) < args.minimum_scenes:
        raise RuntimeError(f"only {len(scenes)} scenes; expected >= {args.minimum_scenes}")
    assignments = {scene: scene_split(scene) for scene in scenes}
    split_scenes = {
        split: [scene for scene in scenes if assignments[scene] == split]
        for split in ("train", "development", "certification")
    }
    if len(scenes) >= 20 and any(not values for values in split_scenes.values()):
        raise RuntimeError("hash split unexpectedly produced an empty scene partition")
    split_records = {
        split: sum(1 for scene, _ in reference_keys if assignments[scene] == split)
        for split in split_scenes
    }
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "method": "sha256_scene_disjoint_85_9_6",
        "source_kind": "base_policy_rollout",
        "base_checkpoint_sha256": next(iter(checkpoint_shas)),
        "rollout_seeds": sorted(roots),
        "num_scenes": len(scenes),
        "num_records_per_seed": len(reference_keys),
        "scene_assignments": assignments,
        "split_scenes": split_scenes,
        "split_record_counts_per_seed": split_records,
        "record_roots": {str(seed): str(root) for seed, root in sorted(roots.items())},
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
