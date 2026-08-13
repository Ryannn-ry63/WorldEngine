#!/usr/bin/env python3
"""Fail-closed audit for one merged DiffusionDrive rollout seed."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def find_records(root):
    candidate = root / "WE_output/openscene_format/diffusiondrive_rollout_records"
    records = sorted(candidate.glob("*_reward.pkl")) if candidate.is_dir() else []
    if not records:
        raise RuntimeError(f"no merged rollout records under {root}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-noise-namespace", required=True)
    parser.add_argument("--minimum-records", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.rollout_root.expanduser().resolve()
    seen = set()
    scenes = Counter()
    selected_rewards = []
    oracle_rewards = []
    parity_errors = []
    config_shas = set()
    code_shas = set()
    for path in find_records(root):
        with path.open("rb") as stream:
            row = pickle.load(stream)
        if row.get("schema_version") != 2 or row.get("record_type") != (
            "diffusiondrive_closed_loop_candidate_reward"
        ):
            raise RuntimeError(f"record schema drifted: {path}")
        if row.get("checkpoint_sha256") != args.expected_checkpoint_sha256:
            raise RuntimeError(f"checkpoint provenance drifted: {path}")
        if row.get("candidate_noise_namespace") != args.expected_noise_namespace:
            raise RuntimeError(f"noise namespace drifted: {path}")
        scene = str(row["rollout_scene_id"])
        identity = (scene, int(row["worldengine_step"]))
        if identity in seen:
            raise RuntimeError(f"duplicate logical rollout frame: {identity}")
        seen.add(identity)
        scenes[scene] += 1
        rewards = np.asarray(row["candidate_rewards"], dtype=np.float32)
        components = np.asarray(row["candidate_reward_components"], dtype=np.float32)
        valid = np.asarray(row["candidate_reward_valid_mask"], dtype=np.bool_)
        if rewards.shape != (20,) or components.shape != (20, 6) or valid.shape != (20,):
            raise RuntimeError(f"candidate reward shape drifted: {path}")
        if not valid.all() or not np.isfinite(rewards).all() or not np.isfinite(components).all():
            raise RuntimeError(f"invalid candidate reward: {path}")
        selected = int(row["selected_index"])
        selected_rewards.append(float(rewards[selected]))
        oracle_rewards.append(float(rewards.max()))
        parity_errors.append(float(row["deployed_candidate_parity_max_abs_error"]))
        config_shas.add(str(row["resolved_config_sha256"]))
        code_shas.add(str(row.get("code_sha")))
    if len(seen) < args.minimum_records:
        raise RuntimeError(f"only {len(seen)} rollout records; expected >= {args.minimum_records}")
    if len(config_shas) != 1 or len(code_shas) != 1 or "None" in code_shas:
        raise RuntimeError("config/code provenance drifted within rollout")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "source_kind": "base_policy_rollout",
        "checkpoint_sha256": args.expected_checkpoint_sha256,
        "noise_namespace": args.expected_noise_namespace,
        "resolved_config_sha256": next(iter(config_shas)),
        "code_sha": next(iter(code_shas)),
        "num_records": len(seen),
        "num_scenes": len(scenes),
        "frames_per_scene": dict(sorted(scenes.items())),
        "mean_selected_reward": float(np.mean(selected_rewards)),
        "mean_oracle_reward": float(np.mean(oracle_rewards)),
        "mean_oracle_headroom": float(
            np.mean(np.asarray(oracle_rewards) - np.asarray(selected_rewards))
        ),
        "maximum_deployed_candidate_parity_error": max(parity_errors),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
