#!/usr/bin/env python3
"""Join rollout records with same-state PDM reward sidecars."""

import argparse
import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np


TRACE_KEYS = (
    "grpo_initial_sample",
    "grpo_transition_action",
    "grpo_final_action",
    "grpo_old_log_probs",
)


def load_object(path):
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    with open(path, "rb") as stream:
        return pickle.load(stream)

def collect_annotations(rollout_root):
    """Merge worker metadata and make sensor paths independent of img_root."""
    annotations = {}
    pattern = "WE_output/openscene_format/meta_datas/*.pkl"
    for metadata_path in sorted(rollout_root.rglob(pattern)):
        payload = load_object(metadata_path)
        infos = payload.get("infos", []) if isinstance(payload, dict) else payload
        sensor_root = metadata_path.parent.parent / "sensor_blobs"
        for info in infos:
            info = dict(info)
            info["lidar_path"] = str(sensor_root / info["lidar_path"])
            info["cams"] = {
                name: dict(camera)
                for name, camera in info["cams"].items()
            }
            for camera in info["cams"].values():
                camera["data_path"] = str(sensor_root / camera["data_path"])
            annotations[info["token"]] = info
    if not annotations:
        raise FileNotFoundError("no rollout OpenScene metadata found")
    return list(annotations.values())


def validate_trace(model_result, state_key):
    initial = np.asarray(model_result["grpo_initial_sample"])
    transition = np.asarray(model_result["grpo_transition_action"])
    final = np.asarray(model_result["grpo_final_action"])
    old_log_probs = np.asarray(model_result["grpo_old_log_probs"])
    modes = final.shape[0]
    if initial.shape != (modes, 8, 2):
        raise ValueError("invalid initial GRPO sample shape for " + state_key)
    if transition.shape != initial.shape:
        raise ValueError("invalid GRPO transition shape for " + state_key)
    if final.shape != (modes, 8, 3):
        raise ValueError("invalid final GRPO action shape for " + state_key)
    if old_log_probs.shape != (modes, 2):
        raise ValueError("invalid old GRPO log-prob shape for " + state_key)
    if not all(np.isfinite(value).all() for value in (
        initial, transition, final, old_log_probs
    )):
        raise ValueError("non-finite GRPO trace for " + state_key)
    return modes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--reward-root", required=True)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rollout_root = Path(args.rollout_root).resolve()
    reward_root = Path(args.reward_root).resolve()
    index_files = sorted(rollout_root.rglob("rollout_index.csv"))
    if not index_files:
        raise FileNotFoundError("no rollout_index.csv under " + str(rollout_root))
    entries = []
    seen_states = set()
    for index_path in index_files:
        with open(index_path, newline="") as stream:
            for row in csv.DictReader(stream):
                record_path = Path(row["record_path"])
                if not record_path.is_absolute():
                    record_path = index_path.parent / record_path
                record = load_object(record_path)
                model_result = record.get("model_result", record)
                if any(key not in model_result for key in TRACE_KEYS):
                    continue
                state_key = f'{record["prefix"]}_{record["step"]}'
                record_version = str(record.get("policy_version", ""))
                if record_version != args.policy_version:
                    raise ValueError(
                        "record policy version mismatch for " + state_key
                    )
                modes = validate_trace(model_result, state_key)
                if state_key in seen_states:
                    raise ValueError("duplicate rollout state: " + state_key)
                reward_matches = (
                    list(reward_root.rglob("grpo_rewards/" + state_key + ".npz"))
                    + list(reward_root.rglob("grpo_rewards/" + state_key + ".pkl"))
                )
                if len(reward_matches) > 1:
                    raise ValueError(
                        "multiple reward sidecars found for " + state_key
                    )
                if not reward_matches:
                    raise FileNotFoundError("reward missing for " + state_key)
                reward_path = reward_matches[0]
                reward = load_object(reward_path)
                reward_version = str(reward.get("policy_version", ""))
                if reward_version != args.policy_version:
                    raise ValueError(
                        "reward policy version mismatch for " + state_key
                    )
                if str(reward.get("state_key", state_key)) != state_key:
                    raise ValueError("reward state mismatch for " + state_key)
                selected_index = int(
                    model_result.get("chosen_ind", record.get("plan_idx", -1))
                )
                reward_selected = int(
                    reward.get("selected_index", selected_index)
                )
                if reward_selected != selected_index:
                    raise ValueError(
                        "selected candidate mismatch for " + state_key
                    )
                score = np.asarray(reward.get("score"))
                if score.shape != (modes,):
                    raise ValueError("reward/trace mode mismatch for " + state_key)
                seen_states.add(state_key)
                entries.append({
                    "token": record["token"],
                    "prefix": record["prefix"],
                    "step": int(record["step"]),
                    "state_key": state_key,
                    "policy_version": args.policy_version,
                    "record_path": str(record_path.resolve()),
                    "reward_path": str(reward_path.resolve()),
                })
    if not entries:
        raise RuntimeError("no complete GRPO rollout/reward pairs found")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    annotations = collect_annotations(rollout_root)
    annotation_path = output.with_suffix(".annotations.pkl")
    annotation_tmp = annotation_path.with_suffix(annotation_path.suffix + ".tmp")
    with open(annotation_tmp, "wb") as stream:
        pickle.dump(
            {"infos": annotations},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(annotation_tmp, annotation_path)
    payload = {
        "policy_version": args.policy_version,
        "num_entries": len(entries),
        "rollout_root": str(rollout_root),
        "reward_root": str(reward_root),
        "ann_file": str(annotation_path),
        "entries": entries,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output)
    print(json.dumps({
        "policy_version": payload["policy_version"],
        "num_entries": payload["num_entries"],
    }))


if __name__ == "__main__":
    main()
