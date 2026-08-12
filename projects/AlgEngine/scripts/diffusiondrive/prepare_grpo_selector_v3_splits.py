#!/usr/bin/env python3
"""Freeze scene-disjoint train/development/certification filters for V3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mmcv
import yaml


def load_tokens(path):
    payload = yaml.safe_load(path.read_text())
    tokens = payload.get("tokens") or payload.get("scenario_tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError(f"filter has no tokens: {path}")
    values = [str(token) for token in tokens]
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate filter tokens: {path}")
    return set(values)


def digest(values):
    text = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def render_filter(tokens):
    payload = {
        "_target_": "navsim.common.dataclasses.SceneFilter",
        "_convert_": "all",
        "tokens": sorted(tokens),
    }
    return yaml.safe_dump(payload, sort_keys=False)


def write_immutable(path, text):
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to overwrite immutable V3 split: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-filter", type=Path, required=True)
    parser.add_argument("--calibration-filter", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-fraction", type=float, default=0.60)
    parser.add_argument("--split-seed", type=int, default=20260811)
    args = parser.parse_args()

    train_filter = args.train_filter.expanduser().resolve()
    calibration_filter = args.calibration_filter.expanduser().resolve()
    annotation_file = args.annotation_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not 0.0 < args.development_fraction < 1.0:
        raise ValueError("development fraction must be in (0, 1)")
    for path in (train_filter, calibration_filter, annotation_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_tokens = load_tokens(train_filter)
    calibration_tokens = load_tokens(calibration_filter)
    if train_tokens & calibration_tokens:
        raise RuntimeError("upstream train/calibration token leakage")
    payload = mmcv.load(str(annotation_file), file_format="pkl")
    infos = payload["infos"] if isinstance(payload, dict) else payload
    all_tokens = train_tokens | calibration_tokens
    scene_to_tokens = defaultdict(set)
    token_to_scene = {}
    for info in infos:
        token = str(info["token"])
        if token not in all_tokens:
            continue
        scene = str(info.get("scene_token") or info["log_token"])
        token_to_scene[token] = scene
        if token in calibration_tokens:
            scene_to_tokens[scene].add(token)
    if set(token_to_scene) != all_tokens:
        missing = sorted(all_tokens - set(token_to_scene))
        raise RuntimeError(
            "annotation coverage drifted for train/calibration tokens: "
            + ", ".join(missing[:10])
        )
    train_scenes = {token_to_scene[token] for token in train_tokens}

    ranked_scenes = sorted(
        scene_to_tokens,
        key=lambda scene: hashlib.sha256(
            f"{args.split_seed}:{scene}".encode("utf-8")
        ).hexdigest(),
    )
    target = args.development_fraction * len(calibration_tokens)
    cumulative = 0
    best_prefix = 1
    best_error = float("inf")
    for prefix, scene in enumerate(ranked_scenes[:-1], start=1):
        cumulative += len(scene_to_tokens[scene])
        error = abs(cumulative - target)
        if error < best_error:
            best_prefix, best_error = prefix, error
    development_scenes = set(ranked_scenes[:best_prefix])
    certification_scenes = set(ranked_scenes[best_prefix:])
    development_tokens = {
        token for scene in development_scenes for token in scene_to_tokens[scene]
    }
    certification_tokens = calibration_tokens - development_tokens

    if not development_tokens or not certification_tokens:
        raise RuntimeError("development/certification split is empty")
    if development_scenes & certification_scenes:
        raise RuntimeError("development/certification scene leakage")
    if train_tokens & calibration_tokens:
        raise RuntimeError("training token leakage into held-out V3 splits")
    if train_scenes & development_scenes:
        raise RuntimeError("training scene leakage into V3 development split")
    if train_scenes & certification_scenes:
        raise RuntimeError("training scene leakage into V3 certification split")

    development_path = output_dir / "navtrain_grpo_development.yaml"
    certification_path = output_dir / "navtrain_grpo_certification.yaml"
    write_immutable(development_path, render_filter(development_tokens))
    write_immutable(certification_path, render_filter(certification_tokens))
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "split_seed": args.split_seed,
        "split_unit": "scene_token_or_log_token",
        "train_tokens": len(train_tokens),
        "train_scenes": len(train_scenes),
        "development_tokens": len(development_tokens),
        "certification_tokens": len(certification_tokens),
        "development_scenes": len(development_scenes),
        "certification_scenes": len(certification_scenes),
        "train_heldout_token_overlap": 0,
        "train_development_scene_overlap": 0,
        "train_certification_scene_overlap": 0,
        "development_certification_token_overlap": 0,
        "development_certification_scene_overlap": 0,
        "train_tokens_sha256": digest(train_tokens),
        "train_scenes_sha256": digest(train_scenes),
        "development_tokens_sha256": digest(development_tokens),
        "certification_tokens_sha256": digest(certification_tokens),
        "development_scenes_sha256": digest(development_scenes),
        "certification_scenes_sha256": digest(certification_scenes),
        "train_filter": str(train_filter),
        "calibration_filter": str(calibration_filter),
        "annotation_file": str(annotation_file),
        "development_filter": str(development_path),
        "certification_filter": str(certification_path),
    }
    write_immutable(
        output_dir / "split_audit_v3.json",
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
