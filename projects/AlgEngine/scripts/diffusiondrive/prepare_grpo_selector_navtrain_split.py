#!/usr/bin/env python3
"""Create the immutable, scene-disjoint selector-GRPO navtrain split.

Only frames present in both the official navtrain filter and the NAVSIM-v1
metric cache are eligible.  Whole OpenScene logs (log_token/scene_token) are
assigned to train or calibration, so neighboring frames can never leak across
model fitting and checkpoint selection.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import mmcv
import yaml
from mmcv import Config
from navsim.common.dataloader import MetricCacheLoader


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/diffusiondrive/e2e_diffusiondrive_grpo_selector.py"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values):
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_filter_tokens(path):
    with path.open("r") as stream:
        payload = yaml.safe_load(stream)
    tokens = payload.get("tokens")
    if tokens is None:
        tokens = payload.get("scenario_tokens")
    if not tokens:
        raise RuntimeError(f"filter has no tokens: {path}")
    return {str(token) for token in tokens}


def split_scenes(scene_to_tokens, train_fraction, split_seed):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    ranked = sorted(
        scene_to_tokens,
        key=lambda scene: hashlib.sha256(
            f"{split_seed}:{scene}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ranked) < 2:
        raise RuntimeError("at least two cache-covered scenes are required")

    target = train_fraction * sum(len(scene_to_tokens[s]) for s in ranked)
    cumulative = 0
    best_prefix = 1
    best_error = float("inf")
    for prefix, scene in enumerate(ranked[:-1], start=1):
        cumulative += len(scene_to_tokens[scene])
        error = abs(cumulative - target)
        if error < best_error:
            best_prefix = prefix
            best_error = error

    train_scenes = set(ranked[:best_prefix])
    calibration_scenes = set(ranked[best_prefix:])
    return train_scenes, calibration_scenes


def yaml_payload(tokens):
    return {
        "_target_": "navsim.common.dataclasses.SceneFilter",
        "_convert_": "all",
        "tokens": sorted(tokens),
    }


def write_immutable(path, payload, serializer):
    rendered = serializer(payload)
    if path.exists():
        if path.read_text() != rendered:
            raise RuntimeError(
                f"refusing to overwrite non-identical immutable split file: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-filter", type=Path)
    parser.add_argument("--navtest-filter", type=Path)
    parser.add_argument("--metric-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--split-seed", type=int, default=20260808)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    cfg = Config.fromfile(str(config_path))
    algengine_root = config_path.parents[2]
    source_filter = (
        args.source_filter
        or algengine_root / "configs/navsim_splits/navtrain_split/navtrain.yaml"
    ).expanduser().resolve()
    navtest_filter = (
        args.navtest_filter
        or algengine_root / "configs/navsim_splits/navtest_split/navtest.yaml"
    ).expanduser().resolve()
    metric_cache = (
        args.metric_cache or Path(cfg.model.planning_head.online_reward.metric_cache_path)
    ).expanduser().resolve()
    annotation_file = Path(cfg.data.train.ann_file).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    for required in (source_filter, navtest_filter, annotation_file, metric_cache):
        if not required.exists():
            raise FileNotFoundError(required)

    source_tokens = load_filter_tokens(source_filter)
    navtest_tokens = load_filter_tokens(navtest_filter)
    cache_tokens = {
        str(token) for token in MetricCacheLoader(metric_cache).metric_cache_paths
    }

    loaded = mmcv.load(str(annotation_file), file_format="pkl")
    infos = loaded["infos"] if isinstance(loaded, dict) else loaded
    scene_to_tokens = defaultdict(list)
    annotation_tokens = set()
    for info in infos:
        token = str(info["token"])
        annotation_tokens.add(token)
        if token not in source_tokens or token not in cache_tokens:
            continue
        scene = str(info.get("scene_token") or info["log_token"])
        scene_to_tokens[scene].append(token)

    if not scene_to_tokens:
        raise RuntimeError("navtrain, annotations, and metric cache have no overlap")
    train_scenes, calibration_scenes = split_scenes(
        scene_to_tokens, args.train_fraction, args.split_seed
    )
    train_tokens = {
        token for scene in train_scenes for token in scene_to_tokens[scene]
    }
    calibration_tokens = {
        token for scene in calibration_scenes for token in scene_to_tokens[scene]
    }
    eligible_tokens = train_tokens | calibration_tokens

    if train_tokens & calibration_tokens:
        raise RuntimeError("train/calibration token leakage")
    if train_scenes & calibration_scenes:
        raise RuntimeError("train/calibration scene leakage")
    navtest_overlap = eligible_tokens & navtest_tokens
    if navtest_overlap:
        raise RuntimeError(
            f"eligible navtrain split overlaps navtest by {len(navtest_overlap)} tokens"
        )
    if eligible_tokens != (
        source_tokens & cache_tokens & annotation_tokens
    ):
        raise RuntimeError("eligible token accounting drifted")

    train_yaml = output_dir / "navtrain_grpo_train.yaml"
    calibration_yaml = output_dir / "navtrain_grpo_calibration.yaml"
    write_immutable(
        train_yaml,
        yaml_payload(train_tokens),
        lambda value: yaml.safe_dump(value, sort_keys=False),
    )
    write_immutable(
        calibration_yaml,
        yaml_payload(calibration_tokens),
        lambda value: yaml.safe_dump(value, sort_keys=False),
    )

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "split_unit": "log_token",
        "split_seed": args.split_seed,
        "requested_train_fraction": args.train_fraction,
        "actual_train_fraction": len(train_tokens) / len(eligible_tokens),
        "config": str(config_path),
        "annotation_file": str(annotation_file),
        "annotation_sha256": sha256_file(annotation_file),
        "source_filter": str(source_filter),
        "source_filter_sha256": sha256_file(source_filter),
        "navtest_filter": str(navtest_filter),
        "navtest_filter_sha256": sha256_file(navtest_filter),
        "metric_cache": str(metric_cache),
        "source_tokens": len(source_tokens),
        "cache_tokens": len(cache_tokens),
        "annotation_tokens": len(annotation_tokens),
        "eligible_tokens": len(eligible_tokens),
        "eligible_scenes": len(scene_to_tokens),
        "train_tokens": len(train_tokens),
        "train_scenes": len(train_scenes),
        "calibration_tokens": len(calibration_tokens),
        "calibration_scenes": len(calibration_scenes),
        "train_calibration_token_overlap": 0,
        "train_calibration_scene_overlap": 0,
        "navtest_token_overlap": 0,
        "eligible_tokens_sha256": sha256_lines(eligible_tokens),
        "train_tokens_sha256": sha256_lines(train_tokens),
        "train_scenes_sha256": sha256_lines(train_scenes),
        "calibration_tokens_sha256": sha256_lines(calibration_tokens),
        "calibration_scenes_sha256": sha256_lines(calibration_scenes),
        "train_filter": str(train_yaml),
        "calibration_filter": str(calibration_yaml),
    }
    audit_path = output_dir / "split_audit.json"
    write_immutable(
        audit_path,
        audit,
        lambda value: json.dumps(value, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
