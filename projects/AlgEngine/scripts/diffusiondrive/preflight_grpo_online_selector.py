#!/usr/bin/env python3
"""Fail-fast contract check for DiffusionDrive online selector GRPO."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mmcv import Config
from mmcv.runner import build_optimizer, load_checkpoint
from mmdet3d.models import build_detector
from navsim.common.dataloader import MetricCacheLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    cfg = Config.fromfile(str(args.config.resolve()))
    checkpoint_path = Path(cfg.load_from).expanduser().resolve()
    expected_sha = cfg.model.planning_head.reference_checkpoint_sha256
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_sha = sha256(checkpoint_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"baseline SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    if Path(cfg.model.planning_head.reference_checkpoint).resolve() != checkpoint_path:
        raise RuntimeError("pi_current and pi_ref are not initialized from one baseline")

    metric_cache_path = Path(
        cfg.model.planning_head.online_reward.metric_cache_path
    ).expanduser().resolve()
    loader = MetricCacheLoader(metric_cache_path)
    if not loader.metric_cache_paths:
        raise RuntimeError(f"empty metric cache: {metric_cache_path}")

    collected_keys = cfg.data.train.pipeline[-1]["keys"]
    forbidden = {
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "score",
    }
    leaked = sorted(forbidden.intersection(collected_keys))
    if leaked:
        raise RuntimeError("fixed-vocabulary PDM keys leaked into pipeline: " + str(leaked))
    if not cfg.data.train.online_candidate_reward:
        raise RuntimeError("dataset online_candidate_reward must be enabled")
    if cfg.runner.type != "EpochBasedRunner":
        raise RuntimeError("formal selector GRPO must use EpochBasedRunner")
    if cfg.total_epochs != 8 or cfg.runner.max_epochs != 8:
        raise RuntimeError("formal selector GRPO must run exactly 8 epochs")
    if cfg.data.samples_per_gpu != 1:
        raise RuntimeError("formal selector GRPO requires batch size 1 per GPU")
    nav_filter_path = Path(cfg.data.train.nav_filter_path).expanduser().resolve()
    if not nav_filter_path.is_file():
        raise FileNotFoundError(nav_filter_path)
    contract = cfg.selector_reward_contract
    if contract.fixed_vocabulary_size is not None:
        raise RuntimeError("formal selector GRPO must reject fixed vocabularies")
    if not contract.generator_frozen:
        raise RuntimeError("formal selector GRPO generator must remain frozen")
    if not contract.pi_old_equals_pi_ref:
        raise RuntimeError("canonical selector GRPO requires pi_old == pi_ref")
    if "old_policy_mode" in cfg.model.planning_head:
        raise RuntimeError("deprecated old_policy_mode leaked into canonical config")
    if contract.imitation_loss or contract.entropy_loss or contract.ranking_loss:
        raise RuntimeError("an auxiliary selector loss leaked into the formal config")
    if contract.trainable_selector_tensors != 10:
        raise RuntimeError("formal config must train exactly 10 selector tensors")


    model = build_detector(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    load_checkpoint(model, str(checkpoint_path), map_location="cpu", strict=False)
    head = model.planning_head
    reference_tensor_count = head.initialize_reference_selector()
    current_state = head._current_selector().state_dict()
    reference_state = head.reference_selector.state_dict()
    max_reference_error = max(
        float((current_state[key] - reference_state[key]).abs().max())
        for key in current_state
    )
    if max_reference_error != 0.0:
        raise RuntimeError(
            f"pi_current/pi_ref baseline initialization drift: {max_reference_error}"
        )

    if any("old_selector" in key for key in model.state_dict()):
        raise RuntimeError("deprecated pi_old snapshot leaked into state_dict")

    optimizer = build_optimizer(model, cfg.optimizer)
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if len(trainable) != 10 or optimized_ids != trainable_ids:
        raise RuntimeError(
            f"expected exactly 10 current-selector tensors, got {len(trainable)}"
        )
    if not all(
        name.startswith(
            "planning_head.diff_decoder.layers.1.task_decoder.plan_cls_branch."
        )
        for name in trainable
    ):
        raise RuntimeError("a non-final-selector tensor is trainable")

    report = {
        "status": "PASS",
        "config": str(args.config.resolve()),
        "baseline_checkpoint": str(checkpoint_path),
        "baseline_sha256": actual_sha,
        "metric_cache": str(metric_cache_path),
        "metric_cache_tokens": len(loader.metric_cache_paths),
        "dynamic_candidates": cfg.model.planning_head.num_anchors,
        "reference_selector_tensors": reference_tensor_count,
        "trainable_current_selector_tensors": len(trainable),
        "max_reference_init_error": max_reference_error,
        "pi_old_equals_pi_ref": True,
        "max_epochs": cfg.runner.max_epochs,
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
