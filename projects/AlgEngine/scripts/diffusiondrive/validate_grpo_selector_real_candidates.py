#!/usr/bin/env python3
"""Validate batched PDM reward on real DiffusionDrive-generated candidates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d_plugin.datasets.builder import build_dataloader
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)


COMPONENT_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--nav-filter", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_scores(candidates, metric_cache):
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=sampling)
    scorer = PDMScorer(proposal_sampling=sampling)
    scores = []
    components = []
    for candidate in candidates:
        result = pdm_score(
            metric_cache,
            Trajectory(candidate),
            sampling,
            simulator,
            scorer,
        )
        scores.append(result.score)
        components.append(
            [
                result.no_at_fault_collisions,
                result.drivable_area_compliance,
                result.ego_progress,
                result.time_to_collision_within_bound,
                result.comfort,
                result.driving_direction_compliance,
            ]
        )
    return (
        np.asarray(scores, dtype=np.float32),
        np.asarray(components, dtype=np.float32),
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("real-candidate parity requires CUDA")
    config_path = args.config.expanduser().resolve()
    nav_filter = args.nav_filter.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for required in (config_path, nav_filter):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.num_tokens < 1:
        raise ValueError("num_tokens must be positive")

    cfg = Config.fromfile(str(config_path))
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    cfg.model.pretrained = None
    cfg.data.train.nav_filter_path = str(nav_filter)
    cfg.data.train.test_mode = False
    cfg.data.samples_per_gpu = 1
    dataset = build_dataset(cfg.data.train)
    if len(dataset) < args.num_tokens:
        raise RuntimeError(
            f"requested {args.num_tokens} tokens from dataset of {len(dataset)}"
        )
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    baseline = Path(cfg.load_from).expanduser().resolve()
    load_checkpoint(model, str(baseline), map_location="cpu")
    head = model.planning_head
    head.initialize_reference_selector()
    head.set_candidate_noise_namespace(
        f"formal_real_candidate_parity_seed{args.noise_seed}"
    )
    # Exercise the GRPO head's formal-training path while keeping perception,
    # GridMask, and the frozen generator deterministic. model.eval() disables
    # stochastic perception layers; head.train(True) alone dispatches online
    # candidate reward generation through _forward_train.
    model.eval()
    head.train(True)
    # Direct planning_head.forward(...) bypasses PyTorch forward hooks. Capture
    # the exact plan_results object consumed by the GRPO loss instead.
    captured = []
    original_loss = head.loss
    def capture_loss(result=None, *loss_args, **loss_kwargs):
        captured.append(result)
        return original_loss(result, *loss_args, **loss_kwargs)
    head.loss = capture_loss
    model = MMDataParallel(model.cuda(), device_ids=[0])
    metric_loader = MetricCacheLoader(
        Path(cfg.model.planning_head.online_reward.metric_cache_path)
    )

    max_score_error = 0.0
    max_component_error = 0.0
    worst_score = None
    worst_component = None
    checked_tokens = []
    with torch.no_grad():
        for data in loader:
            if len(checked_tokens) >= args.num_tokens:
                break
            model(return_loss=True, **data)
            if len(captured) != 1:
                raise RuntimeError("selector loss-capture count drifted")
            result = captured.pop()
            tokens = result["sample_tokens"]
            candidates = (
                result["candidate_trajectories_8"].detach().float().cpu().numpy()
            )
            actual_scores = (
                result["candidate_rewards"].detach().float().cpu().numpy()
            )
            actual_components = (
                result["candidate_reward_components"]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            for batch_index, token in enumerate(tokens):
                expected_scores, expected_components = official_scores(
                    candidates[batch_index],
                    metric_loader.get_from_token(str(token)),
                )
                score_error = np.abs(
                    actual_scores[batch_index] - expected_scores
                )
                component_error = np.abs(
                    actual_components[batch_index] - expected_components
                )
                local_score_max = float(score_error.max())
                local_component_max = float(component_error.max())
                if local_score_max > max_score_error:
                    max_score_error = local_score_max
                    index = int(score_error.argmax())
                    worst_score = {
                        "token": str(token),
                        "candidate_index": index,
                        "actual": float(actual_scores[batch_index, index]),
                        "official": float(expected_scores[index]),
                    }
                if local_component_max > max_component_error:
                    max_component_error = local_component_max
                    flat_index = int(component_error.argmax())
                    candidate_index, component_index = np.unravel_index(
                        flat_index, component_error.shape
                    )
                    worst_component = {
                        "token": str(token),
                        "candidate_index": int(candidate_index),
                        "component": COMPONENT_NAMES[int(component_index)],
                        "actual": float(
                            actual_components[
                                batch_index, candidate_index, component_index
                            ]
                        ),
                        "official": float(
                            expected_components[candidate_index, component_index]
                        ),
                    }
                checked_tokens.append(str(token))
                if len(checked_tokens) >= args.num_tokens:
                    break
    head.loss = original_loss

    unique_tokens = len(set(checked_tokens))
    status = (
        len(checked_tokens) == args.num_tokens
        and unique_tokens == args.num_tokens
        and max_score_error <= args.atol
        and max_component_error <= args.atol
    )
    report = {
        "schema_version": 1,
        "status": "PASS" if status else "FAIL",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "baseline_checkpoint": str(baseline),
        "baseline_sha256": sha256_file(baseline),
        "nav_filter": str(nav_filter),
        "nav_filter_sha256": sha256_file(nav_filter),
        "noise_seed": args.noise_seed,
        "num_tokens": len(checked_tokens),
        "unique_tokens": unique_tokens,
        "num_candidates_per_token": 20,
        "num_candidate_scores": 20 * len(checked_tokens),
        "atol": args.atol,
        "max_score_abs_error": max_score_error,
        "max_component_abs_error": max_component_error,
        "worst_score": worst_score,
        "worst_component": worst_component,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
