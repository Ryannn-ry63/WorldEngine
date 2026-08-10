#!/usr/bin/env python3
"""Parity and timing check for DiffusionDrive online NAVSIM-v1 rewards."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)

from mmdet3d_plugin.navformer.dense_heads.diffusiondrive_online_pdm_reward import (
    OnlineDiffusionDrivePDMReward,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def deterministic_candidates(num_candidates):
    time_s = np.arange(1, 9, dtype=np.float32) * 0.5
    candidates = np.zeros((num_candidates, 8, 3), dtype=np.float32)
    for index in range(num_candidates):
        speed = 1.0 + 0.08 * index
        curvature = (index - (num_candidates - 1) / 2.0) * 0.002
        candidates[index, :, 0] = speed * time_s
        candidates[index, :, 1] = curvature * candidates[index, :, 0] ** 2
        candidates[index, :, 2] = np.arctan(
            2.0 * curvature * candidates[index, :, 0]
        )
    return candidates


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requested but CUDA is unavailable")
    if not args.metric_cache.is_dir():
        raise FileNotFoundError(args.metric_cache)

    loader = MetricCacheLoader(args.metric_cache)
    token = args.token or sorted(loader.metric_cache_paths)[0]
    if token not in loader.metric_cache_paths:
        raise KeyError(f"metric cache does not contain token {token}")
    metric_cache = loader.get_from_token(token)
    candidates = deterministic_candidates(args.num_candidates)
    candidate_tensor = torch.as_tensor(
        candidates, device=torch.device(args.device)
    ).unsqueeze(0)

    online_reward = OnlineDiffusionDrivePDMReward(
        metric_cache_path=str(args.metric_cache),
        num_candidates=args.num_candidates,
        use_cuda_simulator=args.device == "cuda",
    )
    start = time.perf_counter()
    actual = online_reward(candidate_tensor, [token])
    batched_seconds = time.perf_counter() - start

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=sampling)
    scorer = PDMScorer(proposal_sampling=sampling)
    expected_scores = []
    expected_components = []
    start = time.perf_counter()
    for candidate in candidates:
        result = pdm_score(
            metric_cache,
            Trajectory(candidate),
            sampling,
            simulator,
            scorer,
        )
        expected_scores.append(result.score)
        expected_components.append(
            [
                result.no_at_fault_collisions,
                result.drivable_area_compliance,
                result.ego_progress,
                result.time_to_collision_within_bound,
                result.comfort,
                result.driving_direction_compliance,
            ]
        )
    official_loop_seconds = time.perf_counter() - start

    actual_scores = actual["score"][0].cpu().numpy()
    actual_components = actual["components"][0].cpu().numpy()
    expected_scores = np.asarray(expected_scores, dtype=np.float32)
    expected_components = np.asarray(expected_components, dtype=np.float32)
    score_error = float(np.max(np.abs(actual_scores - expected_scores)))
    component_error = float(
        np.max(np.abs(actual_components - expected_components))
    )
    valid = bool(actual["valid_mask"].all().item())
    status = valid and score_error <= args.atol and component_error <= args.atol
    report = {
        "status": "PASS" if status else "FAIL",
        "token": token,
        "device": args.device,
        "num_candidates": args.num_candidates,
        "max_score_abs_error": score_error,
        "max_component_abs_error": component_error,
        "atol": args.atol,
        "batched_seconds": batched_seconds,
        "official_loop_seconds": official_loop_seconds,
        "speedup": official_loop_seconds / max(batched_seconds, 1e-12),
    }
    print(json.dumps(report, sort_keys=True))
    if not status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
