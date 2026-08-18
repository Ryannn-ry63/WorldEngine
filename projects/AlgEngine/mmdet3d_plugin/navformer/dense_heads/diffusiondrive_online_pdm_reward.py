"""Online NAVSIM-v1 PDM rewards for DiffusionDrive dynamic candidates.

This module scores exactly the trajectories produced by the frozen
DiffusionDrive generator. It never indexes a fixed trajectory vocabulary.

For one sample, the PDM reference trajectory and all generated candidates are
simulated together. CUDA execution uses WorldEngine batched TorchSimulator.
The official NAVSIM PDM scorer still operates on CPU map/occupancy objects. A
single scorer pass is post-processed with pairwise progress normalization so
that every candidate matches an independent official pdm_score call.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import (
    get_trajectory_as_array,
    transform_trajectory,
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)


def pairwise_official_scores(
    scorer: PDMScorer,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover official reference-vs-candidate scores from one scorer pass."""

    if scorer._multi_metrics is None or scorer._weighted_metrics is None:
        raise RuntimeError("PDMScorer must score proposals before aggregation")
    if scorer._progress_raw is None:
        raise RuntimeError("PDMScorer did not expose raw progress")

    multi = np.asarray(scorer._multi_metrics, dtype=np.float64)
    weighted = np.asarray(scorer._weighted_metrics, dtype=np.float64).copy()
    progress_raw = np.asarray(scorer._progress_raw, dtype=np.float64)
    if multi.shape[1] < 2:
        raise ValueError("expected one PDM reference and at least one candidate")

    multiplicative = multi.prod(axis=0)
    reference_progress = progress_raw[0]
    threshold = float(scorer._config.progress_distance_threshold)
    weights = np.asarray(
        scorer._config.weighted_metrics_array, dtype=np.float64
    )
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("PDM scorer metric weights must have positive sum")

    num_candidates = multi.shape[1] - 1
    scores = np.full(num_candidates, np.nan, dtype=np.float32)
    components = np.full((num_candidates, 6), np.nan, dtype=np.float32)

    for candidate_index in range(num_candidates):
        proposal_index = candidate_index + 1
        candidate_progress = progress_raw[proposal_index]
        progress_denominator = max(reference_progress, candidate_progress)
        if progress_denominator > threshold:
            normalized_progress = candidate_progress / progress_denominator
        else:
            normalized_progress = 1.0
        normalized_progress *= multiplicative[proposal_index]

        weighted[WeightedMetricIndex.PROGRESS, proposal_index] = (
            normalized_progress
        )
        weighted_score = float(
            np.dot(weighted[:, proposal_index], weights) / weight_sum
        )
        scores[candidate_index] = (
            multiplicative[proposal_index] * weighted_score
        )
        components[candidate_index] = np.asarray(
            [
                multi[MultiMetricIndex.NO_COLLISION, proposal_index],
                multi[MultiMetricIndex.DRIVABLE_AREA, proposal_index],
                normalized_progress,
                weighted[WeightedMetricIndex.TTC, proposal_index],
                weighted[WeightedMetricIndex.COMFORTABLE, proposal_index],
                weighted[WeightedMetricIndex.DRIVING_DIRECTION, proposal_index],
            ],
            dtype=np.float32,
        )

    return scores, components


class OnlineDiffusionDrivePDMReward(nn.Module):
    """Compute one NAVSIM-v1 reward per generated DiffusionDrive candidate."""

    component_names = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
    )

    def __init__(
        self,
        metric_cache_path: str,
        num_candidates: int = 20,
        metric_cache_lru_size: int = 128,
        fail_on_missing_cache: bool = True,
        use_cuda_simulator: bool = True,
    ):
        super().__init__()
        if num_candidates <= 1:
            raise ValueError("online PDM reward requires at least two candidates")
        if metric_cache_lru_size < 0:
            raise ValueError("metric_cache_lru_size must be non-negative")

        self.metric_cache_path = str(metric_cache_path)
        self.num_candidates = int(num_candidates)
        self.metric_cache_lru_size = int(metric_cache_lru_size)
        self.fail_on_missing_cache = bool(fail_on_missing_cache)
        self.use_cuda_simulator = bool(use_cuda_simulator)

        self._sampling = TrajectorySampling(
            num_poses=40, interval_length=0.1
        )
        self._scorer = PDMScorer(
            proposal_sampling=self._sampling,
            config=PDMScorerConfig(),
        )
        self._cpu_simulator = PDMSimulator(proposal_sampling=self._sampling)
        self._torch_simulator = None
        self._torch_simulator_device = None
        self._metric_cache_loader = None
        self._metric_cache_lru = OrderedDict()

    def _ensure_metric_cache_loader(self) -> MetricCacheLoader:
        if self._metric_cache_loader is None:
            cache_path = Path(self.metric_cache_path)
            if not cache_path.is_dir():
                raise FileNotFoundError(
                    f"NAVSIM metric cache does not exist: {cache_path}"
                )
            self._metric_cache_loader = MetricCacheLoader(cache_path)
        return self._metric_cache_loader

    def _get_metric_cache(self, token: str) -> Optional[Any]:
        if token in self._metric_cache_lru:
            value = self._metric_cache_lru.pop(token)
            self._metric_cache_lru[token] = value
            return value

        loader = self._ensure_metric_cache_loader()
        if token not in loader.metric_cache_paths:
            if self.fail_on_missing_cache:
                raise KeyError(f"NAVSIM metric cache missing token {token}")
            return None

        value = loader.get_from_token(token)
        if self.metric_cache_lru_size > 0:
            self._metric_cache_lru[token] = value
            while len(self._metric_cache_lru) > self.metric_cache_lru_size:
                self._metric_cache_lru.popitem(last=False)
        return value

    def clear_metric_cache_lru(self) -> None:
        self._metric_cache_lru.clear()

    def _ensure_torch_simulator(self, device: torch.device):
        if (
            self._torch_simulator is None
            or self._torch_simulator_device != device
        ):
            from worldengine.components.agents.policy.pdm_planner.simulation.torch_simulator import (
                TorchSimulator,
            )

            self._torch_simulator = TorchSimulator(
                proposal_sampling=self._sampling,
                dtype=torch.float64,
                device=device,
            )
            self._torch_simulator_device = device
        return self._torch_simulator

    def _trajectory_states(
        self,
        candidate_trajectories: torch.Tensor,
        metric_cache: Any,
    ) -> np.ndarray:
        initial_ego_state = metric_cache.ego_state
        reference_states = get_trajectory_as_array(
            metric_cache.trajectory,
            self._sampling,
            initial_ego_state.time_point,
        )

        candidate_numpy = (
            candidate_trajectories.detach().to(dtype=torch.float32).cpu().numpy()
        )
        candidate_states = []
        for candidate in candidate_numpy:
            model_trajectory = Trajectory(candidate)
            absolute_trajectory = transform_trajectory(
                model_trajectory, initial_ego_state
            )
            candidate_states.append(
                get_trajectory_as_array(
                    absolute_trajectory,
                    self._sampling,
                    initial_ego_state.time_point,
                )
            )
        return np.stack([reference_states, *candidate_states], axis=0)

    def _simulate(
        self,
        states: np.ndarray,
        initial_ego_state: Any,
        device: torch.device,
    ) -> np.ndarray:
        if self.use_cuda_simulator and device.type == "cuda":
            simulator = self._ensure_torch_simulator(device)
            torch_states = torch.as_tensor(
                states, dtype=torch.float64, device=device
            ).unsqueeze(0)
            simulated = simulator.simulate_proposals(
                torch_states, initial_ego_state, batch_sim=True
            )
            return simulated.squeeze(0).cpu().numpy()

        return self._cpu_simulator.simulate_proposals(
            states, initial_ego_state
        )

    def _score_one(
        self,
        candidates: torch.Tensor,
        token: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        metric_cache = self._get_metric_cache(token)
        if metric_cache is None:
            return None
        states = self._trajectory_states(candidates, metric_cache)
        simulated_states = self._simulate(
            states, metric_cache.ego_state, candidates.device
        )
        self._scorer.score_proposals(
            simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
        )
        return pairwise_official_scores(self._scorer)

    @torch.no_grad()
    def forward(
        self,
        candidate_trajectories: torch.Tensor,
        sample_tokens: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        if candidate_trajectories.ndim != 4:
            raise ValueError(
                "candidate trajectories must have shape (B, 20, 8, 3)"
            )
        batch_size, num_candidates, num_poses, pose_dim = (
            candidate_trajectories.shape
        )
        if num_candidates != self.num_candidates:
            raise ValueError(
                f"expected {self.num_candidates} dynamic candidates, "
                f"got {num_candidates}"
            )
        if (num_poses, pose_dim) != (8, 3):
            raise ValueError(
                "candidate trajectories must use 8 NAVSIM poses with (x,y,heading)"
            )
        if len(sample_tokens) != batch_size:
            raise ValueError(
                f"expected {batch_size} sample tokens, got {len(sample_tokens)}"
            )

        rewards = np.full(
            (batch_size, num_candidates), np.nan, dtype=np.float32
        )
        components = np.full(
            (batch_size, num_candidates, len(self.component_names)),
            np.nan,
            dtype=np.float32,
        )

        for batch_index, token in enumerate(sample_tokens):
            scored = self._score_one(
                candidate_trajectories[batch_index], str(token)
            )
            if scored is None:
                continue
            rewards[batch_index], components[batch_index] = scored

        reward_tensor = torch.as_tensor(
            rewards,
            device=candidate_trajectories.device,
            dtype=torch.float32,
        )
        component_tensor = torch.as_tensor(
            components,
            device=candidate_trajectories.device,
            dtype=torch.float32,
        )
        valid_mask = torch.isfinite(reward_tensor) & torch.isfinite(
            component_tensor
        ).all(dim=-1)
        return {
            "score": reward_tensor.detach(),
            "components": component_tensor.detach(),
            "valid_mask": valid_mask.detach(),
        }
