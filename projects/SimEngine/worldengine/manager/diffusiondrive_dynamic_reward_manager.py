"""Closed-loop PDM reward for DiffusionDrive's dynamic 20-action set.

This manager is opt-in and intentionally does not load the shared 8192-path
vocabulary.  At each scored closed-loop frame it reads the exact candidates
exported by AlgEngine, simulates the PDM reference and all 20 candidates in one
batch, applies official pairwise progress normalization, and writes one
self-contained selector-training record.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import (
    TrajectorySampling,
)

from worldengine.components.agents.policy.pdm_planner.pdm_closed_planner import (
    PDMClosedPlanner,
)
from worldengine.components.agents.policy.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from worldengine.components.agents.policy.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PROGRESS_DISTANCE_THRESHOLD,
    WEIGHTED_METRICS_WEIGHTS,
)
from worldengine.components.agents.policy.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from worldengine.components.agents.policy.pdm_planner.simulation.torch_simulator import (
    TorchSimulator,
)
from worldengine.components.agents.policy.pdm_planner.utils.WE2pdm_utils import (
    WE2NuPlanConverter,
)
from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import (
    PDMDrivableMap,
)
from worldengine.manager.base_manager import BaseManager
from worldengine.manager.dense_reward_manager import DenseRewardManager


logger = logging.getLogger(__name__)

COMPONENT_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def should_load_rollout_sidecar(current_step, num_history, buffer_size):
    """Score only after the first planner action can have been published."""
    return num_history <= current_step < num_history + buffer_size - 1


def pairwise_official_scores(scorer: PDMScorer):
    """Recover official reference-vs-candidate scores from one scorer pass."""
    if scorer._multi_metrics is None or scorer._weighted_metrics is None:
        raise RuntimeError("PDM scorer did not expose metric arrays")
    if scorer._progress_raw is None:
        raise RuntimeError("PDM scorer did not expose raw progress")
    multi = np.asarray(scorer._multi_metrics, dtype=np.float64)
    weighted = np.asarray(scorer._weighted_metrics, dtype=np.float64).copy()
    progress_raw = np.asarray(scorer._progress_raw, dtype=np.float64)
    if multi.shape[1] != 21:
        raise RuntimeError(f"expected reference + 20 candidates, got {multi.shape[1]}")

    multiplicative = multi.prod(axis=0)
    reference_progress = progress_raw[0]
    threshold = float(PROGRESS_DISTANCE_THRESHOLD)
    weights = np.asarray(WEIGHTED_METRICS_WEIGHTS, dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise RuntimeError("PDM scorer weights have non-positive sum")

    scores = np.empty(20, dtype=np.float32)
    components = np.empty((20, len(COMPONENT_NAMES)), dtype=np.float32)
    for candidate_index in range(20):
        proposal_index = candidate_index + 1
        candidate_progress = progress_raw[proposal_index]
        denominator = max(reference_progress, candidate_progress)
        if denominator > threshold:
            normalized_progress = candidate_progress / denominator
        else:
            normalized_progress = 1.0
        normalized_progress *= multiplicative[proposal_index]
        weighted[WeightedMetricIndex.PROGRESS, proposal_index] = normalized_progress
        weighted_score = float(
            np.dot(weighted[:, proposal_index], weights) / weight_sum
        )
        scores[candidate_index] = multiplicative[proposal_index] * weighted_score
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


def expand_candidates_to_40(candidates_8: np.ndarray) -> np.ndarray:
    """Match ``DiffusionPlanningHead._expand_to_40`` exactly in NumPy."""
    candidates_8 = np.asarray(candidates_8, dtype=np.float64)
    if candidates_8.shape != (20, 8, 3):
        raise ValueError(f"candidate shape {candidates_8.shape} != (20, 8, 3)")
    start = np.concatenate(
        [np.zeros((20, 1, 3), dtype=np.float64), candidates_8[:, :-1]], axis=1
    )
    fractions = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64).reshape(1, 1, 4, 1)
    start_xy = start[..., :2][:, :, None, :]
    target_xy = candidates_8[..., :2][:, :, None, :]
    xy = start_xy + fractions * (
        target_xy - start_xy
    )
    heading_delta = candidates_8[..., 2] - start[..., 2]
    heading_delta = np.arctan2(np.sin(heading_delta), np.cos(heading_delta))
    heading = start[..., 2, None] + fractions[..., 0] * heading_delta[..., None]
    heading = np.arctan2(np.sin(heading), np.cos(heading))[..., None]
    intermediate = np.concatenate([xy, heading], axis=-1)
    segments = np.concatenate([intermediate, candidates_8[:, :, None, :]], axis=2)
    return segments.reshape(20, 40, 3)


class DiffusionDriveDynamicRewardManager(DenseRewardManager):
    """Opt-in DenseRewardManager variant for a dynamic 20-trajectory set."""

    PRIORITY = 20

    def __init__(self):
        # Do not call DenseRewardManager.__init__: it allocates the 8192-path
        # vocabulary and its IL target, neither of which belongs to this method.
        BaseManager.__init__(self)
        if not self.engine.global_config.get(
            'diffusiondrive_candidate_sidecar_path'
        ):
            raise ValueError(
                "diffusiondrive_candidate_sidecar_path is required when "
                "dynamic candidate reward is enabled"
            )
        self._future_sampling = TrajectorySampling(
            num_poses=self.engine.global_config['reward_sampling_poses'] * 5 + 1,
            interval_length=0.1,
        )
        self._proposal_sampling = TrajectorySampling(
            num_poses=self.engine.global_config['reward_sampling_poses'] * 5,
            interval_length=0.1,
        )
        self._map_radius = 100
        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=self._future_sampling,
            proposal_sampling=self._proposal_sampling,
            idm_policies=BatchIDMPolicy(
                speed_limit_fraction=[0.15, 0.25, 0.5, 0.75, 1.0],
                fallback_target_velocity=30.0,
                min_gap_to_lead_agent=10.0,
                headway_time=1.5,
                accel_max=4.5,
                decel_max=3.0,
            ),
            lateral_offsets=[-3.5, 3.5],
            map_radius=self._map_radius,
        )
        self.num_history = self.engine.global_config['num_history']
        self.num_future = self.engine.global_config['num_future']
        self.buffer_size = self.engine.global_config['reward_buffer_size']
        self.current_scene = None
        self.converter = None
        self.ego_states_list = []
        self.observations_list = []
        self.detection_tracks_list = []
        self.traffic_light_data_list = []
        self.openloop_pdm_scores_list = []
        self.all_scene_averages = []
        self._cached_roadblock_ids = []
        self.route_roadblock_dict = {}
        self.pkl_paths_df = pd.DataFrame(
            columns=["scene", "step", "planner_step", "pkl_path"]
        )
        self.use_cuda = torch.cuda.is_available()

    def _scene_prefixes(self):
        values = []
        for key in ("id", "token"):
            value = self.current_scene.get(key)
            if value:
                value = str(value)
                values.extend((value, value.rsplit("-", 1)[-1]))
        return tuple(dict.fromkeys(values))

    def _load_sidecar(self):
        root = Path(
            self.engine.global_config['diffusiondrive_candidate_sidecar_path']
        ).expanduser()
        timeout = float(
            self.engine.global_config.get('diffusiondrive_sidecar_timeout_s', 30.0)
        )
        deadline = time.monotonic() + timeout
        planner_steps = (self.current_step + 1, self.current_step)
        prefixes = self._scene_prefixes()
        while True:
            for planner_step in planner_steps:
                matches = []
                for prefix in prefixes:
                    path = root / f"{prefix}_{planner_step}.pkl"
                    if path.is_file():
                        matches.append(path)
                matches = list(dict.fromkeys(matches))
                if len(matches) == 1:
                    with matches[0].open("rb") as stream:
                        payload = pickle.load(stream)
                    if payload.get("schema_version") != 1:
                        raise RuntimeError(f"sidecar schema drifted: {matches[0]}")
                    if payload.get("record_type") != "diffusiondrive_closed_loop_candidate_context":
                        raise RuntimeError(f"sidecar type drifted: {matches[0]}")
                    if int(payload.get("planner_step", -1)) != planner_step:
                        raise RuntimeError(f"sidecar planner step drifted: {matches[0]}")
                    return payload, matches[0]
                if len(matches) > 1:
                    raise RuntimeError(f"ambiguous DiffusionDrive sidecars: {matches}")
            if time.monotonic() >= deadline:
                raise FileNotFoundError(
                    f"no DiffusionDrive sidecar for prefixes={prefixes} "
                    f"planner_steps={planner_steps} under {root}"
                )
            time.sleep(0.1)

    def _score_candidates(self, pdm_closed_trajectory, candidates_8):
        initial_ego_state = self.ego_states_list[self.current_step]
        initial_observation = self.observations_list[self.current_step]
        self._pdm_closed._load_route_dicts(self._cached_roadblock_ids)
        self._pdm_closed._observation.update(
            initial_ego_state,
            initial_observation,
            self.planner_input.traffic_light_data,
            self._pdm_closed._route_lane_dict,
        )
        self._pdm_closed._drivable_area_map = PDMDrivableMap.from_simulation(
            self._map_api, initial_ego_state, self._map_radius
        )
        observations = self._interpolate_observations(
            self.observations_list[
                self.current_step : self.current_step + self.buffer_size
            ]
        )
        traffic_lights = [[] for _ in observations]
        self._pdm_closed._observation.update_detections_tracks(
            observations,
            traffic_lights,
            self._pdm_closed._route_lane_dict,
            compute_traffic_light_data=True,
        )
        current_lane = self._pdm_closed._get_starting_lane(initial_ego_state)
        self._centerline = PDMPath(
            self._pdm_closed._get_discrete_centerline(current_lane)
        )

        local_40 = expand_candidates_to_40(candidates_8)
        global_40 = self.batched_global_transform(
            local_40,
            initial_ego_state.rear_axle.array,
            initial_ego_state.rear_axle.heading,
        )
        current = np.zeros((20, 1, 3), dtype=np.float64)
        current[:, 0, :2] = initial_ego_state.rear_axle.array
        current[:, 0, 2] = initial_ego_state.rear_axle.heading
        proposal_states = np.concatenate([current, global_40], axis=1)

        if self.use_cuda:
            simulator = TorchSimulator(proposal_sampling=self._proposal_sampling)
            tensor = torch.from_numpy(proposal_states).double().unsqueeze(0)
            simulated = simulator.simulate_proposals(
                tensor, initial_ego_state, batch_sim=True
            ).squeeze(0).cpu().numpy()
        else:
            simulator = PDMSimulator(proposal_sampling=self._proposal_sampling)
            simulated = simulator.simulate_proposals(
                proposal_states, initial_ego_state, batch_sim=True
            )

        combined = np.concatenate([[pdm_closed_trajectory], simulated], axis=0)
        self._pdm_closed._scorer.score_proposals(
            combined,
            initial_ego_state,
            self._pdm_closed._observation,
            self._centerline,
            self._pdm_closed._route_lane_dict,
            self._pdm_closed._drivable_area_map,
            self._map_api,
            batch=True,
        )
        return pairwise_official_scores(self._pdm_closed._scorer), local_40

    def _write_record(self, sidecar, sidecar_path, scores, components, local_40):
        selected = int(sidecar['selected_index'])
        deployed = np.asarray(sidecar['deployed_trajectory'], dtype=np.float64)
        parity_error = float(np.max(np.abs(deployed - local_40[selected])))
        if parity_error > 1e-4:
            raise RuntimeError(
                f"deployed/candidate trajectory parity drifted: {parity_error}"
            )
        if not np.isfinite(scores).all() or not np.isfinite(components).all():
            raise RuntimeError("non-finite dynamic candidate PDM reward")
        record = dict(sidecar)
        rollout_scene_id = str(self.current_scene.get("id"))
        origin_token = rollout_scene_id.rsplit("-", 1)[-1]
        if not origin_token or origin_token == rollout_scene_id:
            raise RuntimeError(
                f"cannot recover rare origin token from scene {rollout_scene_id}"
            )
        record.update(
            schema_version=2,
            record_type="diffusiondrive_closed_loop_candidate_reward",
            rollout_scene_id=rollout_scene_id,
            rollout_scene_token=str(self.current_scene.get("token")),
            rollout_origin_token=origin_token,
            worldengine_step=int(self.current_step),
            candidate_rewards=np.asarray(scores, dtype=np.float32),
            candidate_reward_components=np.asarray(components, dtype=np.float32),
            candidate_reward_valid_mask=np.ones(20, dtype=np.bool_),
            reward_component_names=COMPONENT_NAMES,
            sidecar_path=str(sidecar_path.resolve()),
            deployed_candidate_parity_max_abs_error=parity_error,
            deployed_candidate_reward=float(scores[selected]),
            oracle_candidate_reward=float(np.max(scores)),
        )
        output_dir = (
            Path(self.engine.global_config['data_output_dir'])
            / "diffusiondrive_rollout_records"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / (
            f"{sidecar['scene_prefix']}_{sidecar['planner_step']}_reward.pkl"
        )
        temporary_path = final_path.with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, final_path)
        row = pd.DataFrame(
            [{
                "scene": record["rollout_scene_id"],
                "step": self.current_step,
                "planner_step": sidecar["planner_step"],
                "pkl_path": str(final_path.resolve()),
            }]
        )
        self.pkl_paths_df = pd.concat([self.pkl_paths_df, row], ignore_index=True)
        self.pkl_paths_df.to_csv(
            Path(self.engine.global_config['data_output_dir'])
            / "diffusiondrive_rollout_records.csv",
            index=False,
        )

    def step(self):
        self.current_step = self.engine.episode_step
        if self.current_scene is None:
            logger.error("No scene available for DiffusionDrive reward")
            return None
        current_ego_state = self.converter.convert_to_current_ego_state(
            self.current_step
        )
        self.ego_states_list.append(current_ego_state)
        if self.current_step < self.num_history - 1:
            return None

        self.planner_input, planner_initialization = self._get_planner_inputs(
            self.current_scene
        )
        self._pdm_closed.initialize(planner_initialization)
        self._map_api = planner_initialization["map_api"]
        pdm_closed_trajectory = self._pdm_closed.compute_planner_trajectory(
            self.planner_input
        )
        if self.current_step < self.num_history + self.buffer_size - 1:
            for roadblock_id in self.get_current_roadblock_ids():
                roadblock = self._map_api.get_map_object(
                    roadblock_id, SemanticMapLayer.ROADBLOCK
                )
                if roadblock is None:
                    roadblock = self._map_api.get_map_object(
                        roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
                    )
                if roadblock_id not in self._cached_roadblock_ids:
                    self._cached_roadblock_ids.append(roadblock_id)
                    self.route_roadblock_dict[roadblock_id] = roadblock
        self.detection_tracks_list.append(
            self.converter.convert_to_detections_tracks_from_agent_input(
                self.current_step
            )
        )
        if not should_load_rollout_sidecar(
            self.current_step, self.num_history, self.buffer_size
        ):
            return None

        sidecar, sidecar_path = self._load_sidecar()
        candidates_8 = np.asarray(
            sidecar['candidate_trajectories_8'], dtype=np.float64
        )
        (scores, components), local_40 = self._score_candidates(
            pdm_closed_trajectory, candidates_8
        )
        self._write_record(
            sidecar, sidecar_path, scores, components, local_40
        )
        logger.info(
            "Scored DiffusionDrive dynamic action set scene=%s step=%d",
            self.current_scene.get("id"),
            self.current_step,
        )
        return None
