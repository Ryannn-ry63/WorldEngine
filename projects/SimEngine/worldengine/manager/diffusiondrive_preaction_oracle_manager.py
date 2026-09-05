"""Action-before scalar-PDM oracle intervention for DiffusionDrive R1.5.

This manager is opt-in. It scores the exact exported 20-candidate set before
WorldEngine executes the matching action, records the unmodified V3 decision,
and may replace only ``engine.external_actions`` for a frozen intervention.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)

import oracle_r15_common as r15
from worldengine.manager.diffusiondrive_dynamic_reward_manager import (
    COMPONENT_NAMES,
    DiffusionDriveDynamicRewardManager,
)


logger = logging.getLogger(__name__)


def should_score_preaction(current_step: int, num_history: int, buffer_size: int) -> bool:
    """Score decisions 4..11 from their preceding states 3..10."""

    return num_history - 1 <= current_step < num_history + buffer_size - 2


class DiffusionDrivePreActionOracleManager(DiffusionDriveDynamicRewardManager):
    """Pre-action diagnostic manager with optional frozen oracle intervention."""

    def __init__(self):
        super().__init__()
        config = self.engine.global_config
        self.intervention_mode = str(
            config.get("diffusiondrive_preaction_mode", "observe_only")
        )
        if self.intervention_mode not in r15.MODES:
            raise ValueError(f"invalid pre-action mode: {self.intervention_mode}")
        self.behavior_train_seed = int(
            config.get("diffusiondrive_behavior_train_seed", -1)
        )
        if self.behavior_train_seed not in (0, 1):
            raise ValueError("R1.5 behavior train seed must be 0 or 1")
        self.target_manifest_path = str(
            config.get("diffusiondrive_oracle_target_manifest", "") or ""
        )
        expected_sha = str(
            config.get("diffusiondrive_oracle_target_manifest_sha256", "none")
        )
        self.target_manifest_sha256 = expected_sha
        self.targets = {}
        if self.intervention_mode == "observe_only":
            if self.target_manifest_path or expected_sha != "none":
                raise ValueError("observe-only R1.5 must not receive a target manifest")
        else:
            if not self.target_manifest_path or expected_sha == "none":
                raise ValueError("oracle intervention requires a frozen target manifest")
            path = Path(self.target_manifest_path).expanduser().resolve()
            if r15.sha256_file(path) != expected_sha:
                raise RuntimeError("R1.5 target manifest SHA256 drifted")
            payload = json.loads(path.read_text())
            valid_target_methods = {
                "diffusiondrive_selector_oracle_r15_targets_v1",
                "diffusiondrive_selector_causal_branch_pilot_targets_v1",
            }
            if (
                payload.get("status") != "PASS"
                or payload.get("method") not in valid_target_methods
            ):
                raise RuntimeError("invalid R1.5 target manifest")
            matching = [
                row
                for row in payload.get("targets", [])
                if int(row["train_seed"]) == self.behavior_train_seed
            ]
            self.targets = {str(row["scene_id"]): row for row in matching}
            if len(self.targets) != len(matching):
                raise RuntimeError("duplicate R1.5 target scene")
        self._one_shot_applied = False
        self.preaction_paths_df = pd.DataFrame(
            columns=["scene", "state_step", "decision_step", "pkl_path"]
        )

    def reset(self):
        result = super().reset()
        self._one_shot_applied = False
        return result

    def _planner_inputs_preaction(self):
        state_step = int(self.current_step)
        if len(self.ego_states_list) != state_step + 1:
            raise RuntimeError(
                "pre-action ego-state history is not aligned: "
                f"len={len(self.ego_states_list)} state_step={state_step}"
            )
        observations = self.observations_list[: state_step + 1]
        if len(observations) != state_step + 1:
            raise RuntimeError("pre-action observation history is incomplete")
        buffer_size = min(4, state_step + 1)
        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=buffer_size,
            ego_states=self.ego_states_list,
            observations=observations,
            sample_interval=0.5,
        )
        scene = self.current_scene
        timestamp = (
            scene.get("base_timestamp", 0.0)
            + state_step * scene["sample_rate"] * 0.05 * 1e6
        )
        planner_input = PlannerInput(
            iteration=SimulationIteration(
                index=state_step, time_point=TimePoint(timestamp)
            ),
            history=history,
            traffic_light_data=self.converter.convert_to_traffic_lights(state_step),
        )
        initialization = {
            "route_roadblock_dict_ids": self.get_route_roadblocks_ids(),
            "map_api": self.map_api,
        }
        return planner_input, initialization

    def _prepare_preaction_scores(self, candidates_8):
        self.planner_input, initialization = self._planner_inputs_preaction()
        self._pdm_closed.initialize(initialization)
        self._map_api = initialization["map_api"]
        pdm_closed_trajectory = self._pdm_closed.compute_planner_trajectory(
            self.planner_input
        )
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
        return self._score_candidates(pdm_closed_trajectory, candidates_8)

    def _candidate_action(self, local_40, candidate_index):
        client = getattr(self.agent, "client", None)
        convert = getattr(client, "trajectory_from_local_plan", None)
        if convert is None:
            raise RuntimeError("R1.5 requires NAVFormerClient trajectory conversion")
        return convert(np.asarray(local_40[int(candidate_index)], dtype=np.float64))

    def _write_preaction_record(
        self,
        sidecar,
        sidecar_path,
        scores,
        components,
        local_40,
        policy_index,
        raw_oracle_index,
        oracle_index,
        treatment_index,
        deployed_index,
        applied,
        policy_action_error,
        deployed_action_error,
        target_reward_error,
        target_treatment_headroom,
        target_oracle_agreement,
    ):
        record = dict(sidecar)
        record.update(
            schema_version=4,
            record_type="diffusiondrive_preaction_oracle_record",
            rollout_scene_id=str(self.current_scene["id"]),
            rollout_scene_token=str(self.current_scene.get("token")),
            state_step=int(self.current_step),
            decision_step=int(sidecar["planner_step"]),
            worldengine_step=int(self.current_step),
            policy_selected_index=int(policy_index),
            raw_oracle_index=int(raw_oracle_index),
            oracle_index=int(oracle_index),
            treatment_index=int(treatment_index),
            deployed_index=int(deployed_index),
            selected_index=int(policy_index),
            intervention_mode=self.intervention_mode,
            intervention_applied=bool(applied),
            candidate_rewards=np.asarray(scores, dtype=np.float32),
            candidate_reward_components=np.asarray(components, dtype=np.float32),
            candidate_reward_valid_mask=np.ones(20, dtype=np.bool_),
            reward_component_names=COMPONENT_NAMES,
            sidecar_path=str(Path(sidecar_path).resolve()),
            policy_deployed_trajectory=np.asarray(
                sidecar["deployed_trajectory"], dtype=np.float32
            ),
            deployed_trajectory=np.asarray(
                local_40[deployed_index], dtype=np.float32
            ),
            policy_action_parity_max_abs_error=float(policy_action_error),
            deployed_action_parity_max_abs_error=float(deployed_action_error),
            policy_candidate_reward=float(scores[policy_index]),
            oracle_candidate_reward=float(scores[oracle_index]),
            treatment_candidate_reward=float(scores[treatment_index]),
            deployed_candidate_reward=float(scores[deployed_index]),
            target_reward_max_abs_error=target_reward_error,
            target_treatment_current_headroom=target_treatment_headroom,
            target_scored_oracle_agreement=target_oracle_agreement,
            target_manifest_sha256=self.target_manifest_sha256,
        )
        output_dir = (
            Path(self.engine.global_config["data_output_dir"])
            / "diffusiondrive_preaction_records"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / (
            f"{sidecar['scene_prefix']}_{sidecar['planner_step']}_preaction.pkl"
        )
        temporary_path = final_path.with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, final_path)
        row = pd.DataFrame(
            [{
                "scene": record["rollout_scene_id"],
                "state_step": record["state_step"],
                "decision_step": record["decision_step"],
                "pkl_path": str(final_path.resolve()),
            }]
        )
        self.preaction_paths_df = pd.concat(
            [self.preaction_paths_df, row], ignore_index=True
        )
        self.preaction_paths_df.to_csv(
            Path(self.engine.global_config["data_output_dir"])
            / "diffusiondrive_preaction_records.csv",
            index=False,
        )

    def before_step(self):
        super().before_step()
        self.current_step = int(self.engine.episode_step)
        if not should_score_preaction(
            self.current_step, self.num_history, self.buffer_size
        ):
            return None
        sidecar, sidecar_path = self._load_sidecar()
        decision_step = int(sidecar["planner_step"])
        if decision_step != self.current_step + 1:
            raise RuntimeError(
                f"pre-action step drift: state={self.current_step} decision={decision_step}"
            )
        contract = sidecar.get("selector_rollout_contract", {})
        if contract.get("action_score_timing") != "pre_action":
            raise RuntimeError("R1.5 sidecar lacks pre-action contract")
        if contract.get("oracle_intervention_mode") != self.intervention_mode:
            raise RuntimeError("R1.5 sidecar/WorldEngine intervention mode drifted")
        if int(contract.get("behavior_policy_train_seed", -1)) != self.behavior_train_seed:
            raise RuntimeError("R1.5 sidecar behavior seed drifted")
        if (
            str(contract.get("oracle_target_manifest_sha256", "none"))
            != self.target_manifest_sha256
        ):
            raise RuntimeError("R1.5 sidecar target-manifest SHA256 drifted")

        candidates_8 = np.asarray(sidecar["candidate_trajectories_8"], dtype=np.float64)
        if candidates_8.shape != (20, 8, 3) or not np.isfinite(candidates_8).all():
            raise RuntimeError("invalid R1.5 candidate set")
        (scores, components), local_40 = self._prepare_preaction_scores(candidates_8)
        logits = np.asarray(sidecar["current_logits"], dtype=np.float64)
        policy_index = int(sidecar["selected_index"])
        if policy_index != int(np.argmax(logits)):
            raise RuntimeError("R1.5 policy index/logit argmax drifted")
        if r15.max_abs_error(sidecar["deployed_trajectory"], local_40[policy_index]) > r15.ACTION_TOLERANCE:
            raise RuntimeError("R1.5 sidecar/policy candidate parity drifted")
        policy_action = self._candidate_action(local_40, policy_index)
        policy_action_error = r15.trajectory_action_error(
            self.engine.external_actions, policy_action
        )
        if policy_action_error > r15.ACTION_TOLERANCE:
            raise RuntimeError(
                f"R1.5 external policy action parity failed: {policy_action_error}"
            )

        raw_oracle_index = int(np.argmax(scores))
        oracle_index = r15.stable_oracle_index(scores, policy_index)
        target = self.targets.get(str(self.current_scene["id"]))
        target_step = None if target is None else int(target["decision_step"])
        treatment_index = oracle_index
        target_reward_error = None
        target_treatment_headroom = None
        target_oracle_agreement = None
        if target is not None and decision_step == target_step:
            errors = r15.target_context_errors(target, candidates_8, logits)
            if any(value > r15.ARRAY_TOLERANCE for value in errors.values()):
                raise RuntimeError(f"R1.5 frozen target context drifted: {errors}")
            treatment_key = (
                "matched_index"
                if self.intervention_mode == "one_shot_matched"
                else "oracle_index"
            )
            treatment_index = int(target[treatment_key])
            if not 0 <= treatment_index < 20:
                raise RuntimeError("R1.5 frozen treatment index is invalid")
            target_reward_error = r15.max_abs_error(
                target["candidate_rewards"], scores
            )
            target_treatment_headroom = float(
                scores[treatment_index] - scores[policy_index]
            )
            target_oracle_agreement = bool(treatment_index == oracle_index)
        applied = r15.intervention_applies(
            self.intervention_mode,
            decision_step,
            target_step,
            self._one_shot_applied,
        )
        deployed_index = treatment_index if applied else policy_index
        deployed_action = self._candidate_action(local_40, deployed_index)
        if applied:
            self.engine.external_actions = deployed_action
            if self.intervention_mode.startswith("one_shot_"):
                self._one_shot_applied = True
        deployed_action_error = r15.trajectory_action_error(
            self.engine.external_actions, deployed_action
        )
        if deployed_action_error > r15.ACTION_TOLERANCE:
            raise RuntimeError("R1.5 deployed action replacement failed")
        self._write_preaction_record(
            sidecar,
            sidecar_path,
            scores,
            components,
            local_40,
            policy_index,
            raw_oracle_index,
            oracle_index,
            treatment_index,
            deployed_index,
            applied,
            policy_action_error,
            deployed_action_error,
            target_reward_error,
            target_treatment_headroom,
            target_oracle_agreement,
        )
        logger.info(
            "R1.5 pre-action scene=%s state=%d decision=%d mode=%s applied=%s policy=%d oracle=%d treatment=%d",
            self.current_scene["id"],
            self.current_step,
            decision_step,
            self.intervention_mode,
            applied,
            policy_index,
            oracle_index,
            treatment_index,
        )
        return None

    def step(self):
        """Only capture the post-action ego state for the next pre-action score."""

        self.current_step = int(self.engine.episode_step)
        if self.current_scene is None:
            raise RuntimeError("no scene available for R1.5 pre-action manager")
        self.ego_states_list.append(
            self.converter.convert_to_current_ego_state(self.current_step)
        )
        return None
