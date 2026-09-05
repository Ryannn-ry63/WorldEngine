"""V4 method-neutral pre-action observation and one-step intervention manager."""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import oracle_r15_common as r15
import v4_causal_cache_common as v4
from worldengine.manager.diffusiondrive_dynamic_reward_manager import (
    COMPONENT_NAMES,
    DiffusionDriveDynamicRewardManager,
)
from worldengine.manager.diffusiondrive_preaction_oracle_manager import (
    DiffusionDrivePreActionOracleManager,
    should_score_preaction,
)


logger = logging.getLogger(__name__)


class DiffusionDriveV4CausalCacheManager(DiffusionDrivePreActionOracleManager):
    """Record V3 contexts, optionally replace one action, then return to V3."""

    def __init__(self):
        # Reuse audited pre-action scoring helpers without the R1.5 target parser.
        DiffusionDriveDynamicRewardManager.__init__(self)
        config = self.engine.global_config
        self.intervention_mode = str(
            config.get("diffusiondrive_v4_intervention_mode", "observe_only")
        )
        if self.intervention_mode not in {"observe_only", "one_shot_manifest"}:
            raise ValueError(f"invalid V4 intervention mode: {self.intervention_mode}")
        self.collection_id = str(
            config.get("diffusiondrive_v4_collection_id", "") or ""
        )
        if not self.collection_id:
            raise ValueError("V4 collection requires a nonempty collection id")
        self.behavior_train_seed = int(
            config.get("diffusiondrive_behavior_train_seed", -1)
        )
        if self.behavior_train_seed != 0:
            raise ValueError("V4 causal cache freezes scalar V3 train seed 0")
        self.target_manifest_path = str(
            config.get("diffusiondrive_v4_treatment_manifest", "") or ""
        )
        expected_sha = str(
            config.get("diffusiondrive_v4_treatment_manifest_sha256", "none")
        )
        self.target_manifest_sha256 = expected_sha
        self.treatment_kind = "none"
        self.targets = {}
        if self.intervention_mode == "observe_only":
            if self.target_manifest_path or expected_sha != "none":
                raise ValueError(
                    "V4 observe-only collection must not receive a treatment manifest"
                )
        else:
            if not self.target_manifest_path or expected_sha == "none":
                raise ValueError("V4 intervention requires a frozen treatment manifest")
            path, payload = v4.load_treatment(
                self.target_manifest_path, expected_sha
            )
            if str(payload["collection_id"]) != self.collection_id:
                raise RuntimeError("V4 collection/treatment id drifted")
            self.target_manifest_path = str(path)
            self.treatment_kind = str(payload["treatment_kind"])
            rows = payload["targets"]
            self.targets = {str(row["scene_id"]): row for row in rows}
            if len(self.targets) != len(rows):
                raise RuntimeError("duplicate V4 treatment target scene")
        self._one_shot_applied = False
        self.preaction_paths_df = pd.DataFrame(
            columns=["scene", "state_step", "decision_step", "pkl_path"]
        )

    def reset(self):
        result = DiffusionDriveDynamicRewardManager.reset(self)
        self._one_shot_applied = False
        return result

    def _write_v4_record(
        self,
        sidecar,
        sidecar_path,
        scores,
        components,
        local_40,
        policy_index,
        reward_oracle_index,
        treatment_index,
        deployed_index,
        applied,
        policy_action_error,
        deployed_action_error,
        target_reward_error,
    ):
        record = dict(sidecar)
        record.update(
            schema_version=1,
            record_type="diffusiondrive_v4_causal_record",
            rollout_scene_id=str(self.current_scene["id"]),
            rollout_scene_token=str(self.current_scene.get("token")),
            state_step=int(self.current_step),
            decision_step=int(sidecar["planner_step"]),
            worldengine_step=int(self.current_step),
            policy_selected_index=int(policy_index),
            local_reward_oracle_index=int(reward_oracle_index),
            treatment_index=int(treatment_index),
            deployed_index=int(deployed_index),
            selected_index=int(policy_index),
            intervention_mode=self.intervention_mode,
            intervention_applied=bool(applied),
            v4_collection_id=self.collection_id,
            v4_treatment_kind=self.treatment_kind,
            candidate_rewards=np.asarray(scores, dtype=np.float32),
            candidate_reward_components=np.asarray(components, dtype=np.float32),
            candidate_reward_valid_mask=np.ones(
                v4.NUM_CANDIDATES, dtype=np.bool_
            ),
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
            local_reward_oracle_candidate_reward=float(
                scores[reward_oracle_index]
            ),
            treatment_candidate_reward=float(scores[treatment_index]),
            deployed_candidate_reward=float(scores[deployed_index]),
            target_reward_max_abs_error=target_reward_error,
            target_manifest_sha256=self.target_manifest_sha256,
        )
        output_dir = (
            Path(self.engine.global_config["data_output_dir"])
            / "diffusiondrive_v4_causal_records"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / (
            f"{sidecar['scene_prefix']}_{sidecar['planner_step']}_v4causal.pkl"
        )
        temporary_path = final_path.with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as stream:
            pickle.dump(record, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, final_path)
        self.preaction_paths_df = pd.concat(
            [self.preaction_paths_df, pd.DataFrame([{
                "scene": record["rollout_scene_id"],
                "state_step": record["state_step"],
                "decision_step": record["decision_step"],
                "pkl_path": str(final_path.resolve()),
            }])],
            ignore_index=True,
        )
        self.preaction_paths_df.to_csv(
            Path(self.engine.global_config["data_output_dir"])
            / "diffusiondrive_v4_causal_records.csv",
            index=False,
        )

    def before_step(self):
        # Call the dynamic-reward parent; intervention is implemented below.
        DiffusionDriveDynamicRewardManager.before_step(self)
        self.current_step = int(self.engine.episode_step)
        if not should_score_preaction(
            self.current_step, self.num_history, self.buffer_size
        ):
            return None
        sidecar, sidecar_path = self._load_sidecar()
        decision_step = int(sidecar["planner_step"])
        if decision_step != self.current_step + 1:
            raise RuntimeError(
                f"V4 pre-action step drift: state={self.current_step} "
                f"decision={decision_step}"
            )
        contract = sidecar.get("selector_rollout_contract", {})
        expected = {
            "action_score_timing": "pre_action",
            "intervention_mode": self.intervention_mode,
            "behavior_policy_train_seed": self.behavior_train_seed,
            "treatment_manifest_sha256": self.target_manifest_sha256,
            "collection_id": self.collection_id,
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                raise RuntimeError(f"V4 sidecar contract drifted at {key}")

        candidates_8 = np.asarray(
            sidecar["candidate_trajectories_8"], dtype=np.float64
        )
        if (
            candidates_8.shape != (20, 8, 3)
            or not np.isfinite(candidates_8).all()
        ):
            raise RuntimeError("invalid V4 candidate set")
        (scores, components), local_40 = self._prepare_preaction_scores(
            candidates_8
        )
        logits = np.asarray(sidecar["current_logits"], dtype=np.float64)
        policy_index = int(sidecar["selected_index"])
        if policy_index != int(np.argmax(logits)):
            raise RuntimeError("V4 policy index/logit argmax drifted")
        if (
            r15.max_abs_error(
                sidecar["deployed_trajectory"], local_40[policy_index]
            )
            > v4.ACTION_TOLERANCE
        ):
            raise RuntimeError("V4 sidecar/policy candidate parity drifted")
        policy_action = self._candidate_action(local_40, policy_index)
        policy_action_error = r15.trajectory_action_error(
            self.engine.external_actions, policy_action
        )
        if policy_action_error > v4.ACTION_TOLERANCE:
            raise RuntimeError(
                f"V4 external policy action parity failed: {policy_action_error}"
            )

        reward_oracle_index = r15.stable_oracle_index(scores, policy_index)
        target = self.targets.get(str(self.current_scene["id"]))
        target_step = None if target is None else int(target["decision_step"])
        treatment_index = policy_index
        target_reward_error = None
        if target is not None and decision_step == target_step:
            errors = r15.target_context_errors(target, candidates_8, logits)
            if any(value > v4.ARRAY_TOLERANCE for value in errors.values()):
                raise RuntimeError(f"V4 frozen target context drifted: {errors}")
            treatment_index = int(target["treatment_index"])
            if treatment_index not in range(v4.NUM_CANDIDATES):
                raise RuntimeError("V4 frozen treatment index is invalid")
            target_reward_error = r15.max_abs_error(
                target["candidate_rewards"], scores
            )

        applied = bool(
            self.intervention_mode == "one_shot_manifest"
            and target_step is not None
            and decision_step == target_step
            and not self._one_shot_applied
        )
        deployed_index = treatment_index if applied else policy_index
        deployed_action = self._candidate_action(local_40, deployed_index)
        if applied:
            self.engine.external_actions = deployed_action
            self._one_shot_applied = True
        deployed_action_error = r15.trajectory_action_error(
            self.engine.external_actions, deployed_action
        )
        if deployed_action_error > v4.ACTION_TOLERANCE:
            raise RuntimeError("V4 deployed action replacement failed")
        self._write_v4_record(
            sidecar,
            sidecar_path,
            scores,
            components,
            local_40,
            policy_index,
            reward_oracle_index,
            treatment_index,
            deployed_index,
            applied,
            policy_action_error,
            deployed_action_error,
            target_reward_error,
        )
        logger.info(
            "V4 causal scene=%s decision=%d collection=%s applied=%s "
            "policy=%d treatment=%d",
            self.current_scene["id"],
            decision_step,
            self.collection_id,
            applied,
            policy_index,
            treatment_index,
        )
        return None

    def step(self):
        self.current_step = int(self.engine.episode_step)
        if self.current_scene is None:
            raise RuntimeError("no scene available for V4 causal cache manager")
        self.ego_states_list.append(
            self.converter.convert_to_current_ego_state(self.current_step)
        )
        return None
