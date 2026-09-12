"""Observe actual continuous-deployment actions; never score/replace candidates."""
from pathlib import Path

import numpy as np

import cfpi_common as c
import oracle_r15_common as r15
import selector_cfpi_deployment_common as d
from audit_selector_cfpi_deployment import validate_sidecar, action_arrays, action_error
from worldengine.manager.base_manager import BaseManager
from worldengine.manager.diffusiondrive_dynamic_reward_manager import (
    DiffusionDriveDynamicRewardManager, expand_candidates_to_40)
from worldengine.manager.diffusiondrive_sidecar_contract import scene_sidecar_prefixes


class DiffusionDriveCFPIDeploymentManager(BaseManager):
    PRIORITY = 20
    _load_sidecar = DiffusionDriveDynamicRewardManager._load_sidecar

    def __init__(self):
        super().__init__()
        cfg = self.engine.global_config
        self.collection_sha = cfg['diffusiondrive_cfpi_deployment_manifest_sha256']
        self.collection = d.verified_read(dict(path=cfg['diffusiondrive_cfpi_deployment_manifest'],
                                               sha256=self.collection_sha))
        self.routing = d.verified_read(self.collection['routing'])
        self.code_sha = d.verified_read(self.collection['run_contract'])['code_sha']
        if self.collection.get('research_method') in ('selector_feedback_repair_v2', 'selector_decision_feedback_v1'):
            mode = self.collection['react_type']
            policy,navigation = ('idm_policy','idm_navigation') if mode=='R' else ('trajectory_policy','trajectory_navigation')
            if (mode not in ('NR','R') or cfg['agent_policy']!=policy or cfg['agent_navigation']!=navigation
                    or cfg['ego_controller']!='log_play_controller' or cfg['ego_policy']!='env_input_policy'):
                raise RuntimeError('Actual simulator mode/controller differs from feedback contract')
        if cfg['num_history'] != 4 or cfg['num_future'] != 8 or cfg['reward_buffer_size'] != 9:
            raise RuntimeError("Deployment freezes decisions 4..11 at 0.5-second replanning")

    def reset(self):
        self.current_scene = self.engine.managers['scenario_manager'].current_scene
        self.agent = self.engine.managers['agent_manager'].ego_agent

    def _scene_prefixes(self):
        return scene_sidecar_prefixes(self.current_scene)

    def before_step(self):
        self.current_step = int(self.engine.episode_step)
        if self.current_step + 1 not in c.DECISION_STEPS:
            return {}
        sidecar, path = self._load_sidecar()
        if int(sidecar['planner_step']) != self.current_step + 1:
            raise RuntimeError("Deployment action is not aligned to the pre-action state")
        validation = validate_sidecar(sidecar, self.collection, self.routing, self.code_sha)
        if validation['scene_id'] != str(self.current_scene['id']):
            raise RuntimeError("Deployment sidecar belongs to another simulator scene")
        candidates = expand_candidates_to_40(np.asarray(sidecar['candidate_trajectories_8']))
        selected = int(sidecar['selected_index'])
        trajectory_error = r15.max_abs_error(candidates[selected], sidecar['deployed_trajectory'])
        if trajectory_error > c.ACTION_TOLERANCE:
            raise RuntimeError("Selected candidate / published trajectory mismatch")
        expected = self.agent.client.trajectory_from_local_plan(candidates[selected])
        actual_arrays, expected_arrays = action_arrays(self.engine.external_actions), action_arrays(expected)
        error = action_error(actual_arrays, expected_arrays)
        if not np.isfinite(error) or error > c.ACTION_TOLERANCE:
            raise RuntimeError(f"Actually consumed deployment action mismatch: {error}")
        cfg = self.engine.global_config
        plan = Path(cfg['planner_data_path']) / f"{sidecar['scene_prefix']}_{sidecar['planner_step']}.npy"
        output = Path(cfg['data_output_dir']) / 'cfpi_deployment_records'
        record = dict(schema_version=1, collection_sha256=self.collection_sha, scene_id=validation['scene_id'],
                      state_step=self.current_step, validation=validation, sidecar=d.artifact(path), plan=d.artifact(plan),
                      actual_action=actual_arrays, expected_action=expected_arrays,
                      candidate_trajectory_max_abs=trajectory_error)
        # Atomic overwrite of this scene/frame permits deterministic partial-scene retry.
        c.atomic_json(output / f"{sidecar['scene_prefix']}_{sidecar['planner_step']}.json", record)
        return {}
