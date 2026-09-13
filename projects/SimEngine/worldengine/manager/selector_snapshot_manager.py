"""Native-action observation only: no candidate selection or reward evaluation."""
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_snapshot_common as s
from audit_selector_cfpi_deployment import action_arrays, action_error
from worldengine.manager.base_manager import BaseManager
from worldengine.manager.diffusiondrive_dynamic_reward_manager import (
    DiffusionDriveDynamicRewardManager, expand_candidates_to_40)
from worldengine.manager.diffusiondrive_sidecar_contract import scene_sidecar_prefixes


class SelectorSnapshotManager(BaseManager):
    PRIORITY = 20
    _load_sidecar = DiffusionDriveDynamicRewardManager._load_sidecar

    def __init__(self):
        super().__init__()
        cfg=self.engine.global_config
        self.entry=dict(path=cfg['selector_snapshot_manifest'],sha256=cfg['selector_snapshot_manifest_sha256'])
        self.collection=d.verified_read(self.entry)
        self.code_sha=d.verified_read(self.collection['contract'])['code_sha']
        mode=self.collection['mode']
        policies=('idm_policy','idm_navigation') if mode=='R' else ('trajectory_policy','trajectory_navigation')
        if (cfg['agent_policy'],cfg['agent_navigation']) != policies or cfg['ego_controller']!='log_play_controller':
            raise RuntimeError('Native controller/mode mismatch')
        if cfg['ego_policy']!='env_input_policy' or (cfg['num_history'],cfg['num_future'],cfg['reward_buffer_size'])!=(4,8,9):
            raise RuntimeError('Native horizon/policy mismatch')

    def reset(self):
        self.current_scene=self.engine.managers['scenario_manager'].current_scene
        self.agent=self.engine.managers['agent_manager'].ego_agent

    def _scene_prefixes(self):
        return scene_sidecar_prefixes(self.current_scene)

    def before_step(self):
        self.current_step=int(self.engine.episode_step)
        if self.current_step+1 not in c.DECISION_STEPS:
            return {}
        sidecar,path=self._load_sidecar()
        row=s.validate_native(sidecar,self.collection,self.code_sha)
        if row['scene_id']!=str(self.current_scene['id']) or row['decision_step']!=self.current_step+1:
            raise RuntimeError('Native observation alignment mismatch')
        trajectory=expand_candidates_to_40(np.asarray(sidecar['candidate_trajectories_8']))[row['selected_index']]
        error=float(np.max(np.abs(trajectory-np.asarray(sidecar['deployed_trajectory']))))
        expected=action_arrays(self.agent.client.trajectory_from_local_plan(trajectory))
        actual=action_arrays(self.engine.external_actions)
        action_diff=action_error(actual,expected)
        if not np.isfinite(error) or not np.isfinite(action_diff) or error>c.ACTION_TOLERANCE or action_diff>c.ACTION_TOLERANCE:
            raise RuntimeError('Native selected/published/consumed action mismatch')
        cfg=self.engine.global_config
        plan=Path(cfg['planner_data_path'])/f"{sidecar['scene_prefix']}_{row['decision_step']}.npy"
        record=dict(collection=self.entry,validation=row,state_step=self.current_step,
                    sidecar=d.artifact(path),plan=d.artifact(plan),actual_action=actual,
                    expected_action=expected,candidate_trajectory_max_abs=error)
        c.atomic_json(Path(cfg['data_output_dir'])/'snapshot_records'/f"{sidecar['scene_prefix']}_{row['decision_step']}.json",record)
        return {}
