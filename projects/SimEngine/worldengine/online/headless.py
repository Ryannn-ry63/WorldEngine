"""Persistent real SimEngine dynamics, without rendering or offline metrics.

This is a correctness reference for branches, not the full visual main loop.
"""
import copy
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf

from worldengine.engine import engine_utils
from worldengine.manager.agent_manager import BaseAgentManager
from worldengine.manager.map_manager import ScenarioMapManager
from worldengine.manager.scenario_manager import ScenarioManager
from worldengine.common.dataclasses import Trajectory
from .state import SnapshotCodec
from .numerics import check_numeric_runtime


def dynamics_config(reaction, max_steps):
    if reaction not in ('NR', 'R') or max_steps < 1:
        raise ValueError('Use NR or R and a positive max_steps')
    path = Path(__file__).resolve().parents[1] / 'configs/default_runner.yaml'
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    # These keys only concern the runner, disk bridge or CUDA renderer. Dropping
    # unresolved interpolations also prevents accidental legacy path access.
    for key in list(cfg):
        if key in ('hydra', 'defaults') or '${' in str(cfg[key]):
            del cfg[key]
    cfg.update(max_step=max_steps, decision_repeat=1, online_deterministic_rng=True,
               ego_policy='env_input_policy', ego_controller='two_stage_controller',
               ego_navigation='trajectory_navigation', ego_client='base_client',
               agent_policy='idm_policy' if reaction == 'R' else 'trajectory_policy',
               agent_navigation='idm_navigation' if reaction == 'R' else 'trajectory_navigation',
               agent_controller='log_play_controller', with_render_manager=False,
               with_metric_manager=False, with_dense_reward_manager=False,
               with_data_manager=False, visualize_BEV=False, visualize_video=False,
               save_data=False, use_planner_actions=False, store_map=False)
    return OmegaConf.create(cfg)


class HeadlessSimulator:
    def __init__(self, scene_id, scene, reaction='R', max_steps=8, seed=0):
        check_numeric_runtime()
        if engine_utils.engine_initialized():
            raise RuntimeError('Use spawn processes: one SimEngine per process')
        if scene['log_length'] <= max_steps:
            raise ValueError('Scenario is shorter than the requested engineering probe')
        if not np.isclose(scene['sample_rate'] * .05, .5):
            raise ValueError('This adapter currently requires a 0.5 s planning interval')
        np.random.seed(seed)
        random.seed(seed)
        cfg = dynamics_config(reaction, max_steps)
        engine_utils.initialize_global_config(cfg)
        self.engine = engine_utils.initialize_engine(cfg)
        try:
            self.engine.register_manager('scenario_manager', ScenarioManager({scene_id: copy.deepcopy(scene)}))
            self.engine.register_manager('map_manager', ScenarioMapManager())
            self.engine.register_manager('agent_manager', BaseAgentManager())
            # Engine seed selects the only scene (index 0). The traffic seed is
            # independent of scene index and travels in the manager snapshot.
            self.engine.agent_manager.seed(int(seed))
            self.engine.reset()
            self.engine.after_step()
            self.codec = SnapshotCodec(engine=self.engine)
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        return self.codec.capture(self.engine)

    def restore(self, snapshot):
        self.codec.restore(self.engine, snapshot)

    def step(self, action):
        if self.engine.episode_step >= self.engine.global_config.max_step:
            raise RuntimeError('Probe step limit reached')
        if not isinstance(action, Trajectory):
            raise TypeError('Expected a world-coordinate centered-vehicle Trajectory')
        points = np.asarray(action.waypoints)
        headings = np.asarray(action.headings)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError('Trajectory must have at least 3 XY waypoints')
        if headings.shape != (len(points),) or not np.isfinite(points).all() or not np.isfinite(headings).all():
            raise ValueError('Invalid trajectory headings/coordinates')
        # The upstream controller may mutate list-valued trajectories. Protect
        # the candidate bank and ensure branch ordering cannot modify actions.
        self.engine.before_step(copy.deepcopy(action))
        self.engine.step(1)
        self.engine.after_step()
        return self.agent_states()

    def agent_states(self):
        result = {}
        for key, agent in self.engine.agents.items():
            result[key] = dict(position=np.asarray(agent.current_position).tolist(),
                               heading=float(agent.current_heading),
                               velocity=np.asarray(agent.current_velocity).tolist(),
                               traj_step=int(agent.traj_step),
                               acceleration=np.asarray(getattr(agent, 'acceleration', [])).tolist(),
                               angular_velocity=float(agent.current_angular_velocity),
                               tire_steering=float(getattr(agent, 'tire_steering', 0.)))
            if not all(np.isfinite(np.asarray(value)).all() for value in result[key].values()):
                raise RuntimeError('Non-finite agent state: ' + str(key))
        return result

    def branch_group(self, actions, selected, on_main=None):
        if len(actions) != 20 or not 0 <= selected < 20:
            raise ValueError('Strict H1 requires exactly 20 actions and a selected index')
        self.codec.audit_static()
        before = self.snapshot()
        main_states = self.step(actions[selected])  # Execute before any branch.
        main = self.snapshot()
        results = []
        try:
            if on_main is not None:
                on_main(main, main_states)  # Independent receipt before branch feedback.
            for action in actions:
                self.restore(before)
                states = self.step(action)
                results.append((self.snapshot(), states))
            if results[selected][0].state_hash != main.state_hash:
                raise self.codec.mismatch('Selected branch differs from canonical execution',
                                          before, main, results[selected][0], actions[selected])
        finally:
            self.restore(main)
            self.codec.audit_static()
        return before, main, results

    def close(self):
        engine_utils.close_engine()


def diagnostic_candidates(sim):
    """20 current-state-only analytic probes; NOT DiffusionDrive predictions.

    This fixture exercises the real LQR + bicycle with distinct speed/curvature
    requests. It does not read the logged ego future or create training data.
    """
    ego = sim.engine.agent_manager.ego_agent
    origin = np.asarray(ego.current_position)[:2]
    theta = float(ego.current_heading)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    times = np.arange(9) * .5
    actions = []
    for speed_scale in (.5, .8, 1.1, 1.4):
        speed = max(float(ego.current_speed), 2.) * speed_scale
        for bend in (-.02, -.01, 0., .01, .02):
            x = speed * times
            y = bend * x ** 2
            points = np.column_stack([x, y]) @ rotation.T + origin
            headings = theta + np.arctan2(2 * bend * x, np.ones_like(x))
            actions.append(Trajectory(points, headings=headings))
    return actions
