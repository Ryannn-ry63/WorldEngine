"""Live SimEngine to causal PDM adapter. Static route input is read at reset only."""
import copy
import numpy as np
from shapely.geometry import Point
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from worldengine.components.agents.policy.pdm_planner.abstract_pdm_planner import AbstractPDMPlanner
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from .reward import RewardHistory, H1Reward, capture_reward_frame, ego_state
from worldengine.common.dataclasses import Trajectory


class RouteGeometry(AbstractPDMPlanner):
    """Only the upstream route/map helpers; no proposal simulation or expert path."""
    def initialize(self, initialization):
        raise NotImplementedError
    def name(self):
        return 'causal_route_geometry'
    def observation_type(self):
        raise NotImplementedError
    def compute_planner_trajectory(self, current_input):
        raise NotImplementedError


class RewardSession:
    def __init__(self, sim, map_root):
        self.sim = sim
        scene = sim.engine.current_scene
        # Copy only static metadata. Scoring never receives the source scene.
        self.offset = -np.array(scene['metadata']['old_origin_in_current_coordinate'], dtype=float)
        self.actor_types = {k:v['type'] for k,v in scene['object_track'].items()}
        reset_token = scene['metadata']['nuplan_lidar_pc_tokens'][0]
        info = scene['metadata']['openscene_data_infos_dict'][reset_token]
        self.route_ids = tuple(info['roadblock_ids'])
        if not self.route_ids:
            raise ValueError('No route at reset')
        self.map_api = get_maps_api(str(map_root), 'nuplan-maps-v1.0', scene['map'])
        self.history = RewardHistory()
        self.history.append(self.capture())
        route = RouteGeometry(100)
        route._map_api = self.map_api
        route._load_route_dicts(list(self.route_ids))
        initial = ego_state(self.history.frames[0])
        route._drivable_area_map = PDMDrivableMap.from_simulation(self.map_api, initial, 100)
        lane = route._get_starting_lane(initial)
        if lane is None:
            raise ValueError('No starting lane on the registered route')
        self.centerline = PDMPath(route._get_discrete_centerline(lane))
        self.route_lanes = route._route_lane_dict

    def capture(self):
        before = self.sim.snapshot()
        result = capture_reward_frame(self.sim.engine, self.offset, self.actor_types)
        after = self.sim.snapshot()
        if before.state_hash != after.state_hash:
            raise self.sim.codec.mismatch('Reward capture mutated dynamics', before, before, after, None)
        return result

    def warmup_action(self):
        """Current-state/map-only warmup, slowing before the finite route end.

        Trajectory uses centered-vehicle positions at 0.5 s intervals. Start at
        the live pose, use local map tangents, and never extrapolate the route.
        This is engineering history initialization, not a learned action.
        """
        ego = self.sim.engine.agent_manager.ego_agent
        center = np.asarray(ego.current_position, dtype=float)[:2]
        line = self.centerline.linestring
        projected = float(line.project(Point(*(center + self.offset))))
        remaining = max(float(line.length) - projected, 0.)
        # A four-second time-to-end cap gives the real controller room to brake.
        # Do not force a minimum speed or an eight-metre horizon on a short route.
        speed = min(max(float(ego.current_speed), 0.), remaining / 4.)
        times = np.arange(9, dtype=float) * .5
        distances = projected + speed * times
        xy = np.asarray([line.interpolate(float(d)).coords[0] for d in distances])
        xy = xy - self.offset[None, :]
        # Local tangents, not chords from the first point on a curved route.
        lo = np.maximum(distances - .05, 0.)
        hi = np.minimum(distances + .05, line.length)
        tangents = np.asarray([np.asarray(line.interpolate(float(h)).coords[0]) -
                               np.asarray(line.interpolate(float(l)).coords[0])
                               for l, h in zip(lo, hi)])
        headings = np.arctan2(tangents[:, 1], tangents[:, 0])
        if speed <= 1e-6:
            xy[:] = center
            headings[:] = float(ego.current_heading)
        else:
            xy[0] = center
            headings[0] = float(ego.current_heading)
        if not np.isfinite(xy).all() or not np.isfinite(headings).all():
            raise ValueError('Non-finite map warmup trajectory')
        return Trajectory(xy, headings=headings)

    def warmup(self, action):
        self.sim.step(action)
        self.history.append(self.capture())

    def group(self, actions, selected, on_main=None):
        """Execute canonical first, score it independently, replay and score all20.

        Reward snapshots are detached histories and are never changed by a branch.
        The engine snapshot restores complete dynamics, then the canonical history
        is committed only after strict state/component/history parity checks.
        """
        history_before = self.history.snapshot()
        history_hash = self.history.state_hash
        start = history_before[-1]
        # Cover actual trailing history and the next H1 neighbourhood. No missing
        # map fallback; polygons are from the real registered nuPlan map.
        radius = 100 + max(np.linalg.norm(f.ego[:2]-start.ego[:2]) for f in history_before)
        drivable = PDMDrivableMap.from_simulation(self.map_api, ego_state(start), radius)
        adapter = H1Reward(self.centerline, self.route_lanes, drivable, self.map_api)
        main_result = {}
        def canonical(snapshot, states):
            frame = self.capture()
            main_result.update(frame=frame, reward=adapter.score(history_before, frame))
            if on_main is not None:
                on_main(snapshot, states, main_result['reward'])
        before, main, branches = self.sim.branch_group(actions, selected, on_main=canonical)
        rewards, history_hashes = [], []
        try:
            for snapshot, _ in branches:
                self.sim.restore(snapshot)
                frame = self.capture()
                rewards.append(adapter.score(history_before, frame))
                branch_history = RewardHistory()
                branch_history.restore(history_before)
                branch_history.append(frame)
                history_hashes.append(branch_history.state_hash)
            if rewards[selected] != main_result['reward']:
                raise RuntimeError('Selected branch differs from independently scored canonical reward')
            if self.history.state_hash != history_hash:
                raise RuntimeError('Scoring changed pre-action reward history')
            canonical_history = RewardHistory()
            canonical_history.restore(history_before)
            canonical_history.append(main_result['frame'])
            if canonical_history.state_hash != history_hashes[selected]:
                raise RuntimeError('Selected branch reward history differs from canonical history')
        finally:
            self.sim.restore(main)
            self.sim.codec.audit_static()
        self.history.restore(canonical_history.snapshot())
        return dict(main=main_result['reward'], branches=rewards,
            branch_hashes=[s.state_hash for s, _ in branches],
            before_hash=before.state_hash, main_hash=main.state_hash,
            selected_state_parity=main.state_hash==branches[selected][0].state_hash,
            selected_reward_parity=True, selected_history_parity=True,
            history_before_hash=history_hash, history_after_hash=self.history.state_hash,
            reward_std=float(np.std([r['reward'] for r in rewards])),
            component_ranges={k:float(np.ptp([r[k] for r in rewards])) for k in rewards[0]})
