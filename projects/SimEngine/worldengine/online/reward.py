"""Causal H1 adapter around repository PDM components, not official full CL-PDMS.

Only detached current-state records enter scoring. No scene, logged trajectory,
metric cache, candidate scores or future schedule is accepted by this module.
See docs/innovation3/h1_reward.md for the versioned window/reference contract.
"""
import copy
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from shapely import Point
from shapely.affinity import translate
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from worldengine.components.agents.vehicle_model.pacifica_vehicle import get_pacifica_parameters
from worldengine.components.agents.policy.pdm_planner.utils.WE2pdm_utils import tracked_object_type_mapping
from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import StateIndex as S, MultiMetricIndex as M, WeightedMetricIndex as W
from worldengine.components.agents.policy.pdm_planner.scoring.pdm_scorer import PDMScorer, WEIGHTED_METRICS_WEIGHTS
from worldengine.components.agents.policy.pdm_planner.scoring.pdm_comfort_metrics import ego_is_comfortable
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMOccupancyMap
from .state import structural_hash

CONTRACT = dict(name='causal_h1_pdm_components_v1', interval_seconds=0.5,
    comfort_samples=15, comfort_window_seconds=7.0, warmup_transitions=13,
    nc_dac='min_of_actual_t_and_t_plus_1_endpoint_checks_no_substeps',
    ttc='endpoint_current_actor_constant_velocity_1s_at_0_0.5_1.0s',
    progress_reference='current_ego_velocity_constant_heading_0.5s_on_shared_map_centerline',
    progress_normalization='official_batch_pair_max_reference_or_branch_threshold_0.1m',
    direction='actual_15_sample_trailing_window',
    weights=WEIGHTED_METRICS_WEIGHTS.tolist(),
    collision_memory='per_H1_window_no_cross_window_collision_exclusions',
    official_full_closed_loop_pdms=False)


@dataclass(frozen=True)
class RewardFrame:
    step: int
    ego: np.ndarray  # rear axle global pose; body-frame velocity/acceleration
    actors: tuple  # detached nuPlan objects at this actual state only

    @property
    def state_hash(self):
        # Hash detached numerical geometry, not nuPlan cached properties.
        actors = [(a.track_token, int(a.tracked_object_type.value),
                   list(a.box.center), a.box.length, a.box.width, a.box.height,
                   [a.velocity.x, a.velocity.y] if hasattr(a, 'velocity') else [0., 0.]) for a in self.actors]
        return structural_hash((self.step, self.ego, actors))


def capture_reward_frame(engine, global_offset, actor_types):
    """Read live agents; actor_types is static type metadata, never track states."""
    ego = engine.agent_manager.ego_agent
    heading = float(ego.current_heading)
    rot = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
    state = np.zeros(S.size(), dtype=np.float64)
    state[:2] = np.asarray(ego.rear_vehicle.current_position)[:2] + global_offset
    state[S.HEADING] = heading
    state[S.VELOCITY_2D] = rot @ np.asarray(ego.rear_vehicle.current_velocity)[:2]
    state[S.ACCELERATION_2D] = rot @ np.asarray(ego.rear_vehicle.current_acceleration)[:2]
    state[S.STEERING_ANGLE] = ego.tire_steering
    state[S.ANGULAR_VELOCITY] = ego.current_angular_velocity
    state[S.ANGULAR_ACCELERATION] = ego.current_angular_acceleration
    objects = []
    for token, agent in sorted(engine.agent_manager.all_agents.items()):
        if agent is ego:
            continue
        kind = tracked_object_type_mapping[actor_types[token]]
        xy = np.asarray(agent.current_position)[:2] + global_offset
        dimensions = [float(agent.length), float(agent.width), float(agent.height)]
        values = [*xy, float(agent.current_heading), *dimensions, *agent.current_velocity[:2]]
        if not np.isfinite(values).all() or min(dimensions) <= 0:
            raise ValueError('Invalid live actor geometry: ' + str(token))
        box = OrientedBox(StateSE2(*xy, float(agent.current_heading)), *dimensions)
        metadata = SceneObjectMetadata(timestamp_us=int(engine.episode_step*500000),
            token=str(token), track_token=str(token), track_id=None)
        if kind in AGENT_TYPES:
            obj = Agent(kind, box, StateVector2D(*map(float, agent.current_velocity[:2])), metadata)
        else:
            obj = StaticObject(kind, box, metadata)
        objects.append(obj)
    if not np.isfinite(state).all():
        raise ValueError('Non-finite live ego state')
    return RewardFrame(int(engine.episode_step), state.copy(), tuple(objects))


def ego_state(frame):
    x = frame.ego
    return EgoState.build_from_rear_axle(StateSE2(*x[:3]),
        StateVector2D(*x[S.VELOCITY_2D]), StateVector2D(*x[S.ACCELERATION_2D]),
        float(x[S.STEERING_ANGLE]), TimePoint(int(frame.step*500000)), get_pacifica_parameters())


class CurrentObservation:
    """PDM observation protocol, at one actual state plus explicit CV projections."""
    red_light_token = 'red_light'

    def __init__(self, frame, forecast=False):
        self.collided_track_ids = []
        self.unique_objects = {a.track_token: a for a in frame.actors}
        self.maps = []
        for dt in ([0., .5, 1.] if forecast else [0.]):
            polygons = []
            for a in frame.actors:
                velocity = a.velocity if hasattr(a, 'velocity') else StateVector2D(0., 0.)
                polygons.append(translate(a.box.geometry, velocity.x*dt, velocity.y*dt))
            self.maps.append(PDMOccupancyMap(list(self.unique_objects), polygons))

    def __getitem__(self, index):
        return self.maps[index]


class RewardHistory:
    """Explicit reward state separate from dynamics snapshots and renderer handles."""
    def __init__(self):
        self.frames = ()

    def append(self, frame):
        if self.frames and frame.step != self.frames[-1].step + 1:
            raise ValueError('Reward history must be consecutive actual transitions')
        self.frames = (*self.frames, copy.deepcopy(frame))[-15:]

    def snapshot(self):
        return copy.deepcopy(self.frames)

    def restore(self, frames):
        if len(frames) > 15 or any(b.step != a.step+1 for a,b in zip(frames, frames[1:])):
            raise ValueError('Invalid reward history snapshot')
        self.frames = copy.deepcopy(tuple(frames))

    @property
    def state_hash(self):
        return structural_hash(tuple(f.state_hash for f in self.frames))


class H1Reward:
    def __init__(self, centerline, route_lanes, drivable_map, map_api):
        if not route_lanes or drivable_map is None or len(drivable_map) == 0:
            raise ValueError('H1 scoring requires route lanes and drivable polygons')
        if centerline is None or centerline.linestring.length <= 0:
            raise ValueError('H1 scoring requires a map-derived centerline')
        self.centerline, self.route_lanes = centerline, route_lanes
        self.drivable_map, self.map_api = drivable_map, map_api

    def _scorer(self, frames, observation):
        # nuPlan TrajectorySampling rejects zero poses; endpoint-only NC/DAC
        # use the scorer's sampling protocol explicitly, without fake frames.
        sampling = (SimpleNamespace(num_poses=0, interval_length=.5) if len(frames)==1
                    else TrajectorySampling(num_poses=len(frames)-1, interval_length=.5))
        scorer = PDMScorer(sampling)
        scorer._reset(np.stack([f.ego for f in frames])[None], ego_state(frames[0]),
            observation, self.centerline, self.route_lanes, self.drivable_map, self.map_api)
        scorer._calculate_ego_area()
        return scorer

    def score(self, before_history, endpoint):
        frames = (*before_history, endpoint)[-15:]
        if len(frames) != 15 or any(b.step != a.step+1 for a,b in zip(frames,frames[1:])):
            raise ValueError('Comfort requires 15 consecutive actual states; no padding/logged future')
        start = frames[-2]
        # Current tracked-object type, velocity and heading are exact at each endpoint.
        nc, dac = [], []
        for frame in (start, endpoint):
            scorer = self._scorer([frame], CurrentObservation(frame))
            scorer._calculate_no_at_fault_collision()
            scorer._calculate_drivable_area_compliance()
            nc.append(float(scorer._multi_metrics[M.NO_COLLISION, 0]))
            dac.append(float(scorer._multi_metrics[M.DRIVABLE_AREA, 0]))
        # Repeated ego rows only satisfy the scorer shape. Its TTC loop evaluates
        # time_idx=0 exactly once and generates all ego projections itself.
        ttc = self._scorer([endpoint]*3, CurrentObservation(endpoint, forecast=True))
        ttc._calculate_ttc_optimized()
        comfort = bool(ego_is_comfortable(np.stack([f.ego for f in frames])[None], np.arange(15)*.5).all())
        history_scorer = self._scorer(frames, CurrentObservation(endpoint))
        history_scorer._calculate_driving_direction_compliance()
        interval = self._scorer([start, endpoint], CurrentObservation(endpoint))
        interval._calculate_progress()
        interval._calculate_lane_keeping()
        # A fixed same-state CV reference, never an expert future or group winner.
        initial = ego_state(start)
        xy = initial.center.array[:2]
        theta = start.ego[S.HEADING]
        vx, vy = start.ego[S.VELOCITY_2D]
        displacement = np.array([np.cos(theta)*vx-np.sin(theta)*vy,
                                 np.sin(theta)*vx+np.cos(theta)*vy])*.5
        projected = self.centerline.project([Point(*xy), Point(*(xy+displacement))])
        reference = max(float(projected[1]-projected[0]), 0.)
        raw_progress = float(interval._progress_raw[0])
        # Reuse repository aggregation and all its original weights/normalization.
        interval._num_proposals = 2
        interval._progress_raw = np.array([reference, raw_progress])
        interval._multi_metrics = np.tile([[min(nc)], [min(dac)]], (1,2))
        values = interval._weighted_metrics[:,0].copy()
        values[W.TTC] = ttc._weighted_metrics[W.TTC,0]
        values[W.COMFORTABLE] = float(comfort)
        values[W.DRIVING_DIRECTION] = history_scorer._weighted_metrics[W.DRIVING_DIRECTION,0]
        interval._weighted_metrics = np.tile(values[:,None], (1,2))
        with np.errstate(divide='ignore', invalid='ignore'):
            score = float(interval._aggregate_scores(batch=True)[1])
        result = dict(reward=score, NC=min(nc), DAC=min(dac),
            EP=float(interval._weighted_metrics[W.PROGRESS,1]), TTC=float(values[W.TTC]),
            comfort=float(comfort), direction=float(values[W.DRIVING_DIRECTION]),
            progress_m=raw_progress, reference_progress_m=reference)
        if not np.isfinite(list(result.values())).all():
            raise ValueError('Non-finite causal H1 reward')
        return result
