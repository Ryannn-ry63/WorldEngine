"""Pure memory conversion matching NAVFormerClient's deployed trajectory path."""
import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from worldengine.common.dataclasses import Trajectory


def expand_future_poses(candidates):
    values = np.asarray(candidates, dtype=np.float64)
    if values.shape != (20, 8, 3) or not np.isfinite(values).all():
        raise ValueError('Expected 20 finite eight-pose rear-axle-local candidates')
    start = np.concatenate([np.zeros((20, 1, 3)), values[:, :-1]], axis=1)
    fractions = np.array([.2, .4, .6, .8]).reshape(1, 1, 4, 1)
    xy = start[..., :2, None].swapaxes(-1, -2) + fractions * (values[..., :2]-start[..., :2])[:, :, None, :]
    delta = values[..., 2]-start[..., 2]
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    headings = start[..., 2, None] + fractions[..., 0] * delta[..., None]
    headings = np.arctan2(np.sin(headings), np.cos(headings))[..., None]
    intermediate = np.concatenate([xy, headings], axis=-1)
    return np.concatenate([intermediate, values[:, :, None, :]], axis=2).reshape(20, 40, 3)


def to_world_actions(candidates, ego_agent):
    # Preserve the legacy 10 Hz rear->center conversion and 2 Hz downsampling.
    # This only converts an action request; real dynamics remain in SimEngine.
    expanded = expand_future_poses(candidates)
    heading = float(ego_agent.current_heading)
    rotation = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    result = []
    for candidate in expanded:
        poses = np.concatenate([np.zeros((1, 3)), candidate])
        rear_xy = poses[:, :2] @ rotation.T + ego_agent.rear_vehicle.current_position
        headings = poses[:, 2] + heading
        rear_velocity = np.diff(rear_xy, axis=0)/.1
        rear_velocity = np.vstack([rear_velocity, rear_velocity[-1]])
        waypoints = []
        for xy, angle, velocity in zip(rear_xy, headings, rear_velocity):
            state = EgoState.build_from_rear_axle(StateSE2(xy[0], xy[1], angle),
                tire_steering_angle=0., vehicle_parameters=get_pacifica_parameters(),
                time_point=TimePoint(0.), rear_axle_velocity_2d=StateVector2D(*velocity),
                rear_axle_acceleration_2d=StateVector2D(0., 0.))
            waypoints.append([state.waypoint.x, state.waypoint.y])
        waypoints = np.asarray(waypoints)
        velocity = np.diff(waypoints, axis=0)/.1
        velocity = np.vstack([velocity, velocity[-1]])
        angular = np.diff(headings)/.1
        angular = np.append(angular, angular[-1])
        result.append(Trajectory(waypoints=waypoints[::5], velocities=velocity[::5],
                                 headings=headings[::5], angular_velocities=angular[::5]))
    return result
