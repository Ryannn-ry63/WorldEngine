import math

from worldengine.components.agents.controller.abstract_controller import AbstractController
from worldengine.components.agents.controller.motion_model.build_motion_model import build_motion_model
from worldengine.components.agents.controller.tracker.build_tracker import build_tracker
from worldengine.engine.engine_utils import get_engine


class TwoStageController(AbstractController):
    """
    Implements a two stage tracking controller. The two stages comprises of:
        1. an AbstractTracker - This is to simulate a low level controller layer that is present in real AVs.
        2. an AbstractMotionModel - Describes how the AV evolves according to a physical model.
    """

    def __init__(self, agent):
        """
        Constructor for TwoStageController
        :param scenario: Scenario
        :param tracker: The tracker used to compute control actions
        :param motion_model: The motion model to propagate the control actions
        """
        super(TwoStageController, self).__init__(agent)

        self._tracker = build_tracker(self.agent.config)(self.agent)
        self._motion_model = build_motion_model(self.agent.config)(self.agent)

    def step(self):
        """Inherited, see superclass."""
        if self.agent.config.get('online_corrected_bicycle', False) and self.agent.trajectory is not None:
            # Recompute low-level control at its configured 0.1 s resolution;
            # one external action still advances exactly the 0.5 s H1 interval.
            duration = self._motion_model._frame_rate
            count = max(1, math.ceil(duration / self._tracker._discretization_time))
            dt = duration / count
            try:
                self._motion_model._frame_rate = dt
                for index in range(count):
                    self._tracker._online_elapsed = index * dt
                    accel_cmd, steering_rate_cmd = self._tracker.track_trajectory()
                    self._motion_model.propagate_state(accel_cmd, steering_rate_cmd)
            finally:
                self._motion_model._frame_rate = duration
                if hasattr(self._tracker, '_online_elapsed'):
                    del self._tracker._online_elapsed
            return
        if self.agent.trajectory is not None:
            for attr in ['waypoints', 'velocities', 'headings', 'angular_velocities']:
                if hasattr(self.agent.trajectory, attr):
                    value = getattr(self.agent.trajectory, attr)
                    if isinstance(value, list) and len(value) > 1:
                        setattr(self.agent.trajectory, attr, value[1:])

            accel_cmd, steering_rate_cmd = self._tracker.track_trajectory()
        else:
            accel_cmd, steering_rate_cmd = self.agent.lower_action
        self._motion_model.propagate_state(accel_cmd, steering_rate_cmd)
