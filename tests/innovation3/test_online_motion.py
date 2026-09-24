"""Independent unit checks for the opt-in online bicycle corrections."""
from types import SimpleNamespace
import numpy as np
from worldengine.components.agents.controller.motion_model.kinematic_bicycle import KinematicBicycleModel


def model(corrected=True, angle=.2, accel=-2., heading=.7):
 result={}
 rear=SimpleNamespace(current_acceleration=np.array([np.cos(heading),np.sin(heading)])*accel,
  current_heading=heading,current_position=np.zeros(2),current_velocity=np.array([np.cos(heading),np.sin(heading)])*5.,
  current_angular_velocity=0.,update_center_agent=lambda **kw:result.update(kw))
 agent=SimpleNamespace(rear_vehicle=rear,current_tire_steering=angle,MAX_STEERING=1.047,
   vehicle=SimpleNamespace(wheel_base=3.))
 m=KinematicBicycleModel.__new__(KinematicBicycleModel)
 m.agent=agent;m.config={'online_corrected_bicycle':corrected}
 m._frame_rate=.5;m._accel_time_constant=.2;m._steering_angle_time_constant=.05
 return m,result


def test_zero_steering_rate_keeps_nonzero_angle():
 m,r=model();m.propagate_state(-2.,0.)
 np.testing.assert_allclose(r['new_tire_steering'],.2)


def test_rate_has_time_units_and_acceleration_keeps_negative_sign():
 m,r=model();m.propagate_state(-2.,.1)
 np.testing.assert_allclose(r['new_tire_steering'],.2+(.5/(.5+.05))*(.5*.1))
 np.testing.assert_allclose(r['new_rear_acceleration'],[-2.,0.])
 np.testing.assert_allclose(r['new_rear_velocity'],[4.,0.])


def test_legacy_mode_preserves_previous_filter_equations():
 m,r=model(corrected=False);m.propagate_state(-2.,.1)
 np.testing.assert_allclose(r['new_tire_steering'],.2+(.5/(.5+.05))*(.1-.2))
 np.testing.assert_allclose(r['new_rear_acceleration'],[2.+(.5/(.5+.2))*(-2.-2.),0.])


def test_control_substeps_use_elapsed_reference_time_and_restore_interval():
 from worldengine.components.agents.controller.two_stage_controller import TwoStageController
 calls=[]
 tracker=SimpleNamespace(_discretization_time=.1)
 def track():
  calls.append(tracker._online_elapsed)
  return 0.,0.
 tracker.track_trajectory=track
 motion=SimpleNamespace(_frame_rate=.5,propagate_state=lambda a,b:None)
 controller=TwoStageController.__new__(TwoStageController)
 controller.agent=SimpleNamespace(config={'online_corrected_bicycle':True},trajectory=object())
 controller._tracker=tracker;controller._motion_model=motion
 controller.step()
 np.testing.assert_allclose(calls,np.arange(5)*.1)
 assert motion._frame_rate==.5 and not hasattr(tracker,'_online_elapsed')


def test_substep_failure_restores_runtime_fields():
 from worldengine.components.agents.controller.two_stage_controller import TwoStageController
 import pytest
 tracker=SimpleNamespace(_discretization_time=.1,track_trajectory=lambda:(0.,0.))
 def fail(a,b):raise RuntimeError('injected')
 motion=SimpleNamespace(_frame_rate=.5,propagate_state=fail)
 controller=TwoStageController.__new__(TwoStageController)
 controller.agent=SimpleNamespace(config={'online_corrected_bicycle':True},trajectory=object())
 controller._tracker=tracker;controller._motion_model=motion
 with pytest.raises(RuntimeError,match='injected'):controller.step()
 assert motion._frame_rate==.5 and not hasattr(tracker,'_online_elapsed')
