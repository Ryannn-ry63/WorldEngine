"""Execute warmup construction and the real tracker, including route exhaustion."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from shapely.geometry import LineString
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'projects/SimEngine'))
from worldengine.online.reward_runtime import RewardSession
from worldengine.online.headless import HeadlessSimulator
sys.path.insert(0,str(ROOT/'tests/innovation3'))
from test_headless_snapshot import scene_fixture


def session_for(ego, coords, offset=(100.,200.)):
 s=RewardSession.__new__(RewardSession)
 s.sim=SimpleNamespace(engine=SimpleNamespace(agent_manager=SimpleNamespace(ego_agent=ego)))
 s.offset=np.asarray(offset)
 s.centerline=SimpleNamespace(linestring=LineString(coords))
 return s


def test_live_start_time_grid_and_map_offset():
 ego=SimpleNamespace(current_position=np.array([5.,0.]),current_heading=0.,current_speed=4.)
 s=session_for(ego,[(100.,200.),(200.,200.)]); a=s.warmup_action()
 np.testing.assert_allclose(a.waypoints[:,0],5.+np.arange(9)*2.)
 np.testing.assert_allclose(a.waypoints[:,1],0.)
 np.testing.assert_allclose(a.headings,0.)


def test_curved_route_uses_local_map_tangents():
 ego=SimpleNamespace(current_position=np.array([0.,0.]),current_heading=0.,current_speed=4.)
 s=session_for(ego,[(100.,200.),(102.,200.),(102.,210.)]); a=s.warmup_action()
 np.testing.assert_allclose(a.waypoints[0],ego.current_position)
 assert a.headings[0] == ego.current_heading
 assert a.headings[-1] > 1.5
 assert np.isfinite(a.waypoints).all()


def test_route_end_and_stopped_ego_request_hold_at_live_pose():
 for x,speed in [(100.,4.),(4.,0.)]:
  ego=SimpleNamespace(current_position=np.array([x,0.]),current_heading=.3,current_speed=speed)
  a=session_for(ego,[(100.,200.),(200.,200.)]).warmup_action()
  np.testing.assert_allclose(a.waypoints,np.tile(ego.current_position,(9,1)))
  np.testing.assert_allclose(a.headings,.3)


def test_short_route_slows_without_minimum_horizon():
 ego=SimpleNamespace(current_position=np.array([9.,0.]),current_heading=0.,current_speed=8.)
 a=session_for(ego,[(100.,200.),(110.,200.)]).warmup_action()
 assert np.max(a.waypoints[:,0]) <=10.
 assert np.all(np.diff(a.waypoints[:,0])>0.)


def test_real_controller_stationary_action_brakes_and_replays():
 sim=HeadlessSimulator('fixture',scene_fixture(),'R',8,17)
 try:
  ego=sim.engine.agent_manager.ego_agent
  s=session_for(ego,[(0.,0.),(0.01,0.)],offset=(0.,0.))
  # Zero-length remaining route at the actual centered ego pose.
  p=np.asarray(ego.current_position)[:2]
  s.centerline=SimpleNamespace(linestring=LineString([p-[1.,0.],p]))
  a=s.warmup_action(); before=sim.snapshot();v=ego.current_speed
  sim.step(a);after=sim.snapshot()
  assert np.isfinite(sim.engine.agent_manager.ego_agent.current_speed)
  assert sim.engine.agent_manager.ego_agent.current_speed < v
  sim.restore(before);sim.step(a)
  assert sim.snapshot().state_hash==after.state_hash
 finally:sim.close()
