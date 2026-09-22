"""Behavioral reward tests with explicit geometry, no scene/metric cache input."""
from pathlib import Path
import sys
import copy
import pickle
from types import SimpleNamespace
import unittest
from dataclasses import replace
import numpy as np
from shapely.geometry import box
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'projects/SimEngine'))
from worldengine.online.reward import RewardFrame, RewardHistory, H1Reward, ego_state, CONTRACT, capture_reward_frame
from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


def frame(step, speed=4., lateral=0., actors=()):
    s=np.zeros(11);s[:4]=[step*2.,lateral,0.,speed]
    return RewardFrame(step,s,tuple(actors))


def actor(x,y=0.,speed=0.,heading=0.):
    return Agent(TrackedObjectType.VEHICLE,OrientedBox(StateSE2(x,y,heading),4.,2.,1.5),
                 StateVector2D(speed,0.),SceneObjectMetadata(0,'car',None,'car'))


class CausalRewardTest(unittest.TestCase):
    def setUp(self):
        self.map=PDMDrivableMap(['lane'],[SemanticMapLayer.LANE],[box(-100,-5,300,5)])
        self.line=PDMPath([StateSE2(-100.,0.,0.),StateSE2(300.,0.,0.)])
        self.reward=H1Reward(self.line,{'lane':object()},self.map,self.map)
        self.history=tuple(frame(i) for i in range(14))

    def test_constant_speed_on_route_is_comfortable_and_unit_progress(self):
        r=self.reward.score(self.history,frame(14))
        self.assertEqual([r[k] for k in ('NC','DAC','EP','TTC','comfort','direction','reward')],[1.]*7)
        self.assertAlmostEqual(r['progress_m'],2.)
        self.assertAlmostEqual(r['reference_progress_m'],2.)

    def test_full_ego_corner_offroad_is_failure(self):
        end=frame(14,lateral=4.5)
        self.assertEqual(self.reward.score(self.history,end)['DAC'],0.)
        self.assertEqual(self.reward.score(self.history,end)['reward'],0.)

    def test_actual_front_collision_zeroes_score(self):
        end=frame(14,actors=[actor(31)])
        r=self.reward.score(self.history,end)
        self.assertEqual(r['NC'],0.)
        self.assertEqual(r['reward'],0.)

    def test_ttc_checks_one_second_beyond_h1_without_actual_collision(self):
        # Ego front at x=32.049; track rear at x=35. It is hit by 1s projection.
        r=self.reward.score(self.history,frame(14,actors=[actor(37)]))
        self.assertEqual(r['NC'],1.)
        self.assertEqual(r['TTC'],0.)
        self.assertAlmostEqual(r['reward'],7./12.)

    def test_future_actor_projection_uses_velocity(self):
        stationary=self.reward.score(self.history,frame(14,actors=[actor(37)]))
        moving=self.reward.score(self.history,frame(14,actors=[actor(37,speed=4.)]))
        self.assertEqual(stationary['TTC'],0.)
        self.assertEqual(moving['TTC'],1.)

    def test_missing_history_map_and_discontinuous_steps_rejected(self):
        with self.assertRaises(ValueError):self.reward.score(self.history[1:],frame(14))
        with self.assertRaises(ValueError):self.reward.score(self.history,frame(15))
        with self.assertRaises(ValueError):H1Reward(self.line,{},self.map,self.map)
        with self.assertRaises(ValueError):H1Reward(self.line,{'lane':1},None,self.map)

    def test_uncomfortable_actual_history_is_not_zero_padded(self):
        history=copy.deepcopy(self.history)
        for f in history:f.ego[5]=6.
        end=frame(14);end.ego[5]=6.
        self.assertEqual(self.reward.score(history,end)['comfort'],0.)

    def test_reference_is_independent_of_other_candidates(self):
        r1=self.reward.score(self.history,frame(14))
        end=frame(14);end.ego[0]+=2.
        r2=self.reward.score(self.history,end)
        self.assertEqual(r1['reference_progress_m'],r2['reference_progress_m'])
        self.assertEqual(self.reward.score(self.history,frame(14)),r1)

    def test_history_restore_is_detached_and_does_not_mix_branch(self):
        h=RewardHistory()
        for f in self.history:h.append(f)
        old=h.snapshot(); digest=h.state_hash
        h.append(frame(14));h.restore(old)
        self.assertEqual(h.state_hash,digest)
        old[-1].ego[0]+=100
        self.assertEqual(h.state_hash,digest)
        with self.assertRaises(ValueError):h.append(frame(20))

    def test_live_capture_needs_no_scene_or_future_and_copies_rear_state(self):
        rear=SimpleNamespace(current_position=np.array([10.,20.]),
            current_velocity=np.array([1.,3.]),current_acceleration=np.array([2.,4.]))
        ego=SimpleNamespace(rear_vehicle=rear,current_heading=np.pi/2,
            tire_steering=.2,current_angular_velocity=.3,current_angular_acceleration=.4)
        engine=SimpleNamespace(episode_step=7,
            agent_manager=SimpleNamespace(ego_agent=ego,all_agents={'ego':ego}))
        # No current_scene, object_track, logged state or metric cache exists.
        captured=capture_reward_frame(engine,np.array([100.,200.]),{})
        np.testing.assert_allclose(captured.ego[:2],[110.,220.])
        np.testing.assert_allclose(captured.ego[3:7],[3.,-1.,4.,-2.])
        rear.current_position[:]=0
        np.testing.assert_allclose(captured.ego[:2],[110.,220.])

    def test_serialized_history_restores_identical_next_reward(self):
        h=RewardHistory()
        for i in range(14):h.append(frame(i,actors=[actor(200,speed=3.)]))
        digest=h.state_hash
        restored=RewardHistory();restored.restore(pickle.loads(pickle.dumps(h.snapshot())))
        self.assertEqual(restored.state_hash,digest)
        self.assertEqual(self.reward.score(restored.frames,frame(14)),
                         self.reward.score(h.frames,frame(14)))

    def test_dimensions_use_rear_axle_offset(self):
        ego=ego_state(frame(14))
        self.assertAlmostEqual(ego.center.x-ego.rear_axle.x,1.461)
        self.assertEqual(CONTRACT['weights'],[5.,5.,2.,0.,0.])

if __name__=='__main__':unittest.main()
