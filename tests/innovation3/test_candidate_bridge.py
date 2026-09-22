import copy
import ast
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'projects/SimEngine'))
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3.candidate_inputs import from_export, FIELDS
from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector
from worldengine.online.actions import to_world_actions, expand_future_poses
from worldengine.online.headless import HeadlessSimulator
from worldengine.components.agents.client.navformer_client import NAVFormerClient
from test_headless_snapshot import scene_fixture


class CandidateInputsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.model=SceneConditionedTrajectorySetSelector().eval().requires_grad_(False)
        with torch.no_grad():
            self.model.delta_head[-1].weight.normal_(0,.02)
        source={key:torch.randn(shape).numpy() for key,(_,shape) in FIELDS.items()}
        x={name:torch.from_numpy(source[key])[None] for key,(name,_) in FIELDS.items()}
        base=torch.randn(1,20)
        with torch.no_grad(): logits=base+self.model(**x)
        source.update(schema_version=1, reference_logits=base[0].numpy(),
                      current_logits=logits[0].numpy(),selected_indices=np.array(int(logits.argmax()),dtype=np.int64))
        self.result=dict(token='current',diffusiondrive_rollout_context=source)

    def test_current_context_matches_frozen_v3_without_rewards(self):
        # Outer evaluator fields must never become selector inputs.
        self.result['score']=float('nan')
        context,base,error=from_export(self.result,'current',self.model)
        self.assertEqual(set(context),{name for name,_ in FIELDS.values()})
        self.assertEqual(error,0.)
        self.assertEqual(tuple(base.shape),(1,20))
        self.assertTrue(all(not t.requires_grad for t in context.values()))

    def test_stale_token_reward_fields_and_wrong_v3_rejected(self):
        with self.assertRaises(ValueError): from_export(self.result,'stale',self.model)
        bad=copy.deepcopy(self.result);bad['diffusiondrive_rollout_context']['rewards']=np.zeros(20)
        with self.assertRaises(ValueError): from_export(bad,'current',self.model)
        bad=copy.deepcopy(self.result);bad['diffusiondrive_rollout_context']['current_logits']+=10
        with self.assertRaises(RuntimeError): from_export(bad,'current',self.model)

    def test_export_does_not_alias_input_and_accepts_float64_boundary(self):
        source=self.result['diffusiondrive_rollout_context']
        source['candidate_trajectories_8']=source['candidate_trajectories_8'].astype(np.float64)
        context,_,_=from_export(self.result,'current',self.model)
        before=context['candidate_trajectories'].clone()
        source['candidate_trajectories_8'][:]=100
        self.assertTrue(torch.equal(before,context['candidate_trajectories']))
        self.assertEqual(context['candidate_trajectories'].dtype,torch.float32)


class ActionConversionTest(unittest.TestCase):
    def test_matches_legacy_file_client_for_curves_and_rotated_ego(self):
        scene=scene_fixture()
        scene['object_track']['ego']['state']['heading'][:]=.7
        sim=HeadlessSimulator('fixture',scene,'NR',2,17)
        try:
            bank=np.zeros((20,8,3))
            x=np.arange(1,9)*2.
            for i in range(20):
                bend=(i-10)*.002
                bank[i,:,0]=x;bank[i,:,1]=bend*x*x
                bank[i,:,2]=np.arctan2(2*bend*x,np.ones(8))
            # Heading wrap case crosses -pi/pi at a segment boundary.
            bank[0,:,2]=[3.1,-3.1,3.0,-3.0,3.,3.,3.,3.]
            expanded=expand_future_poses(bank)
            np.testing.assert_array_equal(expanded[:,4::5],bank)
            actual=to_world_actions(bank,sim.engine.agent_manager.ego_agent)
            with tempfile.TemporaryDirectory() as directory:
                client=NAVFormerClient.__new__(NAVFormerClient)
                client.agent=sim.engine.agent_manager.ego_agent
                client.config=dict(num_history=1,planner_data_path=directory)
                for i in range(20):
                    np.save(Path(directory)/'fixture_1.npy',expanded[i])
                    expected=client.get_trajectory(0)
                    for key in ('waypoints','headings','velocities','angular_velocities'):
                        np.testing.assert_array_equal(getattr(actual[i],key),getattr(expected,key))
            self.assertEqual(actual[0].waypoints.shape,(9,2))
        finally:sim.close()

    def test_expansion_matches_actual_upstream_method(self):
        # Execute the unmodified upstream tensor-only method without importing
        # the MMDetection/CUDA model. This is independent of our NumPy formula.
        path=ROOT/'projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py'
        tree=ast.parse(path.read_text())
        method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='_expand_to_40')
        module=ast.Module(body=[method],type_ignores=[])
        namespace={'torch':torch}
        exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
        bank=np.random.RandomState(3).normal(size=(20,8,3))
        bank[0,:,2]=[3.1,-3.1,3.,-3.,3.,3.,3.,3.]
        expected=namespace['_expand_to_40'](None,torch.from_numpy(bank)).numpy()
        np.testing.assert_allclose(expand_future_poses(bank),expected,rtol=0,atol=1e-14)

    def test_rejects_bad_shapes_and_nonfinite_actions(self):
        with self.assertRaises(ValueError):expand_future_poses(np.zeros((20,9,3)))
        with self.assertRaises(ValueError):expand_future_poses(np.full((20,8,3),np.nan))
