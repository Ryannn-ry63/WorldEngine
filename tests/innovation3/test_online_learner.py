from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'projects/AlgEngine/scripts/diffusiondrive'))
from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector, exact_group_loss
from innovation3.learner import OnlineV3Learner


def model():
    return SceneConditionedTrajectorySetSelector(feature_dim=16, model_dim=16,
        route_bev_dim=16, context_dim=16, geometry_hidden_dim=8, num_heads=4,
        feedforward_dim=32, num_set_layers=1)


def context():
    return dict(candidate_features=torch.randn(1,20,16),candidate_trajectories=torch.randn(1,20,8,3),
                route_bev_features=torch.randn(1,20,8,16),status_token=torch.randn(1,1,16),
                ego_query=torch.randn(1,1,16),agents_query=torch.randn(1,3,16))


class LearnerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.learner = OnlineV3Learner(model())
        self.x = context(); self.base = torch.randn(1,20)

    def test_real_update_keeps_reference_and_inputs_frozen(self):
        self.x['candidate_features'].requires_grad_(True)
        before = self.learner.state_dict()
        self.learner.choose(self.x, self.base, 'step4')
        result = self.learner.update(torch.arange(20)[None, :], 'step4')
        self.assertTrue(result['optimized'])
        self.assertEqual(self.learner.version, 1)
        self.assertIsNone(self.x['candidate_features'].grad)
        self.assertTrue(all(p.grad is None for p in self.learner.reference.parameters()))
        self.assertTrue(all(torch.equal(v, self.learner.reference.state_dict()[k]) for k,v in before['reference'].items()))
        self.assertTrue(any(not torch.equal(v, self.learner.selector.state_dict()[k]) for k,v in before['selector'].items()))

    def test_matches_existing_exact_group_objective(self):
        self.learner.choose(self.x,self.base,'s')
        _, logits, reference = self.learner.pending
        rewards=torch.arange(20,dtype=torch.float32)[None,:]
        loss,_,_=exact_group_loss(logits,reference,rewards,torch.ones_like(rewards,dtype=torch.bool),1.,1e-3)
        result=self.learner.update(rewards,'s')
        self.assertAlmostEqual(result['loss'],float(loss.detach()),places=7)

    def test_trained_v3_behavior_uses_exact_inference_path(self):
        with torch.no_grad():
            self.learner.selector.delta_head[-1].weight.normal_(0,0.02)
        learner=OnlineV3Learner(self.learner.selector)
        with torch.no_grad():
            expected=(self.base+learner.reference(**self.x)).softmax(-1)
        _,actual,_=learner.choose(self.x,self.base,'s')
        self.assertTrue(torch.equal(expected,actual))

    def test_ties_do_not_apply_weight_decay(self):
        before = self.learner.state_dict()
        self.learner.choose(self.x, self.base, 's')
        result = self.learner.update(torch.ones(1,20), 's')
        self.assertFalse(result['optimized']); self.assertEqual(self.learner.version,0)
        self.assertTrue(all(torch.equal(v,self.learner.selector.state_dict()[k]) for k,v in before['selector'].items()))

    def test_stale_feedback_and_double_choose_rejected(self):
        self.learner.choose(self.x, self.base, 's')
        with self.assertRaises(RuntimeError): self.learner.choose(self.x,self.base,'t')
        with self.assertRaises(RuntimeError): self.learner.update(torch.ones(1,20),'t')
        with self.assertRaises(RuntimeError): self.learner.state_dict()

    def test_resume_restores_action_rng_and_optimizer(self):
        self.learner.choose(self.x,self.base,'s')
        self.learner.update(torch.arange(20)[None,:], 's')
        saved = self.learner.state_dict()
        restored = OnlineV3Learner(model()); restored.load_state_dict(saved)
        left=self.learner.choose(self.x,self.base,'t'); right=restored.choose(self.x,self.base,'t')
        self.assertEqual(left[0],right[0]); self.assertTrue(torch.equal(left[1],right[1]))
        rewards=torch.arange(20).flip(0)[None,:]
        self.learner.update(rewards,'t'); restored.update(rewards,'t')
        self.assertTrue(all(torch.equal(v,restored.selector.state_dict()[k]) for k,v in self.learner.selector.state_dict().items()))

    def test_reward_cannot_be_a_forward_input(self):
        with self.assertRaises(ValueError):
            self.learner.choose(dict(self.x,reward=torch.ones(1,20)),self.base,'s')

    def test_inference_tensors_cloned_before_backward(self):
        with torch.inference_mode(): x=context(); base=torch.zeros(1,20)
        self.learner.choose(x,base,'s')
        self.assertTrue(self.learner.update(torch.arange(20)[None,:],'s')['optimized'])


if __name__ == '__main__':
    unittest.main()
