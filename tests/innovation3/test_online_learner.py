from pathlib import Path
import copy
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'projects/AlgEngine/scripts/diffusiondrive'))
from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector, exact_group_loss
from innovation3.learner import (OnlineV3Learner,
    V3_INITIALIZED_SELECTOR_FINETUNE)


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

    def test_gradient_reducer_runs_only_for_signal_updates(self):
        calls = []
        def reducer(parameters):
            parameters = list(parameters)
            calls.append(sum(parameter.grad is not None for parameter in parameters))
        learner = OnlineV3Learner(model(), gradient_reducer=reducer)
        learner.choose(self.x, self.base, 'signal')
        self.assertTrue(learner.update(torch.arange(20, dtype=torch.float32)[None, :], 'signal')['optimized'])
        self.assertEqual(len(calls), 1)
        self.assertGreater(calls[0], 0)
        learner.choose(self.x, self.base, 'tie')
        self.assertFalse(learner.update(torch.ones(1, 20), 'tie')['optimized'])
        self.assertEqual(len(calls), 1)

    def test_update_coordinator_runs_on_ties_and_controls_global_step(self):
        calls = []
        def coordinator(parameters, local_optimized):
            calls.append(bool(local_optimized))
            return True if len(calls) == 1 else False
        learner = OnlineV3Learner(model(), update_coordinator=coordinator)
        learner.choose(self.x, self.base, 'signal')
        self.assertTrue(learner.update(torch.arange(20, dtype=torch.float32)[None, :], 'signal')['optimized'])
        learner.choose(self.x, self.base, 'tie')
        self.assertFalse(learner.update(torch.ones(1, 20), 'tie')['optimized'])
        self.assertEqual(calls, [True, False])

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

    def test_second_plan_keeps_v3_and_adds_only_new_residual(self):
        trained=model().eval()
        with torch.no_grad():
            trained.delta_head[-1].weight.normal_(0,0.02)
            trained.delta_head[-1].bias.fill_(0.3)
        original={k:v.clone() for k,v in trained.state_dict().items()}
        learner=OnlineV3Learner(trained)
        with torch.no_grad():
            correction=learner.selector(**self.x)
            frozen_logits=self.base+trained(**self.x)
        self.assertTrue(torch.equal(correction,torch.zeros_like(correction)))
        learner.choose(self.x,self.base,'s')
        self.assertTrue(torch.equal(learner.pending[1].detach(),frozen_logits))
        learner.update(torch.arange(20)[None,:],'s')
        self.assertTrue(all(torch.equal(v,trained.state_dict()[k]) for k,v in original.items()))
        self.assertTrue(all(torch.equal(v,learner.reference.state_dict()[k]) for k,v in original.items()))
        self.assertFalse(any(p.requires_grad for p in learner.reference.parameters()))
        optimizer_ids={id(p) for group in learner.optimizer.param_groups for p in group['params']}
        self.assertEqual(optimizer_ids,{id(p) for p in learner.selector.parameters()})
        # After learning the frozen V3 term must still be present in every logit.
        with torch.no_grad():
            expected=(frozen_logits+learner.selector(**self.x)).softmax(-1)
        _,actual,_=learner.choose(self.x,self.base,'next')
        self.assertTrue(torch.equal(actual,expected))

    def test_v3_initialized_finetune_starts_from_complete_v3_without_double_add(self):
        trained = model().eval()
        with torch.no_grad():
            trained.delta_head[-1].weight.normal_(0, 0.02)
            trained.delta_head[-1].bias.fill_(0.3)
        original = {k: v.clone() for k, v in trained.state_dict().items()}
        learner = OnlineV3Learner(
            trained, parameterization=V3_INITIALIZED_SELECTOR_FINETUNE)
        self.assertTrue(all(torch.equal(v, learner.selector.state_dict()[k])
                            for k, v in original.items()))
        self.assertTrue(all(torch.equal(v, learner.reference.state_dict()[k])
                            for k, v in original.items()))
        with torch.no_grad():
            expected = self.base + trained(**self.x)
        selected, probabilities, version = learner.choose(self.x, self.base, 's')
        self.assertEqual(version, 0)
        self.assertTrue(torch.allclose(learner.pending[1].detach(), expected,
                                       atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.equal(probabilities, expected.softmax(-1)))
        self.assertEqual(int(probabilities.argmax()), int(expected.argmax()))
        learner.update(torch.arange(20)[None, :], 's')
        self.assertTrue(any(not torch.equal(v, learner.selector.state_dict()[k])
                            for k, v in original.items()))
        self.assertTrue(all(torch.equal(v, trained.state_dict()[k])
                            for k, v in original.items()))
        for prefix in ('feature_projection', 'geometry_encoder', 'route_cross_attention',
                       'context_cross_attention', 'set_encoder', 'delta_head'):
            self.assertTrue(any(not torch.equal(v, learner.selector.state_dict()[k])
                                for k, v in original.items() if k.startswith(prefix)), prefix)
        optimizer_ids = {id(p) for group in learner.optimizer.param_groups for p in group['params']}
        self.assertEqual(optimizer_ids, {id(p) for p in learner.selector.parameters()})
        self.assertEqual(learner.state_dict()['schema_version'], 3)
        self.assertEqual(learner.state_dict()['parameterization'],
                         V3_INITIALIZED_SELECTOR_FINETUNE)

    def test_parameterization_checkpoint_boundaries_are_not_interchangeable(self):
        state = OnlineV3Learner(
            model(), parameterization=V3_INITIALIZED_SELECTOR_FINETUNE).state_dict()
        with self.assertRaises(RuntimeError):
            self.learner.load_state_dict(state)
        state = self.learner.state_dict()
        with self.assertRaises(RuntimeError):
            OnlineV3Learner(
                model(), parameterization=V3_INITIALIZED_SELECTOR_FINETUNE
            ).load_state_dict(state)

    def test_old_finetune_checkpoint_cannot_resume_as_residual(self):
        saved=self.learner.state_dict()
        saved['schema_version']=1
        with self.assertRaises(RuntimeError): self.learner.load_state_dict(saved)

    def test_schema3_rejects_different_offline_selector_fingerprint(self):
        source = dict(kind='offline_selector_file', sha256='a' * 64)
        trained = model()
        with torch.no_grad():
            trained.delta_head[-1].bias.fill_(0.25)
        learner = OnlineV3Learner(trained, parameterization=V3_INITIALIZED_SELECTOR_FINETUNE,
                                  initialization_source=source)
        state = learner.state_dict()
        other = model()
        with torch.no_grad():
            other.delta_head[-1].bias.fill_(0.75)
        restored = OnlineV3Learner(other, parameterization=V3_INITIALIZED_SELECTOR_FINETUNE,
                                   initialization_source=source)
        with self.assertRaisesRegex(RuntimeError, 'provenance'):
            restored.load_state_dict(state)

    def test_schema3_rejects_different_source_sha_and_tampered_reference(self):
        source = dict(kind='offline_selector_file', sha256='b' * 64)
        template = model()
        learner = OnlineV3Learner(copy.deepcopy(template), parameterization=V3_INITIALIZED_SELECTOR_FINETUNE,
                                  initialization_source=source)
        state = learner.state_dict()
        mismatched = OnlineV3Learner(copy.deepcopy(template), parameterization=V3_INITIALIZED_SELECTOR_FINETUNE,
                                     initialization_source=dict(kind='offline_selector_file', sha256='c' * 64))
        with self.assertRaisesRegex(RuntimeError, 'source'):
            mismatched.load_state_dict(state)
        tampered = copy.deepcopy(state)
        key = next(iter(tampered['reference']))
        tampered['reference'][key] = tampered['reference'][key].clone()
        tampered['reference'][key].view(-1)[0] += 1
        with self.assertRaisesRegex(RuntimeError, 'reference'):
            learner.load_state_dict(tampered)

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

    def test_schema3_restore_is_atomic_and_config_includes_attention_heads(self):
        from innovation3.live_learning import state_digest
        trained = model().eval()
        with torch.no_grad():
            trained.delta_head[-1].weight.normal_(0, .02)
        learner = OnlineV3Learner(trained, parameterization=V3_INITIALIZED_SELECTOR_FINETUNE)
        learner.choose(self.x, self.base, 'first')
        learner.update(torch.arange(20)[None, :], 'first')
        state = learner.state_dict()
        changed_heads = copy.deepcopy(trained)
        changed_heads.num_heads = 2  # same state layout, different config
        bad_arch = OnlineV3Learner(changed_heads, parameterization=V3_INITIALIZED_SELECTOR_FINETUNE)
        with self.assertRaisesRegex(RuntimeError, 'architecture'):
            bad_arch.load_state_dict(state)
        restored = OnlineV3Learner(trained, parameterization=V3_INITIALIZED_SELECTOR_FINETUNE)
        before = state_digest(restored.state_dict())
        for mutation in ('rng', 'moment', 'counter', 'nan', 'reference'):
            bad = copy.deepcopy(state)
            if mutation == 'rng': bad['rng'] = torch.ones(3, dtype=torch.uint8)
            if mutation == 'moment':
                next(iter(bad['optimizer']['state'].values()))['exp_avg'] = torch.ones(1)
            if mutation == 'counter': bad['attempts'] = -1
            if mutation == 'nan': next(iter(bad['selector'].values())).view(-1)[0] = float('nan')
            if mutation == 'reference': next(iter(bad['reference'].values())).view(-1)[0] += 1
            with self.subTest(mutation=mutation):
                with self.assertRaises((ValueError, RuntimeError)):
                    restored.load_state_dict(bad)
                self.assertEqual(state_digest(restored.state_dict()), before)
        restored.load_state_dict(state)
        self.assertEqual(state_digest(restored.state_dict()), state_digest(state))
        left = learner.choose(self.x, self.base, 'next')
        right = restored.choose(self.x, self.base, 'next')
        self.assertEqual(left[0], right[0])
        self.assertTrue(torch.equal(left[1], right[1]))
        learner.update(torch.arange(20).flip(0)[None, :], 'next')
        restored.update(torch.arange(20).flip(0)[None, :], 'next')
        self.assertEqual(state_digest(learner.state_dict()), state_digest(restored.state_dict()))

    def test_legacy_reference_forward_is_reused(self):
        from unittest.mock import patch
        with patch.object(self.learner.reference, 'forward', wraps=self.learner.reference.forward) as call:
            self.learner.choose(self.x, self.base, 'one')
            self.assertEqual(call.call_count, 1)

    def test_reward_cannot_be_a_forward_input(self):
        with self.assertRaises(ValueError):
            self.learner.choose(dict(self.x,reward=torch.ones(1,20)),self.base,'s')

    def test_wrong_input_dtype_is_rejected_before_torch_forward(self):
        bad = dict(self.x, candidate_trajectories=self.x['candidate_trajectories'].double())
        with self.assertRaisesRegex(ValueError, 'dtype/device'):
            self.learner.choose(bad, self.base, 'wrong_dtype')
        self.assertIsNone(self.learner.pending)

    def test_inference_tensors_cloned_before_backward(self):
        with torch.inference_mode(): x=context(); base=torch.zeros(1,20)
        self.learner.choose(x,base,'s')
        self.assertTrue(self.learner.update(torch.arange(20)[None,:],'s')['optimized'])


if __name__ == '__main__':
    unittest.main()
