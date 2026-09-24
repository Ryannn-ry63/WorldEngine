"""Behavioral checks for the same learner and wire gates used by online-probe."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3.live_audit import LiveStepGate
from innovation3.live_learning import LiveLearning, state_digest, learning_evidence
from innovation3.learner import OnlineV3Learner
from test_online_learner import model, context
from test_live_reward_protocol import REWARD_KEYS


def identity(step=13, version=0, state='state'):
    return dict(scene='scene', step=step, policy_version=version,
                state_hash=state, token=str(step))


def receipts(request, rewards, history_before='past', history_after='after'):
    components = [dict.fromkeys(REWARD_KEYS, 1.) for _ in rewards]
    for c, reward in zip(components, rewards): c['reward'] = reward
    selected = request['selected']
    main = dict(kind='main', identity=request['identity'], candidate_hash=request['candidate_hash'],
                selected=selected, next_hash='next'+str(request['identity']['step']),
                reward=copy.deepcopy(components[selected]))
    branches = dict(kind='branches', identity=request['identity'],
        candidate_hash=request['candidate_hash'], selected=selected, main_hash=main['next_hash'],
        branch_hashes=[main['next_hash']]*20, selected_parity=True, canonical_restored=True,
        reward=copy.deepcopy(components[selected]), rewards=components,
        selected_reward_parity=True, selected_history_parity=True,
        history_before_hash=history_before, history_after_hash=history_after)
    return main, branches


class LiveLearningTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(19)
        self.learner = OnlineV3Learner(model().eval())
        self.online = LiveLearning(self.learner)
        self.worker = LiveStepGate()
        self.x = context(); self.base = torch.randn(1,20)
        self.candidates = self.x['candidate_trajectories'][0].tolist()

    def choose(self, ident=None):
        ident = ident or identity()
        self.online.observe(ident); self.worker.observe(ident)
        request = self.online.choose(self.x, self.base, self.candidates)
        self.worker.choose(request)
        return request

    def finish(self, request, values, before='past', after='after'):
        main, branches = receipts(request, values, before, after)
        self.worker.main_executed(main); self.online.main_executed(main)
        self.worker.feedback(branches)
        ack, audit = self.online.update(branches)
        response = self.worker.updated(ack)
        self.online.acknowledged(response)
        return main, audit

    def test_real_optimizer_feedback_then_updated_policy_next_step(self):
        request = self.choose()
        self.assertEqual(request['current_logits'], request['reference_logits'])
        main, first = self.finish(request, [i/19 for i in range(20)])
        self.assertTrue(first['optimized'])
        self.assertGreater(first['same_context_logit_change_after_update'], 0)
        request2 = self.choose(identity(14,1,main['next_hash']))
        self.assertNotEqual(request2['current_logits'], request2['reference_logits'])
        _, second = self.finish(request2, [1.]*20, 'after', 'after2')
        self.assertEqual(second['residual_hash_before'], first['residual_hash_after'])
        self.assertFalse(second['optimized'])
        self.assertEqual(second['version_after'], 1)
        self.assertEqual(second['same_context_logit_change_after_update'], 0.)
        self.assertEqual(second['optimizer_hash_before'], second['optimizer_hash_after'])
        summary = learning_evidence([dict(kind='online_update', **a) for a in (first, second)])
        self.assertTrue(summary['closed_loop_learning_verified'])
        self.assertEqual(summary['actual_optimizer_steps'], 1)
        self.assertEqual(summary['groups_without_signal'], 1)
        self.assertLess(first['times_ns']['action'], first['times_ns']['main'])
        self.assertLess(first['times_ns']['main'], first['times_ns']['feedback'])
        self.assertLess(first['times_ns']['feedback'], first['times_ns']['update_complete'])

    def test_float64_receipt_parity_precedes_float32_reward_conversion(self):
        request = self.choose()
        # 1/3 is not exactly representable in float32; raw parity must stay exact.
        _, audit = self.finish(request, [1./3]*20)
        self.assertFalse(audit['optimized'])

    def test_roundoff_noise_cannot_create_update(self):
        request = self.choose()
        with patch.object(self.learner.optimizer, 'step', wraps=self.learner.optimizer.step) as step:
            _, audit = self.finish(request, [1.+(i%2)*2e-16 for i in range(20)])
        step.assert_not_called()
        self.assertEqual(audit['version_after'], 0)

    def test_invalid_feedback_never_reaches_optimizer(self):
        for bad in ('main_reward','candidate','history','nan','short','state'):
            with self.subTest(bad=bad):
                self.setUp()
                request = self.choose()
                main, branches = receipts(request, [i/19 for i in range(20)])
                self.online.main_executed(main)
                if bad == 'main_reward': branches['rewards'][request['selected']]['reward'] += 1e-12
                if bad == 'candidate': branches['candidate_hash'] = 'wrong'
                if bad == 'history':
                    self.online.wire.history_hash = 'expected'
                if bad == 'nan': branches['rewards'][0]['reward'] = float('nan')
                if bad == 'short': branches['rewards'].pop()
                if bad == 'state': branches['branch_hashes'][request['selected']] = 'wrong'
                with patch.object(self.learner, 'update') as update:
                    with self.assertRaises((ValueError,RuntimeError)): self.online.update(branches)
                update.assert_not_called()
                self.assertEqual(self.learner.version, 0)

    def test_feedback_before_main_and_early_ack_rejected(self):
        request = self.choose()
        main, branches = receipts(request, [0.]*20)
        with self.assertRaises(RuntimeError): self.online.update(branches)
        ack = dict(kind='feedback_ack', identity=request['identity'],
                   candidate_hash=request['candidate_hash'], next_hash=main['next_hash'],
                   policy_version=0, optimized=False)
        with self.assertRaises(RuntimeError): self.worker.updated(ack)

    def test_worker_rejects_wrong_and_duplicate_update_versions(self):
        request = self.choose()
        main, branches = receipts(request, list(range(20)))
        self.worker.main_executed(main); self.worker.feedback(branches)
        self.online.main_executed(main); ack, audit = self.online.update(branches)
        with self.assertRaises(RuntimeError): self.worker.updated(dict(ack, policy_version=2))
        with self.assertRaises(ValueError): self.worker.updated(dict(ack, candidate_hash='stale'))
        response = self.worker.updated(ack)
        with self.assertRaises(RuntimeError): self.worker.updated(ack)
        with self.assertRaises(RuntimeError): self.online.observe(identity(14,1,main['next_hash']))
        with self.assertRaises(RuntimeError): self.online.acknowledged(dict(response, policy_version=0))
        self.online.acknowledged(response)
        with self.assertRaises(RuntimeError): self.online.observe(identity(14,0,main['next_hash']))

    def test_next_state_and_history_must_continue(self):
        request = self.choose()
        main, _ = self.finish(request, list(range(20)))
        with self.assertRaises(RuntimeError): self.worker.observe(identity(14,1,'wrong'))
        request = self.choose(identity(14,1,main['next_hash']))
        main, branches = receipts(request, list(range(20)), history_before='wrong')
        self.online.main_executed(main)
        with self.assertRaises(RuntimeError): self.online.update(branches)

    def test_no_signal_or_final_only_update_is_not_closed_loop_learning_pass(self):
        request = self.choose()
        _, audit = self.finish(request, list(range(20)))
        evidence = learning_evidence([dict(kind='online_update', **audit)])
        self.assertFalse(evidence['closed_loop_learning_verified'])
        self.assertFalse(learning_evidence([])['closed_loop_learning_verified'])

    def test_categorical_rng_is_independent_of_global_candidate_rng(self):
        before = torch.get_rng_state().clone()
        self.choose()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__': unittest.main()
