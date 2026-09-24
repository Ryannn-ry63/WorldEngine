"""Execute the real online CLI loop on CPU with explicit visual/worker doubles.

These tests certify orchestration only; they are never live GPU acceptance.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3 import online_probe
from innovation3.ddp_h1_probe import digest_state
from innovation3.ddp_online_probe import validate_bindings
from innovation3.candidate_inputs import FIELDS
from innovation3.image_transport import metadata_to_wire
from innovation3.live_audit import LiveStepGate
from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector
from test_live_learning import receipts


class WorkerDouble:
    def __init__(self, all_ties=False, stale_ack=False):
        self.wire = LiveStepGate()
        self.index = 0
        self.queue = [dict(kind='ready', reward_adapter=True, online_updates=True,
            warmup_transitions=13, reward_contract=dict(name='causal_h1_pdm_components_v1'),
            scene='fixture', runtime={})]
        self.all_ties, self.stale_ack = all_ties, stale_ack
        self.append_observation()

    def append_observation(self):
        ident = dict(scene='fixture', step=self.index, token=str(self.index),
                     state_hash='state' if self.index <= 13 else 'next'+str(self.index-1),
                     policy_version=self.wire.gate.policy_version)
        if self.index >= 13: self.wire.observe(ident)
        frame = dict(scene_token='fixture', token=str(self.index), frame_idx=self.index,
                     timestamp=self.index*500000)
        self.queue.append(dict(kind='observation', identity=ident,
                               frame=metadata_to_wire(frame), images={}))

    def receive(self):
        if not self.queue: raise RuntimeError('Driver tried to observe before acknowledgement')
        return self.queue.pop(0)

    def send(self, value):
        kind = value['kind']
        if kind == 'warmup':
            self.index += 1; self.append_observation()
        elif kind == 'action':
            self.wire.choose(value)
            values = [1./3]*20 if self.all_ties or self.index > 13 else [i/19 for i in range(20)]
            main, branches = receipts(value, values,
                'past' if self.index == 13 else 'history'+str(self.index-1), 'history'+str(self.index))
            self.wire.main_executed(main); self.wire.feedback(branches)
            self.queue.extend([main, branches])
        elif kind == 'feedback_ack':
            reply = self.wire.updated(value)
            if self.stale_ack: reply['policy_version'] = -1
            self.queue.append(reply)
            self.index += 1
            if self.index < 21: self.append_observation()
        elif kind == 'close':
            self.queue.append(dict(kind='closed', idm_fallbacks=0))
        else: raise AssertionError(kind)


class FrozenDouble:
    def __init__(self):
        self.selector = SceneConditionedTrajectorySetSelector().eval().requires_grad_(False)
        self.module = SimpleNamespace(planning_head=SimpleNamespace(scene_selector=self.selector))
        self.parameters = self.selector.parameters
        self.state_dict = self.selector.state_dict
        self.source = {k: torch.randn(shape).numpy() for k, (_, shape) in FIELDS.items()}
        inputs = {name:torch.from_numpy(self.source[key])[None]
                  for key,(name,_) in FIELDS.items()}
        base = torch.randn(1,20)
        with torch.no_grad(): logits = base+self.selector(**inputs)
        self.source.update(schema_version=1, reference_logits=base[0].numpy(),
            current_logits=logits[0].numpy(), selected_indices=np.array(int(logits.argmax()),dtype=np.int64))

    def __call__(self, token, **kwargs):
        return [dict(token=token, diffusiondrive_rollout_context=copy.deepcopy(self.source))]


class DriverTest(unittest.TestCase):

    def test_ddp_binding_falls_back_to_explicit_physical_ids_without_uuid(self):
        bindings = [
            dict(local_rank=0, current_device=0, visible_devices=['0', '1'], cuda_uuid=None),
            dict(local_rank=1, current_device=1, visible_devices=['0', '1'], cuda_uuid=None),
        ]
        self.assertEqual(validate_bindings(bindings, 2), 'visible_device_local_rank')

    def test_ddp_binding_rejects_duplicate_physical_ids(self):
        bindings = [
            dict(local_rank=0, current_device=0, visible_devices=['0', '0'], cuda_uuid=None),
            dict(local_rank=1, current_device=1, visible_devices=['0', '0'], cuda_uuid=None),
        ]
        with self.assertRaises(RuntimeError):
            validate_bindings(bindings, 2)
    def test_ddp_state_digest_accepts_optimizer_integer_keys_and_sequences(self):
        state = {'optimizer': {'state': {3: {'step': torch.tensor(1.), 'exp_avg': [torch.tensor([2.])]}},
                               'param_groups': [{'params': [3]}]}}
        self.assertEqual(digest_state(state), digest_state(copy.deepcopy(state)))

    def run_driver(self, all_ties=False, stale_ack=False, throughput=False, extra_cfg=None, wrong_scene=False):
        torch.set_num_threads(1); torch.manual_seed(9)
        worker = WorkerDouble(all_ties, stale_ack)
        frozen = FrozenDouble()
        live_module = ModuleType('innovation3.live_inputs')
        live_module.LiveInputs = lambda cfg: SimpleNamespace(
            prepare=lambda frames,cameras: dict(token=frames[-1]['token']), close=lambda:None)
        visual_module = ModuleType('innovation3.visual_model')
        visual_module.configuration = lambda cfg,seed: SimpleNamespace(data=SimpleNamespace(test={}))
        visual_module.load_frozen = lambda cfg,settings:frozen
        parity = ModuleType('innovation3.visual_parity')
        parity.file_oracle = lambda cfg,frames,cameras: dict(token=frames[-1]['token'])
        parity.assert_same = lambda a,b:self.assertEqual(a,b)
        parity.result_parity = lambda a,b:dict(test_double=0.)
        child = SimpleNamespace(wait=lambda timeout:0, poll=lambda:0)
        buffer = SimpleNamespace(fd=0, receive=lambda descriptor,identity:{},
                                 release=lambda identity: None, close=lambda:None)
        sock = SimpleNamespace(fileno=lambda:0, close=lambda:None)
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory)/'settings.json'
            # Runtime metadata is a test artifact, not a source/config edit.
            with settings.open('x') as stream:
                json.dump(dict(simengine_python=sys.executable, **(extra_cfg or {})), stream)
            output = Path(directory)/'report.json'
            argv = ['online-probe', '--settings', str(settings), '--output', str(output), '--steps','8']
            if throughput:
                argv.append('--throughput')
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, {
                    'innovation3.live_inputs':live_module, 'innovation3.visual_model':visual_module,
                    'innovation3.visual_parity':parity}))
                stack.enter_context(patch.object(sys, 'argv', argv))
                stack.enter_context(patch.object(online_probe.socket, 'socketpair', return_value=(sock,sock)))
                stack.enter_context(patch.object(online_probe, 'Channel', return_value=worker))
                stack.enter_context(patch.object(online_probe, 'ImageBuffer', return_value=buffer))
                popen = stack.enter_context(patch.object(online_probe.subprocess, 'Popen', return_value=child))
                stack.enter_context(patch.object(online_probe.subprocess, 'check_output', return_value='test-head'))
                stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='CPU_TEST_DOUBLE'))
                stack.enter_context(patch.object(torch.cuda, 'get_device_properties',
                                                return_value=SimpleNamespace(total_memory=0)))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if wrong_scene:
                    with self.assertRaisesRegex(RuntimeError,'explicitly assigned scene'):
                        online_probe.main()
                    return json.loads(output.read_text())
                if stale_ack:
                    with self.assertRaisesRegex(RuntimeError,'acknowledgement mismatch'):
                        online_probe.main()
                    return json.loads(output.read_text())
                code = online_probe.main()
                report = json.loads(output.read_text())
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index('--seed')+1], str((extra_cfg or {}).get('online_scene_seed',0)))
                self.assertEqual(code, 2 if all_ties else 0)
                self.assertTrue(output.with_suffix('.online.pt').exists())
                self.assertTrue(frozen.selector.training is False)
                return report

    def test_scene_and_seed_namespaces_reach_online_loop(self):
        report = self.run_driver(throughput=True, extra_cfg=dict(visual_scene_id='fixture',
            online_scene_seed=123, online_candidate_seed=456, online_action_seed=789))
        self.assertEqual(report['seed_namespaces'], dict(training_seed=0,scene_seed=123,candidate_seed=456,action_seed=789))

    def test_wrong_worker_scene_fails_before_first_action(self):
        report = self.run_driver(extra_cfg=dict(visual_scene_id='another'),wrong_scene=True)
        self.assertEqual(report['events'],[])
        self.assertEqual(report['status'],'FAIL_STRICT_ONLINE_SELECTOR_ONLY')

    def test_eight_step_driver_uses_update_then_skips_seven_ties(self):
        report = self.run_driver()
        self.assertEqual(report['status'],'PASS_STRICT_ONLINE_SELECTOR_ONLY_PROBE')
        self.assertEqual(report['actual_optimizer_steps'],1)
        self.assertEqual(report['update_attempts'],8)
        self.assertEqual(report['groups_without_signal'],7)
        self.assertEqual(report['candidate_branches'],160)
        groups = [e for e in report['events'] if e['kind']=='online_update']
        self.assertTrue(all(e['version_before']==1 for e in groups[1:]))
        self.assertTrue(all(e['worker_update_receipt']['policy_version']==1 for e in groups))
        self.assertTrue(all(b['times_ns']['observe'] > a['times_ns']['worker_ack_received']
                            for a,b in zip(groups,groups[1:])))

    def test_all_ties_do_not_claim_online_learning(self):
        report = self.run_driver(all_ties=True)
        self.assertEqual(report['status'],'INCOMPLETE_ONLINE_LEARNING_EVIDENCE')
        self.assertFalse(report['real_closed_loop'])
        self.assertFalse(report['online_update'])
        self.assertEqual(report['actual_optimizer_steps'],0)

    def test_worker_bad_ack_fails_and_preserves_report(self):
        report = self.run_driver(stale_ack=True)
        self.assertEqual(report['status'],'FAIL_STRICT_ONLINE_SELECTOR_ONLY')
        self.assertIn('acknowledgement mismatch', report['error'])

    def test_throughput_mode_records_production_like_stage_timings(self):
        report = self.run_driver(throughput=True)
        self.assertEqual(report['status'], 'PASS_STRICT_ONLINE_THROUGHPUT_PROBE')
        self.assertTrue(report['throughput_mode'])
        self.assertEqual(len(report['profile_events']), 8)
        self.assertGreater(report['throughput']['decisions_per_second'], 0.)
        self.assertEqual(report['events'][13]['forward_errors']['mode'],
                         'throughput_no_duplicate_oracle')


if __name__=='__main__': unittest.main()
