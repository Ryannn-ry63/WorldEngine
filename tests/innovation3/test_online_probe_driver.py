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
from unittest.mock import patch, Mock
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3 import online_probe, online_session_probe
from innovation3.learner import V3_INITIALIZED_SELECTOR_FINETUNE
from innovation3.ddp_h1_probe import digest_state
from innovation3.ddp_online_probe import validate_bindings
from innovation3.candidate_inputs import FIELDS
from innovation3.image_transport import metadata_to_wire
from innovation3.live_audit import LiveStepGate
from grpo_selector_v3_cached_common import SceneConditionedTrajectorySetSelector
from test_live_learning import receipts


class WorkerDouble:
    def __init__(self, all_ties=False, stale_ack=False, version=0):
        self.wire = LiveStepGate(policy_version=version)
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
        with torch.no_grad():
            self.selector.delta_head[-1].weight.normal_(0, 0.02)
            self.selector.delta_head[-1].bias.fill_(0.3)
        self.module = SimpleNamespace(planning_head=SimpleNamespace(
            scene_selector=self.selector, set_candidate_noise_namespace=Mock()))
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

    def run_driver(self, all_ties=False, stale_ack=False, throughput=False, extra_cfg=None, wrong_scene=False, session_mode=None, second_ties=False, resident_change=None):
        torch.set_num_threads(1); torch.manual_seed(9)
        worker = WorkerDouble(all_ties, stale_ack)
        frozen = FrozenDouble()
        live_module = ModuleType('innovation3.live_inputs')
        live_module.LiveInputs = lambda cfg: SimpleNamespace(
            prepare=lambda frames,cameras: dict(token=frames[-1]['token']), close=lambda:None)
        visual_module = ModuleType('innovation3.visual_model')
        visual_module.configuration = lambda cfg,seed: SimpleNamespace(
            data=SimpleNamespace(test={}), model=SimpleNamespace(planning_head=SimpleNamespace(
                candidate_noise_namespace='innovation3_visual_probe_seed'+str(seed))))
        visual_module.load_frozen = Mock(return_value=frozen)
        visual_module.reset_temporal_state = Mock()
        parity = ModuleType('innovation3.visual_parity')
        parity.file_oracle = lambda cfg,frames,cameras: dict(token=frames[-1]['token'])
        parity.assert_same = lambda a,b:self.assertEqual(a,b)
        parity.result_parity = lambda a,b:dict(test_double=0.)
        child = SimpleNamespace(wait=lambda timeout:0, poll=lambda:0, pid=123, returncode=0)
        buffer = SimpleNamespace(fd=0, receive=lambda descriptor,identity:{},
                                 release=lambda identity: None, close=lambda:None)
        sock = SimpleNamespace(fileno=lambda:0, close=lambda:None)
        with tempfile.TemporaryDirectory() as directory:
            settings = Path(directory)/'settings.json'
            # Runtime metadata is a test artifact, not a source/config edit.
            with settings.open('x') as stream:
                json.dump(dict(dict(simengine_python=sys.executable, algengine_python=sys.executable,
                    visual_scene_id='fixture', visual_scene_path='/cpu_fixture.pkl'), **(extra_cfg or {})), stream)
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
                channel_mock = stack.enter_context(patch.object(online_probe, 'Channel', return_value=worker))
                stack.enter_context(patch.object(online_probe, '_RESIDENT_MODEL', None))
                stack.enter_context(patch.object(online_probe, '_RESIDENT_LEARNER', None))
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
                if session_mode:
                    # Run the actual session driver and actual online CLI/learner.
                    # Only renderer/model execution and the worker transport are doubles.
                    call_index = [0]
                    def make_worker(*args, **kwargs):
                        version = int(sys.argv[sys.argv.index('--policy-version') + 1])
                        ties = all_ties or (second_ties and call_index[0] == 1)
                        call_index[0] += 1
                        return WorkerDouble(all_ties=ties, version=version)
                    channel_mock.side_effect = make_worker
                    def subprocess_episode(command, **kwargs):
                        with patch.object(sys, 'argv', ['online-probe'] + command[4:]):
                            return SimpleNamespace(returncode=online_probe.main())
                    stack.enter_context(patch.object(online_session_probe.subprocess, 'run',
                                                    side_effect=subprocess_episode))
                    with patch.object(sys, 'argv', ['session', '--settings', str(settings),
                            '--output', str(output), '--episodes', '2', '--steps', '8',
                            '--mode', session_mode, '--throughput']):
                        if resident_change:
                            original_settings = online_session_probe._episode_settings
                            def changed_settings(base, row, path, episode):
                                original_settings(base, row, path, episode)
                                if episode:
                                    modified = json.loads(path.read_text())
                                    modified.update(resident_change)
                                    path.write_text(json.dumps(modified))
                            with patch.object(online_session_probe, '_episode_settings', changed_settings):
                                with self.assertRaisesRegex(RuntimeError, 'Resident learner provenance'):
                                    online_session_probe.main()
                            self.assertEqual(visual_module.load_frozen.call_count, 1)
                            self.assertEqual(visual_module.reset_temporal_state.call_count, 0)
                            return
                        code = online_session_probe.main()
                    summary = json.loads(output.read_text())
                    self.assertEqual(code, 2 if all_ties else 0)
                    self.assertEqual(summary['update_attempts'], 16)
                    self.assertEqual(summary['final_policy_version'], 0 if all_ties else (1 if second_ties else 2))
                    self.assertEqual(visual_module.load_frozen.call_count, 1 if session_mode == 'resident' else 2)
                    self.assertEqual(visual_module.reset_temporal_state.call_count, int(session_mode == 'resident'))
                    namespaces = frozen.module.planning_head.set_candidate_noise_namespace.call_args_list
                    self.assertEqual([x.args[0] for x in namespaces],
                        ['innovation3_visual_probe_seed0', 'innovation3_visual_probe_seed1000003'])
                    reports = [json.loads(Path(e['report']).read_text()) for e in summary['episodes']]
                    self.assertEqual(reports[1]['update_attempts_initial'], 8)
                    self.assertEqual(reports[1]['cumulative_update_attempts'], 16)
                    self.assertEqual(reports[1]['learner_state_hash_initial'], reports[0]['learner_state_hash_final'])
                    state = torch.load(summary['final_checkpoint'], map_location='cpu')['learner']
                    return summary, reports, state
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

    def test_two_episode_resident_matches_checkpoint_rebuild(self):
        resident, resident_reports, resident_state = self.run_driver(session_mode='resident')
        rebuild, rebuild_reports, rebuild_state = self.run_driver(session_mode='rebuild')
        self.assertEqual(resident['status'], 'PASS_ONLINE_SESSION_RESIDENT_PARENT')
        self.assertEqual(rebuild['status'], 'PASS_ONLINE_SESSION_REBUILD_REFERENCE')
        self.assertEqual(digest_state(resident_state), digest_state(rebuild_state))
        for left, right in zip(resident_reports, rebuild_reports):
            self.assertEqual(left['actual_optimizer_steps'], 1)
            self.assertEqual(left['update_attempts'], 8)
            for a, b in zip(left['events'], right['events']):
                if a['kind'] == 'online_update':
                    for key in ('selected', 'candidate_hash', 'version_before', 'version_after',
                                'optimizer_hash_after', 'residual_hash_after'):
                        self.assertEqual(a[key], b[key], key)

    def test_resident_rejects_mode_source_and_config_changes_before_reset(self):
        for change in ({'online_parameterization': 'frozen_v3_plus_zero_residual'},
                       {'baseline': '/different/frozen/model'},
                       {'expected_sha256': {'selector_state': 'a' * 64}}):
            self.run_driver(session_mode='resident', resident_change=change,
                extra_cfg=dict(online_parameterization=V3_INITIALIZED_SELECTOR_FINETUNE))

    def test_initialized_two_episode_resident_rebuild_and_skips(self):
        cfg = dict(online_parameterization=V3_INITIALIZED_SELECTOR_FINETUNE)
        for all_ties, second_ties in ((False, False), (False, True), (True, False)):
            a, ar, ast = self.run_driver(session_mode='resident', extra_cfg=cfg,
                                        all_ties=all_ties, second_ties=second_ties)
            b, br, bst = self.run_driver(session_mode='rebuild', extra_cfg=cfg,
                                        all_ties=all_ties, second_ties=second_ties)
            self.assertEqual(digest_state(ast), digest_state(bst))
            for left, right in zip(ar, br):
                for x, y in zip(left['events'], right['events']):
                    if x['kind'] == 'online_update':
                        for key in ('selected', 'candidate_hash', 'current_logits',
                                    'probabilities', 'online_selector_hash_after', 'optimizer_hash_after'):
                            self.assertEqual(x[key], y[key], key)

    def test_v3_initialized_finetune_parameterization_flows_through_live_probe(self):
        report = self.run_driver(
            throughput=True,
            extra_cfg={'online_parameterization': V3_INITIALIZED_SELECTOR_FINETUNE})
        self.assertEqual(report['status'], 'PASS_STRICT_ONLINE_THROUGHPUT_PROBE')
        self.assertEqual(report['parameterization'], V3_INITIALIZED_SELECTOR_FINETUNE)
        self.assertTrue(report['step0_v3_parity'])
        self.assertTrue(report['online_selector_changed'])

    def test_second_episode_without_signal_preserves_checkpoint_state(self):
        for mode in ('resident', 'rebuild'):
            with self.subTest(mode=mode):
                summary, reports, _ = self.run_driver(session_mode=mode, second_ties=True)
                self.assertTrue(summary['closed_loop_learning_verified'])
                self.assertEqual(reports[1]['actual_optimizer_steps'], 0)
                self.assertEqual(reports[1]['policy_version_initial'], 1)
                self.assertEqual(reports[1]['policy_version'], 1)
                self.assertFalse(reports[1]['online_update'])

    def test_all_tied_session_is_executed_but_not_learning_pass(self):
        summary, reports, _ = self.run_driver(session_mode='resident', all_ties=True)
        self.assertEqual(summary['status'], 'INCOMPLETE_ONLINE_SESSION_RESIDENT_PARENT')
        self.assertFalse(summary['closed_loop_learning_verified'])
        self.assertEqual(len(reports), 2)


if __name__=='__main__': unittest.main()
