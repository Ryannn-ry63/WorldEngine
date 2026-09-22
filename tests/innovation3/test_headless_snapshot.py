"""Real CPU SimEngine fixtures: dynamics, reactive traffic and state restoration."""
from dataclasses import replace
from pathlib import Path
import pickle
import random
import sys
import json
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'projects/SimEngine'))
sys.path.insert(0, str(ROOT / 'projects/AlgEngine/scripts/diffusiondrive'))
from worldengine.online.headless import HeadlessSimulator, diagnostic_candidates
from worldengine.online.state import SnapshotCodec, SnapshotParityError, structural_hash


def scene_fixture():
    n = 12

    def track(key, offset, valid, speed=8., kind='VEHICLE'):
        return dict(type=kind, metadata=dict(object_id=key), state=dict(
            position=np.column_stack([offset + np.arange(n) * .5 * speed, np.zeros(n)]),
            heading=np.zeros(n), velocity=np.tile([speed, 0.], (n, 1)),
            angular_velocity=np.zeros(n), valid=np.asarray(valid, dtype=float),
            length=np.full((n, 1), 4.8), width=np.full((n, 1), 2.), height=np.full((n, 1), 1.7)))

    return dict(id='fixture', name='fixture', dataset='synthetic_engineering_fixture',
                map='fixture', token='fixture', log_length=n, sample_rate=10,
                base_timestamp=0, metadata={}, sdc_id='ego', dynamic_map_states={},
                object_track={
                    'ego': track('ego', 0., [1]*n),
                    'follower': track('follower', -15., [1]*n),
                    'late': track('late', -40., [0]+[1]*(n-1)),
                    'leaving': track('leaving', 80., [1, 1]+[0]*(n-2), speed=16.),
                    'cone': track('cone', 150., [0, 1]+[0]*(n-2), speed=0., kind='TRAFFIC_CONE')},
                map_features={'lane': dict(type='LANE_SURFACE_STREET', roadblock_id='road',
                                          entry_lanes=['lane'], exit_lanes=['lane'],
                                          left_neighbor=[], right_neighbor=[],
                                          polyline=np.array([[-100., 0.], [300., 0.]]))})


class HeadlessSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.sim = HeadlessSimulator('fixture', scene_fixture(), 'R', 8, seed=17)

    def tearDown(self):
        self.sim.close()

    def test_hidden_state_and_rng_restored_with_aliases(self):
        before = self.sim.snapshot()
        agent = self.sim.engine.agents['follower']
        expected_rng = (np.random.random(), random.random(), agent.policy.np_random.random())
        agent.policy.overtake_timer += 100
        agent.navigation.last_and_current_long.append(133.)
        self.assertNotEqual(self.sim.snapshot().state_hash, before.state_hash)
        self.sim.restore(pickle.loads(pickle.dumps(before)))
        self.assertEqual(self.sim.snapshot().state_hash, before.state_hash)
        agent = self.sim.engine.agents['follower']
        self.assertIs(agent.policy.agent, agent)
        self.assertIs(agent.controller.agent, agent)
        self.assertIs(agent.navigation.agent, agent)
        self.assertEqual((np.random.random(), random.random(), agent.policy.np_random.random()), expected_rng)

    def test_spawn_despawn_and_future_rng_repeat(self):
        before = self.sim.snapshot()
        action = diagnostic_candidates(self.sim)[12]
        self.sim.step(action)
        after = self.sim.snapshot()
        self.assertIn('late', self.sim.engine.agents)
        self.assertIn('cone', self.sim.engine.agents)
        self.sim.restore(before)
        self.sim.step(action)
        self.assertEqual(self.sim.snapshot().state_hash, after.state_hash)
        self.sim.step(diagnostic_candidates(self.sim)[12])
        self.assertNotIn('cone', self.sim.engine.agents)
        # Reactive dynamic vehicles persist beyond their logged validity.
        self.assertIn('leaving', self.sim.engine.agents)

    def test_nr_uses_same_physics_and_log_validity(self):
        self.sim.close()
        self.sim = HeadlessSimulator('fixture', scene_fixture(), 'NR', 8, seed=17)
        before = self.sim.snapshot()
        action = diagnostic_candidates(self.sim)[12]
        self.sim.step(action)
        after = self.sim.snapshot()
        self.sim.restore(before)
        self.sim.step(action)
        self.assertEqual(self.sim.snapshot().state_hash, after.state_hash)
        self.sim.step(diagnostic_candidates(self.sim)[12])
        self.assertNotIn('leaving', self.sim.engine.agents)

    def test_full20_preserves_main_and_candidate_bank(self):
        actions = diagnostic_candidates(self.sim)
        original_actions = structural_hash(actions)
        before, main, results = self.sim.branch_group(actions, 12)
        self.assertEqual(len(results), 20)
        self.assertEqual(main.state_hash, results[12][0].state_hash)
        self.assertEqual(main.state_hash, self.sim.snapshot().state_hash)
        self.assertEqual(original_actions, structural_hash(actions))
        self.assertGreater(len({structural_hash(b[1]['ego']) for b in results}), 1)
        # Follower sees the intervention in current ego velocity in the same
        # real IDM engine step. Replayed traffic cannot satisfy this assertion.
        velocities = [b[1]['follower']['velocity'] for b in results]
        self.assertGreater(np.ptp(velocities, axis=0).max(), 1e-6)
        self.sim.restore(before)
        self.assertEqual(before.state_hash, self.sim.snapshot().state_hash)

    def test_failed_branch_restores_canonical_state(self):
        before = self.sim.snapshot()
        actions = diagnostic_candidates(self.sim)
        self.sim.step(actions[12])
        expected = self.sim.snapshot()
        self.sim.restore(before)
        actions[0].waypoints[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.sim.branch_group(actions, 12)
        self.assertEqual(self.sim.snapshot().state_hash, expected.state_hash)

    def test_static_buffers_and_mutation_audit(self):
        lane = self.sim.engine.current_map.road_network.get_lane('lane')
        with self.assertRaises(ValueError):
            lane.center_line_points[0, 0] += 1.
        lane.priority += 1
        with self.assertRaises(RuntimeError):
            self.sim.codec.audit_static()

    def test_bad_snapshot_rejected_before_state_change(self):
        snap = self.sim.snapshot()
        for bad in (replace(snap, schema=0), replace(snap, static_sha256='other'),
                    replace(snap, payload=snap.payload+b'x'), replace(snap, state_hash='wrong')):
            with self.assertRaises(ValueError):
                self.sim.restore(bad)
            self.assertEqual(self.sim.snapshot().state_hash, snap.state_hash)

    def test_parity_failure_keeps_exact_replay_evidence(self):
        from innovation3.snapshot_probe import save_parity_failure
        real_step = self.sim.step
        calls = 0

        def inject_one_ulp_difference(action):
            nonlocal calls
            result = real_step(action)
            calls += 1
            # First call is canonical; call 14 is candidate 12. A one-ULP
            # perturbation must still fail the unchanged strict hash gate.
            if calls == 14:
                ego = self.sim.engine.agent_manager.ego_agent
                ego._cur_pos[0] = np.nextafter(ego._cur_pos[0], np.inf)
            return result

        self.sim.step = inject_one_ulp_difference
        with self.assertRaises(SnapshotParityError) as context:
            self.sim.branch_group(diagnostic_candidates(self.sim), 12)
        error = context.exception
        self.assertEqual(self.sim.snapshot().state_hash, error.expected.state_hash)
        self.assertNotEqual(error.expected.state_hash, error.actual.state_hash)
        self.assertTrue(any(c['path'] == 'agents.ego._cur_pos' for c in error.details['components']))
        with tempfile.TemporaryDirectory() as tmp:
            saved = Path(save_parity_failure(error, Path(tmp)/'report.json'))
            record = json.loads((saved/'diagnosis.json').read_text())
            self.assertEqual(record['actual_hash'], error.actual.state_hash)
            replay = pickle.loads((saved/'replay.pkl').read_bytes())
            self.sim.codec = SnapshotCodec(bundle=(saved/'static_bundle.pkl').read_bytes())
            self.sim.restore(replay['before'])
            real_step(replay['action'])
            self.assertEqual(self.sim.snapshot().state_hash, replay['expected'].state_hash)

    def test_different_bundle_transport_preserves_state(self):
        snap = self.sim.snapshot()
        self.sim.codec = SnapshotCodec(bundle=self.sim.codec.bundle)
        self.sim.restore(snap)
        self.assertEqual(self.sim.snapshot().state_hash, snap.state_hash)
        self.sim.codec.audit_static()

    def test_no_two_engines_or_unknown_manager(self):
        with self.assertRaises(RuntimeError):
            HeadlessSimulator('fixture', scene_fixture(), 'R', 8)
        self.sim.engine._managers['unreviewed_reward_manager'] = object()
        try:
            with self.assertRaises(ValueError):
                self.sim.snapshot()
        finally:
            del self.sim.engine._managers['unreviewed_reward_manager']


class HashTest(unittest.TestCase):
    def test_diagnostic_trace_does_not_change_hash(self):
        value = dict(a=np.arange(8), b=[1, 2, 3])
        trace = []
        self.assertEqual(structural_hash(value), structural_hash(value, trace=trace))
        self.assertTrue(trace)

    def test_cycles_aliases_and_unknown_types(self):
        a = []
        a.append(a)
        self.assertEqual(structural_hash(a), structural_hash(pickle.loads(pickle.dumps(a))))
        shared = [1]
        self.assertNotEqual(structural_hash([shared, shared]), structural_hash([[1], [1]]))
        with self.assertRaises(TypeError):
            structural_hash(object())


if __name__ == '__main__':
    unittest.main()
