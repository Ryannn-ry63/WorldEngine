from dataclasses import asdict
from pathlib import Path
import socket
import struct
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
sys.path.insert(0, str(ROOT/'projects/SimEngine'))
from innovation3.transport import Channel, digest, MAX_FRAME
from innovation3.diagnostic_signal import transition_signal
from innovation3.h1_protocol_probe import diagnostic_context
from innovation3.h1_worker import Session
from test_headless_snapshot import scene_fixture


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.left, self.right = socket.socketpair()
        self.a, self.b = Channel(self.left, timeout=1), Channel(self.right, timeout=1)

    def tearDown(self):
        self.left.close()
        self.right.close()

    def test_exact_numeric_roundtrip(self):
        value = dict(kind='fixture', x=[0., -1.3333333333333333, 1e-300])
        self.a.send(value)
        self.assertEqual(digest(self.b.receive()), digest(value))
        with self.assertRaises(ValueError):
            self.a.send(dict(kind='bad', x=float('nan')))

    def test_partial_frame_disconnect_and_size_rejected(self):
        self.left.sendall(struct.pack('!I', 10)+b'{}')
        self.left.close()
        with self.assertRaises(EOFError):
            self.b.receive()

    def test_oversized_receive_rejected(self):
        self.left.sendall(struct.pack('!I', MAX_FRAME+1))
        with self.assertRaises(ValueError):
            self.b.receive()

    def test_peer_error_and_timeout_are_visible(self):
        self.a.send(dict(kind='error', error='branch failed'))
        with self.assertRaisesRegex(RuntimeError, 'branch failed'):
            self.b.receive()
        self.right.settimeout(.01)
        with self.assertRaises(socket.timeout):
            self.b.receive()


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.session = Session()
        self.session.reset('episode', 'fixture', scene_fixture(), 'R', 4, 17)

    def tearDown(self):
        self.session.close()

    def action(self, obs):
        return dict(kind='action', identity=obs['identity'], candidate_hash=obs['candidate_hash'],
                    selected=12, probabilities=[.05]*20)

    def test_real_feedback_order_independent_canonical_and_reset_version(self):
        obs = self.session.observe()
        events = []
        def emit(message):
            if message['kind']=='main':
                # Called while canonical state is live, before any branch feedback.
                self.assertEqual(message['signal'], transition_signal(obs['states'], self.session.sim.agent_states()))
            events.append(message)
        self.session.act(self.action(obs), emit)
        self.assertEqual([m['kind'] for m in events], ['main','feedback'])
        self.assertEqual(events[0]['signal'], events[1]['rewards'][12])
        with self.assertRaises(RuntimeError):
            self.session.observe()
        with self.assertRaises(RuntimeError):
            self.session.reset('another','fixture',scene_fixture(),'NR',4,17)
        self.session.updated(dict(identity=obs['identity'], policy_version=1, optimized=True))
        next_obs = self.session.observe()
        self.assertEqual(next_obs['identity']['policy_version'],1)
        self.assertEqual(next_obs['identity']['state_hash'],events[0]['next_hash'])
        # Complete the next step without a parameter update, then switch scene.
        self.session.act(self.action(next_obs), lambda m: None)
        self.session.updated(dict(identity=next_obs['identity'],policy_version=1,optimized=False))
        self.session.reset('another','fixture',scene_fixture(),'NR',4,17)
        self.assertEqual(self.session.observe()['identity']['policy_version'],1)

    def test_wrong_candidate_and_duplicate_actions_fail_before_execution(self):
        obs = self.session.observe()
        action = self.action(obs)
        action['candidate_hash'] = 'bad'
        with self.assertRaises(RuntimeError):
            self.session.act(action, lambda m: None)
        self.assertEqual(self.session.sim.engine.episode_step,0)
        self.session.act(self.action(obs), lambda m: None)
        with self.assertRaises(RuntimeError):
            self.session.act(self.action(obs), lambda m: None)
        self.assertEqual(self.session.sim.engine.episode_step,1)

    def test_callback_failure_restores_canonical_and_blocks_next_observation(self):
        obs = self.session.observe()
        def failed(message):
            raise ConnectionError('peer lost')
        with self.assertRaises(ConnectionError):
            self.session.act(self.action(obs), failed)
        self.assertEqual(self.session.sim.engine.episode_step,1)
        self.assertEqual(self.session.sim.snapshot().state_hash,self.session.gate.next_hash)
        with self.assertRaises(RuntimeError):
            self.session.observe()

    def test_context_future_points_float32_and_reward_separation(self):
        obs = self.session.observe()
        x = diagnostic_context(obs,torch.device('cpu'))
        self.assertTrue(all(t.dtype==torch.float32 for t in x.values()))
        self.assertEqual(tuple(x['candidate_trajectories'].shape),(1,20,8,3))
        # First fixture candidate travels at 4 m/s: future t=.5 starts at x=2, not x=0.
        self.assertAlmostEqual(float(x['candidate_trajectories'][0,0,0,0]),2.)
        self.assertAlmostEqual(float(x['candidate_trajectories'][0,0,-1,0]),16.)
        self.assertNotIn('reward',x)
