"""Persistent SimEngine interpreter for the diagnostic H1 protocol probe."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import traceback

from worldengine.online.headless import HeadlessSimulator, diagnostic_candidates
from worldengine.online.state import SnapshotParityError, structural_hash
from .diagnostic_signal import transition_signal
from .paths import checked_path, sha256_file
from .protocol import Action, Feedback, StepGate, StepIdentity
from .transport import Channel, digest


class Session:
    def __init__(self):
        self.sim = None
        self.gate = StepGate(reward_atol=0.)
        self.episode = None
        self.decisions = 0
        self.actions = None
        self.candidate_hash = None
        self.before_states = None

    def reset(self, episode, scene_id, scene, reaction, steps, seed):
        if self.gate.phase != 'ready':
            raise RuntimeError('Cannot reset with unconsumed action or feedback')
        if self.sim is not None:
            self.sim.close()
        self.sim = HeadlessSimulator(scene_id, scene, reaction, steps, seed)
        self.gate = StepGate(policy_version=self.gate.policy_version, reward_atol=0.)
        self.episode, self.decisions = episode, 0

    def observe(self):
        if self.sim is None or self.gate.phase != 'ready':
            raise RuntimeError('Observation must wait for optimizer acknowledgement')
        if self.sim.engine.episode_step >= self.sim.engine.global_config.max_step:
            raise RuntimeError('Episode probe limit reached')
        snap = self.sim.snapshot()
        identity = StepIdentity(self.episode, self.decisions, self.gate.policy_version, snap.state_hash)
        self.gate.observe(identity)
        self.before_states = self.sim.agent_states()
        self.actions = diagnostic_candidates(self.sim)
        bank = [dict(waypoints=a.waypoints.tolist(), headings=a.headings.tolist()) for a in self.actions]
        self.candidate_hash = digest(bank)
        self.action_state_hash = structural_hash(self.actions)
        return dict(kind='observation', identity=asdict(identity), states=self.before_states,
                    candidate_hash=self.candidate_hash, candidates=bank,
                    candidate_source='current_state_analytic_diagnostic')

    def act(self, request, emit):
        identity = StepIdentity(**request['identity'])
        if request['candidate_hash'] != self.candidate_hash:
            raise RuntimeError('Candidate bank mismatch')
        self.gate.choose(Action(identity, request['candidate_hash'], request['selected'],
                                tuple(request['probabilities'])))
        if self.sim.snapshot().state_hash != identity.state_hash:
            raise RuntimeError('Observation state changed before canonical execution')
        main_result = {}

        def canonical(snapshot, states):
            # Evaluate the actual canonical transition, BEFORE branch execution.
            signal = transition_signal(self.before_states, states)
            self.gate.main_executed(snapshot.state_hash)
            main_result.update(next_hash=snapshot.state_hash, signal=signal)
            emit(dict(kind='main', identity=asdict(identity), candidate_hash=self.candidate_hash,
                      next_hash=snapshot.state_hash, signal=signal))

        before, main, branches = self.sim.branch_group(self.actions, request['selected'], on_main=canonical)
        signals = tuple(transition_signal(self.before_states, states) for _, states in branches)
        feedback = Feedback(identity, self.candidate_hash, signals, (True,) * 20,
                            main_result['signal'], main.state_hash,
                            branches[request['selected']][0].state_hash)
        if before.state_hash != identity.state_hash or structural_hash(self.actions) != self.action_state_hash:
            raise RuntimeError('Prestate or candidate bank mutated')
        if self.sim.snapshot().state_hash != main.state_hash:
            raise RuntimeError('Canonical state not restored after branches')
        self.gate.feedback(feedback)
        emit(dict(kind='feedback', **asdict(feedback),
                  branch_hashes=[snapshot.state_hash for snapshot, _ in branches]))

    def updated(self, request):
        if StepIdentity(**request['identity']) != self.gate.identity:
            raise RuntimeError('Stale update acknowledgement')
        self.gate.updated(request['policy_version'], request['optimized'])
        self.decisions += 1
        return dict(kind='updated', policy_version=self.gate.policy_version,
                    next_hash=self.gate.next_hash)

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--scene-count', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--failure-report', type=Path, required=True)
    args = parser.parse_args()
    channel = Channel(socket.socket(fileno=args.fd))
    session = Session()
    try:
        from .snapshot_probe import runtime_evidence, IDMFailures, save_parity_failure
        from worldengine.online.numerics import check_numeric_runtime
        import logging
        idm = IDMFailures()
        logging.getLogger().addHandler(idm)
        health = check_numeric_runtime()
        cfg = json.loads(checked_path(args.settings).read_text())
        source = checked_path(Path(cfg['scenario_root']) / 'original/navtrain_failures_per1/all_scenarios.pkl')
        with source.open('rb') as stream:
            all_scenes = pickle.load(stream)
        ids = sorted(all_scenes)[:args.scene_count]
        if len(ids) != args.scene_count:
            raise ValueError('Not enough source scenes')
        scenes = {k: all_scenes[k] for k in ids}
        del all_scenes
        channel.send(dict(kind='ready', scenes=ids, numeric_health=health,
                          source=dict(path=str(source), bytes=source.stat().st_size, sha256=sha256_file(source)),
                          runtime=runtime_evidence(), pid=os.getpid()))
        while True:
            request = channel.receive()
            kind = request['kind']
            if kind == 'reset':
                session.reset(request['episode'], request['scene_id'], scenes[request['scene_id']],
                              request['reaction'], args.steps, args.seed)
                channel.send(dict(kind='reset', policy_version=session.gate.policy_version))
            elif kind == 'observe':
                channel.send(session.observe())
            elif kind == 'action':
                start = idm.count
                def emit(message):
                    if message['kind'] == 'feedback' and idm.count != start:
                        raise RuntimeError('IDM fallback invalidates protocol probe')
                    channel.send(message)
                session.act(request, emit)
            elif kind == 'updated':
                channel.send(session.updated(request))
            elif kind == 'close':
                if session.gate.phase != 'ready':
                    raise RuntimeError('Cannot finish with pending feedback')
                channel.send(dict(kind='closed', idm_fallbacks=idm.count))
                break
            else:
                raise ValueError('Unknown worker message: ' + kind)
    except Exception as error:
        if isinstance(error, SnapshotParityError):
            save_parity_failure(error, args.failure_report)
        channel.send(dict(kind='error', error=repr(error), traceback=traceback.format_exc()))
        return 1
    finally:
        session.close()
        channel.sock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
