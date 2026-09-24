"""CPU-only window and receipt validation for the bounded live probe."""
from collections import deque
import math
from .protocol import Action, Feedback, StepGate, StepIdentity

REWARD_KEYS = {'reward', 'NC', 'DAC', 'EP', 'TTC', 'comfort', 'direction',
               'progress_m', 'reference_progress_m'}


class LiveWindow:
    def __init__(self):
        self.frames = deque(maxlen=4)
        self.cameras = deque(maxlen=4)

    def append(self, frame, cameras, identity):
        if frame['token'] != identity['token'] or frame['frame_idx'] != identity['step']:
            raise ValueError('Observation identity does not match frame')
        if self.frames:
            previous = self.frames[-1]
            if (frame['scene_token'] != previous['scene_token'] or
                    frame['frame_idx'] != previous['frame_idx'] + 1 or
                    frame['timestamp'] <= previous['timestamp']):
                raise ValueError('Mixed or nonconsecutive live window')
        self.frames.append(frame)
        self.cameras.append(cameras)


def validate_receipts(identity, selected, candidate_hash, main, branches, with_reward):
    if main.get('kind') != 'main' or branches.get('kind') != 'branches':
        raise ValueError('Main execution receipt must precede branch feedback')
    if main.get('identity') != identity or branches.get('identity') != identity:
        raise ValueError('Stale branch feedback')
    for receipt in (main, branches):
        if receipt.get('candidate_hash') != candidate_hash or receipt.get('selected') != selected:
            raise ValueError('Feedback candidate identity mismatch')
    hashes = branches.get('branch_hashes', [])
    if (type(selected) is not int or not 0 <= selected < 20 or len(hashes) != 20 or
            not branches.get('selected_parity') or not branches.get('canonical_restored') or
            not main.get('next_hash') or main['next_hash'] != branches.get('main_hash') or
            hashes[selected] != main['next_hash']):
        raise ValueError('Canonical/selected state receipt mismatch')
    if not with_reward:
        return
    rewards = branches.get('rewards')
    if not isinstance(rewards, list) or len(rewards) != 20:
        raise ValueError('Expected full20 reward feedback')
    for reward in [main.get('reward'), branches.get('reward'), *rewards]:
        if (not isinstance(reward, dict) or set(reward) != REWARD_KEYS or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in reward.values())):
            raise ValueError('Invalid or non-finite reward components')
    if (not branches.get('selected_reward_parity') or not branches.get('selected_history_parity') or
            main['reward'] != branches['reward'] or main['reward'] != rewards[selected] or
            not branches.get('history_before_hash') or not branches.get('history_after_hash')):
        raise ValueError('Independent main reward/history receipt mismatch')


class LiveStepGate:
    """Same strict wire gate in both interpreters; only the parent owns learning."""
    def __init__(self, policy_version=0):
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError('Nonnegative integer policy version required')
        self.gate = StepGate(policy_version=policy_version, reward_atol=0.)
        self.raw_identity = None
        self.main = None
        self.history_hash = None

    def observe(self, identity):
        if type(identity.get('policy_version')) is not int:
            raise ValueError('Integral policy version required')
        self.gate.observe(StepIdentity(identity['scene'], identity['step'],
                                       identity['policy_version'], identity['state_hash']))
        self.raw_identity = dict(identity)
        self.main = None

    def choose(self, request):
        if request.get('kind') != 'action' or request.get('identity') != self.raw_identity:
            raise ValueError('Action wire identity mismatch')
        self.gate.choose(Action(self.gate.identity, request['candidate_hash'],
                                request['selected'], tuple(request['probabilities'])))

    def main_executed(self, main):
        if (main.get('kind') != 'main' or main.get('identity') != self.raw_identity or
                self.gate.action is None or
                main.get('candidate_hash') != self.gate.action.candidate_hash or
                main.get('selected') != self.gate.action.selected):
            raise ValueError('Independent main receipt identity mismatch')
        self.gate.main_executed(main['next_hash'])
        self.main = main

    def feedback(self, branches):
        if self.main is None:
            raise RuntimeError('Feedback arrived before main execution')
        action = self.gate.action
        validate_receipts(self.raw_identity, action.selected, action.candidate_hash,
                          self.main, branches, True)
        if self.history_hash is not None and branches['history_before_hash'] != self.history_hash:
            raise RuntimeError('Reward history did not continue from canonical execution')
        # Preserve original JSON scalars for exact selected/main parity.
        # Conversion to learner precision happens only AFTER this gate.
        self.gate.feedback(Feedback(self.gate.identity, action.candidate_hash,
            tuple(r['reward'] for r in branches['rewards']), (True,) * 20,
            self.main['reward']['reward'], self.main['next_hash'],
            branches['branch_hashes'][action.selected]))
        self.history_hash = branches['history_after_hash']

    def updated(self, ack):
        if self.gate.phase != 'feedback':
            raise RuntimeError('Update acknowledgement requires accepted feedback')
        expected = dict(kind='feedback_ack', identity=self.raw_identity,
                        candidate_hash=self.gate.action.candidate_hash,
                        next_hash=self.gate.next_hash)
        if set(ack) != set(expected) | {'policy_version', 'optimized'} or any(
                ack[k] != v for k, v in expected.items()):
            raise ValueError('Stale or mismatched optimizer acknowledgement')
        if type(ack['policy_version']) is not int:
            raise ValueError('Integral optimizer version required')
        self.gate.updated(ack['policy_version'], ack['optimized'])
        return dict(kind='updated', identity=self.raw_identity,
                    candidate_hash=self.gate.action.candidate_hash,
                    next_hash=self.gate.next_hash, policy_version=self.gate.policy_version)
