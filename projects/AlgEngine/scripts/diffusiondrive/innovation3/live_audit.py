"""CPU-only window and receipt validation for the bounded live probe."""
from collections import deque
import math

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
