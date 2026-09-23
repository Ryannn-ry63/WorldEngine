"""Reject history gaps and inconsistent independently received branch evidence."""
from pathlib import Path
import copy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3.live_audit import LiveWindow, REWARD_KEYS, validate_receipts


def frame(i):
    return dict(token=str(i), frame_idx=i, timestamp=i*500000, scene_token='s')


def identity(i):
    return dict(token=str(i), step=i, scene='s', state_hash='state'+str(i))


def receipts():
    reward = {k: 1. for k in REWARD_KEYS}
    main = dict(kind='main', identity=identity(13), candidate_hash='bank', selected=3,
                next_hash='next', reward=reward)
    feedback = dict(kind='branches', identity=identity(13), candidate_hash='bank', selected=3,
                    main_hash='next', branch_hashes=['next']*20, selected_parity=True,
                    canonical_restored=True, reward=copy.deepcopy(reward),
                    rewards=[copy.deepcopy(reward) for _ in range(20)],
                    selected_reward_parity=True, selected_history_parity=True,
                    history_before_hash='past', history_after_hash='after')
    return main, feedback


class LiveRewardProtocolTest(unittest.TestCase):
    def test_thirteen_warmup_frames_remain_bounded_and_first_action_uses_10_to_13(self):
        window = LiveWindow()
        for i in range(21):  # 13 diagnostic transitions + 8 generated decisions
            window.append(frame(i), {'front': bytes([i])}, identity(i))
            self.assertLessEqual(len(window.frames), 4)
            self.assertEqual(len(window.cameras), len(window.frames))
            if i >= 13:
                self.assertEqual([f['frame_idx'] for f in window.frames], list(range(i-3, i+1)))
                self.assertEqual([c['front'][0] for c in window.cameras], list(range(i-3, i+1)))

    def test_skipped_and_duplicate_frames_rejected(self):
        for bad in (2, 13):
            window = LiveWindow()
            for i in range(3): window.append(frame(i), {}, identity(i))
            with self.assertRaises(ValueError): window.append(frame(bad), {}, identity(bad))

    def test_identity_and_scene_mismatch_rejected(self):
        w = LiveWindow(); w.append(frame(0), {}, identity(0))
        with self.assertRaises(ValueError): w.append(frame(1), {}, identity(2))
        with self.assertRaises(ValueError): w.append(dict(frame(1), scene_token='other'), {}, identity(1))

    def test_valid_full20_independent_receipts(self):
        main, branches = receipts()
        validate_receipts(identity(13), 3, 'bank', main, branches, True)
        validate_receipts(identity(13), 3, 'bank', main, branches, False)

    def test_flags_cannot_hide_main_reward_mismatch(self):
        main, branches = receipts(); main['reward']['reward'] = .5
        with self.assertRaises(ValueError):
            validate_receipts(identity(13), 3, 'bank', main, branches, True)

    def test_flags_cannot_hide_state_or_candidate_mismatch(self):
        for key, value in [('main_hash', 'other'), ('branch_hashes', ['other']*20),
                           ('candidate_hash', 'other'), ('selected', 4), ('identity', identity(12))]:
            main, branches = receipts(); branches[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_receipts(identity(13), 3, 'bank', main, branches, True)

    def test_missing_nonfinite_and_short_reward_groups_rejected(self):
        for change in ('nan', 'missing', 'short', 'history'):
            main, branches = receipts()
            if change == 'nan': branches['rewards'][0]['EP'] = float('nan')
            if change == 'missing': branches['rewards'][0].pop('TTC')
            if change == 'short': branches['rewards'].pop()
            if change == 'history': branches['history_after_hash'] = ''
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_receipts(identity(13), 3, 'bank', main, branches, True)

    def test_main_receipt_must_precede_branch_feedback(self):
        main, branches = receipts()
        with self.assertRaises(ValueError):
            validate_receipts(identity(13), 3, 'bank', branches, main, True)


if __name__ == '__main__': unittest.main()
