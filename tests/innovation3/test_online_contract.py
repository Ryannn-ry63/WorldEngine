import dataclasses
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3.paths import checked_path
from innovation3.protocol import Action, Feedback, StepGate, StepIdentity


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.gate = StepGate()
        self.identity = StepIdentity('scene:attempt', 4, 0, 'state4')
        self.action = Action(self.identity, 'candidates4', 0, (0.05,) * 20)
        self.feedback = Feedback(self.identity, 'candidates4', tuple(range(20)),
                                 (True,) * 20, 0.0, 'state5', 'state5')

    def advance(self):
        self.gate.observe(self.identity)
        self.gate.choose(self.action)
        self.gate.main_executed('state5')

    def test_feedback_cannot_precede_action(self):
        self.gate.observe(self.identity)
        with self.assertRaises(RuntimeError):
            self.gate.feedback(self.feedback)

    def test_next_observation_waits_for_update(self):
        self.advance()
        self.gate.feedback(self.feedback)
        with self.assertRaises(RuntimeError):
            self.gate.observe(StepIdentity('scene:attempt', 5, 1, 'state5'))
        self.gate.updated(1, True)
        self.gate.observe(StepIdentity('scene:attempt', 5, 1, 'state5'))

    def test_stale_reward_or_different_candidates_rejected(self):
        self.advance()
        with self.assertRaises(RuntimeError):
            self.gate.feedback(dataclasses.replace(self.feedback, candidate_hash='another-group'))

    def test_selected_branch_must_match_canonical_state_and_reward(self):
        self.advance()
        for kwargs in [dict(selected_branch_next_hash='other'), dict(main_reward=0.1)]:
            with self.assertRaises(RuntimeError):
                self.gate.feedback(dataclasses.replace(self.feedback, **kwargs))

    def test_failed_branches_do_not_become_zero_reward(self):
        self.advance()
        with self.assertRaises(ValueError):
            self.gate.feedback(dataclasses.replace(self.feedback, valid=(False,) + (True,) * 19))

    def test_no_signal_keeps_actual_policy_version(self):
        self.advance()
        self.gate.feedback(self.feedback)
        self.gate.updated(0, False)
        with self.assertRaises(RuntimeError):
            self.gate.observe(StepIdentity('scene:attempt', 5, 1, 'state5'))
        self.gate.observe(StepIdentity('scene:attempt', 5, 0, 'state5'))

    def test_no_duplicate_transition(self):
        self.advance()
        with self.assertRaises(RuntimeError):
            self.gate.main_executed('state5')


class PathTest(unittest.TestCase):
    def test_own_file_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'file'; p.touch()
            self.assertEqual(checked_path(p), p)
            with self.assertRaises(FileNotFoundError):
                checked_path(Path(directory) / 'missing')

    def test_forbidden_direct_and_symlink_paths(self):
        for part in ['hdd2', 'roboticsystem2']:
            with self.assertRaises(ValueError):
                checked_path('/inspire/' + part + '/missing')
            with tempfile.TemporaryDirectory() as directory:
                p = Path(directory) / 'alias'
                p.symlink_to('/inspire/' + part + '/missing')
                with self.assertRaises(ValueError):
                    checked_path(p)

    def test_intermediate_forbidden_link_cannot_be_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'safe').mkdir(); (root / 'safe/file').touch()
            (root / 'hdd2').symlink_to(root / 'safe')
            (root / 'alias').symlink_to(root / 'hdd2/file')
            with self.assertRaises(ValueError):
                checked_path(root / 'alias')

    def test_symlink_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a').symlink_to(root / 'b'); (root / 'b').symlink_to(root / 'a')
            with self.assertRaises(ValueError):
                checked_path(root / 'a')


if __name__ == '__main__':
    unittest.main()
