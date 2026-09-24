import unittest

from innovation3.session_lifecycle import SessionLedger


class SessionLedgerTest(unittest.TestCase):
    def test_a_b_a_preserves_version_and_attempts(self):
        ledger = SessionLedger('s', policy_version=3, attempts=4)
        receipts = []
        for index, scene in enumerate(('A', 'B', 'A')):
            token = ledger.start_episode(scene, index)
            before = ledger.policy_version
            ledger.record_decision(token, before, before + 1, True)
            ledger.record_decision(token, ledger.policy_version, ledger.policy_version, False)
            receipts.append(ledger.close_episode(2))
        self.assertEqual([x['scene'] for x in receipts], ['A', 'B', 'A'])
        self.assertEqual(ledger.policy_version, 6)
        self.assertEqual(ledger.attempts, 10)
        self.assertEqual(ledger.boundary()['next_episode'], 3)

    def test_stale_episode_and_bad_version_are_rejected(self):
        ledger = SessionLedger('s')
        first = ledger.start_episode('A', 0)
        with self.assertRaises(RuntimeError):
            ledger.record_decision(first, 1, 1, False)
        ledger.record_decision(first, 0, 0, False)
        ledger.close_episode(1)
        second = ledger.start_episode('B', 1)
        with self.assertRaises(RuntimeError):
            ledger.record_decision(first, 0, 0, False)
        with self.assertRaises(RuntimeError):
            ledger.record_decision(second, 1, 1, False)


if __name__ == '__main__':
    unittest.main()
