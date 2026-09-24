"""Fail-closed bookkeeping for continuous online episode boundaries.

The ledger does not own the learner or simulator.  It validates the boundary
contract shared by rebuild and resident session implementations: local
decisions are contiguous, policy versions never regress, and optimizer state
survives a scene transition.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeToken:
    session: str
    episode: int
    scene: str
    seed: int

    @property
    def wire(self):
        return '%s:%d' % (self.session, self.episode)


class SessionLedger:
    def __init__(self, session, policy_version=0, attempts=0):
        if not isinstance(session, str) or not session:
            raise ValueError('Session identifier is required')
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError('Nonnegative integer policy version required')
        if type(attempts) is not int or attempts < 0:
            raise ValueError('Nonnegative integer attempt count required')
        self.session = session
        self.policy_version = policy_version
        self.attempts = attempts
        self.episode = -1
        self.current = None

    def start_episode(self, scene, seed):
        if not isinstance(scene, str) or not scene:
            raise ValueError('Scene identifier is required')
        if type(seed) is not int or seed < 0:
            raise ValueError('Nonnegative integer scene seed required')
        if self.current is not None:
            raise RuntimeError('Previous episode is still open')
        self.episode += 1
        token = EpisodeToken(self.session, self.episode, scene, seed)
        self.current = dict(token=token, decisions=0, attempts_before=self.attempts,
                            version_before=self.policy_version)
        return token

    def record_decision(self, token, version_before, version_after, optimized):
        if self.current is None or token != self.current['token']:
            raise RuntimeError('Decision belongs to a stale or closed episode')
        if type(version_before) is not int or type(version_after) is not int:
            raise ValueError('Policy versions must be integers')
        if version_before != self.policy_version:
            raise RuntimeError('Decision started from a stale policy version')
        if type(optimized) is not bool:
            raise ValueError('Optimizer result must be boolean')
        expected_after = version_before + int(optimized)
        if version_after != expected_after:
            raise RuntimeError('Policy version does not match optimizer result')
        self.current['decisions'] += 1
        self.attempts += 1
        self.policy_version = version_after

    def close_episode(self, expected_decisions):
        if self.current is None:
            raise RuntimeError('No open episode')
        if type(expected_decisions) is not int or expected_decisions < 1:
            raise ValueError('Expected decision count must be positive')
        if self.current['decisions'] != expected_decisions:
            raise RuntimeError('Episode ended with incomplete decisions')
        token = self.current['token']
        receipt = dict(session=token.session, episode=token.episode, scene=token.scene,
                       seed=token.seed, wire=token.wire,
                       decisions=self.current['decisions'],
                       attempts_before=self.current['attempts_before'],
                       attempts_after=self.attempts,
                       policy_version_before=self.current['version_before'],
                       policy_version_after=self.policy_version)
        self.current = None
        return receipt

    def boundary(self):
        if self.current is not None:
            raise RuntimeError('Boundary checkpoint requires a closed episode')
        return dict(session=self.session, next_episode=self.episode + 1,
                    attempts=self.attempts, policy_version=self.policy_version)
