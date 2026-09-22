"""Fail-closed, single-rank causal ordering for a strict full20/H1 step.

An adapter must supply real state/trajectory hashes. This gate does not generate
observations, simulate dynamics, compute rewards, or certify branch equivalence.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StepIdentity:
    episode: str
    decision: int
    policy_version: int
    state_hash: str


@dataclass(frozen=True)
class Action:
    identity: StepIdentity
    candidate_hash: str
    selected: int
    probabilities: tuple


@dataclass(frozen=True)
class Feedback:
    identity: StepIdentity
    candidate_hash: str
    rewards: tuple
    valid: tuple
    main_reward: float
    main_next_hash: str
    selected_branch_next_hash: str


class StepGate:
    def __init__(self, policy_version=0, reward_atol=1e-6):
        if not math.isfinite(reward_atol) or reward_atol < 0:
            raise ValueError("Invalid reward tolerance")
        self.policy_version = policy_version
        self.reward_atol = reward_atol
        self.phase = "ready"
        self.identity = None
        self.next_hash = None
        self.action = None

    def observe(self, identity):
        if self.phase != "ready":
            raise RuntimeError("Previous feedback/update is incomplete")
        if not identity.episode or not identity.state_hash:
            raise ValueError("Missing observation identity")
        if identity.policy_version != self.policy_version:
            raise RuntimeError("Stale observation policy version")
        if self.identity is not None:
            if identity.episode != self.identity.episode:
                raise RuntimeError("New episode requires an explicit gate reset")
            if identity.decision != self.identity.decision + 1:
                raise RuntimeError("Duplicate or skipped decision")
            if identity.state_hash != self.next_hash:
                raise RuntimeError("Observation is not the canonical next state")
        self.identity = identity
        self.phase = "observed"

    def choose(self, action):
        if self.phase != "observed" or action.identity != self.identity:
            raise RuntimeError("Action identity/order mismatch")
        p = action.probabilities
        if (not action.candidate_hash or type(action.selected) is not int
                or not 0 <= action.selected < 20 or len(p) != 20
                or any(not math.isfinite(x) or x < 0 for x in p)
                or not math.isclose(sum(p), 1.0, abs_tol=1e-6)
                or p[action.selected] <= 0):
            raise ValueError("Invalid complete-action selection")
        self.action = action
        self.phase = "chosen"

    def main_executed(self, next_hash):
        if self.phase != "chosen" or not next_hash:
            raise RuntimeError("Canonical transition must follow action selection")
        self.next_hash = next_hash
        self.phase = "executed"

    def feedback(self, value):
        if self.phase != "executed" or value.identity != self.identity:
            raise RuntimeError("Feedback precedes execution or belongs to another step")
        if value.candidate_hash != self.action.candidate_hash:
            raise RuntimeError("Candidate identity mismatch")
        if len(value.rewards) != 20 or len(value.valid) != 20:
            raise ValueError("Strict mode requires exactly 20 branch outcomes")
        # A branch computation failure is not a low-reward action.
        if not all(type(x) is bool and x for x in value.valid):
            raise ValueError("A failed branch invalidates strict full20 feedback")
        if not all(math.isfinite(x) for x in value.rewards + (value.main_reward,)):
            raise ValueError("Non-finite branch reward")
        if not (value.main_next_hash == self.next_hash
                == value.selected_branch_next_hash):
            raise RuntimeError("Selected branch/canonical state parity failed")
        if not math.isclose(value.main_reward, value.rewards[self.action.selected],
                            rel_tol=0, abs_tol=self.reward_atol):
            raise RuntimeError("Selected branch/canonical reward parity failed")
        self.phase = "feedback"

    def updated(self, policy_version, optimized):
        if self.phase != "feedback" or type(optimized) is not bool:
            raise RuntimeError("Update requires accepted feedback")
        if policy_version != self.policy_version + int(optimized):
            raise RuntimeError("Update version does not match actual optimizer step")
        self.policy_version = policy_version
        self.phase = "ready"
