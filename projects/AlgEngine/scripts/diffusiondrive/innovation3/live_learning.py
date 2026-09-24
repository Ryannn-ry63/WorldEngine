"""Audited learner boundary, independent of renderer imports for CPU testing."""
import hashlib
import time
import torch
from .live_audit import LiveStepGate
from .transport import digest


def state_digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().contiguous()
            h.update(str((x.dtype, tuple(x.shape))).encode())
            h.update(x.numpy().tobytes())
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(repr(k).encode()); visit(x[k])
        elif isinstance(x, (tuple, list)):
            for v in x: visit(v)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


class LiveLearning:
    def __init__(self, learner):
        self.learner = learner
        self.wire = LiveStepGate()
        self.reference_hash = state_digest(learner.reference.state_dict())
        self.last_residual_hash = state_digest(learner.selector.state_dict())
        self.awaiting_ack = None

    def observe(self, identity):
        if self.awaiting_ack is not None:
            raise RuntimeError('Next observation precedes worker update acknowledgement')
        self.wire.observe(identity)
        if identity['policy_version'] != self.learner.version:
            raise RuntimeError('Worker and learner versions disagree')
        self.times = dict(observe=time.monotonic_ns())

    def choose(self, context, base, candidates):
        learner = self.learner
        if self.wire.gate.phase != 'observed':
            raise RuntimeError('Selection requires the current observation')
        self.before_hash = state_digest(learner.selector.state_dict())
        if self.before_hash != self.last_residual_hash:
            raise RuntimeError('Next action did not retain the last updated residual')
        self.context = {k: v.detach().clone() for k, v in context.items()}
        self.context_hash = state_digest(self.context)
        with torch.no_grad():
            self.frozen = base + learner.reference(**self.context)
            self.before_logits = self.frozen + learner.selector(**self.context)
        selected, probabilities, version = learner.choose(context, base, self.wire.gate.identity)
        if not torch.equal(probabilities, self.before_logits.softmax(-1)):
            raise RuntimeError('Action probabilities do not match audited current logits')
        if version == 0 and not torch.equal(self.before_logits, self.frozen):
            raise RuntimeError('Step-zero residual must be exactly zero')
        request = dict(kind='action', identity=self.wire.raw_identity,
            candidates=candidates, candidate_hash=digest(candidates), selected=selected,
            probabilities=probabilities[0].tolist(), current_logits=self.before_logits[0].tolist(),
            reference_logits=self.frozen[0].tolist())
        self.wire.choose(request)
        self.times['action'] = time.monotonic_ns()
        self.action_audit = dict(residual_hash_before=self.before_hash,
            context_hash=self.context_hash, version_before=version,
            current_logits=request['current_logits'], reference_logits=request['reference_logits'],
            probabilities=request['probabilities'],
            current_vs_frozen_probability_max_abs=float((probabilities-self.frozen.softmax(-1)).abs().max()))
        return request

    def main_executed(self, main):
        self.wire.main_executed(main)
        self.times['main'] = time.monotonic_ns()

    def update(self, branches):
        self.wire.feedback(branches)
        self.times['feedback'] = time.monotonic_ns()
        learner = self.learner
        optimizer_before = state_digest(learner.optimizer.state_dict())
        update = learner.update([[r['reward'] for r in branches['rewards']]], self.wire.gate.identity)
        # The host digests synchronize device work before any acknowledgement.
        after_hash = state_digest(learner.selector.state_dict())
        optimizer_after = state_digest(learner.optimizer.state_dict())
        with torch.no_grad():
            after_logits = self.frozen + learner.selector(**self.context)
        if not torch.isfinite(after_logits).all():
            raise RuntimeError('Non-finite post-update logits')
        delta = float((after_logits-self.before_logits).abs().max())
        if not update['optimized'] and (after_hash != self.before_hash or
                optimizer_after != optimizer_before or delta != 0):
            raise RuntimeError('No-signal step changed residual or optimizer state')
        if state_digest(learner.reference.state_dict()) != self.reference_hash or any(
                p.grad is not None or p.requires_grad for p in learner.reference.parameters()):
            raise RuntimeError('Frozen V3 reference changed or received gradients')
        if state_digest(self.context) != self.context_hash:
            raise RuntimeError('Feedback/update mutated live context')
        self.last_residual_hash = after_hash
        self.times['update_complete'] = time.monotonic_ns()
        ack = dict(kind='feedback_ack', identity=self.wire.raw_identity,
                   candidate_hash=self.wire.gate.action.candidate_hash,
                   next_hash=self.wire.gate.next_hash,
                   policy_version=update['policy_version'], optimized=update['optimized'])
        self.awaiting_ack = self.wire.updated(ack)
        audit = dict(self.action_audit, update=update,
            residual_hash_after=after_hash, optimizer_hash_before=optimizer_before,
            optimizer_hash_after=optimizer_after, post_update_logits=after_logits[0].tolist(),
            same_context_logit_change_after_update=delta,
            version_after=learner.version, optimized=update['optimized'],
            times_ns=dict(self.times))
        # Release graph-free copies between decisions; pending graph is consumed.
        self.context = None
        return ack, audit

    def acknowledged(self, response):
        if self.awaiting_ack is None or response != self.awaiting_ack:
            raise RuntimeError('Worker update acknowledgement mismatch')
        self.awaiting_ack = None


def learning_evidence(events):
    groups = [e for e in events if e.get('kind') == 'online_update']
    updates = sum(e['optimized'] for e in groups)
    changed = any(e['optimized'] and e['same_context_logit_change_after_update'] > 0
                  for e in groups)
    consumed = any(e['version_before'] > 0 and e['current_vs_frozen_probability_max_abs'] > 0
                   for e in groups)
    return dict(actual_optimizer_steps=updates, update_attempts=len(groups),
                groups_with_signal=updates, groups_without_signal=len(groups)-updates,
                local_groups_with_signal=sum(e['update'].get('local_group_signal', e['optimized']) for e in groups),
                globally_optimized_groups=updates,
                update_changed_logits=changed, next_action_used_updated_policy=consumed,
                closed_loop_learning_verified=bool(updates and changed and consumed))
