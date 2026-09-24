"""One-update-per-feedback selector learner (single rank, no simulator).

The historical parameterization keeps a frozen V3 reference and trains a new
zero-output correction.  The paper-facing parameterization initializes the
trainable selector from the complete offline V3 checkpoint and continues
fine-tuning that selector.  Both keep the generator, perception stack and
original base logits frozen.
"""
import copy
import hashlib
import json
import math
import re
from pathlib import Path

import torch

from grpo_selector_v3_cached_common import (clip_grad_norm_cpu_, exact_group_loss,
                                            normalized_advantage)

CONTEXT_KEYS = frozenset(("candidate_features", "candidate_trajectories",
                          "route_bev_features", "status_token", "ego_query", "agents_query"))
FROZEN_V3_PLUS_ZERO_RESIDUAL = "frozen_v3_plus_zero_residual"
V3_INITIALIZED_SELECTOR_FINETUNE = "v3_initialized_selector_finetune"
PARAMETERIZATIONS = frozenset((FROZEN_V3_PLUS_ZERO_RESIDUAL,
                               V3_INITIALIZED_SELECTOR_FINETUNE))


def _state_fingerprint(state):
    """Stable tensor fingerprint used for checkpoint provenance and CPU fixtures."""
    digest = hashlib.sha256()
    for key in sorted(state, key=str):
        value = state[key]
        if not torch.is_tensor(value):
            digest.update(repr((key, value)).encode())
            continue
        value = value.detach().cpu().contiguous()
        digest.update(str(key).encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _architecture_descriptor(module):
    # Import aliases differ between lightweight CPU tools and mmdet. The
    # constructor config, not the import module name or repr, defines V3.
    return {
        "class": module.__class__.__qualname__,
        "config": copy.deepcopy(module.config_dict()),
        "state_layout": [(k, str(v.dtype), tuple(v.shape))
                         for k, v in module.state_dict().items()],
    }


def _sha256_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class OnlineV3Learner:
    def __init__(self, selector, learning_rate=3e-5, kl_weight=1e-3, seed=0,
                 gradient_reducer=None, update_coordinator=None,
                 parameterization=FROZEN_V3_PLUS_ZERO_RESIDUAL,
                 initialization_source=None):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if not math.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("Invalid KL coefficient")
        if parameterization not in PARAMETERIZATIONS:
            raise ValueError("Unknown online selector parameterization: %s" % parameterization)
        if initialization_source is not None and not isinstance(initialization_source, dict):
            raise TypeError("initialization_source must be a mapping")
        self.parameterization = parameterization
        self._architecture = _architecture_descriptor(selector)
        self._architecture_sha256 = _sha256_json(self._architecture)
        selector_fingerprint = _state_fingerprint(selector.state_dict())
        source = copy.deepcopy(initialization_source or {})
        source.setdefault("kind", "selector_state_fingerprint")
        source.setdefault("selector_sha256", selector_fingerprint)
        if source['kind'] not in ('selector_state_fingerprint', 'offline_selector_file'):
            raise ValueError('Unknown initialization source kind')
        if source['kind'] == 'offline_selector_file' and not re.fullmatch(
                '[0-9a-f]{64}', str(source.get('sha256', ''))):
            raise ValueError('File initialization requires a SHA256')
        source["kind"] = str(source["kind"])
        if source.get("sha256") is not None:
            source["sha256"] = str(source["sha256"])
        if source.get("path") is not None:
            source["path"] = str(source["path"])
        # A file source is validated before model copies are made.  The
        # fingerprint remains the authoritative no-file fixture identity.
        if source.get("path") and source.get("sha256"):
            path = Path(source["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != source["sha256"]:
                raise ValueError("Initialization selector source SHA256 mismatch")
        source["selector_sha256"] = selector_fingerprint
        self.initialization = {
            "source": source,
            "selector_init_sha256": selector_fingerprint,
            "reference_init_sha256": selector_fingerprint,
            "architecture": copy.deepcopy(self._architecture),
            "architecture_sha256": self._architecture_sha256,
        }
        self.reference = copy.deepcopy(selector).eval().requires_grad_(False)
        self.selector = copy.deepcopy(selector).eval().requires_grad_(True)
        if parameterization == FROZEN_V3_PLUS_ZERO_RESIDUAL:
            # Historical mode: zero only the NEW online correction, never the
            # trained V3 reference.
            torch.nn.init.zeros_(self.selector.delta_head[-1].weight)
            torch.nn.init.zeros_(self.selector.delta_head[-1].bias)
        self.optimizer = torch.optim.AdamW(self.selector.parameters(), lr=learning_rate,
                                           weight_decay=1e-4)
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.version = 0
        self.attempts = 0
        self.rng = torch.Generator(device=next(selector.parameters()).device)
        self.rng.manual_seed(seed)
        self.pending = None
        if gradient_reducer is not None and not callable(gradient_reducer):
            raise TypeError("gradient_reducer must be callable")
        if update_coordinator is not None and not callable(update_coordinator):
            raise TypeError("update_coordinator must be callable")
        self.gradient_reducer = gradient_reducer
        self.update_coordinator = update_coordinator

    @staticmethod
    def state_fingerprint(state):
        return _state_fingerprint(state)

    def provenance(self):
        return copy.deepcopy(self.initialization)

    def reference_logits(self, base_logits, inputs):
        """Frozen V3 logits, shared by action auditing and the KL reference."""
        return base_logits + self.reference(**inputs)

    def current_logits(self, base_logits, inputs, reference_logits=None):
        """Current behavior logits for either online parameterization."""
        selector_logits = self.selector(**inputs)
        if self.parameterization == FROZEN_V3_PLUS_ZERO_RESIDUAL:
            if reference_logits is None:
                reference_logits = self.reference_logits(base_logits, inputs)
            return reference_logits + selector_logits
        return base_logits + selector_logits

    # Kept private alias for existing probes/tests.
    def _current_logits(self, base_logits, inputs, reference_logits=None):
        return self.current_logits(base_logits, inputs, reference_logits)

    def choose(self, context, base_logits, decision_id):
        if self.pending is not None:
            raise RuntimeError("Feedback/update required before next action")
        if set(context) != CONTEXT_KEYS:
            raise ValueError("Unexpected selector inputs; reward/labels cannot enter forward")
        if tuple(base_logits.shape) != (1, 20):
            raise ValueError("Single-rank pilot requires one complete 20-action group")
        parameter = next(self.selector.parameters())
        if any(v.dtype != parameter.dtype or v.device != parameter.device
               for v in [base_logits, *context.values()]):
            raise ValueError('Selector inputs must match model dtype/device; normalize at the observation adapter')
        # clone outside inference_mode: inference tensors cannot be saved for backward.
        inputs = {k: v.detach().clone() for k, v in context.items()}
        base = base_logits.detach().clone()
        if not torch.isfinite(base).all() or not all(torch.isfinite(v).all() for v in inputs.values()):
            raise ValueError("Non-finite live inputs")
        with torch.no_grad():
            # Compute the frozen reference once.  Legacy behavior reuses this
            # exact no-grad value instead of adding a second reference forward.
            reference_logits = self.reference_logits(base, inputs)
            behavior_logits = self.current_logits(base, inputs, reference_logits)
        logits = self.current_logits(base, inputs, reference_logits)
        if not all(torch.isfinite(v).all() for v in (logits, reference_logits, behavior_logits)):
            raise ValueError("Non-finite selector logits")
        # PyTorch 2.0 eval/no_grad and autograd Transformer kernels can differ
        # numerically. Actions must use the exact frozen-V3 inference path.
        drift = float((behavior_logits - logits.detach()).abs().max())
        if not torch.allclose(behavior_logits, logits.detach(), atol=1e-5, rtol=1e-4):
            raise RuntimeError("Training/inference selector logits diverged: " + str(drift))
        self.logit_drift = drift
        probabilities = behavior_logits.softmax(-1)
        selected = int(torch.multinomial(probabilities, 1, generator=self.rng).item())
        self.pending = (decision_id, logits, reference_logits)
        return selected, probabilities, self.version

    def update(self, rewards, decision_id):
        if self.pending is None or self.pending[0] != decision_id:
            raise RuntimeError("Stale, duplicate, or missing feedback")
        _, logits, reference_logits = self.pending
        rewards = torch.as_tensor(rewards, device=logits.device, dtype=logits.dtype).detach()
        if tuple(rewards.shape) != (1, 20) or not torch.isfinite(rewards).all():
            raise ValueError("Strict full20 requires all finite rewards")
        valid = torch.ones_like(rewards, dtype=torch.bool)
        _, active = normalized_advantage(rewards, valid)
        self.optimizer.zero_grad(set_to_none=True)
        loss, policy, kl = exact_group_loss(logits, reference_logits, rewards, valid, 1.0, self.kl_weight)
        optimized = bool(active.any())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite online loss")
        if optimized:
            loss.backward()
        globally_optimized = optimized
        if self.update_coordinator is not None:
            globally_optimized = bool(self.update_coordinator(
                self.selector.parameters(), optimized))
        elif optimized and self.gradient_reducer is not None:
            self.gradient_reducer(self.selector.parameters())
        if globally_optimized:
            # A distributed coordinator may have supplied zero gradients for a
            # locally tied group whose peer has a real signal.
            clip_grad_norm_cpu_(self.selector.parameters(), 1.0)
            self.optimizer.step()
            self.version += 1
        # A step is skipped only when every rank (or the single rank) has no
        # group signal; no AdamW weight decay or fake version increment.
        self.attempts += 1
        self.pending = None
        return dict(policy_version=self.version, optimized=globally_optimized, attempts=self.attempts,
                    local_group_signal=optimized,
                    reward_std=float(rewards.std(unbiased=False)),
                    skip_reason=None if globally_optimized else 'no_group_signal_std_le_1e-6',
                    loss=float(loss.detach()), policy_loss=float(policy.detach()), kl=float(kl.detach()),
                    training_inference_max_abs_logit_drift=self.logit_drift)

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError("Cannot checkpoint an unconsumed action")
        if self.parameterization == FROZEN_V3_PLUS_ZERO_RESIDUAL:
            # Keep schema2 byte/field semantics for historical replay.
            return copy.deepcopy(dict(schema_version=2,
                                      parameterization=self.parameterization,
                                      selector=self.selector.state_dict(),
                                      reference=self.reference.state_dict(), optimizer=self.optimizer.state_dict(),
                                      rng=self.rng.get_state(), version=self.version, attempts=self.attempts,
                                      kl_weight=self.kl_weight))
        return copy.deepcopy(dict(schema_version=3,
                                  parameterization=self.parameterization,
                                  initialization=self.provenance(),
                                  selector=self.selector.state_dict(),
                                  reference=self.reference.state_dict(), optimizer=self.optimizer.state_dict(),
                                  rng=self.rng.get_state(), version=self.version, attempts=self.attempts,
                                  learning_rate=self.learning_rate, kl_weight=self.kl_weight,
                                  rng_device=self.rng.device.type, code_contract='innovation3_online_selector_v3'))

    def _validate_schema3(self, state):
        if state.get('schema_version') != 3 or state.get('parameterization') != self.parameterization:
            raise RuntimeError("Invalid checkpoint boundary/schema")
        if self.parameterization != V3_INITIALIZED_SELECTOR_FINETUNE:
            raise RuntimeError("Schema3 is reserved for initialized selector fine-tuning")
        if state.get('kl_weight') != self.kl_weight or state.get('learning_rate') != self.learning_rate:
            raise ValueError("Checkpoint optimizer configuration mismatch")
        saved = state.get('initialization')
        if not isinstance(saved, dict):
            raise RuntimeError("Schema3 checkpoint lacks initialization provenance")
        current = self.initialization
        for key in ('selector_init_sha256', 'reference_init_sha256', 'architecture_sha256'):
            if saved.get(key) != current.get(key):
                raise RuntimeError("Checkpoint initialization provenance mismatch: " + key)
        if saved.get('architecture') != current.get('architecture'):
            raise RuntimeError("Checkpoint selector architecture/config mismatch")
        saved_source = saved.get('source', {})
        current_source = current.get('source', {})
        for key in ('kind', 'sha256', 'selector_sha256', 'payload_schema', 'method', 'scene_selector_config'):
            if saved_source.get(key) != current_source.get(key):
                raise RuntimeError("Checkpoint initialization source mismatch: " + key)
        if _state_fingerprint(state.get('reference', {})) != current['reference_init_sha256']:
            raise RuntimeError("Checkpoint frozen reference fingerprint mismatch")
        if state.get('code_contract') != 'innovation3_online_selector_v3':
            raise RuntimeError('Checkpoint code contract mismatch')

    def load_state_dict(self, state):
        if self.pending is not None:
            raise RuntimeError("Cannot restore with a pending action")
        if not isinstance(state, dict):
            raise RuntimeError("Invalid checkpoint boundary/schema")
        if state.get('schema_version') == 3:
            self._validate_schema3(state)
            if state.get('rng_device') != self.rng.device.type:
                raise RuntimeError('Checkpoint RNG backend mismatch')
        else:
            expected_schema = 2 if self.parameterization == FROZEN_V3_PLUS_ZERO_RESIDUAL else 3
            if (state.get('schema_version') != expected_schema
                    or state.get('parameterization') != self.parameterization):
                raise RuntimeError("Invalid checkpoint boundary/schema")
            if state.get('kl_weight') != self.kl_weight:
                raise ValueError("Checkpoint KL configuration mismatch")
        # Validate model keys/shapes before any module mutation.
        for target, name in ((self.selector, 'selector'), (self.reference, 'reference')):
            payload = state.get(name)
            if not isinstance(payload, dict):
                raise RuntimeError("Checkpoint %s state is missing" % name)
            expected = target.state_dict()
            if set(payload) != set(expected) or any(not torch.is_tensor(payload[k]) or tuple(payload[k].shape) != tuple(expected[k].shape)
                                                    or payload[k].dtype != expected[k].dtype for k in expected):
                raise RuntimeError("Checkpoint %s structure mismatch" % name)
        if any(type(state.get(k)) is not int or state[k] < 0 for k in ('version', 'attempts')):
            raise RuntimeError('Invalid checkpoint counters')
        if state['version'] > state['attempts']:
            raise RuntimeError('Checkpoint version exceeds attempts')
        for name in ('selector', 'reference'):
            if not all(torch.isfinite(v).all() for v in state[name].values()):
                raise RuntimeError('Non-finite checkpoint ' + name)
        # Stage every potentially failing load on isolated objects. A bad RNG,
        # optimizer, tensor, or counter must leave the live learner untouched.
        staged_selector = copy.deepcopy(self.selector)
        staged_selector.load_state_dict(state['selector'], strict=True)
        staged_optimizer = torch.optim.AdamW(staged_selector.parameters(),
                                            lr=self.learning_rate, weight_decay=1e-4)
        optimizer_state = state.get('optimizer', {})
        groups = optimizer_state.get('param_groups', [])
        expected_ids = [i for group in self.optimizer.state_dict()['param_groups'] for i in group['params']]
        saved_ids = [i for group in groups for i in group.get('params', [])]
        if (saved_ids != expected_ids or not set(optimizer_state.get('state', {})).issubset(saved_ids)
                or (state['version'] > 0 and not optimizer_state.get('state'))):
            raise RuntimeError('Checkpoint optimizer parameter structure mismatch')
        staged_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        for group, expected_group in zip(staged_optimizer.param_groups, self.optimizer.param_groups):
            for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize'):
                if state['schema_version'] == 3 and group.get(key) != expected_group.get(key):
                    raise RuntimeError('Checkpoint optimizer configuration mismatch: ' + key)
            for parameter in group['params']:
                slot = staged_optimizer.state.get(parameter, {})
                if not slot:
                    continue
                if set(slot) != {'step', 'exp_avg', 'exp_avg_sq'}:
                    raise RuntimeError('Invalid AdamW checkpoint slot')
                for key in ('exp_avg', 'exp_avg_sq'):
                    value = slot[key]
                    if (not torch.is_tensor(value) or value.shape != parameter.shape
                            or not torch.isfinite(value).all()):
                        raise RuntimeError('Invalid AdamW checkpoint moment')
                step = slot['step']
                if (not torch.is_tensor(step) or step.numel() != 1
                        or not torch.isfinite(step).all()
                        or float(step) < 0 or float(step) != int(float(step))
                        or float(step) > state['version']):
                    raise RuntimeError('Invalid AdamW checkpoint step')
        staged_rng = torch.Generator(device=self.rng.device)
        staged_rng.set_state(state['rng'].cpu())
        # All validation succeeded. Keep module identity for external users.
        self.selector.load_state_dict(staged_selector.state_dict(), strict=True)
        self.reference.load_state_dict(state['reference'], strict=True)
        self.optimizer.load_state_dict(staged_optimizer.state_dict())
        self.rng.set_state(staged_rng.get_state())
        self.version, self.attempts = state['version'], state['attempts']
