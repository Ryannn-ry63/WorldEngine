"""One-update-per-feedback standard V3 learner (single rank, no simulator).

Keep the registered trained V3 frozen. A separate V3-shaped residual starts
with zero output and is the only trainable module. Its encoder is initialized
from V3, while the final projection is zeroed. Context must be freshly computed
from the live observation. The frozen generator supplies original base logits.
"""
import copy
import math
import torch
from grpo_selector_v3_cached_common import exact_group_loss, normalized_advantage

CONTEXT_KEYS = frozenset(("candidate_features", "candidate_trajectories",
                          "route_bev_features", "status_token", "ego_query", "agents_query"))


class OnlineV3Learner:
    def __init__(self, selector, learning_rate=3e-5, kl_weight=1e-3, seed=0):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if not math.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("Invalid KL coefficient")
        self.reference = copy.deepcopy(selector).eval().requires_grad_(False)
        self.selector = copy.deepcopy(selector).eval().requires_grad_(True)
        # Zero only the NEW online correction, never the trained V3 reference.
        torch.nn.init.zeros_(self.selector.delta_head[-1].weight)
        torch.nn.init.zeros_(self.selector.delta_head[-1].bias)
        self.optimizer = torch.optim.AdamW(self.selector.parameters(), lr=learning_rate,
                                          weight_decay=1e-4)
        self.kl_weight = kl_weight
        self.version = 0
        self.attempts = 0
        self.rng = torch.Generator(device=next(selector.parameters()).device)
        self.rng.manual_seed(seed)
        self.pending = None

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
            reference_logits = base + self.reference(**inputs)
            behavior_logits = reference_logits + self.selector(**inputs)
        logits = reference_logits + self.selector(**inputs)
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
            torch.nn.utils.clip_grad_norm_(self.selector.parameters(), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.version += 1
        # Ties are recorded as no-signal; no AdamW weight decay or fake version increment.
        self.attempts += 1
        self.pending = None
        return dict(policy_version=self.version, optimized=optimized, attempts=self.attempts,
                    loss=float(loss.detach()), policy_loss=float(policy.detach()), kl=float(kl.detach()),
                    training_inference_max_abs_logit_drift=self.logit_drift)

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError("Cannot checkpoint an unconsumed action")
        return copy.deepcopy(dict(schema_version=2, parameterization="frozen_v3_plus_zero_residual",
                                  selector=self.selector.state_dict(),
                                  reference=self.reference.state_dict(), optimizer=self.optimizer.state_dict(),
                                  rng=self.rng.get_state(), version=self.version, attempts=self.attempts,
                                  kl_weight=self.kl_weight))

    def load_state_dict(self, state):
        if (self.pending is not None or state['schema_version'] != 2
                or state.get('parameterization') != 'frozen_v3_plus_zero_residual'):
            raise RuntimeError("Invalid checkpoint boundary/schema")
        if state['kl_weight'] != self.kl_weight:
            raise ValueError("Checkpoint KL configuration mismatch")
        self.selector.load_state_dict(state['selector'], strict=True)
        self.reference.load_state_dict(state['reference'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.rng.set_state(state['rng'].cpu())
        self.version, self.attempts = state['version'], state['attempts']
