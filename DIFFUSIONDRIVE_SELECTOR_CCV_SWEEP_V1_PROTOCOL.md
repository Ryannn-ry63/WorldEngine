# DiffusionDrive Candidate Causal-Value Sweep v1

Status: **FROZEN BEFORE ANY CCV-SWEEP OUTCOME**

## Question

The preceding causal-branch pilot showed that a single local scalar-reward
oracle action rescues frozen-V3 failures, but it did not uniquely outperform an
equal-magnitude non-improving perturbation.  This experiment identifies whether
the remaining ambiguity is caused by local-reward horizon/ranking mismatch,
reward ties, or diffusion-action neighbourhood sensitivity.

## Frozen scope

- Reuse the 33 train-only targets frozen by causal pilot
  `causal_prefixfix_20260903`: 9 failed and 24 solved targets.
- Freeze scalar V3 train seed 0, checkpoint, perception, generator, base
  selector, residual selector, 20 candidates, candidate-noise namespace, and
  each target decision step.
- For each target and candidate index `0..19`, replace exactly one action with
  that candidate, then immediately return control to frozen V3.
- The resulting closed-loop score is diagnostic candidate causal value
  `Q^V3(s,a)`. It is not consumed as training data in v1.
- Official scalar PDM is the local reward. Reward components are descriptive
  diagnostics only. No navtest, external data, or new supervision is used.

## Required sentinel

Before the formal sweep, run the prior smoke8 targets with per-scene policy,
oracle, and matched treatment manifests.  All scene scores, NC, DAC, EP and
success labels must reproduce their corresponding prior runs within `1e-6`.
Candidate/logit context must reproduce within `1e-5`, and action replacement
parity must be at most `1e-4`.

## Primary decomposition

For every target:

- `q_policy`: causal value of V3's selected candidate;
- `q_reward_oracle`: value of the prior stable scalar-reward oracle;
- `q_reward_set`: best causal value among all candidates tied for maximum local
  reward within `1e-8`;
- `q_star`: best causal value among all 20 candidates.

The following identity must hold within `1e-8`:

`q_star-q_policy = (q_star-q_reward_set) + (q_reward_set-q_reward_oracle) + (q_reward_oracle-q_policy)`.

The three right-hand terms are respectively the reward ranking/horizon gap,
reward-tie resolution gap, and current-selector gap.

## Predeclared gates

Primary gates use the 9 failed targets and origin-log cluster bootstrap with
10,000 repetitions and seed 20260903.  A gap is established only when:

1. mean gap is at least `0.05`;
2. positive gaps occur in at least 3 origin logs; and
3. the 95% bootstrap lower bound is above zero.

Geometry neighbourhood structure is established on all 33 targets when
leave-one-candidate-out inverse-distance 3-NN has mean per-scene Spearman at
least `0.25`, origin-log bootstrap lower bound above zero, and recalls a causal
best candidate in its predicted top-3 for at least half of targets.

## Decisions

- Ranking/horizon gap passes: authorize long-horizon causal-advantage method
  design. Add basin-aware aggregation only if the geometry gate also passes.
- Ranking gap fails and tie gap passes: authorize scalar coarse ranking plus a
  diffusion-set tie breaker.
- Both gaps fail and selector gap passes: retain scalar reward and improve the
  selector representation/optimization without new causal labels.
- A point signal without bootstrap/log support: expand train-only targets or a
  frozen noise namespace before method training.
- No actionable gap: stop this method branch.

No decision in this diagnostic automatically authorizes using `Q^V3` as a
training label. That requires a separately frozen protocol and leakage audit.

