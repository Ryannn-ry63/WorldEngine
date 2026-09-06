# Implementation validation — 2026-09-06

Scope: software + source audit + read-only cached diagnostics.
No new real-model training, simulator rollout, rare efficacy result or confirmation result.
Current node: one RTX4090 (49,140 MiB); formal entrypoint requires eight H100s.

- 25 new rare CPU tests passed. Includes exact zero-adapter parity, frozen V3 tensors,
  cached/online semantics, input allowlist, teacher-to-student KL,
  500-step synthetic optimizer interruption/resume equivalence,
  complete condition inventory, unique three-seed winner, confirmation,
  initial-state-only bridge checks, truthful exposure metadata, and cumulative budget.
- 27 existing CFPI deployment tests passed after the rare hooks.
- Python compilation, shell syntax and git diff whitespace checks passed.
- Real rare config resolved and real OOF bank loaded with the production environment.
- Three bridge manifests materialized: 64 scenes each, OOF fold routing, start decision4,
  64 initial-context audit references each. No simulator launched.
- Full historical common-reference dependency hashing completed in software audit01.
- Existing scalar/gate checkpoints have **963 identical non-selector tensors**.
- Real correction64 + replay512 CPU preflight passed:
  zero adapter and original V3 scores exactly equal;
  cached/online maximum absolute difference0;
  teacher maximum absolute recompute difference6.866455078125e-5 <1e-4;
  no argmax mismatch.

Development artifacts:
`experiments/diffusiondrive/selector_rare_v1/runs/rare_software_audit_20260906_01/`.
Its contract was created during implementation, before subsequent validation hardening.
It is not a formal run and must not be used for GPU continuation.
Use a new formal run ID with the final code. New formal audit performs fresh dependency hashing.

## Fixed posthoc OOF/refit diagnostic

All192 original OOF action predictions reproduced on their frozen cached states.
Action disagreement between OOF and full64 refit:

| Fixed policy | Same64-state disagreement |
|---|---:|
| Q-GRPO T1 seed0 | 75.00% |
| Q-GRPO T1 seed2 | 78.125% |
| local-GRPO T1 seed2 | 34.375% |

The refits reduce their in-sample exact-group objective, but this is not evidence of
better unseen-state or continuous closed-loop behavior. This strengthens the reason
to separate refit from takeover timing in the fixed bridge; it does not identify a causal root cause.
The conditions were chosen posthoc from known previous results, not sampled as an efficacy study.

## Budget caveat

Historical-runtime conservative rollout forecast: bridge4.60, screen38.12,
confirmation21.61 GPUh; total64.33 before training/preflight.
The64 GPUh allowance is a cap, not a claim that every phase fits.
No budget expansion, seed reduction or cohort trimming was performed.
