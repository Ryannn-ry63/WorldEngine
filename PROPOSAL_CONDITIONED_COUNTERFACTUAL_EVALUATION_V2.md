# PCRA V2: Decoupled Counterfactual Opportunity-Risk Evaluation

## Frozen claim

DiffusionDrive still generates exactly 20 candidates with the immutable epoch-100
perception/generator policy. The immutable proposal32 selector exposes one proposal
`p`; the original selector exposes incumbent `b`. PCRA V2 only learns whether the
deployed system should replace `b` with `p`.

Proposal32 SHA256:

```text
562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6
```

## Method

PCRA V1 trained only a small head on the frozen proposal representation. V2 adds a
separate-parameter copy of the complete relational scene/set encoder. It is
initialized exactly from proposal32, while the proposal branch remains bitwise
frozen. A shared directed scorer evaluates `(p,b)` and `(b,p)`.

For scalar official-PDM gain `delta = PDM(p) - PDM(b)`:

```text
opportunity target = max(delta, 0)
risk target        = max(-delta, 0)
evidence           = softplus(f(p,b)) - softplus(f(b,p))
```

The main loss is the unweighted mean of the two squared value errors. Deployment
accepts only when `p != b` and `evidence > 0`; there is no threshold search. Reward
components are diagnostics only and are never model inputs.

## Data contract

The main arm retains the existing V3 rare-rollout/rarelog mixture exactly:

```text
50% paired common
50% hard = real rare + senior-v1 filtered synthetic rollout
```

The equal-source arm is a non-promotable ablation with one-third common, one-third
real rare, and one-third synthetic. It uses the same total example budget.

## Fixed search

| Arm | Encoder | Loss | Source risk | Promotable |
|---|---|---|---|---|
| PCRA V1 | frozen | regret BCE | rarelog | no |
| independent-regret-BCE | trainable | regret BCE | rarelog | no |
| frozen-opportunity-risk | frozen | opportunity-risk | rarelog | no |
| independent-signed-gain | trainable | signed gain MSE | rarelog | no |
| independent-opportunity-risk | trainable | opportunity-risk | rarelog | yes |
| independent-opportunity-risk-equal-source | trainable | opportunity-risk | equal source | no |

Every new arm uses train split, seed 0, 16 epochs, batch 64, LR `1e-4`, final epoch,
and fixed zero threshold. No epoch, margin, coefficient, or seed selection is allowed.

## Implementation validation

The focused selector regression suite passes 56 tests. A CPU audit against the
real proposal32 checkpoint verifies its frozen SHA, exact-zero evaluator encoder
initialization error, 82 evaluator-only trainable tensors, and 80 frozen proposal
tensors. Shell syntax, Python compilation, and trainer argument parsing also pass.
The current implementation host is a single RTX 4090, so no 8xH100 search or
development score is claimed here.

## Commands

From the isolated worktree root:

```bash
./run_diffusiondrive_selector_pcra_v2_smoke.sh
./run_diffusiondrive_selector_pcra_v2_search_8hopper.sh
./run_diffusiondrive_selector_pcra_v2_development_8hopper.sh SEARCH_ROOT
```

The development command exits `2` when the predeclared gate fails. That is a valid
negative experimental result, not an infrastructure failure. A failure stops PCRA
V2 before closed-loop or certification data are consumed.

## Gate

The only promotable arm must satisfy all previous PCRA thresholds: equal-stratum
PDM at least best proposal plus `0.005`; common within `0.005` of the frozen
reference; rare and synthetic within `0.020` of their best proposal; common
degraded fraction at most `0.10`; proposal SHA and override threshold fixed.

If it passes, use `select_pcra_v2_closed_loop_development.py` for the unchanged
three-seed closed-loop/SR gate. Do not promote a control arm or relax a threshold
post hoc.
