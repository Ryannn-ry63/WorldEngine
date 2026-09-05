# IVPS V1 Fixed-Budget Search Result — 2026-08-29

## Decision

IVPS V1 does not pass its predeclared offline development gate.  No verifier
margin may advance to closed-loop development, certification, or formal
three-seed evaluation.  The verifier threshold, reward temperature, data
mixture, and gates must not be changed after observing this result.

The experiment supports the proposal–incumbent complementarity diagnosis, but
rejects the current all-candidate soft-BCE verifier as the solution.  The best
development score is instead the 32-epoch scalar-preference proposal control.

## Immutable provenance

- Search root: `experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z`
- Gate audit: `experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/development_gate.json`
- Data: train split for optimization; log-disjoint development for evaluation.
- Certification consumed: false.
- All six training reports and all six development evaluations: `status=PASS`.
- Proposal: relational B01, scalar preference weight 1.0, 16 epochs.
- Verifier: frozen proposal, 16 epochs, scalar-PDM temperature 0.05, fixed
  deployment threshold 0, margins `{0.00, 0.01, 0.03, 0.05}`.
- Compute control: proposal-only, 32 epochs.

## Development results

All gains are against the frozen V3 reference.  Coverage and precision are for
common-stratum proposal overrides.

| Arm | Equal PDM | Common PDM | Common gain | Rare PDM | Rare gain | Synthetic PDM | Common coverage | Common precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proposal-16 | 0.801626 | 0.878352 | -0.048672 | 0.725469 | +0.440074 | 0.801057 | — | — |
| IVPS m=0.00 | 0.796310 | 0.887342 | -0.039682 | 0.702206 | +0.416811 | 0.799381 | 0.710757 | 0.469308 |
| IVPS m=0.01 | 0.793378 | 0.890879 | -0.036145 | 0.694548 | +0.409152 | 0.794708 | 0.668744 | 0.488436 |
| IVPS m=0.03 | 0.789056 | 0.893950 | -0.033074 | 0.678510 | +0.393114 | 0.794708 | 0.574561 | 0.517477 |
| IVPS m=0.05 | 0.782234 | 0.896213 | -0.030810 | 0.657437 | +0.372042 | 0.793053 | 0.466066 | 0.492818 |
| proposal-32 control | **0.807756** | 0.891603 | -0.035421 | **0.727905** | +0.442509 | **0.803761** | — | — |

Relative to proposal-16, the 32-epoch compute control improves common by
0.013250, rare by 0.002436, synthetic by 0.002704, and their equal mean by
0.006130.  This is a compute effect, not evidence for IVPS.

## Frozen gate audit

Eligible verifier margins: **0/4**.

Every margin fails all of the following important requirements:

- equal-stratum PDM at least proposal-16 + 0.005;
- common PDM within 0.005 of the frozen reference;
- rare PDM within 0.020 of proposal-16;
- common degraded fraction at most 0.10.

Only m=0.03 passes the common override-precision threshold of 0.50.  Its common
degraded fraction is still 0.234534 and its equal score is 0.012570 below the
proposal baseline, so precision alone is not a sufficient promotion signal.

## Failure attribution

### The verifier traces a trade-off instead of resolving it

Increasing the reward margin monotonically reduces common coverage and raises
common selected PDM, but simultaneously rejects useful rare proposals.  From
m=0.00 to m=0.05, common PDM rises by 0.008871 while rare PDM falls by 0.044769.
The equal score therefore falls monotonically.

### Binary correctness is not decision value

On m=0.03, common override precision is 0.517477, yet 40.82% of accepted common
overrides are degrading and the final common PDM remains 0.033074 below the
incumbent.  On real-rare, override precision is much higher at 0.742567, but the
verifier misses 6.97% of beneficial proposals and falls back to an extremely
weak reference; rare PDM becomes 0.046959 worse than always using the proposal.

The all-candidate BCE asks whether each candidate beats the incumbent as a
classification problem.  Deployment instead faces a cost-sensitive decision on
the single realized proposal.  False acceptance and false rejection have
different scalar-PDM regret, and their cost changes with incumbent quality.  A
count-calibrated sign classifier therefore need not maximize selected PDM.

### The proposal–incumbent complementarity remains real

The forbidden-at-deployment oracle that selects the proposal exactly when its
observed PDM beats the incumbent obtains:

| Stratum | Proposal-16 | Oracle verified |
|---|---:|---:|
| common | 0.878352 | 0.945905 |
| real-rare | 0.725469 | 0.757798 |
| synthetic | 0.801057 | 0.810515 |
| equal mean | 0.801626 | 0.838073 |

The available arbitration headroom is therefore 0.036447 equal-stratum PDM.
IVPS fails because its verifier does not recover this headroom, not because the
incumbent and proposal lack complementary decisions.

## Consequence for V3

Stop IVPS V1 and retain the following supported findings:

1. scalar tie-aware preference remains the strongest learned causal factor;
2. additional scalar-proposal optimization helps modestly under matched data;
3. proposal verification has a large oracle ceiling;
4. the next arbitration formulation must be proposal-conditioned and
   scalar-regret/cost sensitive, rather than another threshold sweep or
   all-candidate sign-classification loss.

This report does not authorize a new experiment.  Any successor must receive a
new method definition, controls, fixed budget, and development gate before it is
run.
