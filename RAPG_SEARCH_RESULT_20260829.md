# RAPG V1 Fixed-Budget Search Result — 2026-08-29

## Decision

The RAPG V1 search is a valid negative result for the graph hypothesis and a
positive result for the scalar tie-aware preference objective.

No arm passes the predeclared offline development gate because every learned
selector reduces common-stratum top-1 PDM relative to the frozen V3 reference.
Therefore this search must not advance to closed-loop development or formal
three-seed evaluation. Thresholds must not be relaxed after observing these
results.

The full reference-anchored graph should not be presented as the main method:
`B11` does not beat the objective-only `B01` control, and graph-only `B10` is
worse than the exact-GRPO relational baseline `B00`.

## Immutable provenance

- Search root: `experiments/diffusiondrive/selector_rapg_v1/search/search_20260829T105023Z`
- Offline shortlist audit: `experiments/diffusiondrive/selector_rapg_v1/search/search_20260829T105023Z/development_shortlist.json`
- Data split: held-out development only; certification was not consumed.
- Budget: nine arms, 16 epochs per arm, one fixed training seed.
- All nine training reports and all nine offline evaluations completed with
  `status=PASS`.
- Primary scalar objective: official scalar PDM only. Component rewards were
  recorded for diagnosis and were not used to tune the primary objective.

## Held-out development results

The equal-weight score is the mean selected PDM over common, real-rare, and
synthetic strata. Stratum gains are measured against the frozen V3 reference,
not against `B00`.

| Rank | Arm | Method | Equal PDM | Delta vs B00 | Common gain | Rare gain | Synthetic gain |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | B01-l1.00 | relational + scalar preference | 0.804517 | +0.034059 | -0.054241 | +0.451707 | +0.711974 |
| 2 | B11-l1.00 | RAPG + scalar preference | 0.799363 | +0.028905 | -0.053943 | +0.439587 | +0.708334 |
| 3 | B11-l0.50 | RAPG + scalar preference | 0.791992 | +0.021535 | -0.057488 | +0.442079 | +0.687278 |
| 4 | B01-l0.50 | relational + scalar preference | 0.791390 | +0.020933 | -0.061129 | +0.450403 | +0.680787 |
| 5 | B01-l0.25 | relational + scalar preference | 0.791020 | +0.020563 | -0.070039 | +0.453588 | +0.685401 |
| 6 | B11-l0.25 | RAPG + scalar preference | 0.785922 | +0.015464 | -0.054543 | +0.428920 | +0.679281 |
| 7 | B00 | relational + exact scalar GRPO | 0.770458 | +0.000000 | -0.086972 | +0.436840 | +0.657395 |
| 8 | Unary control | capacity-matched unary + exact scalar GRPO | 0.767490 | -0.002968 | -0.085420 | +0.428809 | +0.654971 |
| 9 | B10 | RAPG + exact scalar GRPO | 0.763425 | -0.007033 | -0.110028 | +0.421547 | +0.674646 |

Frozen-reference selected PDM is 0.927024 on common, 0.285395 on real-rare,
and 0.091691 on synthetic. The search therefore exposes a distributional
trade-off that an equal-weight average hides: large gains on rare pools are
purchased by replacing many already-good common decisions.

## Gate audit

The predeclared shortlist requires all of the following:

1. Positive top-1 reward gain against the frozen reference in every stratum.
2. No stratum selected-PDM drop greater than 0.005 against `B00`.
3. Common degraded-fraction no more than 0.02 above `B00`.

Eligible arms: **0/8 candidates**. The first condition fails for every arm due
to negative common gain. This is sufficient to stop the pipeline before
closed-loop development, independently of candidate ranking.

## Causal attribution

### What worked

The scalar tie-aware preference objective is the only supported positive
factor. At weight 1.00, `B01` improves equal-weight selected PDM by 0.034059
over `B00`, while also improving all three strata relative to `B00`. It reduces
common degraded fraction from 0.633657 to 0.393583 and reduces common KL from
3.8691 to 2.6171.

### What did not work

The graph is not the source of the gain:

- `B10 - B00 = -0.007033` equal-weight PDM.
- `B11-l1.00 - B01-l1.00 = -0.005154` equal-weight PDM.
- The capacity-matched unary control is also close to `B00`, so extra parameter
  count does not explain the preference-objective improvement.

### Bottleneck

The dominant failure is overconfident policy replacement, not insufficient
training or a missing reward component. On common development data:

| Arm | Selection disagreement | KL to reference | Selection margin | Degraded fraction |
|---|---:|---:|---:|---:|
| B00 | 0.871653 | 3.8691 | 10.3748 | 0.633657 |
| B01-l1.00 | 0.829409 | 2.6171 | 2.3423 | 0.393583 |
| B11-l1.00 | 0.824331 | 2.6401 | 2.3275 | 0.365651 |

For comparison, the frozen reference margin is 1.8987. Preference training
substantially softens the learned selector, but it still changes more than 82%
of common selections and remains worse than the incumbent on common PDM.

## Consequence for V3

The next V3 hypothesis should be framed as **conservative scalar improvement**:
learn when there is reliable scalar-PDM evidence to override the frozen
selector, and otherwise preserve its decision. It should retain the successful
tie-aware scalar preference formulation, remove the unsupported graph claim,
and make abstention/reference preservation a first-class part of the method.

This is a new hypothesis and must receive a new frozen protocol before running
experiments. This result does not authorize post-hoc gates, component-specific
reward patches, certification-set tuning, or further RAPG graph variants.
