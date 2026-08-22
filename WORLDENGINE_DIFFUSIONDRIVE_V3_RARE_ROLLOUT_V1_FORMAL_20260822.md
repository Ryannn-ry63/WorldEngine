# DiffusionDrive V3 rare-rollout v1 formal result

## Frozen result

- Code branch: `diffusiondrive-selector-grpo-v3-rare-rollout-v1`
- Code commit before this documentation-only freeze: `79c1084`
- Baseline: immutable epoch-100 DiffusionDrive, SHA256 `1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514`
- Selector: fresh zero-initialized V3 scene selector; only its 54 tensors are trained
- Formal budget: 304,272 examples and 4,800 optimizer steps per seed
- Seeds: 0, 1, 2
- Artifact root: `experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1`
- Full checksums: `WORLDENGINE_DIFFUSIONDRIVE_V3_RARE_ROLLOUT_V1_ARTIFACT_MANIFEST_20260822.json`

All aggregate metrics below are paired-seed means on a 0--100 scale.

| Method | OP-PDMS navtest | OP-PDMS rare | CL valid rate | CL-PDMS |
|---|---:|---:|---:|---:|
| epoch100 | 85.71 | 58.80 | 76.82 | 63.13 |
| common V3 progress fix | 87.07 | 60.43 | 80.05 | 68.31 |
| rare-original frozen | 86.13 | 69.63 | 92.27 | 73.31 |
| rare-original tuned (4x compute) | 87.27 | 68.14 | 89.97 | 75.73 |
| rare-rollout v1 | 84.22 | 67.52 | 95.62 | 75.17 |

The primary causal comparison is `rare-rollout v1 - rare-original frozen`.
Both runs start from the same epoch-100 checkpoint, train a fresh V3 selector,
and use the same optimizer, hyperparameters, examples, steps and evaluation
seeds. `rare-original tuned` is a secondary 4x-compute reference, not the
compute-matched baseline.

## Primary causal delta

### Closed-loop Reactive

| Metric | rare-original frozen | rare-rollout v1 | Delta |
|---|---:|---:|---:|
| NC | 95.14 | 97.45 | +2.31 |
| DAC | 97.45 | 98.26 | +0.81 |
| EP | 49.67 | 49.21 | -0.46 |
| TTC | 91.78 | 94.44 | +2.66 |
| Comfort | 100.00 | 100.00 | 0.00 |
| PDM | 73.31 | 75.17 | +1.86 |

Reactive PDM deltas by training seed are +3.97, +0.02 and +1.59 points.
All three seeds are non-negative, although seed 1 is effectively flat.

### Closed-loop NonReactive

| Metric | rare-original frozen | rare-rollout v1 | Delta |
|---|---:|---:|---:|
| NC | 96.18 | 97.34 | +1.16 |
| DAC | 97.57 | 98.61 | +1.04 |
| EP | 50.32 | 48.77 | -1.55 |
| TTC | 92.25 | 93.87 | +1.62 |
| Comfort | 100.00 | 100.00 | 0.00 |
| PDM | 74.15 | 74.91 | +0.76 |

## Interpretation

Across 867 paired seed-by-scene Reactive comparisons, rare-rollout rescues
40 `invalid -> valid` cases and loses 11 `valid -> invalid` cases. Among the
789 cases valid under both methods, EP decreases by 2.56 points and PDM by
0.85 points on average. The aggregate gain therefore comes primarily from
recovering unsafe/invalid cases, while the learned selector is conservative
on already-valid cases.

This result shape is consistent with the senior reference: its
`syn rea -> syn reactive + generalization` step improves CL-PDMS from 68.29
to 70.12 (+1.83), essentially the same magnitude as this run's +1.86.
The approximately five-point EP drop seen in some informal comparisons comes
from comparing against a selected 4x-compute tuned seed; it is not the paired,
compute-matched causal delta.

The current result supports the claim that online rare rollout improves
closed-loop safety and validity. It does not support a universal open-loop or
per-scene quality improvement claim, and all open-loop regressions remain
reported rather than hidden.

## Frozen data contract

- Raw online records: 49,376
- Collectable rare origins: 6,172
- Explicitly excluded short origins: 99
- Filtered senior-v1 synthetic records: 9,869
- Real rare rows: 6,271
- Hard pool: 16,140 rows
- Training sampling: globally 50% paired common and 50% hard
- Source policy: immutable epoch-100 DiffusionDrive, not a trained V3 policy
- navtest overlap: zero by the inherited rare-original split audit

The ignored binary artifacts remain in the shared filesystem and are frozen
by path plus SHA256 in the companion manifest. They must not be overwritten;
new experiments use distinct roots and validate these hashes before reuse.
