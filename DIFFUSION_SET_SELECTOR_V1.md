# Diffusion-Set Selector Post-Training V1

## Core claim

The generator, perception stack, frozen base selector, and the original 20
candidates remain unchanged. For scene `x`, diffusion noise `epsilon_d`
produces candidate set `C_d`. The trainable V3 residual selector is evaluated
independently on every set and may choose a different candidate in each draw.

The proposed arm optimizes scalar top-1 gain over the frozen V3 selector:

```text
g_d = R(top1_theta(C_d)) - R(top1_V3(C_d))
J = 0.5 * mean(g) + 0.5 * softmin_tau=0.05(g) - 1e-3 * KL
```

The hard top-1 reward is used in the forward pass and the categorical expected
reward supplies the straight-through gradient. With three draws, the bounded
mean-risk mixture keeps every draw weight in `[1/6, 2/3]`. No reward component,
new label, evaluator network, or inference-time extra draw is used.

## Fixed eight-arm attribution test

| Arm | Question isolated |
| --- | --- |
| `repeat_grpo` | Does matched extra diffusion data alone explain the gain? |
| `mean_soft` | Is raw reward scale sufficient? |
| `mean_top1` | Is train/deployment top-1 mismatch sufficient? |
| `pure_softmin_top1` | Does unrestricted lower-tail risk help or collapse? |
| `bounded_relative_top1` | Full proposed method |
| `bounded_relative_soft` | Is straight-through top-1 necessary? |
| `bounded_relative_top1_shuffled` | Does true same-scene grouping matter? |
| `bounded_absolute_top1` | Does V3-relative rather than absolute risk matter? |

All arms use train noise seeds `0/1/2`, unseen development seeds `3/4/5`,
rare/common 50/50, 8 epochs, 6339 outer groups per epoch, batch 64, learning
rate `3e-5`, and one fixed epoch-8 checkpoint. Paired uncertainty is
bootstrapped over logs.

## Run

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_diffusiondrive_selector_diffusion_set_v1_8hopper.sh
```

The runner first audits scene-context invariance and per-token GRPO gradient
structure, then launches the eight matched arms in parallel. It writes:

```text
experiments/diffusiondrive/diffusion_set_selector_v1/search/<SEARCH_ID>/
  training_mechanism_audit.json
  trials/<ARM>/report.json
  trials/<ARM>/development_evaluation.json
  development_gate.json
```

Formal attribution requires the proposed arm to beat both `repeat_grpo` and
`mean_top1` by at least `0.001` equal-stratum PDM with a positive paired-log
bootstrap lower bound, and to beat the same-stratum shuffled grouping by the
same rule. Pure soft-min replaces the bounded objective only when it beats the
bounded arm by `0.001`, has a positive paired-log lower bound, and its training
draw weights do not collapse.

## Machine decisions

- `AUTHORIZE_BOUNDED_RELATIVE_TOP1_FORMAL_REPLICAS`: efficacy and all causal
  controls pass; train three formal selector seeds.
- `AUTHORIZE_BOUNDED_SOFT_MATCHED_CONTROLS`: soft estimator is preferable;
  run only its matched shuffled and absolute controls.
- `EFFICACY_WITHOUT_REFERENCE_ATTRIBUTION`: grouping works but the V3-relative
  claim does not; run only the matched absolute control.
- `AUTHORIZE_PURE_SOFTMIN_MATCHED_SHUFFLE_CONTROL`: pure risk wins without
  collapse; run only its shuffled control.
- `EFFICACY_WITHOUT_DIFFUSION_GROUPING_ATTRIBUTION`: score improves but the
  diffusion-specific claim fails; do not run formal evaluation.
- `NON_DIFFUSION_CONTROL_WINS`: retain the simpler control and reject the
  diffusion-set method claim.
- `STOP_DIFFUSION_SET_OBJECTIVE_AFTER_DEVELOPMENT`: stop objective-only current
  frame changes and move to the pre-registered temporal/history audit.

Certification and the formal open-loop/closed-loop suites are forbidden until
the development gate explicitly authorizes formal replicas.
