# DiffusionDrive Selector Root-Cause Protocol (R0/R1)

Date: 2026-09-02

## Question

The current question is narrower than “does GRPO work for diffusion?”:

> When a frozen DiffusionDrive generator exposes 20 trajectories, does the
> deployed V3 selector still encounter failure states in which a materially
> better candidate is present, and does its probability support suppress those
> candidates strongly enough to motivate a diffusion-set-specific post-training
> objective?

R0/R1 do not introduce a new loss. They establish whether the selector-side
problem exists under the deployed policy before R2/R3 are authorized.

## Attribution contract

- Generator, perception, epoch-100 selector, and candidate count are frozen.
- The scalar rare-tuned V3 is the primary clean-attribution policy.
- Gate-conditioned V3 is a performance baseline only. It cannot authorize a
  new method by itself because it contains structured gate supervision.
- The official pairwise PDM scalar is the only target. Reward components are
  descriptive diagnostics and never selector inputs.
- CL development is diagnostic only. No R0/R1 record may be used for training.
- Proposal history remains parked and is not run by this protocol.
- R0 is epoch-100, non-reactive counterfactual replay. R1 is trained-selector,
  Reactive on-policy rollout. Their difference is not an occupancy-shift
  estimate because both policy and simulator mode differ.

## Stages

### R0 — retrospective replay

Replay scalar V3 seed0 and gate-conditioned seed0 on all 49,376 immutable
epoch-100 NR candidate records. Report success/failure-stratified selected
reward, oracle20 reward, top5/top8 capture, oracle rank/probability, entropy,
margin, headroom, and component diagnostics. Bootstrap by scenario and also
report log-level intervals.

R0 is evidence about ranking on old visited candidate sets only.

### R1 — trained-selector on-policy diagnosis

For each policy, run a fixed first-eight-scenario smoke and then all 58 Reactive
CL-development scenarios. The simulator scores all 20 candidates for eight
planning frames per scenario. Scalar and gate policies share evaluation seed 0
and the same candidate-noise namespace.

Every record is fail-closed audited against:

- materialized checkpoint SHA256 and checkpoint-manifest SHA256;
- policy family and training seed;
- complete resolved rollout contract;
- fixed 20-candidate shapes and finite rewards;
- deployed action equals `argmax(current_logits)`;
- nonzero trained-selector residual exists in the collection;
- exact deployed-trajectory/candidate parity;
- 8 frames per scenario, successful runner reports, completion ledgers, worker
  coverage, source/config/code fingerprints, raw observations, and sidecars;
- Reactive merged metric coverage and `success := NC == 1 and DAC == 1`.

## Seed0 decision

The decision uses only failed Reactive scenarios from scalar V3 seed0 and
scenario-bootstrap 95% intervals:

- pass: lower95(mean oracle20-selected) > 0.01 and
  lower95(fraction of frames with headroom > 0.005) > 0.25;
- clear fail: both corresponding upper95 bounds fail their thresholds;
- otherwise: borderline.

Sensitivity at headroom > 0.02 is always reported.

Decisions:

- `AUTHORIZE_R1_SCALAR_SEED1_SEED2`: collect scalar seeds1/2, then run the
  probability-support/temperature diagnosis before designing R2;
- `AUTHORIZE_R1_SCALAR_SEED1_BORDERLINE`: collect scalar seed1 only and rerun
  the gate;
- `STOP_SELECTOR_ONLY_NO_ON_POLICY_HEADROOM`: stop selector-only optimization
  and revisit candidate generation;
- `STOP_R1_NO_FAILED_REACTIVE_SCENARIOS`: report the closed-loop ceiling but do
  not infer a selector mechanism.

R2/R3 and a new loss are intentionally absent from the implementation. Passing
R1 authorizes their design; it does not pre-commit to full-feedback training.

## Commands

From the research worktree:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1
```

Check an 8-Hopper instance:

```bash
./run_diffusiondrive_selector_root_cause_r1_8hopper.sh preflight
```

Run the complete seed0 protocol, resume-safe under the same run id:

```bash
./run_diffusiondrive_selector_root_cause_r1_8hopper.sh seed0 root_cause_seed0_v1
```

Individual-stage forms are printed by running the script without arguments.
Development collection requires its matching smoke audit. A scalar seed1/2
smoke or development run additionally requires the prior `development_gate.json`; the
runner rejects unauthorized follow-up seeds and all gate-conditioned seed1/2
follow-ups.

Outputs live under:

```text
experiments/diffusiondrive/selector_root_cause_r1/
├── r0/<run_id>/r0_replay.json
├── runs/<run_id>/{smoke8,cl_dev58}/<policy>/seed0/collection_audit.json
└── analysis/<run_id>/development_gate.json
```

