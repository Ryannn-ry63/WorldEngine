# DiffusionDrive trajectory-set reasoner GRPO v1

## Scope and provenance

- Branch: `diffusiondrive-selector-trajectory-set-reasoner-grpo-v1`.
- Clean implementation base: `8dc8bbf8c68f9699453fe554c09b9221aa346f34`.
- Frozen epoch-100 generator/reference SHA256:
  `1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514`.
- ReCogDrive concept reference:
  `xiaomi-research/recogdrive@6b8d8f5e01346c71094651c81dcaf66405dbc04e`.
  Only same-state grouped samples, group-relative credit, and a frozen
  reference-policy constraint are conceptual references.
- Official DiffusionDriveV2 concept reference:
  `hustvl/DiffusionDriveV2@1cd12a1e155c34dcc471261835444c8d5587580b`.
  Only within-anchor comparison and candidate-generation/selection separation
  are conceptual references.
- No source was copied from either upstream. The historical Stage41--61 chain,
  local-frontier/router patches, gates, generator GRPO code, and synthetic
  rollouts are not implementation sources.

This branch returns to the frozen ungated V3 selector line. The paused setwise
study remains isolated at commit `ed56c97716a6db056d26cb88190c4cb74ee43b1a`;
the hierarchical generator jobs and their artifacts remain read-only.

## Method

The generator, perception, BEV/query encoders, plan anchors, original
classifier, and frozen epoch-100 reference logits do not change. The trainable
selector receives only deployment-available inputs:

- final candidate feature for each of 20 trajectories;
- all eight `(x, y, yaw)` waypoints and derived kinematics;
- frozen BEV features sampled along those eight waypoints;
- frozen status, ego, and agent context.

Each candidate is first reasoned over in time. A shared scene cross-attention
then conditions each trajectory, and relation-aware attention compares all 20
candidates using directed relative position, yaw, distance, and path-length
features. The final scalar head is zero initialized and adds a residual to the
frozen reference logits. Rewards, PDM components, future labels, and gates are
not accepted by the inference interface.

Training keeps the audited exact full-action objective:

```text
A_i = zscore(r_i among valid candidates in the same 20-action set)
L_policy = -mean_groups sum_i pi(i) A_i
L = L_policy + 1e-3 KL(pi || pi_epoch100)
```

## Architecture-first experiment

All arms use the same rare-original/same-log-common 50/50 sampler, three fixed
noise caches, learning rate `1e-4`, KL `1e-3`, batch size 64, and exact reward.
Checkpoints are fixed at epochs 16/32/64 (4,800/9,600/19,200 optimizer steps).
There is no metric-specific reward patching or formal-result retuning.

The five development arms are:

1. capacity-matched legacy V3 with four ordinary set-attention layers;
2. temporal reasoning only;
3. pairwise relational reasoning only;
4. full reasoner without scene context;
5. full temporal + scene + pairwise reasoner.

The log-disjoint rare development cache only creates a three-candidate
shortlist. It does not promote the main method. The fixed-candidate ceiling
report records reference reward, selected reward, oracle reward, oracle rank,
regret, and realized headroom for attribution.

## Closed-loop promotion contract

Final comparison is paired across seeds 0/1/2. The primary metric is the mean
of CL-NonReactive PDM and CL-Reactive PDM. A candidate is mainline-eligible only
when all of these frozen rules pass:

- mean closed-loop PDM gain is at least `+0.01`;
- at least two of three paired seeds improve;
- mean Success Rate drop is no worse than `-0.035`;
- mean NC and DAC drops are each no worse than `-0.02`;
- mean navtest OpenLoop PDM drop is no worse than `-0.01`;
- no seed loses more than `-0.03` mean closed-loop PDM.

The highest closed-loop result is always reported as a Pareto result even if a
guardrail fails. Gate-conditioned V3 remains an external Pareto/high-score
ablation and is not silently merged into the ungated main method.

## Commands

Run from this worktree. Both H100 and H200 are accepted, but one allocation
must be homogeneous sm90 and must expose exactly eight GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_diffusiondrive_trajectory_set_reasoner_v1_smoke_8hopper.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_diffusiondrive_trajectory_set_reasoner_v1_architecture_8hopper.sh
```

After reading `shortlist.json` and locking a label/epoch, independently train
seeds 0/1/2 from epoch-100 on the full rare-original training split:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_diffusiondrive_trajectory_set_reasoner_v1_formal_train_8hopper.sh \
  /absolute/path/to/architecture_RUN LABEL LOCKED_EPOCH
```

This produces three independently trained and audited checkpoints. Evaluate
each checkpoint on the four formal blocks with its matching seed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./run_diffusiondrive_trajectory_set_reasoner_v1_eval_8hopper.sh \
  CHECKPOINT EXPECTED_SHA256 MODEL_NAME EVAL_SEED NOTE
```

The standalone materializer is only for inspecting one development checkpoint;
it does not replace independent formal training:

```bash
./run_diffusiondrive_trajectory_set_reasoner_v1_materialize.sh \
  /absolute/path/to/architecture_RUN LABEL LOCKED_EPOCH
```

Run the formal evaluator for seeds 0/1/2, then collect without manual metric
transcription and apply the frozen promotion rules:

```bash
python projects/AlgEngine/scripts/diffusiondrive/collect_trajectory_set_reasoner_formal_metrics.py \
  --summary SEED0_SUMMARY --summary SEED1_SUMMARY --summary SEED2_SUMMARY \
  --output candidate_metrics.json

python projects/AlgEngine/scripts/diffusiondrive/audit_trajectory_set_reasoner_promotion.py \
  --baseline frozen_v3_metrics.json \
  --candidate reasoner=candidate_metrics.json \
  --output promotion.json
```

Smoke success authorizes the architecture study, not formal promotion.
