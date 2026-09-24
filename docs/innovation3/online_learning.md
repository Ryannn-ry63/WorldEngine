# Innovation 3 online acceptance stages

## Engineering pilot after the selector-only H100 probe

The next acceptance stage is `ddp-h1-probe`. It is a bounded engineering
check, not formal training: every rank owns a distinct synthetic H1 context,
averages selector gradients, checks exact rank state parity, and restores a
scene-boundary learner checkpoint (including optimizer and RNG state). It
does not certify rendering, live reward, throughput, or paper metrics.

Use a fresh output basename on a target with at least two idle GPUs:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/code
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python \
  projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py \
  --settings ../registry/online_runtime.json \
  --output ../registry/ddp_h1_probe_h100_v1.json \
  --devices 0,1,2,3,4,5,6,7 --steps 4 --seed 0 ddp-h1-probe
```

The expected status is `PASS_DDP_H1_PILOT_ONLY`. A successful pilot still has
`formal_ready=false` and must not be used as a formal training checkpoint.

## Production-like throughput pilot

After DDP synthetic parity, measure the real single-GPU loop without the
correctness-only file oracle or duplicate forward. Reward/state checks,
full20 branches, online updates, frozen-module checks and acknowledgement
ordering remain enabled. Use a new output basename:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/code
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python \
  projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py \
  --settings ../registry/online_runtime.json \
  --output ../registry/online_throughput_h100_v1.json \
  --devices 0 --steps 8 --seed 0 online-throughput-probe
```

The expected status is `PASS_STRICT_ONLINE_THROUGHPUT_PROBE`. The report
contains per-decision stage timings and `decisions_per_second`; it remains a
bounded pilot with `formal_ready=false`, not a formal performance claim.

## Real-feedback multi-rank pilot

The next stage is `ddp-online-throughput-probe`. It launches one resident real
online loop per rank, binds each rank to one listed GPU, and all-reduces only
the selector residual gradients. Signal presence is synchronized every step:
if one rank has a valid group signal, all ranks perform the same zero-filled
gradient update; only a globally signal-free step is skipped. Start with two
GPUs and a fresh basename:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/code
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python \
  projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py \
  --settings ../registry/online_runtime.json \
  --output ../registry/ddp_online_throughput_h100_v4.json \
  --devices 0,1 --steps 8 --seed 0 ddp-online-throughput-probe
```

Expected status is `PASS_DDP_REAL_ONLINE_THROUGHPUT_PILOT`. This remains an
engineering pilot with `formal_ready=false` and `used_for_formal_training=false`.
The launcher keeps both listed devices visible to NCCL, binds each rank with
`LOCAL_RANK` before process-group/model construction, and pins only its separate
renderer worker to the rank's physical GPU. The final report records distinct
CUDA UUIDs when exposed by the installed PyTorch; otherwise it verifies the
explicit physical IDs and local-rank permutation. Do not jump to eight GPUs until
both rank reports, real reward, residual parity and clean worker exit pass on the
two-rank run.

The two-rank pilot has passed. The next bounded scale check uses eight GPUs and
is still an engineering pilot, not formal training:

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/code
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python \
  projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py \
  --settings ../registry/online_runtime.json \
  --output ../registry/ddp_online_throughput_h100_8rank_v1.json \
  --devices 0,1,2,3,4,5,6,7 --steps 8 --seed 0 ddp-online-throughput-probe
```

Expected status is `PASS_DDP_REAL_ONLINE_THROUGHPUT_PILOT`; retain
`formal_ready=false` and `used_for_formal_training=false`.

## Strict single-GPU online acceptance

The new `online-probe` connects live observations and generated candidate context
to `OnlineV3Learner`, with `LiveStepGate` in both interpreters. The frozen epoch100
generator and registered trained V3 remain unchanged; only a separate zero-output
V3-shaped residual belongs to AdamW. Local CPU tests are not H100 acceptance.

## Run

From the code root, use the registered runtime settings and a new private report:

```bash
python projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py \
  --settings /absolute/private/online_runtime.json \
  --output /absolute/private/fresh_online_report.json \
  --devices 0 --steps 8 --seed 0 online-probe
```

This is a bounded one-scene acceptance run, not a production training launcher.
The launcher reuses the configured AlgEngine/SimEngine interpreters, extension
bootstrap, Prescott numeric policy and single-thread BLAS. Local work uses CPU;
the live run needs one available H100 for generator and renderer.

The established visual and live-reward modes remain separate. Only this new mode
passes `--online-updates` to the worker. All modes retain the same physics, IDM,
map, source scene and `causal_h1_pdm_components_v1` reward contract.

## What must be evidenced

1. Thirteen diagnostic warmup transitions populate the actual reward history and
   bounded four-frame visual window. They do not update the learner.
2. At each of eight decisions, frozen inference exports the current candidate
   group and reward-free context. File/memory input and forward parity remain
   acceptance checks. The temporary file oracle and duplicate forward are not
   production transport or throughput measurements.
3. The online policy samples categorically at temperature 1 using a private
   learner RNG. At version 0, logits/probabilities equal frozen V3 exactly.
   This does **not** mean the sampled index equals the frozen evaluator's argmax.
   The earlier live-reward run used argmax, so its two signal groups need not
   recur at the same decisions in this run.
4. Both gates bind scene, step, policy version, state, candidate identity,
   selected index and probabilities. The independent main receipt precedes the
   full20 branch feedback. Original reward precision is retained for exact
   selected/main reward equality; only accepted rewards enter learner precision.
5. Feedback is checked before one update attempt. A no-signal group
   (learner-precision group std <= 1e-6) skips AdamW entirely and keeps the policy
   version. Parameters, optimizer state and same-context logits must be unchanged.
   Host global-norm clipping reuses the existing H100-compatible V3 helper.
6. Logs contain current/reference/post-update logits, probabilities, optimizer
   and residual hashes, update counters and monotonic event times. Same-context
   logit change means after minus **before** this update, not after minus V3.
   The next action must retain the previous residual hash and consume its version.
7. The worker validates the update acknowledgement and responds with that version
   before its next observation is accepted. Canonical state and reward history
   remain continuous. All generator parameters stay frozen with no gradients;
   the full model state and source hashes are checked at completion.

## Status and limits

`PASS_STRICT_ONLINE_SELECTOR_ONLY_PROBE` requires a real optimizer step, changed
same-context logits, and a later action whose probabilities reflect the residual.
`INCOMPLETE_ONLINE_LEARNING_EVIDENCE` (exit 2) preserves the run but does not claim
learning acceptance, for example when all groups tie or only the final step learns.
It must not be turned into PASS by changing reward weights or forcing an update.
Protocol, finite-value, source, state/history or frozen-weight failures stop the run
and preserve a FAIL report. Always use a fresh basename for retries.

The private `.online.pt` contains learner/optimizer/RNG state with a disposable
probe marker and source/reward provenance. It is not a formal training artifact
or complete simulator/renderer resume checkpoint. Formal data manifests, parallel
branch pools, eight-rank DDP, full recovery, paired frozen evaluation and performance
claims still require separate work. The scorer remains causal H1 components, not
official full-episode closed-loop PDMS.

CPU regression includes actual SimEngine and reward-component tests, plus explicit
test doubles for the full eight-step visual entrypoint. The doubles verify control
flow and reporting, and cannot certify rendering, actual generated candidates or
GPU updates. A private CPU integration probe may combine real dynamics/reward with
analytic diagnostic candidates and stub context; its evidence has the same limit.
