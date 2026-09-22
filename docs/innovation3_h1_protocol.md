# Strict H1 protocol probe and live candidate boundaries

`innovation3_runtime.py ... h1-protocol-probe` starts an AlgEngine learner process
and one persistent SimEngine interpreter using an inherited Unix socket. JSON
messages have a bounded length and schema version, with EOF/error/timeout checks.
There is no frame-file polling or dataset/dataloader construction per step. This
stage sends control/state messages only; camera shared memory is not implemented.

The probe uses the registered trained V3, frozen, plus a separately initialized
zero-output residual. All diagnostic input tensors are float32. It checks the
same-state full20 physical transitions, independent canonical and selected-branch
feedback, causal message ordering, learning-induced logit changes, unchanged V3,
and serialized learner optimizer/action-RNG restoration. Episode resets preserve
the current policy version. A tied group does not run AdamW or invent a version.
The engine rejects observations/resets before an update acknowledgement.

The candidate bank is analytic and tokens/base logits are diagnostic stubs.
The feedback is explicitly `diagnostic_negative_velocity_change_squared_v1`;
it is NOT a driving objective and exposes none of the PDM metric names. It
must never supply formal training data. Saved weights are wrapped with
`DISPOSABLE_PROTOCOL_PROBE_NOT_TRAINING`. No test split, PDM cache, or logged
future is read by the diagnostic feedback function. The simulator itself still
uses its normal scene traffic spawning/navigation, as in the snapshot probe.

Success is `PASS_H1_PROTOCOL_PROBE_ONLY`. The report continues to say
`real_generator=false`, `online_reward_verified=false`,
`real_closed_loop_verified=false`, and `formal_ready=false`. This is not a
formal training run, full simulator/renderer resume, throughput study, or DDP test.

## Interfaces for the next visual stage

- `innovation3.candidate_inputs.from_export` accepts the existing frozen
  DiffusionDrive context export, checks current sample token/shapes and frozen
  V3 logits, and isolates only the six selector inputs and original generator
  base logits. It never adds the already-trained V3 term twice. Its caller
  must bind that export to a fresh live observation and StepIdentity.
- `worldengine.online.actions.to_world_actions` expands eight local future rear
  poses to 40, uses the same rear-to-center conversion as NAVFormerClient, then
  downsamples to nine requested tracking poses including the current pose.
  Coordinate conversion is not a substitute for physical branch simulation.
- Tests compare conversion against the real file client for all 20 curved/wrapped
  trajectories and execute the original tensor-only `_expand_to_40` method as a
  separate oracle. These are fixture checks, not GPU/file-vs-memory forward parity.

## Explicit next acceptance gates

1. One H100: render main observation in memory, preserve the four-frame history,
   BEV/tracking reset semantics, bind generator noise to step/scene, and compare
   original file and memory inputs/candidates/logits/actions on identical frames.
   Keep one dataset/pipeline/model resident; do not turn the old polling runner
   into an online-training claim.
2. Build the H1 adapter around the existing PDM scoring implementation with fixed
   weights. NC needs true boxes and fault rules; DAC needs actual drivable polygons;
   comfort needs real history; EP needs an explicit same-state reference; TTC needs
   a declared prediction horizon. Any H1 temporal adaptation must be separately
   identified and audited, not called the unchanged full-horizon official metric.
   The existing dense reward simulates PDM proposals: it cannot replace the real
   20-way reactive SimEngine transitions.
3. Add the reward/history state to the snapshot protocol and verify independent
   canonical/selected reward parity. Missing map/history must fail, not score 1.
4. Single-rank rendered H1 full chain, then eight-H100 synchronized optimization,
   exact rank checksums, interruption/resume, and throughput. Branch pooling is
   still serial within the SimEngine process in this diagnostic stage.

The current bicycle integrator first updates position from the old velocity.
Thus H1 branches can share the same next position while their next velocities
and reactive responses differ. Audit component variance and genuine optimizer
steps; do not invent smooth surrogate PDM weights to manufacture H1 signal.
Original V3/evaluator and the authoritative 0922plan.md stay unchanged.
