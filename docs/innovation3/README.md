# Innovation 3: live selector updates

This branch starts from the clean standard V3 contribution and preserves the
upstream Git history. Memory rendering and deterministic actor RNG are opt-in; existing evaluation
keeps its file-based behavior and upstream RNG defaults. A map namedtuple also
exports its declared class name so trusted snapshots can be pickled across spawn.

## Implemented and bounded validation

- `innovation3.paths`: checks configured paths and every symlink hop, rejecting
  unavailable old project mounts. Hashes are streamed, not loaded into memory.
- `innovation3.protocol`: strict full20 step identities, canonical transition,
  branch parity receipt, update ordering, and stale-feedback rejection.
- `innovation3.learner`: frozen standard V3 plus a zero-output online residual,
  existing exact-group loss, one update per feedback, and optimizer/RNG state.
- `RenderManager.get_observations(persist=False, cache=False)`: in-memory images
  without calling DataManager or retaining all rendered frames. Defaults unchanged.
- `worldengine.online`: a persistent CPU dynamics adapter using real SimEngine
  agents, LQR/bicycle ego control, log-play traffic (NR), or IDM traffic (R).
  It captures controller/navigation/history/spawn/RNG state; the static scene/map
  bundle stays in memory and dynamic snapshots reference its objects by identity.
  Static arrays are read-only and full static-state audits reject branch mutation.
- `innovation3.snapshot_probe`: full20 H1 transition checks, reverse candidate
  ordering, and replay in a separate persistent **spawn** worker. Candidate inputs
  are explicit current-state-only diagnostic trajectories, not model predictions.
- `innovation3.preflight` / `innovation3.smoke`: explicit reports distinguishing
  path/CUDA checks and synthetic learner validation from real closed-loop results.

The active second-version protocol freezes the **trained V3** and adds a new
online correction: `current_logits = frozen_V3_logits + online_residual`.
The correction uses a separate V3-shaped module, copies V3's encoder initialization,
and zeros only its final output projection. The existing V3 weights are preserved;
only the new correction belongs to the optimizer. Initial actions exactly reproduce
V3. In PyTorch 2.0, action selection uses the no-grad eval path; a separate autograd
forward records numerical differences (atol 1e-5, rtol 1e-4 guard). No-signal groups
skip AdamW entirely and record an unchanged policy version.

Learner checkpoint schema 2 explicitly records `frozen_v3_plus_zero_residual`.
Schema-1 checkpoints from the earlier V3-finetuning prototype are rejected. Historical
smoke reports for that prototype remain historical evidence, not validation of the
second-version parameterization.

## Commands

From this repository's root, set `SETTINGS` to a private absolute runtime JSON and
`REPORT` to a new file outside the code repository. The JSON extends the selector
runtime with `scenario_root`, `asset_root`, `map_root`, `selector_state`, and
`expected_sha256` for baseline, selector_state and anchors.

```bash
python3 projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py \
  --settings "$SETTINGS" --output "$REPORT" --devices 0 preflight
```

For the learner-only CUDA smoke, use another fresh `REPORT` and replace `preflight`
with `learner-smoke`. It loads registered V3 weights, performs eight synthetic
updates, then validates a serialized optimizer/RNG restoration with a ninth
update. It does not save a deployable trained model. Use `--devices 0,1,2,3,4,5,6,7`
for an eight-device **preflight only**; this does not claim DDP was tested.

For the CPU-only dynamics gate (GPUs can remain occupied):

```bash
python3 projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py \
  --settings "$SETTINGS" --output "$REPORT" --scene-count 2 --steps 8 snapshot-probe
```

This deterministically selects the first two sorted scenes from the registered
`scenario_root/original/navtrain_failures_per1/all_scenarios.pkl`, without filtering
by rollout success or reward. Both NR/R are tested. Reports and incremental event
logs are saved outside code; existing report names are refused. Source provenance
does not establish log-disjointness or image/asset coverage. These diagnostics
must not be used as online training samples or a performance evaluation.

Existing regression tests, in the configured AlgEngine environment with this
repository's SimEngine and owned dependencies on PYTHONPATH:

```bash
python -m pytest tests/selector_v2_v3 tests/innovation3 \
  --ignore=tests/innovation3/test_headless_snapshot.py
```

Actual dynamics tests run in the configured **SimEngine** environment, with the
same explicit source paths used by the runtime launcher:

```bash
python -m unittest discover -s tests/innovation3 -p test_headless_snapshot.py -v
```

The dynamics fixtures check actual IDM response to ego interventions, delayed
actor spawning, NR/R despawn semantics, hidden state and RNG restoration, candidate
immutability, static-map corruption, and canonical recovery after a branch fails.
Snapshot payloads are trusted local pickles, not an untrusted network format.

## Not yet implemented or certified

There is intentionally no production training command yet. Remaining work:

1. Verify log-disjoint scene/image/map/asset coverage; directory existence is not coverage.
2. Implement live observation preprocessing, temporal state and bounded shared-memory IPC.
3. Extend the verified headless dynamics boundary to the rendered main process
   and add explicit future-free reward histories/accumulators. Snapshot schema 1
   deliberately rejects engines with renderer/data/metric/reward managers: it
   does not certify complete rendered-main or reward parity. Do not remove that
   guard without implementing and testing the missing state adapters.
4. Define and test H1 reward windows, progress reference and component semantics.
5. Connect generator, protocol gate, actual simulator feedback and learner; validate
   one real episode and then DDP, synchronized scene-boundary recovery and throughput.
   Export and inference must preserve both frozen V3 and the new residual; the old
   single-V3 materializer cannot represent this parameterization unchanged.
6. Test longer branches and controls after strict H1 validation. A value/TD head is
   optional if long-term feedback is insufficient, and must be a separate condition.

Existing SimEngine metrics are not an off-the-shelf future-free H1 reward:
MetricManager caches expert future, while comfort uses temporal filtering and TTC
uses a horizon. Preserve the official full evaluator and build/audit the online
reward separately; never silently label a short-horizon approximation official PDMS.
The standard evaluation launcher uses `log_play_controller` (perfect tracking).
The physical-controller condition must use `two_stage_controller` with
`kinematic_bicycle` for both main and branch, paired with a matching V3 baseline.
Keep the existing 29-column evaluator unchanged and report the physical-controller
comparison as an additional controlled evaluation.

SimEngine has a process-global engine singleton. Multiple live engines cannot be
constructed in one process; branch workers require spawn-based process isolation.
Never fork a process after creating its CUDA rendering context.

## Dynamics gate boundaries and throughput

The canonical execution in `snapshot-probe` is headless. It executes the selected
action first, evaluates all 20 branches from the pre-action snapshot, verifies
selected-branch state equality, and restores the canonical post-action state.
There is no reward, generator forward, optimizer update, or DDP in this probe.
The branch loop is a serial correctness reference; the extra spawned process
tests serialization, not parallel throughput. Reported group timings include
strict dynamic hashes and full static-state audits. They must not be presented
as training throughput or multiplied by GPU count to estimate scaling.

The upstream bicycle model integrates position from current velocity before
updating speed and steering. Consequently H1 positions can coincide across
candidates even when next velocities/steering differ. Reward distinguishability
requires separate validation; passing snapshot parity does not prove H1 is a
sufficient learning signal. Preserve the official dynamics for paired controls.

The simulator retains upstream traffic route priors and actor validity schedules.
This is distinct from allowing an online reward or selector to read logged ego
future; that information boundary remains to be implemented and audited.


## Investigating a parity failure

A target-instance rerun can fail even after a local probe passed. A local PASS
must not override that failure or be interpreted as evidence that it is fixed.
Keep the failed report and rerun using a new report name on the failing instance.
The probe now records CPU, numeric library versions/thread pools, selected runtime
settings, Git HEAD, and SHA256 of the relevant source files.

On a state parity failure the strict gate still stops immediately. Alongside the
report, `<report-stem>.failure/` stores the static scene/map bundle, pre-action
snapshot, canonical and branch snapshots, and selected action. `diagnosis.json`
records component hashes, the first differing hash events, and artifact checksums.
These trusted local pickles are private debugging evidence, never training data.
They are written only on failure. A one-ULP fault-injection test verifies that
small numerical mismatches are still rejected, the canonical state is restored,
and saved pre-action/action evidence can reproduce the unmodified execution.

This adds diagnostics; it does not establish the cause of any previously observed
failure. Do not loosen tolerances, silently normalize the canonical execution, or
advance training until the failing target-instance behavior is explained and a
correction is verified there.

### Visual probe asset configuration

`asset_root` must directly contain `<asset_id>/background/<asset_id>.ckpt`.
For a WE_processed layout, use its `navtrain/assets` directory for this training-source
probe. `visual_scene_id` optionally pins a scene in the registered
`original/navtrain_failures_per1/all_scenarios.pkl`; absent that setting, the first
sorted scene is retained. Missing assets fail explicitly, without switching scenes
or borrowing another asset from the same log. Audit and record the exact scene/asset
pair before GPU execution. Asset availability alone is not a training-set coverage
or log-disjointness certification. Existing headless probes keep their original scene selection.
