# Innovation 3: live selector updates

This branch starts from the clean standard V3 contribution and preserves the
upstream Git history. All changes are additive except an opt-in memory rendering
API; existing evaluation keeps its file-based defaults.

## Implemented and bounded validation

- `innovation3.paths`: checks configured paths and every symlink hop, rejecting
  unavailable old project mounts. Hashes are streamed, not loaded into memory.
- `innovation3.protocol`: strict full20 step identities, canonical transition,
  branch parity receipt, update ordering, and stale-feedback rejection.
- `innovation3.learner`: single-rank standard V3 initialization, frozen reference,
  existing exact-group loss, one update per feedback, and optimizer/RNG state.
- `RenderManager.get_observations(persist=False, cache=False)`: in-memory images
  without calling DataManager or retaining all rendered frames. Defaults unchanged.
- `innovation3.preflight` / `innovation3.smoke`: explicit reports distinguishing
  path/CUDA checks and synthetic learner validation from real closed-loop results.

The learner copies the **trained V3** into the current selector and frozen KL
reference. It does not reset V3's trained residual head. In PyTorch 2.0, action
selection uses the no-grad eval path for exact initial inference parity; a separate
autograd forward records numerical differences (atol 1e-5, rtol 1e-4 guard).
No-signal groups skip AdamW entirely and record an unchanged policy version.

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

Tests: `python -m pytest tests/selector_v2_v3 tests/innovation3` in the configured
AlgEngine environment with this repository's SimEngine on PYTHONPATH.

## Not yet implemented or certified

There is intentionally no production training command yet. Remaining work:

1. Verify log-disjoint scene/image/map/asset coverage; directory existence is not coverage.
2. Implement live observation preprocessing, temporal state and bounded shared-memory IPC.
3. Snapshot all dynamic state, controllers, navigation, random generators, traffic spawn
   state and metric histories; validate selected-branch parity on NR and R.
4. Define and test H1 reward windows, progress reference and component semantics.
5. Connect generator, protocol gate, actual simulator feedback and learner; validate
   one real episode and then DDP, synchronized scene-boundary recovery and throughput.
6. Add value/TD, longer branches and experimental controls after strict H1 validation.

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
