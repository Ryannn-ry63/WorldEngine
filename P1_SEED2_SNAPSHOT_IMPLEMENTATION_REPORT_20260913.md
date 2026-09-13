# P1 seed2 native full-evaluation snapshot — implementation handoff

Date: 2026-09-13. Branch: `research/selector-p1-seed2-snapshot-v1`.
Parent: `6ff2abb` (`research/selector-decision-feedback-v1`).

## Delivered

Implemented the approved observed-best snapshot, without training or launching formal GPU experiments.
The selected P1 seed2 step500 checkpoint is frozen and exported into a native full-model checkpoint.
Only 54 scene-selector tensors change; 963 generator/perception/reference tensors remain bitwise equal
to the paired historical V3 rare_tuned seed0 checkpoint. Existing historical files and P3
`STOP_EFFECT_FAIL` remain unchanged. The old decision-feedback worktree is clean.

Four-H100 runner: offline audit → CUDA preflight → deterministic bridge8 → 12 formal conditions →
complete report/archive. Default independent cumulative cap: 192 GPU-hours; not an ETA.
No training, Q/reward inference, CFPI routing, or performance-conditioned suppression of results.
An observation-only manager checks selected trajectory against actual controller actions.
Bridge failures stop for engineering review; this is not permission to search noise seeds.

The complete command sequence and restart rules are in [research/p1_snapshot/README.md](research/p1_snapshot/README.md).
Use formal run ID `p1_seed2_full_20260913_v1`, not a software audit ID.

## Verification actually completed

- Snapshot unit/integration tests: **19 PASS**.
- Decision-feedback regression: **19 PASS**.
- Feedback regression: **40 PASS** (existing test emits an unclosed-lock ResourceWarning, no failure).
- Continuous-deployment regression: **27 PASS**.
- Rare-protocol regression: **25 PASS**.
- Total: **130 tests PASS**; includes synthetic action/coverage audits and negative-result reporting,
  not GPU simulation acceptance tests.
- CLOSED and OPEN native configurations parsed and pretty-printed with MMCV on CPU;
  custom GPU/plugin imports were deliberately deferred to instance preflight.
- Python AST and launcher shell syntax checked.
- Final real-cache CPU audit passed twice under the same immutable run ID:
  `experiments/diffusiondrive/selector_p1_snapshot_v1/runs/software_audit_20260913_v3`.
- Real-cache comparison uses the first 16 rows from the archived 256-row cache:
  max score difference **0**, argmax mismatches **0**, optimizer steps **0**.
- Exact counts: open-navtest **12146**, open-rare **288**, full **288**,
  development **58**, common **64**, bridge **8**.
- Archived **726** implementation/config/document files with hashes and reproduction manifest.
- Final code inventory digest:
  `47923e3578fce3aa6e12c6e76f8242234e1748e6feae653aff88737017f17618`.
- Selected selector digest:
  `331ac4459cef19febff865080e3015f955cae0d2ebbd04bed58e207b2b8ebc6b`.
- Software-audit native export digest:
  `7895612dc85c35ee2856f1cd5238ed2947424a5b952b83fd0977e6ed0e912f60`.
- Audit ledger: **0 GPU-hours**, no active child. Earlier software audit v1/v2 are retained development
  artifacts with superseded code hashes; do not use them to launch formal experiments.

## Still to run on the user's four-card instance

CUDA/extension preflight, CUDA parity, actual bridge rollouts and every formal evaluation condition.
CPU parity is not evidence that closed-loop native export matches routed deployment; bridge8 is
specifically required before full evaluation. No new performance claim is made by this implementation.

The final report excludes aggregate CSV rows, separates full288/development58/remainder230/common64,
keeps all safety components, rescued/broken scenes and conditional log-cluster intervals, and links
the prior three-seed report. The historical aggregate-row success denominator error is explained
without modifying old results. A development-selected single seed is not a three-seed paper result;
historically exposed remainder230 is not a blind test.

The next milestone is an auditable complete snapshot, positive or negative. Further optimization and
paper-level multi-seed evidence require a subsequent research decision after reviewing that snapshot.
