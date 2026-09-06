# Overnight screen → conditional confirmation

This is an external supervisor, not a change to the frozen rare experiment.
No original model, source, training setting, effect threshold, code inventory or
budget contract is edited. Automation events and its source hash are stored in
the existing run's `overnight_events.jsonl`; the original ledger remains owned
by the original stage runner.

Wait for **bridge COMPLETE** first. In the rare worktree on the same8-H100 instance:

```bash
nohup bash run_diffusiondrive_selector_rare_overnight.sh rare_retention_20260906_v1 >> rare_retention_20260906_v1_overnight.log 2>&1 &
```

The same command can be used for a retry; `>>` retains earlier terminal output.

Observe with:

```bash
tail -n 50 rare_retention_20260906_v1_overnight.log
```

Screen is launched if its final report does not already exist. Any failed child
stops the sequence. A complete screen with no qualifying winner stops normally.
Only the original fixed winner (all3 seeds, unchanged checkpoint hashes) can
advance. The original confirm stage independently repeats its full screen audit,
model checks, hardware preflight and per-condition budget forecasts.

The same cumulative64-GPUh cap and phase limits apply. Running unattended does
**not** authorize expansion, truncated evaluation, refitting or winner replacement.
Budget or engineering failure can still prevent finishing overnight.
When resuming, complete screen reports are reused subject to the original confirm
audit; an already-complete confirmation is revalidated CPU-only.

The wrapper never starts bridge, stops unrelated jobs or waits for a running bridge.
It rejects a second simultaneous overnight supervisor for the same run.
SIGINT/SIGTERM are forwarded only to its current original runner, which performs
its existing owned-process cleanup. Do not launch a second manual screen/confirm
for the same run while this supervisor is active.
`nohup` protects against terminal disconnect, not instance shutdown or scheduler eviction.

Validation:17 synthetic orchestration tests pass. The active
`rare_retention_20260906_v1` implementation inventory (475 files) was independently
recomputed and exactly matches its original frozen SHA:
`c59e3ecd4a870239b32e2b1f0cae33d9bfda18d4a1036c84e13e8925ee097403`.
No re-audit, new experiment ID, or change to the running bridge is required.
