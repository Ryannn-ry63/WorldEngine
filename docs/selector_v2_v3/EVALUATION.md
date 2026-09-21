# Full evaluation and oracle protocol

## Formal model/ablation table

The current frozen benchmark contains 12,146 navtest scene tokens and 288 rare scene tokens.
A formal row has 29 columns: Model Name, Note, ADE4s, FDE4s, then six metrics
(NC, DAC, EP, TTC, Comfort, PDM) for each of navtest open-loop, rare open-loop,
rare closed-loop NR, rare closed-loop R, and finally Reactive Success Rate.
EP/PDM in the closed-loop headings can be labelled EP*/PDM* to match the paper table.
ADE/FDE are metres; scores are fractions. Multiply score columns by 100 only for a labelled percentage table.
Reactive Success Rate is the fraction of scenes satisfying **NC == 1 AND DAC == 1**.
It is not the product of the two averages.

Freeze a JSON protocol **from the benchmark's input manifest before evaluation**, with
`schema_version: 1`, `navtest_tokens: [the 12146 IDs]`, `rare_tokens: [the 288 IDs]`.
Never infer expected coverage from an incomplete output CSV. Scene identifiers must be token IDs;
any scenario-name mapping must be explicitly audited before table assembly.

Run both open-loop splits with official NAVSIM rescoring and both closed-loop reactions.
The launcher requires a fresh output directory per attempt; it deliberately does not guess resume safety.
For existing long experiments, use their original identity-aware resume launchers.

```bash
bash run_selector.sh --settings /absolute/selector.local.json --devices 0,1,2,3 openloop \
  --config projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py \
  --checkpoint /absolute/checkpoint.pth --split navtest --eval-seed 0 --output /absolute/runs/model_navtest
```

Repeat with `--split rare` and a different output. For closed-loop:

```bash
bash run_selector.sh --settings /absolute/selector.local.json --devices 0,1,2,3 closedloop \
  --config projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py \
  --checkpoint /absolute/checkpoint.pth --reaction NR --eval-seed 0 \
  --scenarios /absolute/rare288/all_scenarios.pkl --assets /absolute/navtest_failures/assets \
  --output /absolute/runs/model_NR
```

Repeat with `--reaction R`. This native launcher is a new integration: a local path or unit test pass does not
certify H100 rendering. Perform a scoped end-to-end validation on the target environment before a long run.

Record evaluator artifacts with `record_selector_evaluation.py` (see `--help`), specifying the checkpoint,
protocol, evaluation seed and all five CSVs. This binds supplied attribution and hashes; the operator must ensure
the paths come from that model's actual commands. Then:

```bash
bash run_selector.sh --settings /absolute/selector.local.json --devices 0 tool build_selector_formal_table \
  --protocol /absolute/protocol.json --evaluation-manifest /absolute/evaluation.json \
  --model-name v3_rarelog_s0 --note 'rarelog; fixed budget; train seed0; eval seed0' \
  --output /absolute/reports/v3_rarelog_s0.json
```

The assembler requires exact token sets, valid finite results and matching hashes. It excludes named aggregate
rows (`average`, `overall_average`) and records their removal. Legacy means containing aggregate rows must not be
silently relabelled as newly verified results. Three seeds require three accepted rows, then a separate mean/sample-SD report.
A successful process, cache certification, smoke test, bridge8 or common64 run is not a formal table.

## A: full candidate oracle (planned experiment, not a reported result)

Use the same immutable epoch-100 generator, no additional training, the same 20 candidates per token and
candidate-noise seeds 0/1/2 for paired comparisons. Compare the original scorer's selection against the maximum
official PDM-reward candidate in that exact set. Record candidate tensors/hashes and deterministic tie handling.
Evaluate all navtest and rare tokens, report full open-loop components, ADE/FDE, oracle gap, scene win/tie coverage,
and paired uncertainty clustered by log. This measures available candidate quality versus selection; it does not
prove the global bottleneck is exclusively selection.

The oracle uses future ground truth and is not a deployable rule method. This experiment's scope is complete
open-loop analysis; do not invent NR/R scores for it or present training-cache best-of-20 as full test-set oracle.
If adding V2/V3 scorers later, verify generator/candidate identity before interpreting selection-only differences.
