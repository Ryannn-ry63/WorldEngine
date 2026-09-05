# V3-RAPG: Reference-Anchored Preference Graph Selector

## 1. Frozen research question

The paper studies **strong generation but weak selection** under one immutable
action set.  The epoch-100 DiffusionDrive perception stack, denoiser, anchors,
20 generated trajectories and original selector are frozen.  Training changes
only a fresh scene-conditioned residual selector; inference is still plain
argmax over the same 20 trajectories.  PDM rewards, PDM components and future
labels are never selector inputs.

The main method uses only the official scalar PDM reward.  The existing
gate-conditioned/EP result is renamed **Conditional Quality Reweighting (CQR)**
and retained as a diagnostic Pareto baseline.  It is not the proposed loss and
must always be reported with its rare/success trade-off.

## 2. Method

RAPG reuses the audited `relational_only` trajectory-set encoder.  For candidate
tokens `z_i`, directed trajectory relation `g_ij` and a shared edge network `f`,

```text
a_ij = 0.5 [f(z_i, z_j, g_ij) - f(z_j, z_i, g_ji)]
b    = argmax_i reference_logits_i
G_i  = a_i,b + mean_{j != i} a_i,j
l_i  = reference_logits_i + unary_delta_i + G_i
```

`a` is exactly antisymmetric, its diagonal is zero, unavailable-candidate edges
are masked, and the complete operation is permutation equivariant when the
candidate set and reference logits are permuted together.  The unary and edge
last layers are zero initialized, so step zero exactly reproduces the frozen
reference policy.

The scalar tie-aware objective uses normalized official rewards `A_i`:

```text
q_i    = softmax(A)_i
y_ij   = sigmoid(A_i - A_j)
w_ij   = q_i + q_j
L      = L_exact_complete_action_GRPO
       + lambda_pref * weighted_BCE(l_i - l_j, y_ij; w_ij)
       + 1e-3 * KL(pi || pi_ref)
```

The KL term is implemented inside the exact GRPO term and is not counted twice.
Reward ties yield target 0.5.  Invalid and diagonal pairs have zero weight.  The
preference function has no reward-component argument by construction.

## 3. Causal experiment matrix

All search arms use LR `1e-4`, batch size `64`, temperature `1`, 16 epochs and
the same train-split mixture.  Preference weights are swept over
`{0.25, 0.5, 1.0}`.

| Arm | Structure | Objective | Purpose |
| --- | --- | --- | --- |
| original V3 | V3 | exact scalar GRPO | original selector baseline |
| CQR | V3 | component-conditioned | diagnostic/Pareto only |
| B00 | relational encoder | exact scalar GRPO | clean structure baseline |
| B01 | relational encoder | exact + scalar preference | objective effect |
| B10 | RAPG | exact scalar GRPO | graph-structure effect |
| B11 | RAPG | exact + scalar preference | full method |
| unary control | capacity-matched unary | exact scalar GRPO | parameter-count control |

The control adds 72,533 parameters versus RAPG's 72,403 at full dimension, a
0.18% difference.

## 4. Leakage-safe stages

1. Search trains only hard-pool rows marked `train` (13,649 hard rows, 743 logs).
2. Offline development evaluates common, real-rare and synthetic strata with
   equal stratum weight.  A candidate is eligible only if all strata improve
   over the frozen reference, no selected-reward stratum falls more than 0.005
   below B00, and common degraded fraction is at most B00 + 0.02.
3. The top three variants enter Reactive CL-dev built from the 588 real-rare
   development tokens in 80 isolated logs.  Selection maximizes three-seed
   CL-R subject to success at least CQR minus 0.02.
4. After architecture and lambda are locked, certification is consumed once on
   414 real-rare tokens in 52 different logs.
5. The locked method is retrained on `all` rows for the compute-matched formal
   B00/B01/B10/B11 comparison.  No hyperparameter changes are allowed after
   formal evaluation.

Generate immutable CL filters with:

```bash
python projects/AlgEngine/scripts/diffusiondrive/prepare_rapg_closed_loop_splits.py \
  --data-manifest experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data/manifest.json \
  --hard-pool experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data/hard_pool.jsonl \
  --output-dir experiments/diffusiondrive/selector_rapg_v1/closed_loop_splits
```

## 5. Execution

Run a real-cache smoke on one Hopper GPU:

```bash
./run_diffusiondrive_selector_rapg_v1_smoke.sh
```

Run the nine-arm train-only search on eight Hopper GPUs:

```bash
./run_diffusiondrive_selector_rapg_v1_search_8hopper.sh
```

After locking `lambda_pref`, run three seeds of B00/B01/B10/B11:

```bash
RAPG_LOCKED_PREFERENCE_WEIGHT=0.5 \
  ./run_diffusiondrive_selector_rapg_v1_formal_8hopper.sh
```

Materialize each final selector state with `materialize_rapg_selector.py`, then
run its four official blocks with:

```bash
./run_diffusiondrive_selector_rapg_v1_formal_eval_8hopper.sh \
  CHECKPOINT CHECKPOINT_SHA MODEL_NAME EVAL_SEED NOTE \
  reference_anchored_preference_graph
```

Collect the three independent summaries using `collect_rapg_formal_metrics.py`.
The collector maps `openloop_failures.score` to the paper's rare PDM field and
produces the exact input schema consumed by `audit_rapg_formal_promotion.py`.

`evaluate_rapg_offline.py` evaluates either `development` or `certification`.
`select_rapg_offline_development.py` refuses certification payloads.  Use
`select_rapg_closed_loop_development.py` for the CL-dev decision and
`audit_rapg_formal_promotion.py` for the final gate.

## 6. Frozen formal gate

The full method is promoted only when all conditions hold across paired seeds
0/1/2:

- mean `(CL-NR + CL-R) / 2 >= 0.800`;
- mean CL gain over CQR at least `+0.005`;
- at least two seeds are non-negative versus CQR;
- no seed loses more than `0.020` CL versus CQR;
- mean success at least `0.862`;
- mean navtest PDM at least `0.846`;
- mean rare PDM at least `0.607`.

Report selected reward, oracle regret, top-1 gain, epsilon-optimal hit rates,
selection-change/degraded fractions, preference margins, noise-view agreement,
all official open-/closed-loop blocks and the full CQR trade-off.

If B11 does not beat both B01 and B10, do not package their combination as the
main contribution.  Stop adding component-specific reward patches; report the
winning causal factor or the negative result.
