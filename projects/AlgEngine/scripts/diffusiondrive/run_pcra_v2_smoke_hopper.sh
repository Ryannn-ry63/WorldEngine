#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
# Smoke is intentionally single-GPU even on an eight-GPU Hopper allocation.
export CUDA_VISIBLE_DEVICES=0
rapg_preflight 1
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_pcra_v2/smoke/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
PROPOSAL32="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt"
mkdir -p "${ROOT}"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_proposal_conditioned_counterfactual_evaluator.py" \
    "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
    --data-manifest "${RAPG_DATA_ROOT}/manifest.json" --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
    --proposal-checkpoint "${PROPOSAL32}" --output-dir "${ROOT}" \
    --counterfactual-loss opportunity_risk --train-evaluator-encoder \
    --source-risk original_mixture --override-threshold 0.0 --data-split train \
    --seed 0 --epochs 1 --examples-per-cache-epoch 6 --checkpoint-epochs 1 \
    --batch-size 3 --smoke-limit-hard-pool 2 --device cuda \
    >"${ROOT}/train.log" 2>&1
"${ALGENGINE_PYTHON}" - "${ROOT}/report.json" <<'PY_REPORT'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS" and row["schema_version"]==6
assert row["frozen_parameter_max_abs_delta"]==0.0
assert row["evaluator_initialization_max_abs_delta"]==0.0
assert row["counterfactual_evaluation_diagnostics"]["active_groups"]>=0
print("PASS PCRA V2 smoke",sys.argv[1])
PY_REPORT
