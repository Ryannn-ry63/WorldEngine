#!/usr/bin/env bash
# Fixed PCRA V2 factorial search; only independent opportunity-risk is promotable.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8

PCRA_V2_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_pcra_v2"
RUN_ID="${PCRA_V2_SEARCH_ID:-search_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${PCRA_V2_ROOT}/search/${RUN_ID}"
PCRA_V1_SEARCH="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_pcra_v1/search/search_20260829T171222Z"
PROPOSAL32="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt"
PROPOSAL32_SHA256=562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6
PROPOSAL48="${PCRA_V1_SEARCH}/proposal_compute48/epoch_48_scene_selector.pt"
PCRA_V1="${PCRA_V1_SEARCH}/pair_regret/epoch_16_scene_selector.pt"
for path in "${PROPOSAL32}" "${PROPOSAL48}" "${PCRA_V1}"; do
    [[ -f "${path}" ]] || { echo "missing frozen PCRA input: ${path}" >&2; exit 1; }
done
[[ "$(sha256sum "${PROPOSAL32}" | awk '{print $1}')" == "${PROPOSAL32_SHA256}" ]] || {
    echo "frozen proposal32 SHA256 mismatch" >&2
    exit 1
}
[[ ! -e "${ROOT}" ]] || { echo "PCRA V2 search root exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
COMMON_ARGS=("${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" --data-manifest "${RAPG_DATA_ROOT}/manifest.json" --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl")

train_v2() {
    local gpu="$1" label="$2" loss="$3" train_encoder="$4" source_risk="$5"
    local output="${ROOT}/${label}"
    local encoder_arg=--train-evaluator-encoder
    [[ "${train_encoder}" == true ]] || encoder_arg=--no-train-evaluator-encoder
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_proposal_conditioned_counterfactual_evaluator.py" \
        "${COMMON_ARGS[@]}" --proposal-checkpoint "${PROPOSAL32}" \
        --output-dir "${output}" --counterfactual-loss "${loss}" \
        "${encoder_arg}" --source-risk "${source_risk}" --override-threshold 0.0 \
        --data-split train --seed 0 --epochs 16 --checkpoint-epochs 16 \
        --learning-rate 1e-4 --batch-size 64 --device cuda \
        >"${output}/train.log" 2>&1
}

train_v2 0 independent_regret_bce regret_bce true original_mixture &
train_v2 1 frozen_opportunity_risk opportunity_risk false original_mixture &
train_v2 2 independent_signed_gain signed_gain true original_mixture &
train_v2 3 independent_opportunity_risk opportunity_risk true original_mixture &
train_v2 4 independent_opportunity_risk_equal_source opportunity_risk true equal_strata &
wait

for label in independent_regret_bce frozen_opportunity_risk independent_signed_gain independent_opportunity_risk independent_opportunity_risk_equal_source; do
    checkpoint="${ROOT}/${label}/epoch_16_scene_selector.pt"
    report="${ROOT}/${label}/report.json"
    [[ -f "${checkpoint}" && -f "${report}" ]] || {
        echo "missing PCRA V2 artifact: ${label}" >&2
        exit 1
    }
    "${ALGENGINE_PYTHON}" - "${report}" <<'PY_REPORT'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert row["schema_version"]==6
assert row["frozen_parameter_max_abs_delta"]==0.0
assert row["evaluator_initialization_max_abs_delta"]==0.0
assert row["override_threshold"]==0.0
diag=row["counterfactual_evaluation_diagnostics"]
assert diag["reward_components_consumed"] is False
assert diag["proposal_frozen"] is True
PY_REPORT
done
printf '%s  %s\n' "${PROPOSAL32_SHA256}" "${PROPOSAL32}" >"${ROOT}/proposal32.sha256"
printf '%s\n' "${PROPOSAL48}" >"${ROOT}/proposal48.path"
printf '%s\n' "${PCRA_V1}" >"${ROOT}/pcra_v1.path"
echo "PASS PCRA V2 fixed search: ${ROOT}"
