#!/usr/bin/env bash
# Evaluate a completed IVPS search and apply the frozen development gate.
set -Eeuo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 SEARCH_ROOT" >&2; exit 2; }
SEARCH_ROOT="$(readlink -f "$1")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)

evaluate_one() {
    local gpu="$1" label="$2" epoch="$3"
    local checkpoint="${SEARCH_ROOT}/${label}/epoch_${epoch}_scene_selector.pt"
    [[ -f "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; return 1; }
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/evaluate_rapg_offline.py" "${CACHE_ARGS[@]}" \
        --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
        --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
        --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
        --checkpoint "${checkpoint}" --split development \
        --output "${SEARCH_ROOT}/${label}/development_evaluation.json" \
        --batch-size 64 --device cuda \
        >"${SEARCH_ROOT}/${label}/development_evaluation.log" 2>&1
}

evaluate_one 0 proposal_b01_l100 16 &
evaluate_one 1 ivps_m000 16 &
evaluate_one 2 ivps_m010 16 &
evaluate_one 3 ivps_m030 16 &
evaluate_one 4 ivps_m050 16 &
evaluate_one 5 proposal_compute32 32 &
wait

"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/select_ivps_offline_development.py" \
    --proposal-baseline "${SEARCH_ROOT}/proposal_b01_l100/development_evaluation.json" \
    --candidate "m000=${SEARCH_ROOT}/ivps_m000/development_evaluation.json" \
    --candidate "m010=${SEARCH_ROOT}/ivps_m010/development_evaluation.json" \
    --candidate "m030=${SEARCH_ROOT}/ivps_m030/development_evaluation.json" \
    --candidate "m050=${SEARCH_ROOT}/ivps_m050/development_evaluation.json" \
    --output "${SEARCH_ROOT}/development_gate.json"

echo "PASS IVPS development audit: ${SEARCH_ROOT}/development_gate.json"
