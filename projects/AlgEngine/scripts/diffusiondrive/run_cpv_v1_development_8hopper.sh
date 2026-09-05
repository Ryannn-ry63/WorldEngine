#!/usr/bin/env bash
# Evaluate the fixed CPV E1 run once; S/C/D are valid zero-exit outcomes.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8

ROOT="${1:-${CPV_SEARCH_ROOT:-}}"
[[ -n "${ROOT}" ]] || { echo "usage: $0 SEARCH_ROOT" >&2; exit 2; }
if [[ ! -d "${ROOT}" ]]; then
    echo "CPV search root does not exist: ${ROOT}" >&2
    echo "Available runs:" >&2
    find "${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_cpv_v1/search" \
        -mindepth 1 -maxdepth 1 -type d -printf '  %p\n' 2>/dev/null | sort >&2 || true
    exit 2
fi
ROOT="$(cd "${ROOT}" && pwd)"
PCRA_V1_SEARCH="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_pcra_v1/search/search_20260829T171222Z"
PROPOSAL32="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt"
PROPOSAL48="${PCRA_V1_SEARCH}/proposal_compute48/epoch_48_scene_selector.pt"
A0="${PCRA_V1_SEARCH}/pair_regret/epoch_16_scene_selector.pt"
PROPOSAL32_SHA256=562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6
[[ "$(sha256sum "${PROPOSAL32}" | awk '{print $1}')" == "${PROPOSAL32_SHA256}" ]] || exit 1
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
EVAL_ROOT="${ROOT}/offline_development"
[[ ! -e "${ROOT}/development_gate.json" ]] || {
    echo "development gate already exists; refusing to reread development: ${ROOT}/development_gate.json" >&2
    exit 2
}
mkdir -p "${EVAL_ROOT}"

evaluate_one() {
    local gpu="$1" label="$2" checkpoint="$3"
    [[ -f "${checkpoint}" ]] || { echo "missing CPV checkpoint: ${checkpoint}" >&2; return 1; }
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/evaluate_rapg_offline.py" \
        "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
        --data-manifest "${RAPG_DATA_ROOT}/manifest.json" --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
        --checkpoint "${checkpoint}" --split development --output "${EVAL_ROOT}/${label}.json" \
        --batch-size 64 --device cuda >"${EVAL_ROOT}/${label}.log" 2>&1
}

pids=()
evaluate_one 0 proposal32 "${PROPOSAL32}" & pids+=("$!")
evaluate_one 1 proposal48 "${PROPOSAL48}" & pids+=("$!")
evaluate_one 2 A0_global_regret "${A0}" & pids+=("$!")
evaluate_one 3 A1_source_regret "${ROOT}/A1_source_regret/epoch_16_scene_selector.pt" & pids+=("$!")
evaluate_one 4 A2_sign_regret "${ROOT}/A2_sign_regret/epoch_16_scene_selector.pt" & pids+=("$!")
evaluate_one 5 A3_source_sign_regret "${ROOT}/A3_source_sign_regret/epoch_16_scene_selector.pt" & pids+=("$!")
for pid in "${pids[@]}"; do
    wait "${pid}"
done

"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/select_cpv_offline_development.py" \
    --proposal-32 "${EVAL_ROOT}/proposal32.json" \
    --proposal-48 "${EVAL_ROOT}/proposal48.json" \
    --a0 "${EVAL_ROOT}/A0_global_regret.json" \
    --a1 "${EVAL_ROOT}/A1_source_regret.json" \
    --a2 "${EVAL_ROOT}/A2_sign_regret.json" \
    --a3 "${EVAL_ROOT}/A3_source_sign_regret.json" \
    --output "${ROOT}/development_gate.json" \
    --decision-ledger "${RAPG_SOURCE_WORLDENGINE_ROOT}/DIFFUSIONDRIVE_SELECTOR_DECISION_LEDGER.json"

"${ALGENGINE_PYTHON}" - "${ROOT}/development_gate.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
print(
    "VALID CPV E1 RESULT: outcome={outcome}; authorized_followup={followup}; gate={gate}".format(
        outcome=report["stage_outcome"],
        followup=report["authorized_followup_stage"],
        gate="PASS" if report["development_gate_passed"] else "FAIL",
    )
)
PY
