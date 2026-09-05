#!/usr/bin/env bash
# Log-disjoint PCRA development evaluation; exits nonzero when the frozen gate fails.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8

ROOT="${1:-${PCRA_SEARCH_ROOT:-}}"
[[ -n "${ROOT}" ]] || { echo "usage: $0 SEARCH_ROOT" >&2; exit 2; }
ROOT="$(cd "${ROOT}" && pwd)"
PROPOSAL32="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt"
PROPOSAL32_SHA256=562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6
[[ "$(sha256sum "${PROPOSAL32}" | awk '{print $1}')" == "${PROPOSAL32_SHA256}" ]] || {
    echo "frozen proposal32 SHA256 mismatch" >&2
    exit 1
}
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
EVAL_ROOT="${ROOT}/offline_development"
mkdir -p "${EVAL_ROOT}"

evaluate_one() {
    local gpu="$1" label="$2" checkpoint="$3"
    [[ -f "${checkpoint}" ]] || { echo "missing PCRA checkpoint: ${checkpoint}" >&2; return 1; }
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/evaluate_rapg_offline.py" "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" --data-manifest "${RAPG_DATA_ROOT}/manifest.json" --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" --checkpoint "${checkpoint}" --split development --output "${EVAL_ROOT}/${label}.json" --batch-size 64 --device cuda >"${EVAL_ROOT}/${label}.log" 2>&1
}

evaluate_one 0 proposal32 "${PROPOSAL32}" &
evaluate_one 1 proposal48 "${ROOT}/proposal_compute48/epoch_48_scene_selector.pt" &
evaluate_one 2 legacy_all "${ROOT}/legacy_all_candidates/epoch_16_scene_selector.pt" &
evaluate_one 3 pair_sign "${ROOT}/pair_sign/epoch_16_scene_selector.pt" &
evaluate_one 4 pair_regret "${ROOT}/pair_regret/epoch_16_scene_selector.pt" &
evaluate_one 5 pair_regret_context "${ROOT}/pair_regret_context/epoch_16_scene_selector.pt" &
wait

"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/select_pcra_offline_development.py" --proposal-32 "${EVAL_ROOT}/proposal32.json" --proposal-48 "${EVAL_ROOT}/proposal48.json" --control "legacy_all=${EVAL_ROOT}/legacy_all.json" --control "pair_sign=${EVAL_ROOT}/pair_sign.json" --candidate "pair_regret=${EVAL_ROOT}/pair_regret.json" --candidate "pair_regret_context=${EVAL_ROOT}/pair_regret_context.json" --output "${ROOT}/development_gate.json"

"${ALGENGINE_PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p["development_gate_passed"] else 2)' "${ROOT}/development_gate.json"
echo "PASS PCRA offline development gate: ${ROOT}/development_gate.json"
