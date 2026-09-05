#!/usr/bin/env bash
# Frozen PCRA search: equal-compute proposal control plus three actual-pair arms.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8

PCRA_EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_pcra_v1"
RUN_ID="${PCRA_SEARCH_ID:-search_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${PCRA_EXPERIMENT_ROOT}/search/${RUN_ID}"
PROPOSAL32="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt"
PROPOSAL32_SHA256=562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6
[[ -f "${PROPOSAL32}" ]] || { echo "missing frozen proposal32: ${PROPOSAL32}" >&2; exit 1; }
[[ "$(sha256sum "${PROPOSAL32}" | awk '{print $1}')" == "${PROPOSAL32_SHA256}" ]] || {
    echo "frozen proposal32 SHA256 mismatch" >&2
    exit 1
}
[[ ! -e "${ROOT}" ]] || { echo "PCRA search root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
COMMON_ARGS=("${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" --data-manifest "${RAPG_DATA_ROOT}/manifest.json" --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl")

train_proposal48() {
    local output="${ROOT}/proposal_compute48"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_reference_anchored_preference_graph.py" "${COMMON_ARGS[@]}" --output-dir "${output}" --architecture trajectory_set_reasoner --ablation relational_only --objective official_pdm_plus_scalar_preference --preference-weight 1.0 --data-split train --seed 0 --epochs 48 --checkpoint-epochs 48 --learning-rate 1e-4 --batch-size 64 --device cuda >"${output}/train.log" 2>&1
}

train_legacy_all() {
    local output="${ROOT}/legacy_all_candidates"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES=1 "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_incumbent_verified_preference_selector.py" "${COMMON_ARGS[@]}" --proposal-checkpoint "${PROPOSAL32}" --output-dir "${output}" --verifier-reward-margin 0.0 --verifier-reward-temperature 0.05 --override-threshold 0.0 --data-split train --seed 0 --epochs 16 --checkpoint-epochs 16 --learning-rate 1e-4 --batch-size 64 --device cuda >"${output}/train.log" 2>&1
}

train_pcra() {
    local gpu="$1" label="$2" loss="$3" context="$4"
    local output="${ROOT}/${label}"
    local context_args=()
    [[ "${context}" == "true" ]] && context_args+=(--use-decision-context)
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_proposal_conditioned_regret_arbitrator.py" "${COMMON_ARGS[@]}" --proposal-checkpoint "${PROPOSAL32}" --output-dir "${output}" --arbiter-loss "${loss}" "${context_args[@]}" --override-threshold 0.0 --data-split train --seed 0 --epochs 16 --checkpoint-epochs 16 --learning-rate 1e-4 --batch-size 64 --device cuda >"${output}/train.log" 2>&1
}

train_proposal48 &
train_legacy_all &
train_pcra 2 pair_sign sign false &
train_pcra 3 pair_regret regret false &
train_pcra 4 pair_regret_context regret true &
wait

required=(
    "${ROOT}/proposal_compute48/epoch_48_scene_selector.pt"
    "${ROOT}/proposal_compute48/report.json"
    "${ROOT}/legacy_all_candidates/epoch_16_scene_selector.pt"
    "${ROOT}/legacy_all_candidates/report.json"
    "${ROOT}/pair_sign/epoch_16_scene_selector.pt"
    "${ROOT}/pair_sign/report.json"
    "${ROOT}/pair_regret/epoch_16_scene_selector.pt"
    "${ROOT}/pair_regret/report.json"
    "${ROOT}/pair_regret_context/epoch_16_scene_selector.pt"
    "${ROOT}/pair_regret_context/report.json"
)
for path in "${required[@]}"; do
    [[ -f "${path}" ]] || { echo "missing PCRA search artifact: ${path}" >&2; exit 1; }
done
printf '%s  %s\n' "${PROPOSAL32_SHA256}" "${PROPOSAL32}" >"${ROOT}/proposal32.sha256"
echo "PASS PCRA frozen search: ${ROOT}"
