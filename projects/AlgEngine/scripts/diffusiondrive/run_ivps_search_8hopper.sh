#!/usr/bin/env bash
# Frozen two-stage IVPS search: scalar proposal, then four verifier margins.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8

IVPS_EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1"
RUN_ID="${IVPS_SEARCH_ID:-search_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${IVPS_EXPERIMENT_ROOT}/search/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "IVPS search root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}/proposal_b01_l100"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)

CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/train_reference_anchored_preference_graph.py" \
    "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
    --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
    --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
    --output-dir "${ROOT}/proposal_b01_l100" \
    --architecture trajectory_set_reasoner --ablation relational_only \
    --objective official_pdm_plus_scalar_preference --preference-weight 1.0 \
    --data-split train --seed 0 --epochs 16 --checkpoint-epochs 16 \
    --batch-size 64 --device cuda \
    >"${ROOT}/proposal_b01_l100/train.log" 2>&1

PROPOSAL="${ROOT}/proposal_b01_l100/epoch_16_scene_selector.pt"
[[ -f "${PROPOSAL}" ]] || { echo "missing IVPS proposal: ${PROPOSAL}" >&2; exit 1; }

run_verifier() {
    local gpu="$1" label="$2" margin="$3"
    mkdir -p "${ROOT}/${label}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/train_incumbent_verified_preference_selector.py" \
        "${CACHE_ARGS[@]}" \
        --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
        --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
        --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
        --proposal-checkpoint "${PROPOSAL}" --output-dir "${ROOT}/${label}" \
        --verifier-reward-margin "${margin}" --verifier-reward-temperature 0.05 \
        --override-threshold 0.0 --data-split train --seed 0 --epochs 16 \
        --checkpoint-epochs 16 --batch-size 64 --device cuda \
        >"${ROOT}/${label}/train.log" 2>&1
}

run_verifier 0 ivps_m000 0.00 &
run_verifier 1 ivps_m010 0.01 &
run_verifier 2 ivps_m030 0.03 &
run_verifier 3 ivps_m050 0.05 &

mkdir -p "${ROOT}/proposal_compute32"
CUDA_VISIBLE_DEVICES=4 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/train_reference_anchored_preference_graph.py" \
    "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
    --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
    --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
    --output-dir "${ROOT}/proposal_compute32" \
    --architecture trajectory_set_reasoner --ablation relational_only \
    --objective official_pdm_plus_scalar_preference --preference-weight 1.0 \
    --data-split train --seed 0 --epochs 32 --checkpoint-epochs 32 \
    --batch-size 64 --device cuda \
    >"${ROOT}/proposal_compute32/train.log" 2>&1 &
wait

echo "PASS IVPS fixed-budget search: ${ROOT}"
