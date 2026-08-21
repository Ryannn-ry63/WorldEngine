#!/usr/bin/env bash
# One-H100 V2 smoke over the already-built V3 rollout data manifest.

set -Eeo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/projects/AlgEngine/scripts/diffusiondrive"
export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
export DIFFUSIONDRIVE_GRPO_CONFIG="${ROOT_DIR}/projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2_rare.py"
. "${SCRIPT_DIR}/grpo_selector_v2_rare_h100_env.sh"

DATA_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data"
REAL_CACHE="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/full"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2_rare_rollout_v1/local_smoke/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${ROOT}/train"
LOG="${ROOT}/smoke.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL V2 rare-rollout smoke stage=${CURRENT_STAGE} exit=${rc}" >&2; echo "persistent_log: ${LOG}" >&2' ERR
for path in "${DATA_ROOT}/manifest.json" "${DATA_ROOT}/hard_pool.jsonl" \
    "${DATA_ROOT}/synthetic_cache.pt"; do
    [[ -f "${path}" ]] || { echo "Missing V3 rollout data: ${path}" >&2; exit 1; }
done
"${ALGENGINE_PYTHON}" "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible
CURRENT_STAGE=tiny_cached_training
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_grpo_selector_v2_cached_rare_rollout.py" \
    --real-cache "0=${REAL_CACHE}/train_seed0/cache.pt" \
    --real-cache "1=${REAL_CACHE}/train_seed1/cache.pt" \
    --real-cache "2=${REAL_CACHE}/train_seed2/cache.pt" \
    --synthetic-cache "${DATA_ROOT}/synthetic_cache.pt" \
    --data-manifest "${DATA_ROOT}/manifest.json" --hard-pool "${DATA_ROOT}/hard_pool.jsonl" \
    --hard-pool-split train --smoke-limit-hard-pool 4 --output-dir "${ROOT}/train" \
    --temperature 1 --learning-rate 1e-4 --kl-weight 1e-3 --seed 0 --epochs 1 \
    --examples-per-cache-epoch 12 --checkpoint-epochs 1 --batch-size 4 --device cuda \
    --method-name exact_group_grpo_v2_rare_rollout_smoke_v1
CURRENT_STAGE=materialize
STATE="${ROOT}/train/epoch_1_selector.pt"
STATE_SHA="$(sha256sum "${STATE}" | awk '{print $1}')"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v2_rare.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" --selector-state "${STATE}" \
    --expected-selector-sha256 "${STATE_SHA}" \
    --expected-method exact_group_grpo_v2_rare_rollout_smoke_v1 \
    --release-name e2e_diffusiondrive_grpo_selector_v2_rare_rollout_smoke \
    --output "${ROOT}/checkpoint.pth" --manifest "${ROOT}/checkpoint_manifest.json"
CURRENT_STAGE=checkpoint_audit
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v2_rare_checkpoint.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" --checkpoint "${ROOT}/checkpoint.pth" \
    --output "${ROOT}/checkpoint_audit.json"
CURRENT_STAGE=complete
trap - ERR
echo "PASS one-H100 V2 rare-rollout smoke"
echo "output: ${ROOT}"
echo "persistent_log: ${LOG}"
