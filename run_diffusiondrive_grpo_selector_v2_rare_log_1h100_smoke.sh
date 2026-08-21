#!/usr/bin/env bash
# One-H100 integration smoke for shared rare-log caches -> V2 train -> checkpoint audit.

set -Eeo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/projects/AlgEngine/scripts/diffusiondrive"
export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
export DIFFUSIONDRIVE_GRPO_CONFIG="${ROOT_DIR}/projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2_rare.py"
. "${SCRIPT_DIR}/grpo_selector_v2_rare_h100_env.sh"

SHARED_SOURCE="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1"
TUNING_SPLIT="${SHARED_SOURCE}/tuning_split"
SHARED_CACHE="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/tuning"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2_rare_original_v1/local_smoke/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${ROOT}/train"
LOG="${ROOT}/smoke.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL V2 rare-log smoke stage=${CURRENT_STAGE} exit=${rc}" >&2; echo "persistent_log: ${LOG}" >&2' ERR

"${ALGENGINE_PYTHON}" "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible
CURRENT_STAGE=tiny_cached_training
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_grpo_selector_v2_cached_rare_original.py" \
    --train-cache "${SHARED_CACHE}/train_seed0/cache.pt" \
    --train-cache "${SHARED_CACHE}/train_seed1/cache.pt" \
    --train-cache "${SHARED_CACHE}/train_seed2/cache.pt" \
    --pair-manifest "${TUNING_SPLIT}/train/pairs.jsonl" \
    --rare-data-audit "${TUNING_SPLIT}/train/rare_data_audit.json" \
    --output-dir "${ROOT}/train" --temperature 1 --learning-rate 1e-4 \
    --kl-weight 1e-3 --sampling-mode rare_balanced --seed 0 --epochs 2 \
    --examples-per-cache-epoch 6339 --checkpoint-epochs 2 --batch-size 64 --device cuda \
    --method-name exact_group_grpo_v2_rare_original_smoke_v1
CURRENT_STAGE=materialize
STATE="${ROOT}/train/epoch_2_selector.pt"
STATE_SHA="$(sha256sum "${STATE}" | awk '{print $1}')"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v2_rare.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" --selector-state "${STATE}" \
    --expected-selector-sha256 "${STATE_SHA}" \
    --expected-method exact_group_grpo_v2_rare_original_smoke_v1 \
    --release-name e2e_diffusiondrive_grpo_selector_v2_rare_original_smoke \
    --output "${ROOT}/checkpoint.pth" --manifest "${ROOT}/checkpoint_manifest.json"
CURRENT_STAGE=checkpoint_audit
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v2_rare_checkpoint.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" --checkpoint "${ROOT}/checkpoint.pth" \
    --output "${ROOT}/checkpoint_audit.json"
CURRENT_STAGE=complete
trap - ERR
echo "PASS one-H100 V2 rare-log smoke"
echo "output: ${ROOT}"
echo "persistent_log: ${LOG}"
