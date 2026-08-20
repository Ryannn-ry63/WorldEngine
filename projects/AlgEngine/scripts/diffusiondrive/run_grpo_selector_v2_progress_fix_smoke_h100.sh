#!/usr/bin/env bash
set -Eeo pipefail

export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

RUN_ID="${V2_PROGRESS_FIX_SMOKE_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2_progress_fix_v1/local_smoke/${RUN_ID}"
CACHE_ROOT="${ROOT}/cache"
NAV_FILTER="${DIFFUSIONDRIVE_GRPO_NAV_FILTER_PATH}"
VALIDATOR="${SCRIPT_DIR}/validate_grpo_selector_pdm_progress_fix.py"
REWARD_FILE="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
PARITY_TOKENS="${V2_PROGRESS_FIX_SMOKE_PARITY_TOKENS:-16}"
CACHE_TOKENS="${V2_PROGRESS_FIX_SMOKE_CACHE_TOKENS:-2}"
mkdir -p "${ROOT}"
LOG_FILE="${ROOT}/smoke.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL one-H100 corrected V2 smoke stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

CURRENT_STAGE=mmcv_cuda_preflight
"${ALGENGINE_PYTHON}" "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible

CURRENT_STAGE=unit_regression
"${ALGENGINE_PYTHON}" -m pytest -q \
    "${ALGENGINE_ROOT}/scripts/tests/test_navsim_online_pdm_sampling.py" \
    "${ALGENGINE_ROOT}/tests/test_grpo_selector_v2_progress_fix.py" \
    "${ALGENGINE_ROOT}/tests/test_diffusion_grpo_selector_v2_sweep.py" \
    "${ALGENGINE_ROOT}/tests/test_diffusion_grpo_selector_v2_formal.py"

CURRENT_STAGE=real_candidate_cpu_cuda_parity
cd "${ALGENGINE_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${VALIDATOR}" \
    "${DIFFUSIONDRIVE_GRPO_CONFIG}" --nav-filter "${NAV_FILTER}" \
    --num-tokens "${PARITY_TOKENS}" --noise-seed 20260820 --atol 1e-5 \
    --output "${ROOT}/parity_report.json"

export V2_PROGRESS_FIX_CACHE_ROOT="${CACHE_ROOT}"
for cache_name in train_seed0 calibration_seed0 calibration_seed1 calibration_seed2; do
    CURRENT_STAGE="rescore_${cache_name}"
    "${SCRIPT_DIR}/run_grpo_selector_v2_progress_fix_rescore_cache_h100.sh" \
        "${cache_name}" "${CACHE_TOKENS}"
done

CURRENT_STAGE=selector_train_step
TRAIN_ROOT="${ROOT}/train"
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/train_grpo_selector_v2_cached.py" \
    --train-cache "${CACHE_ROOT}/train_seed0/cache.pt" \
    --calibration-cache "${CACHE_ROOT}/calibration_seed0/cache.pt" \
    --calibration-cache "${CACHE_ROOT}/calibration_seed1/cache.pt" \
    --calibration-cache "${CACHE_ROOT}/calibration_seed2/cache.pt" \
    --output-dir "${TRAIN_ROOT}" --temperature 1 --learning-rate 1e-3 \
    --kl-weight 1e-3 --seed 0 --epochs 1 --checkpoint-epochs 1 --device cuda

CURRENT_STAGE=materialize
SELECTOR_STATE="${TRAIN_ROOT}/epoch_1_selector.pt"
SELECTOR_SHA="$(sha256sum "${SELECTOR_STATE}" | awk '{print $1}')"
REWARD_SHA="$(sha256sum "${REWARD_FILE}" | awk '{print $1}')"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v2_replica.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --selector-state "${SELECTOR_STATE}" --expected-selector-sha256 "${SELECTOR_SHA}" \
    --train-seed 0 --temperature 1 --learning-rate 1e-3 --kl-weight 1e-3 \
    --epoch 1 --expected-reward-contract navsim_pairwise_raw_progress_then_candidate_gate_v1 \
    --expected-reward-sha256 "${REWARD_SHA}" \
    --experiment-method e2e_diffusiondrive_grpo_selector_v2_progress_fix_v1_smoke \
    --output "${ROOT}/checkpoint.pth" --manifest "${ROOT}/checkpoint_manifest.json"

CURRENT_STAGE=checkpoint_audit
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_checkpoint.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --checkpoint "${ROOT}/checkpoint.pth" --output "${ROOT}/checkpoint_audit.json"

CURRENT_STAGE=complete
"${ALGENGINE_PYTHON}" -c '
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); payload={"schema_version":1,"status":"PASS","parity_report":str(root/"parity_report.json"),"training_report":str(root/"train/report.json"),"checkpoint_manifest":str(root/"checkpoint_manifest.json"),"checkpoint_audit":str(root/"checkpoint_audit.json")}
(root/"smoke_report.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
' "${ROOT}"
trap - ERR
echo "PASS one-H100 V2 progress normalization end-to-end smoke"
echo "output: ${ROOT}"
echo "persistent_log: ${LOG_FILE}"
