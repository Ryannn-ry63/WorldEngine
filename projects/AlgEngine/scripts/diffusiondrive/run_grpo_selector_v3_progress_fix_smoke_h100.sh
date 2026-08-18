#!/usr/bin/env bash
set -Eeo pipefail

export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
export DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME=grpo_selector_v3_progress_fix_v1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
. "${SCRIPT_DIR}/grpo_selector_v3_experiment_env.sh"

RUN_ID="${V3_PROGRESS_FIX_SMOKE_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${V3_ROOT}/local_parity_smoke/${RUN_ID}"
CONFIG="${DIFFUSIONDRIVE_GRPO_CONFIG}"
NAV_FILTER="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_navtrain_split_v2/navtrain_grpo_train.yaml"
VALIDATOR="${SCRIPT_DIR}/validate_grpo_selector_pdm_progress_fix.py"
UNIT_TEST="${ALGENGINE_ROOT}/scripts/tests/test_navsim_online_pdm_sampling.py"
NUM_TOKENS="${V3_PROGRESS_FIX_SMOKE_TOKENS:-16}"

mkdir -p "${ROOT}"
LOG_FILE="${ROOT}/smoke.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL V3 progress-fix smoke stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

for path in "${CONFIG}" "${NAV_FILTER}" "${VALIDATOR}" "${UNIT_TEST}"     "${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"; do
    [[ -f "${path}" ]] || { echo "Missing progress-fix smoke input: ${path}" >&2; exit 1; }
done
if ! [[ "${NUM_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "V3_PROGRESS_FIX_SMOKE_TOKENS must be a positive integer" >&2
    exit 2
fi

"${ALGENGINE_PYTHON}" "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py"     --extension "${WORLDENGINE_MMCV_EXTENSION}"     --expected-capability sm_90 --all-visible

CURRENT_STAGE=unit_regression
"${ALGENGINE_PYTHON}" -m pytest -q "${UNIT_TEST}"

CURRENT_STAGE=real_candidate_cpu_cuda_parity
cd "${ALGENGINE_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${VALIDATOR}"     "${CONFIG}"     --nav-filter "${NAV_FILTER}"     --num-tokens "${NUM_TOKENS}"     --noise-seed 20260818     --atol 1e-5     --output "${ROOT}/parity_report.json"

CURRENT_STAGE=complete
trap - ERR
echo "PASS one-H100 V3 PDM progress normalization smoke"
echo "report: ${ROOT}/parity_report.json"
echo "persistent_log: ${LOG_FILE}"
