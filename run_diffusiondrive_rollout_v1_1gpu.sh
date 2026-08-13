#!/usr/bin/env bash
# One-click gated preflight -> smoke -> pilot for one visible H100.

set -Eeo pipefail

WORLDENGINE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
LAUNCHER="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_1gpu_h100.sh"
ROLLOUT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1"
LOG_ROOT="${ROLLOUT_ROOT}/logs"
DATA_TYPE=navtrain_50pct_collision
ASSET_NAME=navtrain
SEED=0
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
PREFLIGHT_RUN_ID="r1preflight1g_${ATTEMPT_ID}"
SMOKE_RUN_ID="r1single1g_${ATTEMPT_ID}"
PILOT_RUN_ID="r1pilot1g_${ATTEMPT_ID}"

mkdir -p "${LOG_ROOT}"
WRAPPER_LOG="${LOG_ROOT}/$(date -u +%Y%m%dT%H%M%SZ)_1gpu_preflight_smoke_pilot.log"
exec > >(tee -a "${WRAPPER_LOG}") 2>&1

CURRENT_STAGE=bootstrap
trap 'rc=$?; echo "FAIL one-GPU rollout stage=${CURRENT_STAGE} exit=${rc}" >&2; echo "wrapper_log: ${WRAPPER_LOG}" >&2' ERR

cd "${WORLDENGINE_ROOT}"
[[ -x "${LAUNCHER}" ]] || {
    echo "Missing executable one-GPU launcher: ${LAUNCHER}" >&2
    exit 1
}

echo "START DiffusionDrive rollout-v1 one-GPU gate"
echo "No manual conda activation is required."
echo "data_type=${DATA_TYPE} asset_name=${ASSET_NAME} seed=${SEED}"
echo "attempt_id=${ATTEMPT_ID}"
echo "algengine_python=/root/miniconda3/envs/algengine/bin/python"
echo "simengine_python=/root/miniconda3/envs/simengine/bin/python"
echo "wrapper_log=${WRAPPER_LOG}"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free \
    --format=csv,noheader

CURRENT_STAGE=preflight
"${LAUNCHER}" preflight "${SEED}" "${DATA_TYPE}" "${ASSET_NAME}" "${PREFLIGHT_RUN_ID}"

CURRENT_STAGE=smoke
"${LAUNCHER}" smoke "${SEED}" "${DATA_TYPE}" "${ASSET_NAME}" "${SMOKE_RUN_ID}"
SMOKE_AUDIT="${ROLLOUT_ROOT}/rollouts/${DATA_TYPE}/seed${SEED}/${SMOKE_RUN_ID}/rollout_audit.json"
[[ -f "${SMOKE_AUDIT}" ]] || { echo "Missing smoke audit: ${SMOKE_AUDIT}" >&2; exit 1; }
echo "PASS smoke audit: ${SMOKE_AUDIT}"

CURRENT_STAGE=pilot
"${LAUNCHER}" pilot "${SEED}" "${DATA_TYPE}" "${ASSET_NAME}" "${PILOT_RUN_ID}"
PILOT_AUDIT="${ROLLOUT_ROOT}/rollouts/${DATA_TYPE}/seed${SEED}/${PILOT_RUN_ID}/rollout_audit.json"
[[ -f "${PILOT_AUDIT}" ]] || { echo "Missing pilot audit: ${PILOT_AUDIT}" >&2; exit 1; }
echo "PASS pilot audit: ${PILOT_AUDIT}"

CURRENT_STAGE=complete
trap - ERR
echo "PASS DiffusionDrive rollout-v1 one-GPU preflight, smoke, and pilot"
echo "smoke_audit: ${SMOKE_AUDIT}"
echo "pilot_audit: ${PILOT_AUDIT}"
echo "wrapper_log: ${WRAPPER_LOG}"
