#!/usr/bin/env bash
set -Eeo pipefail

CHECKPOINT="${1:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/diffusiondrive/grpo_online_selector/ddgrpo_online_pi_ref_pilot_lr3e6_s0/iter_150.pth}"
MODEL_NAME="${2:-e2e_diffusiondrive_grpo_selector_iter150}"
EXPECTED_SHA256="99157e1bd6b64eb0960ab8bd263e0f209164bf4237d42dd0932f86e1b3e1263d"

LOG_DIR=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/diffusiondrive/grpo_online_selector/closedloop_launcher_logs
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ)_${MODEL_NAME}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=launcher_init
trap 'rc=$?; echo "FAIL stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
echo "DiffusionDrive GRPO selector closed-loop launcher"
echo "persistent_log: ${LOG_FILE}"
echo "checkpoint: ${CHECKPOINT}"
echo "model_name: ${MODEL_NAME}"

CURRENT_STAGE=select_read_only_python_environments
echo "PASS stage=${CURRENT_STAGE}"

export WORLDENGINE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export DIFFUSIONDRIVE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export H100_SUPPORT_DIR="${SIMENGINE_ROOT}/scripts/diffusiondrive"
export DIFFUSIONDRIVE_BOOTSTRAP="${H100_SUPPORT_DIR}/algengine_worker_bootstrap"
export ALGENGINE_PYTHON=/root/miniconda3/envs/algengine/bin/python
export SIMENGINE_PYTHON=/root/miniconda3/envs/simengine/bin/python

export WORLDENGINE_DIFFUSIONDRIVE_MMCV_BOOTSTRAP=1
export WORLDENGINE_MMCV_EXTENSION="${WORLDENGINE_ROOT}/artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so"
export WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP=1
export WORLDENGINE_GSPLAT_EXTENSION="${WORLDENGINE_ROOT}/artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${H100_SUPPORT_DIR}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${PYTHONPATH:-}"

CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_online_selector.py"
SCENARIO="${WORLDENGINE_ROOT}/data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl"
ASSETS="${WORLDENGINE_ROOT}/data/sim_engine/assets/navtest_failures/assets"
RAY_RUNNER="${ALGENGINE_ROOT}/scripts/diffusiondrive/run_ray_distributed_testing_diffusiondrive_h100.sh"

CURRENT_STAGE=static_preflight
for path in "${CHECKPOINT}" "${CONFIG}" "${SCENARIO}" "${WORLDENGINE_MMCV_EXTENSION}" "${WORLDENGINE_GSPLAT_EXTENSION}" "${H100_SUPPORT_DIR}/preflight_mmcv_cuda.py" "${H100_SUPPORT_DIR}/preflight_gsplat_cuda.py"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}"
        exit 1
    fi
done
if [[ ! -d "${ASSETS}" ]]; then
    echo "Missing navtest_failures assets: ${ASSETS}"
    exit 1
fi
for python_bin in "${ALGENGINE_PYTHON}" "${SIMENGINE_PYTHON}"; do
    if [[ ! -x "${python_bin}" ]]; then
        echo "Missing read-only environment Python: ${python_bin}"
        exit 1
    fi
done
ACTUAL_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "Checkpoint SHA256 mismatch: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}"
    exit 1
fi
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "Expected exactly 8 visible H100 GPUs, found ${GPU_COUNT}"
    exit 1
fi
echo "PASS stage=${CURRENT_STAGE} gpu_count=${GPU_COUNT} checkpoint_sha256=${ACTUAL_SHA256}"

CURRENT_STAGE=algengine_mmcv_h100_preflight
echo "[0a/4] ${CURRENT_STAGE}"
"${ALGENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_mmcv_cuda.py" --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible

CURRENT_STAGE=simengine_gsplat_ray_h100_preflight
echo "[0b/4] ${CURRENT_STAGE}"
"${SIMENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_gsplat_cuda.py" --extension "${WORLDENGINE_GSPLAT_EXTENSION}" --expected-capability sm_90 --ray-workers 8

cd "${ALGENGINE_ROOT}"
CURRENT_STAGE=closed_loop_nonreactive
echo "[1/4] navtest_failures CL-NonReactive"
"${RAY_RUNNER}" "${CONFIG}" "${CHECKPOINT}" "${MODEL_NAME}" navtest_failures NR

CURRENT_STAGE=closed_loop_reactive
echo "[2/4] navtest_failures CL-Reactive"
sleep 10
"${RAY_RUNNER}" "${CONFIG}" "${CHECKPOINT}" "${MODEL_NAME}" navtest_failures R

CURRENT_STAGE=complete
trap - ERR
echo "[4/4] PASS DiffusionDrive GRPO selector closed-loop table evaluation"
echo "checkpoint: ${CHECKPOINT}"
echo "checkpoint_sha256: ${ACTUAL_SHA256}"
echo "NR: ${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${MODEL_NAME}/navtest_failures_NR"
echo "R:  ${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${MODEL_NAME}/navtest_failures_R"
echo "persistent_log: ${LOG_FILE}"
