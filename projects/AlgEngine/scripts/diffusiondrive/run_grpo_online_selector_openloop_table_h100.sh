#!/usr/bin/env bash
set -eo pipefail

CHECKPOINT="${1:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/diffusiondrive/grpo_online_selector/ddgrpo_online_pi_ref_pilot_lr3e6_s0/iter_150.pth}"
EXPECTED_SHA256="99157e1bd6b64eb0960ab8bd263e0f209164bf4237d42dd0932f86e1b3e1263d"

set +u
PS1="${PS1:-diffusiondrive-h100}"
. /inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/dotfiles/.bashrc
set -u

export WORLDENGINE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export ALGENGINE_ENV=/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine
export PATH="${ALGENGINE_ENV}/bin:${PATH}"
export PYTHON_BIN="${ALGENGINE_ENV}/bin/python"
export DIFFUSIONDRIVE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export MMCV_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/mmcv
export NUPLAN_DEVKIT_ROOT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/nuplan-devkit
export SENIOR_NAVSIM_PARENT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/E2E/navsim_v1
export NAVSIM_DEVKIT_ROOT="${SENIOR_NAVSIM_PARENT}/navsim"
export DIFFUSIONDRIVE_BOOTSTRAP="${SIMENGINE_ROOT}/scripts/diffusiondrive/algengine_worker_bootstrap"
export WORLDENGINE_DIFFUSIONDRIVE_MMCV_BOOTSTRAP=1
export WORLDENGINE_MMCV_EXTENSION="${WORLDENGINE_ROOT}/artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so"
export NAVSIM_EXP_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp
export NAVSIM_METRIC_CACHE_PATH_TRAIN="${NAVSIM_EXP_ROOT}/metric_cache_trainval"
export NAVSIM_METRIC_CACHE_PATH_EVAL="${NAVSIM_EXP_ROOT}/metric_cache_navtest"
export NAVSIM_METRIC_CACHE_PATH="${NAVSIM_METRIC_CACHE_PATH_EVAL}"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${MMCV_ROOT}:${NUPLAN_DEVKIT_ROOT}:${NAVSIM_DEVKIT_ROOT}:${SENIOR_NAVSIM_PARENT}:${PYTHONPATH:-}"
export NAVSIM_OFFICIAL_RESCORE=auto

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing deployment checkpoint: ${CHECKPOINT}"
    exit 1
fi
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
if [[ ! -f "${WORLDENGINE_MMCV_EXTENSION}" ]]; then
    echo "Missing H100 mmcv extension: ${WORLDENGINE_MMCV_EXTENSION}"
    exit 1
fi

CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_online_selector.py"
cd "${ALGENGINE_ROOT}"

echo "[1/2] OpenLoop-navtest with frozen GRPO selector checkpoint"
MASTER_PORT="${MASTER_PORT_NAVTEST:-28620}" ./scripts/e2e_dist_eval.sh "${CONFIG}" "${CHECKPOINT}" 8

echo "[2/2] navtest_failures OpenLoop with the same frozen checkpoint"
MASTER_PORT="${MASTER_PORT_FAILURES:-28621}" ./scripts/e2e_dist_eval_navtest_failures.sh "${CONFIG}" "${CHECKPOINT}" 8

echo "PASS DiffusionDrive GRPO selector OpenLoop table evaluation"
echo "checkpoint: ${CHECKPOINT}"
echo "checkpoint_sha256: ${ACTUAL_SHA256}"
