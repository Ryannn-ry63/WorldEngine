#!/usr/bin/env bash
# Shared read-only senior-environment bootstrap for DiffusionDrive-only jobs.

set +u
PS1="${PS1:-diffusiondrive-h100}"
. /inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/dotfiles/.bashrc
set -u

# Honor the code root selected by an isolated Git worktree.  Keep the
# historical main-worktree path only as a fallback for legacy launchers.
export WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine}"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export ALGENGINE_ENV="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}"
export ALGENGINE_PYTHON="${ALGENGINE_ENV}/bin/python"
export ALGENGINE_TORCHRUN="${ALGENGINE_ENV}/bin/torchrun"
export PATH="${ALGENGINE_ENV}/bin:${PATH}"
export DIFFUSIONDRIVE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export MMCV_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/mmcv
export NUPLAN_DEVKIT_ROOT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/nuplan-devkit
export SENIOR_NAVSIM_PARENT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/E2E/navsim_v1
export DIFFUSIONDRIVE_BOOTSTRAP="${SIMENGINE_ROOT}/scripts/diffusiondrive/algengine_worker_bootstrap"
export WORLDENGINE_DIFFUSIONDRIVE_MMCV_BOOTSTRAP=1
export WORLDENGINE_MMCV_EXTENSION="${WORLDENGINE_ROOT}/artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so"
export NAVSIM_EXP_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp
export NAVSIM_METRIC_CACHE_PATH_TRAIN="${NAVSIM_EXP_ROOT}/metric_cache_trainval"
export NAVSIM_METRIC_CACHE_PATH_EVAL="${NAVSIM_EXP_ROOT}/metric_cache_navtest"
export NAVSIM_METRIC_CACHE_PATH="${NAVSIM_METRIC_CACHE_PATH_EVAL}"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${MMCV_ROOT}:${NUPLAN_DEVKIT_ROOT}:${SENIOR_NAVSIM_PARENT}:${PYTHONPATH:-}"
export DIFFUSIONDRIVE_GRPO_SPLIT_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_navtrain_split_v2"
export DIFFUSIONDRIVE_GRPO_NAV_FILTER_PATH="${DIFFUSIONDRIVE_GRPO_SPLIT_ROOT}/navtrain_grpo_train.yaml"
export DIFFUSIONDRIVE_GRPO_CALIBRATION_FILTER_PATH="${DIFFUSIONDRIVE_GRPO_SPLIT_ROOT}/navtrain_grpo_calibration.yaml"
export DIFFUSIONDRIVE_GRPO_CONFIG="${DIFFUSIONDRIVE_GRPO_CONFIG:-${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector.py}"
export DIFFUSIONDRIVE_GRPO_BASELINE=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EXPECTED_GPU_COUNT="${DIFFUSIONDRIVE_EXPECTED_GPU_COUNT:-8}"
if [[ ! "${EXPECTED_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DIFFUSIONDRIVE_EXPECTED_GPU_COUNT must be a positive integer" >&2
    return 1
fi
if [[ ! -f "${WORLDENGINE_MMCV_EXTENSION}" ]]; then
    echo "Missing multi-arch mmcv extension: ${WORLDENGINE_MMCV_EXTENSION}" >&2
    return 1
fi
if [[ ! -x "${ALGENGINE_PYTHON}" || ! -x "${ALGENGINE_TORCHRUN}" ]]; then
    echo "Missing senior algengine executables under ${ALGENGINE_ENV}" >&2
    return 1
fi
GPU_COUNT="$("${ALGENGINE_PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${GPU_COUNT}" -ne "${EXPECTED_GPU_COUNT}" ]]; then
    echo "Expected exactly ${EXPECTED_GPU_COUNT} visible H100 GPU(s), found ${GPU_COUNT}" >&2
    return 1
fi
