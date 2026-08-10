#!/usr/bin/env bash
set -eo pipefail

MODE="${1:-smoke}"
RUN_NAME="${2:-${MODE}_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash"
    exit 2
fi
case "${MODE}" in
    smoke) DEFAULT_MAX_ITERS=32; DEFAULT_CHECKPOINT_INTERVAL=32 ;;
    pilot) DEFAULT_MAX_ITERS=250; DEFAULT_CHECKPOINT_INTERVAL=50 ;;
    formal) DEFAULT_MAX_ITERS=2000; DEFAULT_CHECKPOINT_INTERVAL=250 ;;
    *) echo "MODE must be smoke, pilot, or formal"; exit 2 ;;
esac

# The shared dotfiles intentionally return from non-interactive shells when
# PS1 is empty. Give this process a harmless prompt and keep nounset disabled
# while conda's shell hook is evaluated; strict nounset resumes immediately.
set +u
PS1="${PS1:-diffusiondrive-h100}"
. /inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/dotfiles/.bashrc
set -u

export WORLDENGINE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export ALGENGINE_ENV=/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine
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
# The senior dotfiles point the generic NAVSIM cache variable at navtest.
# Selector-GRPO trains on navtrain, so pin every train-cache alias after
# sourcing the dotfiles and keep config/audit/parity on one immutable cache.
export NAVSIM_EXP_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp
export NAVSIM_METRIC_CACHE_PATH_TRAIN="${NAVSIM_EXP_ROOT}/metric_cache_trainval"
export NAVSIM_METRIC_CACHE_PATH_EVAL="${NAVSIM_EXP_ROOT}/metric_cache_navtest"
export NAVSIM_METRIC_CACHE_PATH="${NAVSIM_METRIC_CACHE_PATH_EVAL}"
# Keep our DiffusionDrive/WorldEngine code first while reusing only the senior
# nuPlan package and algengine interpreter.  In particular, do not inherit the
# senior WorldEngine/mmcv source paths ahead of this checkout.
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${MMCV_ROOT}:${NUPLAN_DEVKIT_ROOT}:${SENIOR_NAVSIM_PARENT}:${PYTHONPATH:-}"
export DIFFUSIONDRIVE_GRPO_MAX_ITERS="${DIFFUSIONDRIVE_GRPO_MAX_ITERS:-${DEFAULT_MAX_ITERS}}"
export DIFFUSIONDRIVE_GRPO_CHECKPOINT_INTERVAL="${DIFFUSIONDRIVE_GRPO_CHECKPOINT_INTERVAL:-${DEFAULT_CHECKPOINT_INTERVAL}}"
export DIFFUSIONDRIVE_GRPO_MAX_KEEP_CKPTS="${DIFFUSIONDRIVE_GRPO_MAX_KEEP_CKPTS:-6}"
export DIFFUSIONDRIVE_GRPO_LR="${DIFFUSIONDRIVE_GRPO_LR:-1e-5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MASTER_PORT="${MASTER_PORT:-23456}"

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${GPU_COUNT}" -ne 8 ]]; then
    echo "Expected exactly 8 visible H100 GPUs, found ${GPU_COUNT}"
    exit 1
fi
if [[ ! -f "${WORLDENGINE_MMCV_EXTENSION}" ]]; then
    echo "Missing multi-arch mmcv extension: ${WORLDENGINE_MMCV_EXTENSION}"
    exit 1
fi
if [[ ! -x "${ALGENGINE_PYTHON}" || ! -x "${ALGENGINE_TORCHRUN}" ]]; then
    echo "Missing senior algengine executables under ${ALGENGINE_ENV}"
    exit 1
fi

CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_online_selector.py"
WORK_DIR="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_online_selector/${RUN_NAME}"
METRIC_CACHE="${NAVSIM_METRIC_CACHE_PATH_TRAIN}"
mkdir -p "${WORK_DIR}"

echo "DiffusionDrive train metric cache: ${METRIC_CACHE}"
echo "DiffusionDrive selector GRPO: lr=${DIFFUSIONDRIVE_GRPO_LR}, iters=${DIFFUSIONDRIVE_GRPO_MAX_ITERS}, checkpoint_interval=${DIFFUSIONDRIVE_GRPO_CHECKPOINT_INTERVAL}"
echo "[1/4] Dataset/cache coverage audit"
cd "${ALGENGINE_ROOT}"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/audit_grpo_online_dataset.py --config "${CONFIG}"

echo "[2/4] CPU model/checkpoint contract preflight"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/preflight_grpo_online_selector.py --config "${CONFIG}"

echo "[3/4] H100 batched-PDM parity"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/validate_online_pdm_reward.py \
    --metric-cache "${METRIC_CACHE}" \
    --device cuda \
    --num-candidates 20 \
    --atol 1e-5

echo "[4/4] 8xH100 selector-GRPO ${MODE} (${DIFFUSIONDRIVE_GRPO_MAX_ITERS} iters)"
"${ALGENGINE_TORCHRUN}" \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT}" \
    scripts/train.py \
    "${CONFIG}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR}" \
    --no-validate

CHECKPOINT="${WORK_DIR}/iter_${DIFFUSIONDRIVE_GRPO_MAX_ITERS}.pth"
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Expected final checkpoint not found: ${CHECKPOINT}"
    exit 1
fi
echo "PASS DiffusionDrive online selector GRPO ${MODE}"
echo "checkpoint: ${CHECKPOINT}"
