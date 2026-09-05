#!/usr/bin/env bash
# Reproducible environment and immutable inputs for RAPG selector experiments.
set +u
PS1="${PS1:-rapg-selector}"
. /inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/dotfiles/.bashrc
set -u
ulimit -c 0

RAPG_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORLDENGINE_ROOT="$(cd "${RAPG_SCRIPT_DIR}/../../../.." && pwd)"
export RAPG_SOURCE_WORLDENGINE_ROOT="${RAPG_SOURCE_WORLDENGINE_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine}"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export ALGENGINE_ENV="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}"
export ALGENGINE_PYTHON="${ALGENGINE_ENV}/bin/python"
export PATH="${ALGENGINE_ENV}/bin:${PATH}"
export DIFFUSIONDRIVE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export MMCV_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/mmcv
export NUPLAN_DEVKIT_ROOT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/nuplan-devkit
export SENIOR_NAVSIM_PARENT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/E2E/navsim_v1
export DIFFUSIONDRIVE_BOOTSTRAP="${SIMENGINE_ROOT}/scripts/diffusiondrive/algengine_worker_bootstrap"
export WORLDENGINE_DIFFUSIONDRIVE_MMCV_BOOTSTRAP=1
export WORLDENGINE_MMCV_EXTENSION="${RAPG_SOURCE_WORLDENGINE_ROOT}/artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${MMCV_ROOT}:${NUPLAN_DEVKIT_ROOT}:${SENIOR_NAVSIM_PARENT}:${PYTHONPATH:-}"

export RAPG_CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_rapg.py"
export DIFFUSIONDRIVE_GRPO_CONFIG="${RAPG_CONFIG}"
export RAPG_BASELINE=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth
export RAPG_BASELINE_SHA256=1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514
export RAPG_DATA_ROOT="${RAPG_SOURCE_WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data"
export RAPG_REAL_CACHE_ROOT="${RAPG_SOURCE_WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/full"
export RAPG_EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_rapg_v1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

rapg_real_cache_args() {
    printf '%s\n' \
        "--real-cache" "0=${RAPG_REAL_CACHE_ROOT}/train_seed0/cache.pt" \
        "--real-cache" "1=${RAPG_REAL_CACHE_ROOT}/train_seed1/cache.pt" \
        "--real-cache" "2=${RAPG_REAL_CACHE_ROOT}/train_seed2/cache.pt"
}

rapg_ensure_link() {
    local target="$1"
    local source="$2"
    if [[ -L "${target}" ]]; then
        [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
            echo "Wrong isolated-worktree link: ${target}" >&2
            return 1
        }
        return
    fi
    [[ ! -e "${target}" ]] || {
        echo "Refusing to replace existing isolated-worktree path: ${target}" >&2
        return 1
    }
    ln -s "${source}" "${target}"
}

rapg_preflight() {
    local expected_gpus="$1"
    local hardware_contract="${2:-hopper}"
    [[ "${hardware_contract}" == "hopper" || "${hardware_contract}" == "smoke" ]] || {
        echo "Unknown RAPG hardware contract: ${hardware_contract}" >&2
        return 1
    }
    rapg_ensure_link "${WORLDENGINE_ROOT}/data" "${RAPG_SOURCE_WORLDENGINE_ROOT}/data"
    local required=(
        "${ALGENGINE_PYTHON}" "${RAPG_CONFIG}" "${RAPG_BASELINE}"
        "${RAPG_DATA_ROOT}/manifest.json" "${RAPG_DATA_ROOT}/hard_pool.jsonl"
        "${RAPG_DATA_ROOT}/synthetic_cache.pt"
        "${RAPG_REAL_CACHE_ROOT}/train_seed0/cache.pt"
        "${RAPG_REAL_CACHE_ROOT}/train_seed1/cache.pt"
        "${RAPG_REAL_CACHE_ROOT}/train_seed2/cache.pt"
    )
    for path in "${required[@]}"; do
        [[ -e "${path}" ]] || { echo "Missing RAPG dependency: ${path}" >&2; return 1; }
    done
    [[ "$(sha256sum "${RAPG_BASELINE}" | awk '{print $1}')" == "${RAPG_BASELINE_SHA256}" ]] || {
        echo "Epoch-100 baseline SHA256 mismatch" >&2
        return 1
    }
    mkdir -p "${RAPG_EXPERIMENT_ROOT}"
    "${ALGENGINE_PYTHON}" - "${expected_gpus}" "${hardware_contract}" <<'PY'
import json,sys,torch
expected=int(sys.argv[1])
hardware_contract=sys.argv[2]
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names)!=expected:
    raise RuntimeError(f"expected {expected} visible GPUs, got {len(names)}")
if hardware_contract == "hopper":
    if any(not any(family in name.upper() for family in ("H100","H200")) for name in names):
        raise RuntimeError(f"RAPG contract requires Hopper GPUs, got {names}")
    if any(torch.cuda.get_device_capability(i)!=(9,0) for i in range(len(names))):
        raise RuntimeError("RAPG contract requires sm_90")
print(json.dumps({"status":"PASS","hardware_contract":hardware_contract,"devices":names,"torch":torch.__version__},sort_keys=True))
PY
}
