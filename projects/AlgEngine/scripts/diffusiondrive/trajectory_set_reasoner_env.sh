#!/usr/bin/env bash
# Isolated H100/H200 runtime for trajectory_set_reasoner_grpo_v1.
set +u
PS1="${PS1:-trajectory-set-reasoner}"
. /inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/dotfiles/.bashrc
set -u
ulimit -c 0

REASONER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORLDENGINE_ROOT="$(cd "${REASONER_SCRIPT_DIR}/../../../.." && pwd)"
export REASONER_SOURCE_WORLDENGINE_ROOT="${REASONER_SOURCE_WORLDENGINE_ROOT:-/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine}"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export ALGENGINE_ENV="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}"
export ALGENGINE_PYTHON="${ALGENGINE_ENV}/bin/python"
export ALGENGINE_TORCHRUN="${ALGENGINE_ENV}/bin/torchrun"
export SIMENGINE_PYTHON="${SIMENGINE_PYTHON:-/root/miniconda3/envs/simengine/bin/python}"
export PATH="${ALGENGINE_ENV}/bin:${PATH}"
export DIFFUSIONDRIVE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive
export MMCV_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/mmcv
export NUPLAN_DEVKIT_ROOT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/nuplan-devkit
export SENIOR_NAVSIM_PARENT=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/E2E/navsim_v1
export DIFFUSIONDRIVE_BOOTSTRAP="${SIMENGINE_ROOT}/scripts/diffusiondrive/algengine_worker_bootstrap"
export WORLDENGINE_DIFFUSIONDRIVE_MMCV_BOOTSTRAP=1
export WORLDENGINE_MMCV_EXTENSION="${REASONER_SOURCE_WORLDENGINE_ROOT}/artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so"
export NAVSIM_EXP_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp
export NAVSIM_METRIC_CACHE_PATH_TRAIN="${NAVSIM_EXP_ROOT}/metric_cache_trainval"
export NAVSIM_METRIC_CACHE_PATH_EVAL="${NAVSIM_EXP_ROOT}/metric_cache_navtest"
export NAVSIM_METRIC_CACHE_PATH="${NAVSIM_METRIC_CACHE_PATH_EVAL}"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${MMCV_ROOT}:${NUPLAN_DEVKIT_ROOT}:${SENIOR_NAVSIM_PARENT}:${PYTHONPATH:-}"

export REASONER_CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_trajectory_set_reasoner.py"
export REASONER_BASELINE=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth
export REASONER_BASELINE_SHA256=1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514
export REASONER_SOURCE_ROOT="${REASONER_SOURCE_WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1"
export REASONER_PAIR_ROOT="${REASONER_SOURCE_ROOT}/rare_data"
export REASONER_TUNING_ROOT="${REASONER_SOURCE_ROOT}/tuning_split"
export REASONER_CACHE_ROOT="${REASONER_SOURCE_WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache"
export REASONER_METRIC_CACHE_INDEX="${REASONER_SOURCE_ROOT}/metric_cache_navtrain_full"
export REASONER_EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/trajectory_set_reasoner_grpo_v1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

reasoner_ensure_link() {
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

reasoner_setup_assets() {
    reasoner_ensure_link "${WORLDENGINE_ROOT}/data" "${REASONER_SOURCE_WORLDENGINE_ROOT}/data"
    mkdir -p "${REASONER_EXPERIMENT_ROOT}"
}

reasoner_hopper_preflight() {
    local expected_gpu_count="$1"
    reasoner_setup_assets
    local required=(
        "${ALGENGINE_PYTHON}" "${ALGENGINE_TORCHRUN}" "${REASONER_CONFIG}"
        "${REASONER_BASELINE}" "${WORLDENGINE_MMCV_EXTENSION}"
        "${REASONER_PAIR_ROOT}/pairs.jsonl"
        "${REASONER_PAIR_ROOT}/rare_data_audit.json"
    )
    for path in "${required[@]}"; do
        [[ -e "${path}" ]] || { echo "Missing reasoner dependency: ${path}" >&2; return 1; }
    done
    [[ "$(sha256sum "${REASONER_BASELINE}" | awk '{print $1}')" == "${REASONER_BASELINE_SHA256}" ]] || {
        echo "Epoch-100 SHA256 mismatch" >&2
        return 1
    }
    mkdir -p "${REASONER_EXPERIMENT_ROOT}"
    "${ALGENGINE_PYTHON}" - "${expected_gpu_count}" <<'PY'
import json,sys,torch
expected=int(sys.argv[1])
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names)!=expected:
    raise RuntimeError(f"expected {expected} visible GPUs, got {len(names)}")
families=[]
for name in names:
    matches=[value for value in ("H100","H200") if value in name.upper()]
    if len(matches)!=1:
        raise RuntimeError(f"formal hardware must be H100 or H200, got {name!r}")
    families.append(matches[0].lower())
if len(set(families))!=1:
    raise RuntimeError(f"mixed H100/H200 allocation is forbidden: {names}")
capabilities=[torch.cuda.get_device_capability(i) for i in range(len(names))]
if any(value!=(9,0) for value in capabilities):
    raise RuntimeError(f"expected sm_90 Hopper, got {capabilities}")
if torch.version.cuda!="11.8" or not str(torch.__version__).startswith("2.0.1+cu118"):
    raise RuntimeError(f"environment drift: torch={torch.__version__}, runtime={torch.version.cuda}")
print(json.dumps({"status":"PASS","hardware_family":families[0],"device_names":names,"capabilities":capabilities,"torch_version":torch.__version__,"torch_cuda_runtime":torch.version.cuda,"visible_device_count":len(names)},sort_keys=True))
PY
}
