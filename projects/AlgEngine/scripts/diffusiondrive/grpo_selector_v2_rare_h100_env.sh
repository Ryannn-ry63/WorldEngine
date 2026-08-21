#!/usr/bin/env bash
# Reuse the senior environment while keeping code in the isolated V2 worktree.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

export WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export DIFFUSIONDRIVE_BOOTSTRAP="${SIMENGINE_ROOT}/scripts/diffusiondrive/algengine_worker_bootstrap"
export WORLDENGINE_MMCV_EXTENSION="${WORLDENGINE_ROOT}/artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so"
export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${MMCV_ROOT}:${NUPLAN_DEVKIT_ROOT}:${SENIOR_NAVSIM_PARENT}:${PYTHONPATH:-}"
