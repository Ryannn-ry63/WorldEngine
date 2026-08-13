#!/usr/bin/env bash
# Formal rollout-v1 collection: one noise seed on exactly eight visible H100s.

set -Eeo pipefail

SEED="${1:?usage: bash run_diffusiondrive_rollout_v1_collect_8h100.sh SEED}"
if [[ ! "${SEED}" =~ ^[0-8]$ ]]; then
    echo "SEED must be one integer from 0 through 8" >&2
    exit 2
fi

WORLDENGINE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
EXPECTED_TAG=diffusiondrive-selector-grpo-rollout-v1-formal-collection-ready-20260813
DATA_TYPE=navtrain_50pct_collision
ASSET_NAME=navtrain
RUN_ID=r1full
LAUNCHER="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh"
ROLLOUT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1/rollouts/${DATA_TYPE}/seed${SEED}/${RUN_ID}"

cd "${WORLDENGINE_ROOT}"
[[ -x "${LAUNCHER}" ]] || { echo "Missing rollout launcher: ${LAUNCHER}" >&2; exit 1; }

TRACKED_STATUS="$(git status --porcelain --untracked-files=no)"
if [[ -n "${TRACKED_STATUS}" ]]; then
    echo "Refusing formal collection from a dirty tracked worktree:" >&2
    echo "${TRACKED_STATUS}" >&2
    exit 1
fi
EXPECTED_SHA="$(git rev-list -n 1 "${EXPECTED_TAG}" 2>/dev/null)" || {
    echo "Missing frozen formal-collection tag: ${EXPECTED_TAG}" >&2
    exit 1
}
ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo "HEAD ${ACTUAL_SHA} != frozen ${EXPECTED_TAG} ${EXPECTED_SHA}" >&2
    exit 1
fi
if [[ -e "${ROLLOUT_ROOT}" ]]; then
    echo "Immutable formal rollout target already exists: ${ROLLOUT_ROOT}" >&2
    exit 1
fi

echo "START formal DiffusionDrive rollout-v1 collection"
echo "seed=${SEED} data_type=${DATA_TYPE} asset_name=${ASSET_NAME} run_id=${RUN_ID}"
echo "code_sha=${ACTUAL_SHA} frozen_tag=${EXPECTED_TAG}"
echo "Each formal seed requires exactly eight visible H100 GPUs."

exec "${LAUNCHER}" collect "${SEED}" "${DATA_TYPE}" "${ASSET_NAME}" "${RUN_ID}"
