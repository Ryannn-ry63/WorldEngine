#!/usr/bin/env bash
# Frozen entry point for the single-allocation rollout-v1 finish pipeline.

set -Eeo pipefail

MODE="${1:-run}"
if [[ "${MODE}" != "run" && "${MODE}" != "preflight" ]]; then
    echo "usage: ./run_diffusiondrive_rollout_v1_finish_8h100.sh [preflight|run]" >&2
    exit 2
fi

WORLDENGINE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
EXPECTED_TAG=diffusiondrive-selector-grpo-rollout-v1-finish-ready-r2-20260814
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"
PIPELINE="${SCRIPT_DIR}/run_grpo_selector_rollout_v1_finish_h100.sh"
AUDITOR="${SCRIPT_DIR}/audit_grpo_selector_rollout_v1_finish.py"
ALGENGINE_PYTHON=/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1"

cd "${WORLDENGINE_ROOT}"
for file in "${PIPELINE}" "${AUDITOR}" "${ALGENGINE_PYTHON}"; do
    [[ -e "${file}" ]] || { echo "Missing finish entry-point input: ${file}" >&2; exit 1; }
done

TRACKED_STATUS="$(git status --porcelain --untracked-files=no)"
if [[ -n "${TRACKED_STATUS}" ]]; then
    echo "Refusing rollout-v1 finish from a dirty tracked worktree:" >&2
    echo "${TRACKED_STATUS}" >&2
    exit 1
fi
EXPECTED_SHA="$(git rev-list -n 1 "${EXPECTED_TAG}" 2>/dev/null)" || {
    echo "Missing frozen finish tag: ${EXPECTED_TAG}" >&2
    exit 1
}
ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo "HEAD ${ACTUAL_SHA} != frozen ${EXPECTED_TAG} ${EXPECTED_SHA}" >&2
    exit 1
fi

if [[ "${MODE}" == "preflight" ]]; then
    bash -n "${BASH_SOURCE[0]}" "${PIPELINE}" \
        "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_development_h100.sh" \
        "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_certify_h100.sh" \
        "${SCRIPT_DIR}/run_grpo_selector_rollout_v1_formal_seed_h100.sh"
    "${ALGENGINE_PYTHON}" -m py_compile "${AUDITOR}"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
        --worldengine-root "${WORLDENGINE_ROOT}" --stage inputs
    echo "PASS local rollout-v1 finish preflight"
    echo "code_sha=${ACTUAL_SHA} frozen_tag=${EXPECTED_TAG}"
    exit 0
fi

echo "START rollout-v1 development + one-shot certification + formal seeds 0,1,2"
echo "code_sha=${ACTUAL_SHA} frozen_tag=${EXPECTED_TAG}"
echo "This command requires exactly eight visible H100 GPUs and runs all formal seeds sequentially."
exec "${PIPELINE}"
