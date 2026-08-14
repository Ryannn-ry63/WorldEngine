#!/usr/bin/env bash
# One-H100 smoke for the frozen rollout-v1 finish path. Never reads certification.

set -Eeo pipefail

WORLDENGINE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
EXPECTED_TAG=diffusiondrive-selector-grpo-rollout-v1-finish-ready-r2-20260814
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1"
AUDITOR="${SCRIPT_DIR}/audit_grpo_selector_rollout_v1_finish.py"
METHOD=scene_conditioned_exact_group_grpo_rollout_v1

cd "${WORLDENGINE_ROOT}"
TRACKED_STATUS="$(git status --porcelain --untracked-files=no)"
if [[ -n "${TRACKED_STATUS}" ]]; then
    echo "Refusing one-H100 smoke from a dirty tracked worktree:" >&2
    echo "${TRACKED_STATUS}" >&2
    exit 1
fi
EXPECTED_SHA="$(git rev-list -n 1 "${EXPECTED_TAG}" 2>/dev/null)" || {
    echo "Missing frozen smoke tag: ${EXPECTED_TAG}" >&2
    exit 1
}
ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
    echo "HEAD ${ACTUAL_SHA} != frozen ${EXPECTED_TAG} ${EXPECTED_SHA}" >&2
    exit 1
fi

export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

SMOKE_ID="smoke_$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="${ROOT}/local_finish_smoke/${SMOKE_ID}"
TRAIN_ROOT="${OUTPUT}/train"
CHECKPOINT="${OUTPUT}/checkpoint.pth"
MANIFEST="${OUTPUT}/checkpoint_manifest.json"
STATUS="${OUTPUT}/status.txt"
mkdir -p "${TRAIN_ROOT}"
LOG_FILE="${OUTPUT}/smoke.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=input_audit
trap 'rc=$?; echo "FAIL stage=${CURRENT_STAGE} exit=${rc}" > "${STATUS}"; echo "FAIL one-H100 finish smoke stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

"${ALGENGINE_PYTHON}" "${AUDITOR}" --root "${ROOT}" \
    --worldengine-root "${WORLDENGINE_ROOT}" --stage inputs

CURRENT_STAGE=h100_optimizer_preflight
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/preflight_grpo_selector_v3_h100_optimizer.py" \
    --ablation full

CURRENT_STAGE=real_cache_one_epoch
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/train_grpo_selector_v3_cached.py" \
    --train-cache "${ROOT}/cache/train_seed0/cache.pt" \
    --train-cache "${ROOT}/cache/train_seed1/cache.pt" \
    --train-cache "${ROOT}/cache/train_seed2/cache.pt" \
    --development-cache "${ROOT}/cache/development_seed3/cache.pt" \
    --development-cache "${ROOT}/cache/development_seed4/cache.pt" \
    --development-cache "${ROOT}/cache/development_seed5/cache.pt" \
    --output-dir "${TRAIN_ROOT}" --temperature 1.0 \
    --learning-rate 1e-4 --kl-weight 0 --seed 314159 \
    --epochs 1 --checkpoint-epochs 1 --batch-size 64 \
    --device cuda --ablation full --method-name "${METHOD}"

SELECTOR_STATE="${TRAIN_ROOT}/epoch_1_scene_selector.pt"
SELECTOR_SHA="$(sha256sum "${SELECTOR_STATE}" | awk '{print $1}')"
CURRENT_STAGE=checkpoint_materialize
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --scene-selector-state "${SELECTOR_STATE}" \
    --expected-selector-sha256 "${SELECTOR_SHA}" \
    --expected-method "${METHOD}" \
    --release-name e2e_diffusiondrive_grpo_selector_rollout_v1_local_smoke \
    --output "${CHECKPOINT}" --manifest "${MANIFEST}"

CURRENT_STAGE=checkpoint_audit
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}/checkpoint_audit.json"

CURRENT_STAGE=complete
echo "PASS" > "${STATUS}"
trap - ERR
echo "PASS one-H100 rollout-v1 finish smoke"
echo "output: ${OUTPUT}"
echo "persistent_log: ${LOG_FILE}"
