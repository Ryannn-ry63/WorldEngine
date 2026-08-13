#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1"
CACHE_ROOT="${ROOT}/cache"
SELECTION="${ROOT}/development/selection.json"
OUTPUT="${ROOT}/certification"
METHOD=scene_conditioned_exact_group_grpo_rollout_v1
mkdir -p "${OUTPUT}/logs"
LOG_FILE="${OUTPUT}/logs/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'rc=$?; echo "FAIL exit=${rc}" > "${OUTPUT}/status.txt"; echo "FAIL rollout-v1 certification line=${LINENO} command=${BASH_COMMAND}" >&2' ERR
[[ -f "${SELECTION}" ]] || { echo "Missing development selection" >&2; exit 1; }
for seed in 6 7 8; do
    [[ -f "${CACHE_ROOT}/certification_seed${seed}/cache.pt" ]] || {
        echo "Missing certification cache seed${seed}" >&2; exit 1;
    }
done

CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/certify_grpo_selector_v3.py" \
    --selection "${SELECTION}" \
    --certification-cache "${CACHE_ROOT}/certification_seed6/cache.pt" \
    --certification-cache "${CACHE_ROOT}/certification_seed7/cache.pt" \
    --certification-cache "${CACHE_ROOT}/certification_seed8/cache.pt" \
    --output "${OUTPUT}/report.json" --device cuda --batch-size 64 \
    --bootstrap-replicates 10000
SELECTOR_STATE="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["scene_selector_state"])' "${SELECTION}")"
SELECTOR_SHA="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["scene_selector_state_sha256"])' "${SELECTION}")"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --scene-selector-state "${SELECTOR_STATE}" \
    --expected-selector-sha256 "${SELECTOR_SHA}" \
    --expected-method "${METHOD}" \
    --release-name e2e_diffusiondrive_grpo_selector_rollout_v1 \
    --output "${OUTPUT}/selected_checkpoint.pth" \
    --manifest "${OUTPUT}/checkpoint_manifest.json"
echo "PASS" > "${OUTPUT}/status.txt"
trap - ERR
echo "PASS DiffusionDrive selector rollout-v1 certification"
echo "checkpoint: ${OUTPUT}/selected_checkpoint.pth"
echo "persistent_log: ${LOG_FILE}"
