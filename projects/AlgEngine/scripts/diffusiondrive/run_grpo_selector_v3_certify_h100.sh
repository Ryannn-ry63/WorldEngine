#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3"
CACHE_ROOT="${ROOT}/cache"
SWEEP_ROOT="${ROOT}/sweep"
OUTPUT_ROOT="${ROOT}/certification"
SELECTION="${SWEEP_ROOT}/selection.json"
STATUS_FILE="${OUTPUT_ROOT}/status.txt"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
printf 'RUNNING started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
CURRENT_STAGE=preflight
trap 'rc=$?; printf "FAIL stage=%s exit=%s\n" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2' ERR

for file in "${SELECTION}" \
    "${CACHE_ROOT}/certification_seed6/cache.pt" \
    "${CACHE_ROOT}/certification_seed7/cache.pt" \
    "${CACHE_ROOT}/certification_seed8/cache.pt" \
    "${SCRIPT_DIR}/certify_grpo_selector_v3.py" \
    "${SCRIPT_DIR}/materialize_grpo_selector_v3.py"; do
    [[ -f "${file}" ]] || { echo "Missing V3 certification input: ${file}" >&2; exit 1; }
done

CURRENT_STAGE=one_shot_certification
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/certify_grpo_selector_v3.py" \
    --selection "${SELECTION}" \
    --certification-cache "${CACHE_ROOT}/certification_seed6/cache.pt" \
    --certification-cache "${CACHE_ROOT}/certification_seed7/cache.pt" \
    --certification-cache "${CACHE_ROOT}/certification_seed8/cache.pt" \
    --output "${OUTPUT_ROOT}/report.json" --device cuda \
    --batch-size 64 --bootstrap-replicates 10000

CURRENT_STAGE=materialize_checkpoint
SELECTOR_STATE="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["scene_selector_state"])' "${SELECTION}")"
SELECTOR_SHA="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["scene_selector_state_sha256"])' "${SELECTION}")"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --scene-selector-state "${SELECTOR_STATE}" \
    --expected-selector-sha256 "${SELECTOR_SHA}" \
    --output "${OUTPUT_ROOT}/selected_checkpoint.pth" \
    --manifest "${OUTPUT_ROOT}/checkpoint_manifest.json"

CURRENT_STAGE=complete
printf 'PASS completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive selector V3 certification and materialization"
echo "report: ${OUTPUT_ROOT}/report.json"
echo "checkpoint: ${OUTPUT_ROOT}/selected_checkpoint.pth"
echo "persistent_log: ${LOG_FILE}"
