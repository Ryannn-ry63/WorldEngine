#!/usr/bin/env bash
# One-command gate for a one-H100 instance:
# prepare/audit the three formal scenario lanes, then run the online smoke.

set -Eeo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1"
LOG_DIR="${ROOT}/logs"
STATUS_FILE="${ROOT}/prepare_smoke_status.txt"
CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/prepare_smoke_1h100_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=prepare_scenarios
printf 'RUNNING code_sha=%s stage=%s started_utc=%s\n' \
    "${CODE_SHA}" "${CURRENT_STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap 'rc=$?; printf "FAIL code_sha=%s stage=%s exit=%s completed_utc=%s\n" "${CODE_SHA}" "${CURRENT_STAGE}" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"; echo "FAIL one-H100 rare-rollout prepare+smoke stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

cd "${WORLDENGINE_ROOT}"
echo "[1/2] Preparing and auditing three rare-rollout scenario lanes..."
"${WORLDENGINE_ROOT}/run_diffusiondrive_grpo_selector_v3_rare_rollout_prepare_scenarios_local.sh"

CURRENT_STAGE=end_to_end_smoke
echo "[2/2] Running one-H100 rare-rollout end-to-end smoke..."
"${WORLDENGINE_ROOT}/run_diffusiondrive_grpo_selector_v3_rare_rollout_1h100_smoke.sh"

CURRENT_STAGE=complete
printf 'PASS code_sha=%s completed_utc=%s\n' \
    "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS one-H100 DiffusionDrive V3 rare-rollout prepare + smoke"
echo "status: ${STATUS_FILE}"
echo "persistent_log: ${LOG_FILE}"
