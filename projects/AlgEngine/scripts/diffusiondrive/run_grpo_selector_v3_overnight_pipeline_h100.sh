#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
. "${SCRIPT_DIR}/grpo_selector_v3_experiment_env.sh"
CACHE_ROOT="${V3_ROOT}/cache"
OUTPUT_ROOT="${V3_ROOT}/overnight_pipeline"
STATUS_FILE="${OUTPUT_ROOT}/status.txt"
LOG_DIR="${OUTPUT_ROOT}/logs"
WAIT_SECONDS="${V3_WAIT_INTERVAL_SECONDS:-60}"
TIMEOUT_SECONDS="${V3_WAIT_TIMEOUT_SECONDS:-21600}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
printf 'RUNNING attempt_id=%s code_sha=%s started_utc=%s\n' \
    "${ATTEMPT_ID}" "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${STATUS_FILE}"
CURRENT_STAGE=wait_cache_bundles
trap 'rc=$?; printf "FAIL attempt_id=%s code_sha=%s stage=%s exit=%s\n" "${ATTEMPT_ID}" "${CODE_SHA}" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL V3 overnight pipeline attempt_id=${ATTEMPT_ID} code_sha=${CODE_SHA} stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2' ERR

wait_for_bundle() {
    local bundle="$1"
    local status="${CACHE_ROOT}/bundle${bundle}.status"
    local elapsed=0
    while true; do
        if [[ -f "${status}" ]] && grep -q '^PASS ' "${status}"; then
            echo "PASS prerequisite bundle ${bundle}: $(cat "${status}")"
            return
        fi
        if [[ -f "${status}" ]] && grep -q '^FAIL ' "${status}"; then
            echo "Bundle ${bundle} failed: $(cat "${status}")" >&2
            return 1
        fi
        if (( elapsed >= TIMEOUT_SECONDS )); then
            echo "Timed out waiting for bundle ${bundle} after ${elapsed}s" >&2
            return 1
        fi
        echo "WAIT bundle=${bundle} elapsed=${elapsed}s status=$(cat "${status}" 2>/dev/null || echo MISSING)"
        sleep "${WAIT_SECONDS}"
        elapsed=$((elapsed + WAIT_SECONDS))
    done
}

for bundle in 0 1 2; do
    wait_for_bundle "${bundle}"
done

for seed in 0 1 2; do
    [[ -f "${CACHE_ROOT}/train_seed${seed}/cache.pt" ]] || {
        echo "Missing train cache seed ${seed}" >&2; exit 1;
    }
done
for seed in 3 4 5; do
    [[ -f "${CACHE_ROOT}/development_seed${seed}/cache.pt" ]] || {
        echo "Missing development cache seed ${seed}" >&2; exit 1;
    }
done
for seed in 6 7 8; do
    [[ -f "${CACHE_ROOT}/certification_seed${seed}/cache.pt" ]] || {
        echo "Missing certification cache seed ${seed}" >&2; exit 1;
    }
done

CURRENT_STAGE=h100_optimizer_preflight
"${SCRIPT_DIR}/run_grpo_selector_v3_optimizer_preflight_h100.sh"

CURRENT_STAGE=sweep
"${SCRIPT_DIR}/run_grpo_selector_v3_sweep_h100.sh"

CURRENT_STAGE=certification
"${SCRIPT_DIR}/run_grpo_selector_v3_certify_h100.sh"

CURRENT_STAGE=complete
printf 'PASS attempt_id=%s code_sha=%s completed_utc=%s\n' \
    "${ATTEMPT_ID}" "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive selector V3 overnight pipeline"
echo "selection: ${V3_ROOT}/sweep/selection.json"
echo "certification: ${V3_ROOT}/certification/report.json"
echo "persistent_log: ${LOG_FILE}"
