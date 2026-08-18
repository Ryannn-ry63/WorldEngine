#!/usr/bin/env bash
set -Eeo pipefail

SEED="${1:?usage: run_grpo_selector_v3_formal_wait_h100.sh SEED}"
if [[ ! "${SEED}" =~ ^[012]$ ]]; then
    echo "SEED must be 0, 1, or 2" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
. "${SCRIPT_DIR}/grpo_selector_v3_experiment_env.sh"
PIPELINE_STATUS="${V3_ROOT}/overnight_pipeline/status.txt"
CERT_STATUS="${V3_ROOT}/certification/status.txt"
CHECKPOINT="${V3_ROOT}/certification/selected_checkpoint.pth"
MANIFEST="${V3_ROOT}/certification/checkpoint_manifest.json"
WAIT_SECONDS="${V3_WAIT_INTERVAL_SECONDS:-60}"
TIMEOUT_SECONDS="${V3_FORMAL_WAIT_TIMEOUT_SECONDS:-43200}"
EXPECTED_CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
LOG_DIR="${V3_ROOT}/formal_wait/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

elapsed=0
while true; do
    if [[ -f "${CERT_STATUS}" ]] && grep -q '^PASS ' "${CERT_STATUS}" \
        && [[ -f "${CHECKPOINT}" && -f "${MANIFEST}" ]]; then
        echo "PASS certification prerequisite for formal seed ${SEED}"
        break
    fi
    if [[ -f "${PIPELINE_STATUS}" ]] \
        && grep -q '^FAIL ' "${PIPELINE_STATUS}" \
        && grep -q "code_sha=${EXPECTED_CODE_SHA}" "${PIPELINE_STATUS}"; then
        echo "Current-code pipeline failed: $(cat "${PIPELINE_STATUS}")" >&2
        exit 1
    fi
    if (( elapsed >= TIMEOUT_SECONDS )); then
        echo "Timed out waiting for certification after ${elapsed}s" >&2
        exit 1
    fi
    echo "WAIT formal_seed=${SEED} elapsed=${elapsed}s pipeline=$(cat "${PIPELINE_STATUS}" 2>/dev/null || echo MISSING) certification=$(cat "${CERT_STATUS}" 2>/dev/null || echo MISSING)"
    sleep "${WAIT_SECONDS}"
    elapsed=$((elapsed + WAIT_SECONDS))
done

exec "${SCRIPT_DIR}/run_grpo_selector_v3_formal_seed_h100.sh" "${SEED}"
