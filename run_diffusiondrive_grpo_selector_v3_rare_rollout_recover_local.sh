#!/usr/bin/env bash
# CPU-only recovery for the three completed formal collection lanes.

set -Eeo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
ALGENGINE_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_PYTHON:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}"
SIMENGINE_PYTHON="${DIFFUSIONDRIVE_SIMENGINE_PYTHON:-/root/miniconda3/envs/simengine/bin/python}"
COLLECTION_CODE_SHA="${DIFFUSIONDRIVE_RARE_ROLLOUT_COLLECTION_CODE_SHA:-4f79dfaa16e16c60dd1650d2e53cf6fb915dc952}"
NOISE_NAMESPACE="diffusiondrive_v3_rare_rollout_v1"
SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_rollout_v1"
SCENARIO_ROOT="${SOURCE_ROOT}/scenarios"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1"
COLLECTION_ROOT="${ROOT}/collection/formal"
AUDITOR="${ALGENGINE_ROOT}/scripts/diffusiondrive/audit_grpo_selector_v3_rare_rollout_collection.py"
MERGER="${SIMENGINE_ROOT}/scripts/merge_simulation_results.py"
LOG_DIR="${ROOT}/logs"
STATUS="${ROOT}/recover_collection_status.txt"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/recover_collection_local_$(date -u +%Y%m%dT%H%M%SZ).log"
export PYTHONPATH="${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${PYTHONPATH:-}"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=static_preflight
RECOVERY_CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
printf 'RUNNING recovery_code_sha=%s collection_code_sha=%s\n' \
    "${RECOVERY_CODE_SHA}" "${COLLECTION_CODE_SHA}" > "${STATUS}"
trap 'rc=$?; printf "FAIL stage=%s exit=%s recovery_code_sha=%s collection_code_sha=%s\n" "${CURRENT_STAGE}" "${rc}" "${RECOVERY_CODE_SHA}" "${COLLECTION_CODE_SHA}" > "${STATUS}"; echo "FAIL rare-rollout local collection recovery stage=${CURRENT_STAGE} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

for path in "${ALGENGINE_PYTHON}" "${SIMENGINE_PYTHON}" "${AUDITOR}" "${MERGER}"; do
    [[ -e "${path}" ]] || { echo "Missing recovery dependency: ${path}" >&2; exit 1; }
done
[[ "${COLLECTION_CODE_SHA}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "Invalid collection code SHA: ${COLLECTION_CODE_SHA}" >&2
    exit 2
}

for lane in 0 1 2; do
    lane_root="${COLLECTION_ROOT}/lane${lane}"
    scenario_file="${SCENARIO_ROOT}/scenario_shard_0${lane}_of_03.pkl"
    [[ -d "${lane_root}" && -f "${scenario_file}" ]] || {
        echo "Missing lane ${lane} inputs" >&2
        exit 1
    }

    CURRENT_STAGE="lane${lane}_premerge_audit"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" \
        --rollout-root "${lane_root}" \
        --scenario-file "${scenario_file}" \
        --layout split \
        --expected-noise-namespace "${NOISE_NAMESPACE}" \
        --expected-code-sha "${COLLECTION_CODE_SHA}" \
        --expected-workers 8 \
        --output "${lane_root}/premerge_collection_audit.json"

    CURRENT_STAGE="lane${lane}_merge"
    "${SIMENGINE_PYTHON}" "${MERGER}" \
        --test_path "${lane_root}" --react_type NR --num-splits 8

    CURRENT_STAGE="lane${lane}_final_audit"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" \
        --rollout-root "${lane_root}" \
        --scenario-file "${scenario_file}" \
        --layout merged \
        --expected-noise-namespace "${NOISE_NAMESPACE}" \
        --expected-code-sha "${COLLECTION_CODE_SHA}" \
        --expected-workers 8 \
        --output "${lane_root}/collection_audit.json"
    echo "PASS recovered rare-rollout collection lane ${lane}"
done

CURRENT_STAGE=complete
printf 'PASS recovery_code_sha=%s collection_code_sha=%s\n' \
    "${RECOVERY_CODE_SHA}" "${COLLECTION_CODE_SHA}" > "${STATUS}"
trap - ERR
echo "PASS DiffusionDrive V3 rare-rollout local collection recovery"
echo "status: ${STATUS}"
echo "persistent_log: ${LOG_FILE}"
