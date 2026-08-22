#!/usr/bin/env bash
# Merge the frozen reactive cache with three completed BWM collection lanes.

set -Eeo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"
ALGENGINE_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_PYTHON:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_bwm_v1"
BASE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1"
REAL_CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/full"
SCENARIO_AUDIT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_rollout_bwm_v1/scenarios/bwm_scenario_audit.json"
BUILDER="${SCRIPT_DIR}/build_grpo_selector_v3_rare_rollout_bwm_data.py"
mkdir -p "${ROOT}/logs"
LOG_FILE="${ROOT}/logs/build_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=build_bwm_mixture
trap 'rc=$?; echo "FAIL BWM data build stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
args=(
    --base-manifest "${BASE_ROOT}/data/manifest.json"
    --scenario-audit "${SCENARIO_AUDIT}"
    --real-cache "0=${REAL_CACHE_ROOT}/train_seed0/cache.pt"
    --real-cache "1=${REAL_CACHE_ROOT}/train_seed1/cache.pt"
    --real-cache "2=${REAL_CACHE_ROOT}/train_seed2/cache.pt"
    --output-dir "${ROOT}/data"
    --minimum-new-synthetic 1
)
for lane in 0 1 2; do
    args+=(
        --lane-root "${ROOT}/collection/formal/lane${lane}"
        --lane-audit "${ROOT}/collection/formal/lane${lane}/collection_audit.json"
    )
done
"${ALGENGINE_PYTHON}" "${BUILDER}" "${args[@]}"

CURRENT_STAGE=complete
trap - ERR
echo "PASS DiffusionDrive V3 rare-rollout BWM data build"
echo "manifest: ${ROOT}/data/manifest.json"
echo "persistent_log: ${LOG_FILE}"
