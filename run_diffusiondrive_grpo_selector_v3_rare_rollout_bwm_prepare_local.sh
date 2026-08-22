#!/usr/bin/env bash
# Prepare three immutable BWM scenario lanes without loading a model or GPU.

set -Eeo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"
ALGENGINE_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_PYTHON:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}"
SENIOR_ROOT="/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine"
SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_rollout_bwm_v1"
SCENARIO_ROOT="${SOURCE_ROOT}/scenarios"
PAIR_MANIFEST="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1/rare_data/pairs.jsonl"
NAVTEST_FILTER="${WORLDENGINE_ROOT}/projects/AlgEngine/configs/navsim_splits/navtest_split/navtest.yaml"
PREPARER="${SCRIPT_DIR}/prepare_grpo_selector_v3_rare_rollout_bwm_scenarios.py"
mkdir -p "${SOURCE_ROOT}/logs"
LOG_FILE="${SOURCE_ROOT}/logs/prepare_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=prepare_bwm_scenarios
trap 'rc=$?; echo "FAIL BWM scenario preparation stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
"${ALGENGINE_PYTHON}" "${PREPARER}" \
    --source "bwm_collision=${SENIOR_ROOT}/data/sim_engine/scenarios/augmented/navtrain_50pct_collision/all_scenarios.pkl" \
    --source "bwm_low_ep=${SENIOR_ROOT}/data/sim_engine/scenarios/augmented/navtrain_50pct_ep_1pct/all_scenarios.pkl" \
    --source "bwm_offroad=${SENIOR_ROOT}/data/sim_engine/scenarios/augmented/navtrain_50pct_off_road/all_scenarios.pkl" \
    --pair-manifest "${PAIR_MANIFEST}" --navtest-filter "${NAVTEST_FILTER}" \
    --output-dir "${SCENARIO_ROOT}" --num-lanes 3

CURRENT_STAGE=complete
trap - ERR
echo "PASS DiffusionDrive V3 BWM scenario preparation"
echo "audit: ${SCENARIO_ROOT}/bwm_scenario_audit.json"
echo "persistent_log: ${LOG_FILE}"
