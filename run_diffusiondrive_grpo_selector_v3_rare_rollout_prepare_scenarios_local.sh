#!/usr/bin/env bash
# CPU/local preparation of the three immutable SimEngine scenario lanes.

set -Eeo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
ALGENGINE_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_PYTHON:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}"
SIMENGINE_PYTHON="${DIFFUSIONDRIVE_SIMENGINE_PYTHON:-/root/miniconda3/envs/simengine/bin/python}"
DIFFUSIONDRIVE_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"
NUPLAN_DEVKIT_ROOT="/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/nuplan-devkit"
export WORLDENGINE_ROOT ALGENGINE_ROOT SIMENGINE_ROOT
export PYTHONPATH="${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${NUPLAN_DEVKIT_ROOT}:${PYTHONPATH:-}"

RARE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1/rare_data"
SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_rollout_v1"
SCENARIO_ROOT="${SOURCE_ROOT}/scenarios"
CONVERTER="${SIMENGINE_ROOT}/worldengine/utils/dataset_utils/nuplan/digitaltwin_nuplan_converter_navsim_filter.py"
AUDITOR="${ALGENGINE_ROOT}/scripts/diffusiondrive/prepare_grpo_selector_v3_rare_rollout_scenarios.py"
RARE_FILTER="${RARE_ROOT}/rare_tokens.yaml"
RARE_PAIRS="${RARE_ROOT}/pairs.jsonl"
RARE_AUDIT="${RARE_ROOT}/rare_data_audit.json"
ASSET_ROOT="${WORLDENGINE_ROOT}/data/sim_engine/assets/navtrain"
NUPLAN_ROOT="${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/nuplan-v1.1"
NUPLAN_DB_ROOT="${NUPLAN_ROOT}/splits/all_sensor"
NUPLAN_MAP_ROOT="${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/maps"
OPENSCENE_ROOT="${WORLDENGINE_ROOT}/data/raw/openscene-v1.1"
OUTPUT_AUDIT="${SCENARIO_ROOT}/rare_rollout_scenario_audit.json"
LOG_DIR="${SOURCE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/prepare_local_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=static_preflight
trap 'rc=$?; echo "FAIL rare-rollout local preparation stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
for path in "${CONVERTER}" "${AUDITOR}" "${RARE_FILTER}" "${RARE_PAIRS}" "${RARE_AUDIT}"; do
    [[ -f "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
[[ -d "${ASSET_ROOT}/configs" && -d "${ASSET_ROOT}/assets" ]] || {
    echo "Missing full-navtrain DigitalTwin assets: ${ASSET_ROOT}" >&2
    exit 1
}
for path in "${NUPLAN_ROOT}" "${NUPLAN_DB_ROOT}" "${NUPLAN_MAP_ROOT}" "${OPENSCENE_ROOT}"; do
    [[ -e "${path}" ]] || { echo "Missing conversion data root: ${path}" >&2; exit 1; }
done
[[ -x "${ALGENGINE_PYTHON}" ]] || { echo "Missing AlgEngine Python" >&2; exit 1; }
[[ -x "${SIMENGINE_PYTHON}" ]] || { echo "Missing SimEngine Python" >&2; exit 1; }

if [[ -f "${OUTPUT_AUDIT}" ]]; then
    CURRENT_STAGE=reuse_audit
    if "${ALGENGINE_PYTHON}" "${AUDITOR}" \
        --converter-manifest "${SCENARIO_ROOT}/scenario_shards_manifest.json" \
        --rare-pairs "${RARE_PAIRS}" \
        --rare-data-audit "${RARE_AUDIT}" \
        --rare-filter "${RARE_FILTER}" \
        --asset-config-root "${ASSET_ROOT}/configs" \
        --expected-lanes 3 --expected-scenarios 6271 \
        --output "${OUTPUT_AUDIT}"; then
        trap - ERR
        echo "SKIP verified rare-rollout scenario preparation"
        echo "audit: ${OUTPUT_AUDIT}"
        exit 0
    fi
    echo "Existing scenario audit failed revalidation; rebuilding from chunks" >&2
fi

CURRENT_STAGE=convert_rare_scenarios
mkdir -p "${SCENARIO_ROOT}"
cd "${SIMENGINE_ROOT}"
"${SIMENGINE_PYTHON}" "${CONVERTER}" \
    --digitaltwin-asset-root "${ASSET_ROOT}" \
    --navsim-filters "${RARE_FILTER}" \
    --nuplan-root-path "${NUPLAN_ROOT}" \
    --nuplan-db-path "${NUPLAN_DB_ROOT}" \
    --nuplan-map-root "${NUPLAN_MAP_ROOT}" \
    --openscene-dataroot "${OPENSCENE_ROOT}" \
    --out-dir "${SCENARIO_ROOT}" \
    --num-processes "${DIFFUSIONDRIVE_RARE_ROLLOUT_CONVERTER_PROCESSES:-1}" \
    --num-splits 3 \
    --shards-only \
    --resume-chunks \
    --expected-scenarios 6271

CURRENT_STAGE=audit_scenario_lanes
"${ALGENGINE_PYTHON}" "${AUDITOR}" \
    --converter-manifest "${SCENARIO_ROOT}/scenario_shards_manifest.json" \
    --rare-pairs "${RARE_PAIRS}" \
    --rare-data-audit "${RARE_AUDIT}" \
    --rare-filter "${RARE_FILTER}" \
    --asset-config-root "${ASSET_ROOT}/configs" \
    --expected-lanes 3 --expected-scenarios 6271 \
    --output "${OUTPUT_AUDIT}"

CURRENT_STAGE=complete
trap - ERR
echo "PASS DiffusionDrive V3 rare-rollout local scenario preparation"
echo "scenario_root: ${SCENARIO_ROOT}"
echo "audit: ${OUTPUT_AUDIT}"
echo "persistent_log: ${LOG_FILE}"
