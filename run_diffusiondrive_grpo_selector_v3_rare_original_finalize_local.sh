#!/usr/bin/env bash
set -Eeuo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"
ALGENGINE_PYTHON="${DIFFUSIONDRIVE_ALGENGINE_PYTHON_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}"
PREPARE="${SCRIPT_DIR}/prepare_grpo_selector_v3_rare_original_data.py"
SPLITTER="${SCRIPT_DIR}/split_grpo_selector_v3_rare_original.py"
SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1"
MINING_ROOT="${SOURCE_ROOT}/mining"
FULL_INDEX="${SOURCE_ROOT}/metric_cache_navtrain_full"
RARE_DATA_ROOT="${SOURCE_ROOT}/rare_data"
TUNING_SPLIT_ROOT="${SOURCE_ROOT}/tuning_split"
ANNOTATION="${WORLDENGINE_ROOT}/data/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl"
FULL_FILTER="${WORLDENGINE_ROOT}/projects/AlgEngine/configs/navsim_splits/navtrain_split/navtrain.yaml"
LOG_DIR="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/logs"
LOG_FILE="${LOG_DIR}/mine_finalize_local_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=inputs
trap 'rc=$?; echo "FAIL local rare-original finalize stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

for path in "${ALGENGINE_PYTHON}" "${PREPARE}" "${SPLITTER}" \
    "${FULL_INDEX}/metadata/metric_cache.csv" "${ANNOTATION}" "${FULL_FILTER}"; do
    [[ -e "${path}" ]] || {
        echo "Missing local-finalize input: ${path}" >&2
        exit 1
    }
done

for seed in 0 1 2; do
    seed_root="${MINING_ROOT}/seed${seed}"
    CURRENT_STAGE="audit_score_seed${seed}"
    "${ALGENGINE_PYTHON}" "${PREPARE}" audit-score \
        --score-csv "${seed_root}/official_scores/pdm_scores_merged.csv" \
        --submission "${seed_root}/navsim_submission.pkl" \
        --cache-index "${FULL_INDEX}" \
        --expected-tokens 103288 \
        --output "${seed_root}/score_audit.json"
done

CURRENT_STAGE=rare_common_mining
"${ALGENGINE_PYTHON}" "${PREPARE}" mine \
    --score "0=${MINING_ROOT}/seed0/official_scores/pdm_scores_merged.csv" \
    --score "1=${MINING_ROOT}/seed1/official_scores/pdm_scores_merged.csv" \
    --score "2=${MINING_ROOT}/seed2/official_scores/pdm_scores_merged.csv" \
    --cache-index "${FULL_INDEX}" \
    --annotation-file "${ANNOTATION}" \
    --base-filter "${FULL_FILTER}" \
    --output-dir "${RARE_DATA_ROOT}" \
    --expected-tokens 103288 \
    --ego-progress-percentile 1 \
    --pair-seed 20260818

CURRENT_STAGE=log_disjoint_tuning_split
"${ALGENGINE_PYTHON}" "${SPLITTER}" \
    --pair-manifest "${RARE_DATA_ROOT}/pairs.jsonl" \
    --rare-data-audit "${RARE_DATA_ROOT}/rare_data_audit.json" \
    --base-filter "${FULL_FILTER}" \
    --output-dir "${TUNING_SPLIT_ROOT}" \
    --split-seed 20260819 \
    --train-fraction 0.8 \
    --development-fraction 0.1

CURRENT_STAGE=complete
trap - ERR
echo "PASS local rare-original mining finalize"
echo "persistent_log: ${LOG_FILE}"
