#!/usr/bin/env bash
set -Eeo pipefail

export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
RUN_ID="${RARE_ORIGINAL_SMOKE_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1"
FULL_INDEX="${SOURCE_ROOT}/metric_cache_navtrain_full"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/local_smoke/${RUN_ID}"
SUBSET="${ROOT}/subset"
MINING="${ROOT}/mining"
RARE_DATA="${ROOT}/rare_data"
CACHE_ROOT="${ROOT}/cache"
TRAIN_ROOT="${ROOT}/train"
CONFIG="${DIFFUSIONDRIVE_GRPO_CONFIG}"
BASELINE="${DIFFUSIONDRIVE_GRPO_BASELINE}"
FULL_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtrain_split/navtrain.yaml"
NAVTEST_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtest_split/navtest.yaml"
ANNOTATION="${WORLDENGINE_ROOT}/data/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl"
TRAIN_IMAGE_ROOT="${WORLDENGINE_ROOT}/data/raw/openscene-v1.1/sensor_blobs/trainval"
PHYSICAL_CACHE="${WORLDENGINE_ROOT%/WorldEngine}/exp/metric_cache"
PREPARE="${SCRIPT_DIR}/prepare_grpo_selector_v3_rare_original_data.py"
SUBSET_TOOL="${SCRIPT_DIR}/prepare_grpo_selector_v3_rare_original_smoke_subset.py"
TRAINER="${SCRIPT_DIR}/train_grpo_selector_v3_cached_rare_original.py"
CONTEXT_EXTRACTOR="${SCRIPT_DIR}/extract_grpo_selector_context_cache.py"
CONTEXT_HELPER="${SCRIPT_DIR}/extract_grpo_selector_diagnostic_cache.py"
PLANNING_HEAD="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py"
SCENE_SELECTOR="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
PDM_REWARD="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
NAVFORMER="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/detectors/navformer.py"
TEST_SCRIPT="${ALGENGINE_ROOT}/scripts/test.py"

mkdir -p "${ROOT}"
LOG_FILE="${ROOT}/smoke.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL one-H100 rare-original smoke stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

for path in "${CONFIG}" "${BASELINE}" "${FULL_FILTER}" "${NAVTEST_FILTER}" \
    "${ANNOTATION}" "${PREPARE}" "${SUBSET_TOOL}" "${TRAINER}" "${CONTEXT_EXTRACTOR}" "${CONTEXT_HELPER}" \
    "${PLANNING_HEAD}" "${SCENE_SELECTOR}" "${PDM_REWARD}" "${NAVFORMER}" "${TEST_SCRIPT}"; do
    [[ -f "${path}" ]] || { echo "Missing smoke input: ${path}" >&2; exit 1; }
done
[[ "$(sha256sum "${BASELINE}" | awk '{print $1}')" == "${BASELINE_SHA256}" ]] || {
    echo "Baseline SHA256 mismatch" >&2
    exit 1
}
"${ALGENGINE_PYTHON}" \
    "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" \
    --expected-capability sm_90 --all-visible

CURRENT_STAGE=full_index
"${ALGENGINE_PYTHON}" "${PREPARE}" index \
    --physical-cache-root "${PHYSICAL_CACHE}" \
    --navtrain-filter "${FULL_FILTER}" \
    --navtest-filter "${NAVTEST_FILTER}" \
    --annotation-file "${ANNOTATION}" \
    --output-dir "${FULL_INDEX}" \
    --expected-tokens 103288 \
    --expected-logs 1192 \
    --expected-physical-tokens 115434

CURRENT_STAGE=smoke_subset
"${ALGENGINE_PYTHON}" "${SUBSET_TOOL}" \
    --source-index "${FULL_INDEX}" \
    --annotation-file "${ANNOTATION}" \
    --base-filter "${FULL_FILTER}" \
    --output-dir "${SUBSET}" \
    --num-tokens 64 \
    --expected-source-tokens 103288 \
    --seed 20260818

run_inference() {
    local seed="$1"
    local checkpoint="$2"
    local namespace="$3"
    local output="$4"
    local port="$5"
    if [[ -f "${output}" ]]; then
        echo "REUSE smoke inference: ${output}"
        return
    fi
    export DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE="${namespace}"
    export NAVSIM_METRIC_CACHE_PATH_TRAIN="${SUBSET}"
    cd "${ALGENGINE_ROOT}"
    "${ALGENGINE_TORCHRUN}" \
        --nproc_per_node=1 \
        --master_port="${port}" \
        "${TEST_SCRIPT}" "${CONFIG}" "${checkpoint}" \
        --launcher pytorch \
        --seed "${seed}" \
        --out "${output}" \
        --tmpdir "$(dirname "${output}")/tmp_collect" \
        --cfg-options \
            data.workers_per_gpu=2 \
            data.test.ann_file="${ANNOTATION}" \
            data.test.nav_filter_path="${SUBSET}/subset_filter.yaml" \
            data.test.metric_cache_path="${SUBSET}" \
            data.test.pipeline.0.img_root="${TRAIN_IMAGE_ROOT}"
    cd "${WORLDENGINE_ROOT}"
}

for seed in 0 1 2; do
    seed_root="${MINING}/seed${seed}"
    mkdir -p "${seed_root}"
    namespace="selector_context_train_seed${seed}"
    CURRENT_STAGE="base_inference_seed${seed}"
    run_inference "${seed}" "${BASELINE}" "${namespace}" \
        "${seed_root}/inference_results.pkl" "$((29900 + seed))"

    CURRENT_STAGE="submission_seed${seed}"
    "${ALGENGINE_PYTHON}" "${PREPARE}" submission \
        --results "${seed_root}/inference_results.pkl" \
        --cache-index "${SUBSET}" \
        --checkpoint "${BASELINE}" \
        --config "${CONFIG}" \
        --provenance-file "${TEST_SCRIPT}" \
        --provenance-file "${PLANNING_HEAD}" \
        --provenance-file "${SCENE_SELECTOR}" \
        --provenance-file "${NAVFORMER}" \
        --expected-checkpoint-sha256 "${BASELINE_SHA256}" \
        --noise-seed "${seed}" \
        --noise-namespace "${namespace}" \
        --output "${seed_root}/navsim_submission.pkl" \
        --manifest "${seed_root}/submission_manifest.json"

    CURRENT_STAGE="official_score_seed${seed}"
    if [[ ! -f "${seed_root}/official_scores/pdm_scores_merged.csv" ]]; then
        NAVSIM_DEVKIT_ROOT="${DIFFUSIONDRIVE_ROOT}" \
        PYTHON_BIN="${ALGENGINE_PYTHON}" \
        NAVSIM_RESCORE_SPLIT=navtrain \
        NAVSIM_RESCORE_SHARDS=1 \
            bash "${ALGENGINE_ROOT}/scripts/e2e_navsim_official_rescore.sh" \
                "${seed_root}/navsim_submission.pkl" \
                "${SUBSET}" \
                "${seed_root}/official_scores"
    fi
    "${ALGENGINE_PYTHON}" "${PREPARE}" audit-score \
        --score-csv "${seed_root}/official_scores/pdm_scores_merged.csv" \
        --submission "${seed_root}/navsim_submission.pkl" \
        --cache-index "${SUBSET}" \
        --expected-tokens 64 \
        --output "${seed_root}/score_audit.json"
done

CURRENT_STAGE=rare_mining
"${ALGENGINE_PYTHON}" "${PREPARE}" mine \
    --score "0=${MINING}/seed0/official_scores/pdm_scores_merged.csv" \
    --score "1=${MINING}/seed1/official_scores/pdm_scores_merged.csv" \
    --score "2=${MINING}/seed2/official_scores/pdm_scores_merged.csv" \
    --cache-index "${SUBSET}" \
    --annotation-file "${ANNOTATION}" \
    --base-filter "${SUBSET}/subset_filter.yaml" \
    --output-dir "${RARE_DATA}" \
    --expected-tokens 64 \
    --ego-progress-percentile 10 \
    --pair-seed 20260818
UNION_COUNT="$("${ALGENGINE_PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["union_count"])' \
    "${RARE_DATA}/rare_data_audit.json")"

for seed in 0 1 2; do
    output="${CACHE_ROOT}/train_seed${seed}"
    CURRENT_STAGE="context_cache_seed${seed}"
    if [[ ! -f "${output}/cache.pt" || ! -f "${output}/manifest.json" ]]; then
        mkdir -p "${output}"
        export NAVSIM_METRIC_CACHE_PATH_TRAIN="${SUBSET}"
        cd "${ALGENGINE_ROOT}"
        "${ALGENGINE_TORCHRUN}" \
            --nproc_per_node=1 \
            --master_port="$((29910 + seed))" \
            "${SCRIPT_DIR}/extract_grpo_selector_context_cache.py" \
            "${CONFIG}" "${BASELINE}" \
            --expected-checkpoint-sha256 "${BASELINE_SHA256}" \
            --nav-filter "${RARE_DATA}/rare_common_union.yaml" \
            --split train \
            --noise-seed "${seed}" \
            --expected-num-tokens "${UNION_COUNT}" \
            --output-dir "${output}" \
            --workers-per-gpu 2 \
            --launcher pytorch
        cd "${WORLDENGINE_ROOT}"
    fi
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" "${ALGENGINE_PYTHON}" -c '
import sys
from pathlib import Path
import grpo_selector_v3_cached_common as common
cache,manifest=common.load_cache(sys.argv[1],"train")
assert int(manifest["noise_seed"])==int(sys.argv[2])
assert len(cache["tokens"])==int(sys.argv[3])
expected={str(Path(path).resolve()) for path in sys.argv[4:]}
recorded=manifest.get("implementation_files",{})
assert set(recorded)==expected
assert all(recorded[path]==common.sha256_file(path) for path in expected)
' "${output}/cache.pt" "${seed}" "${UNION_COUNT}" \
        "${SCRIPT_DIR}/extract_grpo_selector_context_cache.py" \
        "${SCRIPT_DIR}/extract_grpo_selector_diagnostic_cache.py" \
        "${PDM_REWARD}" "${PLANNING_HEAD}" "${SCENE_SELECTOR}" "${NAVFORMER}"
done

CURRENT_STAGE=two_epoch_training
"${ALGENGINE_PYTHON}" "${TRAINER}" \
    --train-cache "${CACHE_ROOT}/train_seed0/cache.pt" \
    --train-cache "${CACHE_ROOT}/train_seed1/cache.pt" \
    --train-cache "${CACHE_ROOT}/train_seed2/cache.pt" \
    --pair-manifest "${RARE_DATA}/pairs.jsonl" \
    --rare-data-audit "${RARE_DATA}/rare_data_audit.json" \
    --output-dir "${TRAIN_ROOT}" \
    --temperature 1.0 \
    --learning-rate 0.0001 \
    --kl-weight 0.001 \
    --sampling-mode rare_balanced \
    --seed 0 \
    --epochs 2 \
    --examples-per-cache-epoch 64 \
    --checkpoint-epochs 2 \
    --batch-size 8 \
    --device cuda \
    --ablation full

SELECTOR="${TRAIN_ROOT}/epoch_2_scene_selector.pt"
SELECTOR_SHA="$(sha256sum "${SELECTOR}" | awk '{print $1}')"
CURRENT_STAGE=materialize
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_grpo_selector_v3.py" \
    --baseline "${BASELINE}" \
    --scene-selector-state "${SELECTOR}" \
    --expected-selector-sha256 "${SELECTOR_SHA}" \
    --expected-method "scene_conditioned_exact_group_grpo_v3_rare_original_v1" \
    --release-name "e2e_diffusiondrive_grpo_selector_v3_rare_original_v1_smoke" \
    --output "${ROOT}/checkpoint.pth" \
    --manifest "${ROOT}/checkpoint_manifest.json"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    --baseline "${BASELINE}" \
    --checkpoint "${ROOT}/checkpoint.pth" \
    --output "${ROOT}/checkpoint_audit.json"

CURRENT_STAGE=trained_checkpoint_inference
run_inference 0 "${ROOT}/checkpoint.pth" "rare_original_smoke_trained_eval" \
    "${ROOT}/trained_inference_results.pkl" 29920
"${ALGENGINE_PYTHON}" -c '
import pickle,sys
import numpy as np
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import prepare_grpo_selector_v3_rare_original_data as prepare
rows=prepare.flatten_results(pickle.load(open(sys.argv[1],"rb")))
assert len(rows)==64
assert len({str(row["token"]) for row in rows})==64
assert all(np.asarray(row["trajectory"]).shape==(40,3) for row in rows)
assert all(np.isfinite(np.asarray(row["trajectory"])).all() for row in rows)
print("PASS trained checkpoint inference rows=64")
' "${ROOT}/trained_inference_results.pkl" "${SCRIPT_DIR}"

CURRENT_STAGE=complete
trap - ERR
echo "PASS one-H100 V3 rare-original end-to-end smoke"
echo "output: ${ROOT}"
echo "persistent_log: ${LOG_FILE}"
