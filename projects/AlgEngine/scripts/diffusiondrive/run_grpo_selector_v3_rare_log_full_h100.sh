#!/usr/bin/env bash
set -Eeo pipefail

MODE="${1:-all}"
case "${MODE}" in
    preflight|mine|cache|train|eval|all) ;;
    *)
        echo "Usage: $0 [preflight|mine|cache|train|eval|all]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
EXPECTED_NAVTRAIN_TOKENS=103288
EXPECTED_NAVTRAIN_LOGS=1192
EXPECTED_PHYSICAL_TOKENS=115434
FORMAL_METHOD="scene_conditioned_exact_group_grpo_v3_rare_log_v1"
CONFIG="${DIFFUSIONDRIVE_GRPO_CONFIG}"
BASELINE="${DIFFUSIONDRIVE_GRPO_BASELINE}"
FULL_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtrain_split/navtrain.yaml"
NAVTEST_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtest_split/navtest.yaml"
ANNOTATION="${WORLDENGINE_ROOT}/data/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl"
TRAIN_IMAGE_ROOT="${WORLDENGINE_ROOT}/data/raw/openscene-v1.1/sensor_blobs/trainval"
PHYSICAL_CACHE="${WORLDENGINE_ROOT%/WorldEngine}/exp/metric_cache"
SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_log_v1"
FULL_INDEX="${SOURCE_ROOT}/metric_cache_navtrain_full"
MINING_ROOT="${SOURCE_ROOT}/mining"
RARE_DATA_ROOT="${SOURCE_ROOT}/rare_data"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_log_v1"
CACHE_ROOT="${ROOT}/cache"
REPLICA_ROOT="${ROOT}/replicas"
FORMAL_ROOT="${ROOT}/formal"
LOG_DIR="${ROOT}/logs"
STATUS_FILE="${ROOT}/status.txt"
PREPARE="${SCRIPT_DIR}/prepare_grpo_selector_v3_rare_log_data.py"
TRAINER="${SCRIPT_DIR}/train_grpo_selector_v3_cached_rare_log.py"
CONTEXT_EXTRACTOR="${SCRIPT_DIR}/extract_grpo_selector_context_cache.py"
CONTEXT_HELPER="${SCRIPT_DIR}/extract_grpo_selector_diagnostic_cache.py"
MATERIALIZE="${SCRIPT_DIR}/materialize_grpo_selector_v3.py"
CHECKPOINT_AUDIT="${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py"
SUMMARIZE="${SCRIPT_DIR}/summarize_grpo_selector_v3_rare_log.py"
PLANNING_HEAD="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py"
SCENE_SELECTOR="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
NAVFORMER="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/detectors/navformer.py"
TEST_SCRIPT="${ALGENGINE_ROOT}/scripts/test.py"

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${MODE}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=bootstrap
CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
printf 'RUNNING mode=%s code_sha=%s started_utc=%s\n' \
    "${MODE}" "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap 'rc=$?; printf "FAIL mode=%s stage=%s exit=%s\n" "${MODE}" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL rare-log-v1 mode=${MODE} stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

json_number() {
    "${ALGENGINE_PYTHON}" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
        "$1" "$2"
}

run_preflight() {
    CURRENT_STAGE=preflight_paths
    for path in \
        "${CONFIG}" "${BASELINE}" "${FULL_FILTER}" "${NAVTEST_FILTER}" \
        "${ANNOTATION}" "${PREPARE}" "${TRAINER}" "${CONTEXT_EXTRACTOR}" "${CONTEXT_HELPER}" "${MATERIALIZE}" \
        "${CHECKPOINT_AUDIT}" "${SUMMARIZE}" "${PLANNING_HEAD}" \
        "${SCENE_SELECTOR}" "${NAVFORMER}" "${TEST_SCRIPT}"; do
        require_file "${path}"
    done
    [[ -d "${TRAIN_IMAGE_ROOT}" ]] || {
        echo "Missing navtrain image root: ${TRAIN_IMAGE_ROOT}" >&2
        exit 1
    }
    [[ -d "${PHYSICAL_CACHE}" ]] || {
        echo "Missing physical metric cache: ${PHYSICAL_CACHE}" >&2
        exit 1
    }
    actual_sha="$(sha256sum "${BASELINE}" | awk '{print $1}')"
    [[ "${actual_sha}" == "${BASELINE_SHA256}" ]] || {
        echo "Baseline SHA256 mismatch: ${actual_sha}" >&2
        exit 1
    }

    CURRENT_STAGE=preflight_cuda
    "${ALGENGINE_PYTHON}" \
        "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
        --extension "${WORLDENGINE_MMCV_EXTENSION}" \
        --expected-capability sm_90 --all-visible

    CURRENT_STAGE=full_navtrain_index
    "${ALGENGINE_PYTHON}" "${PREPARE}" index \
        --physical-cache-root "${PHYSICAL_CACHE}" \
        --navtrain-filter "${FULL_FILTER}" \
        --navtest-filter "${NAVTEST_FILTER}" \
        --annotation-file "${ANNOTATION}" \
        --output-dir "${FULL_INDEX}" \
        --expected-tokens "${EXPECTED_NAVTRAIN_TOKENS}" \
        --expected-logs "${EXPECTED_NAVTRAIN_LOGS}" \
        --expected-physical-tokens "${EXPECTED_PHYSICAL_TOKENS}"
    echo "PASS full-navtrain index: ${FULL_INDEX}"
}

build_submission() {
    local seed="$1"
    local seed_root="$2"
    local namespace="selector_context_train_seed${seed}"
    "${ALGENGINE_PYTHON}" "${PREPARE}" submission \
        --results "${seed_root}/inference_results.pkl" \
        --cache-index "${FULL_INDEX}" \
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
}

run_mining_seed() {
    local seed="$1"
    local seed_root="${MINING_ROOT}/seed${seed}"
    local namespace="selector_context_train_seed${seed}"
    local results="${seed_root}/inference_results.pkl"
    local submission="${seed_root}/navsim_submission.pkl"
    local submission_manifest="${seed_root}/submission_manifest.json"
    local score_root="${seed_root}/official_scores"
    local score_csv="${score_root}/pdm_scores_merged.csv"
    local score_audit="${seed_root}/score_audit.json"
    mkdir -p "${seed_root}"

    CURRENT_STAGE="mining_seed${seed}_inference"
    if [[ ! -f "${results}" || ! -f "${submission}" || ! -f "${submission_manifest}" ]]; then
        export DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE="${namespace}"
        export NAVSIM_METRIC_CACHE_PATH_TRAIN="${FULL_INDEX}"
        cd "${ALGENGINE_ROOT}"
        "${ALGENGINE_TORCHRUN}" \
            --nproc_per_node=8 \
            --master_port="$((29700 + seed))" \
            "${TEST_SCRIPT}" "${CONFIG}" "${BASELINE}" \
            --launcher pytorch \
            --seed "${seed}" \
            --out "${results}" \
            --tmpdir "${seed_root}/tmp_collect" \
            --cfg-options \
                data.workers_per_gpu=2 \
                data.test.ann_file="${ANNOTATION}" \
                data.test.nav_filter_path="${FULL_FILTER}" \
                data.test.metric_cache_path="${FULL_INDEX}" \
                data.test.pipeline.0.img_root="${TRAIN_IMAGE_ROOT}"
        cd "${WORLDENGINE_ROOT}"
    else
        echo "REUSE inference candidate seed ${seed}; validating full provenance"
    fi

    CURRENT_STAGE="mining_seed${seed}_submission"
    build_submission "${seed}" "${seed_root}"

    CURRENT_STAGE="mining_seed${seed}_official_score"
    if [[ ! -f "${score_csv}" ]]; then
        NAVSIM_DEVKIT_ROOT="${DIFFUSIONDRIVE_ROOT}" \
        PYTHON_BIN="${ALGENGINE_PYTHON}" \
        NAVSIM_RESCORE_SPLIT=navtrain \
        NAVSIM_RESCORE_SHARDS=8 \
            bash "${ALGENGINE_ROOT}/scripts/e2e_navsim_official_rescore.sh" \
                "${submission}" "${FULL_INDEX}" "${score_root}"
    else
        echo "REUSE official score seed ${seed}; validating exact token coverage"
    fi

    CURRENT_STAGE="mining_seed${seed}_audit"
    "${ALGENGINE_PYTHON}" "${PREPARE}" audit-score \
        --score-csv "${score_csv}" \
        --submission "${submission}" \
        --cache-index "${FULL_INDEX}" \
        --expected-tokens "${EXPECTED_NAVTRAIN_TOKENS}" \
        --output "${score_audit}"
    echo "PASS full-navtrain mining seed ${seed}"
}

run_mining() {
    for seed in 0 1 2; do
        run_mining_seed "${seed}"
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
        --expected-tokens "${EXPECTED_NAVTRAIN_TOKENS}" \
        --ego-progress-percentile 1 \
        --pair-seed 20260818
    echo "PASS rare/common mining: ${RARE_DATA_ROOT}/rare_data_audit.json"
}

validate_context_cache() {
    local output="$1"
    local seed="$2"
    local expected_count="$3"
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" "${ALGENGINE_PYTHON}" -c '
import sys
from pathlib import Path
import grpo_selector_v3_cached_common as common
cache, manifest = common.load_cache(sys.argv[1], "train")
expected_seed, expected_count, expected_filter_sha = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
assert int(manifest["noise_seed"]) == expected_seed
assert len(cache["tokens"]) == expected_count == int(manifest["num_tokens"])
assert manifest["nav_filter_sha256"] == expected_filter_sha
expected_files={str(Path(path).resolve()) for path in sys.argv[5:]}
recorded=manifest.get("implementation_files", {})
assert set(recorded)==expected_files
assert all(recorded[path]==common.sha256_file(path) for path in expected_files)
print("PASS verified context cache seed", expected_seed, "tokens", expected_count)
' "${output}/cache.pt" "${seed}" "${expected_count}" \
        "$(sha256sum "${RARE_DATA_ROOT}/rare_common_union.yaml" | awk '{print $1}')" \
        "${SCRIPT_DIR}/extract_grpo_selector_context_cache.py" \
        "${SCRIPT_DIR}/extract_grpo_selector_diagnostic_cache.py" \
        "${PLANNING_HEAD}" "${SCENE_SELECTOR}" "${NAVFORMER}"
}

run_cache_seed() {
    local seed="$1"
    local expected_count="$2"
    local output="${CACHE_ROOT}/train_seed${seed}"
    CURRENT_STAGE="context_cache_seed${seed}"
    if [[ -f "${output}/cache.pt" && -f "${output}/manifest.json" ]]; then
        validate_context_cache "${output}" "${seed}" "${expected_count}"
        return
    fi
    mkdir -p "${output}"
    export NAVSIM_METRIC_CACHE_PATH_TRAIN="${FULL_INDEX}"
    cd "${ALGENGINE_ROOT}"
    "${ALGENGINE_TORCHRUN}" \
        --nproc_per_node=8 \
        --master_port="$((29800 + seed))" \
        "${SCRIPT_DIR}/extract_grpo_selector_context_cache.py" \
        "${CONFIG}" "${BASELINE}" \
        --expected-checkpoint-sha256 "${BASELINE_SHA256}" \
        --nav-filter "${RARE_DATA_ROOT}/rare_common_union.yaml" \
        --split train \
        --noise-seed "${seed}" \
        --expected-num-tokens "${expected_count}" \
        --output-dir "${output}" \
        --workers-per-gpu 2 \
        --launcher pytorch
    cd "${WORLDENGINE_ROOT}"
    validate_context_cache "${output}" "${seed}" "${expected_count}"
}

run_caches() {
    require_file "${RARE_DATA_ROOT}/rare_data_audit.json"
    require_file "${RARE_DATA_ROOT}/rare_common_union.yaml"
    expected_count="$(json_number "${RARE_DATA_ROOT}/rare_data_audit.json" union_count)"
    for seed in 0 1 2; do
        run_cache_seed "${seed}" "${expected_count}"
    done
    echo "PASS three fixed-noise rare/common context caches"
}

training_ready() {
    local seed="$1"
    local train_root="${REPLICA_ROOT}/seed${seed}/train"
    local checkpoint_root="${REPLICA_ROOT}/seed${seed}"
    local report="${train_root}/report.json"
    local manifest="${checkpoint_root}/checkpoint_manifest.json"
    local checkpoint="${checkpoint_root}/checkpoint.pth"
    [[ -f "${report}" && -f "${manifest}" && -f "${checkpoint}" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import hashlib,json,sys
report=json.load(open(sys.argv[1]))
manifest=json.load(open(sys.argv[2]))
checkpoint=sys.argv[3]
digest=hashlib.sha256(open(checkpoint,"rb").read()).hexdigest()
assert report["status"]=="PASS"
assert report["method"]==sys.argv[4]
implementation=report.get("implementation_files", {})
assert len(implementation)==4
for path, expected in implementation.items():
    assert hashlib.sha256(open(path,"rb").read()).hexdigest()==expected
assert report["sampling"]["total_examples"]==304272
assert report["sampling"]["total_optimizer_steps"]==4800
assert report["sampling"]["class_examples"]["rare"]==152136
assert report["sampling"]["class_examples"]["common"]==152136
assert manifest["checkpoint_sha256"]==digest
' "${report}" "${manifest}" "${checkpoint}" "${FORMAL_METHOD}"
}

train_seed() {
    local seed="$1"
    local seed_root="${REPLICA_ROOT}/seed${seed}"
    local train_root="${seed_root}/train"
    local checkpoint="${seed_root}/checkpoint.pth"
    local manifest="${seed_root}/checkpoint_manifest.json"
    mkdir -p "${train_root}"
    if training_ready "${seed}"; then
        echo "REUSE verified V3 rare-log replica seed ${seed}"
    else
        CUDA_VISIBLE_DEVICES="${seed}" "${ALGENGINE_PYTHON}" "${TRAINER}" \
            --train-cache "${CACHE_ROOT}/train_seed0/cache.pt" \
            --train-cache "${CACHE_ROOT}/train_seed1/cache.pt" \
            --train-cache "${CACHE_ROOT}/train_seed2/cache.pt" \
            --pair-manifest "${RARE_DATA_ROOT}/pairs.jsonl" \
            --rare-data-audit "${RARE_DATA_ROOT}/rare_data_audit.json" \
            --output-dir "${train_root}" \
            --temperature 1.0 \
            --learning-rate 0.0001 \
            --kl-weight 0.0 \
            --seed "${seed}" \
            --epochs 16 \
            --examples-per-cache-epoch 6339 \
            --checkpoint-epochs 16 \
            --batch-size 64 \
            --device cuda \
            --ablation full \
            --formal-contract
        selector="${train_root}/epoch_16_scene_selector.pt"
        selector_sha="$(sha256sum "${selector}" | awk '{print $1}')"
        "${ALGENGINE_PYTHON}" "${MATERIALIZE}" \
            --baseline "${BASELINE}" \
            --scene-selector-state "${selector}" \
            --expected-selector-sha256 "${selector_sha}" \
            --expected-method "${FORMAL_METHOD}" \
            --release-name "e2e_diffusiondrive_grpo_selector_v3_rare_log_v1_s${seed}" \
            --output "${checkpoint}" \
            --manifest "${manifest}"
    fi
    "${ALGENGINE_PYTHON}" "${CHECKPOINT_AUDIT}" \
        --baseline "${BASELINE}" \
        --checkpoint "${checkpoint}" \
        --output "${seed_root}/checkpoint_audit.json"
    training_ready "${seed}"
    echo "PASS V3 rare-log replica seed ${seed}"
}

run_training() {
    for seed in 0 1 2; do
        require_file "${CACHE_ROOT}/train_seed${seed}/cache.pt"
    done
    require_file "${RARE_DATA_ROOT}/pairs.jsonl"
    PIDS=()
    for seed in 0 1 2; do
        (
            train_seed "${seed}"
        ) > "${LOG_DIR}/train_seed${seed}.log" 2>&1 &
        PIDS+=("$!")
    done
    failed=0
    for pid in "${PIDS[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "One or more replica trainings failed; inspect ${LOG_DIR}/train_seed*.log" >&2
        return 1
    fi
    for seed in 0 1 2; do
        training_ready "${seed}"
    done
    echo "PASS all three V3 rare-log replicas"
}

evaluation_ready() {
    local seed="$1"
    local model="e2e_diffusiondrive_grpo_selector_v3_rare_log_v1_s${seed}"
    local summary="${FORMAL_ROOT}/formal_eval/${model}/summary.json"
    local manifest="${REPLICA_ROOT}/seed${seed}/checkpoint_manifest.json"
    [[ -f "${summary}" && -f "${manifest}" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import json,sys
summary=json.load(open(sys.argv[1]))
manifest=json.load(open(sys.argv[2]))
assert summary["status"]=="PASS"
assert int(summary["eval_seed"])==int(sys.argv[3])
assert summary["checkpoint_sha256"]==manifest["checkpoint_sha256"]
' "${summary}" "${manifest}" "${seed}"
}

run_evaluation_seed() {
    local seed="$1"
    local seed_root="${REPLICA_ROOT}/seed${seed}"
    local checkpoint="${seed_root}/checkpoint.pth"
    local manifest="${seed_root}/checkpoint_manifest.json"
    local model="e2e_diffusiondrive_grpo_selector_v3_rare_log_v1_s${seed}"
    if evaluation_ready "${seed}"; then
        echo "REUSE verified formal evaluation seed ${seed}"
        return
    fi
    checkpoint_sha="$(json_number "${manifest}" checkpoint_sha256)"
    export DIFFUSIONDRIVE_GRPO_FORMAL_ROOT="${FORMAL_ROOT}"
    export DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE=1.0
    export DIFFUSIONDRIVE_GRPO_KL_WEIGHT=0.0
    "${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh" \
        "${checkpoint}" "${checkpoint_sha}" "${model}" "${seed}" \
        "V3 rare-log-v1; full navtrain; same-log strict-common 1:1; train/eval seed=${seed}"
    evaluation_ready "${seed}"
}

run_evaluations() {
    for seed in 0 1 2; do
        CURRENT_STAGE="formal_eval_seed${seed}"
        run_evaluation_seed "${seed}"
    done

    CURRENT_STAGE=formal_aggregate
    aggregate_args=()
    for seed in 0 1 2; do
        require_file "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_s${seed}/summary.json"
        require_file "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/formal_eval/e2e_diffusiondrive_reference_paired_s${seed}/summary.json"
        aggregate_args+=(
            --rare "${seed}=${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_log_v1_s${seed}/summary.json"
            --common "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_s${seed}/summary.json"
            --base "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/formal_eval/e2e_diffusiondrive_reference_paired_s${seed}/summary.json"
        )
    done
    "${ALGENGINE_PYTHON}" "${SUMMARIZE}" \
        "${aggregate_args[@]}" \
        --output "${FORMAL_ROOT}/rare_vs_common_vs_base.json"
    echo "PASS formal aggregate: ${FORMAL_ROOT}/rare_vs_common_vs_base.json"
}

case "${MODE}" in
    preflight)
        run_preflight
        ;;
    mine)
        run_preflight
        run_mining
        ;;
    cache)
        run_preflight
        run_caches
        ;;
    train)
        run_preflight
        run_training
        ;;
    eval)
        run_preflight
        run_evaluations
        ;;
    all)
        run_preflight
        run_mining
        run_caches
        CURRENT_STAGE=replica_training
        run_training
        run_evaluations
        ;;
esac

CURRENT_STAGE=complete
printf 'PASS mode=%s code_sha=%s completed_utc=%s\n' \
    "${MODE}" "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive selector V3 rare-log-v1 mode ${MODE}"
echo "status: ${STATUS_FILE}"
echo "persistent_log: ${LOG_FILE}"
