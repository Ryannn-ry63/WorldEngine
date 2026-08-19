#!/usr/bin/env bash
set -Eeo pipefail

MODE="${1:-all}"
case "${MODE}" in
    preflight|mine|mine-seed0|mine-seed1|mine-seed2|cache|cache-lane0|cache-lane1|cache-lane2|sweep|train|eval|eval-paired-common|eval-rare-frozen|eval-rare-tuned|summarize|all) ;;
    *)
        echo "Usage: $0 [preflight|mine|mine-seed{0,1,2}|cache|cache-lane{0,1,2}|sweep|train|eval|eval-{paired-common,rare-frozen,rare-tuned}|summarize|all]" >&2
        exit 2
        ;;
esac

export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
EXPECTED_NAVTRAIN_TOKENS=103288
EXPECTED_NAVTRAIN_LOGS=1192
EXPECTED_PHYSICAL_TOKENS=115434
CONFIG="${DIFFUSIONDRIVE_GRPO_CONFIG}"
BASELINE="${DIFFUSIONDRIVE_GRPO_BASELINE}"
FULL_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtrain_split/navtrain.yaml"
NAVTEST_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtest_split/navtest.yaml"
ANNOTATION="${WORLDENGINE_ROOT}/data/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl"
TRAIN_IMAGE_ROOT="${WORLDENGINE_ROOT}/data/raw/openscene-v1.1/sensor_blobs/trainval"
PHYSICAL_CACHE="${WORLDENGINE_ROOT%/WorldEngine}/exp/metric_cache"

SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1"
FULL_INDEX="${SOURCE_ROOT}/metric_cache_navtrain_full"
MINING_ROOT="${SOURCE_ROOT}/mining"
RARE_DATA_ROOT="${SOURCE_ROOT}/rare_data"
TUNING_SPLIT_ROOT="${SOURCE_ROOT}/tuning_split"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1"
CACHE_ROOT="${ROOT}/cache"
SWEEP_ROOT="${ROOT}/sweep"
CERTIFICATION_ROOT="${ROOT}/certification"
MODEL_ROOT="${ROOT}/models"
FORMAL_ROOT="${ROOT}/formal"
LOG_DIR="${ROOT}/logs"
if [[ "${MODE}" == "all" ]]; then
    STATUS_FILE="${ROOT}/status.txt"
else
    STATUS_FILE="${ROOT}/status_${MODE}.txt"
fi

PREPARE="${SCRIPT_DIR}/prepare_grpo_selector_v3_rare_original_data.py"
SPLIT_TOOL="${SCRIPT_DIR}/split_grpo_selector_v3_rare_original.py"
TRAINER="${SCRIPT_DIR}/train_grpo_selector_v3_cached_rare_original.py"
EVALUATOR="${SCRIPT_DIR}/evaluate_grpo_selector_v3_rare_original.py"
SELECTOR="${SCRIPT_DIR}/select_grpo_selector_v3_rare_original.py"
CERTIFIER="${SCRIPT_DIR}/certify_grpo_selector_v3_rare_original.py"
SUMMARIZER="${SCRIPT_DIR}/summarize_grpo_selector_v3_rare_original_full.py"
CONTEXT_EXTRACTOR="${SCRIPT_DIR}/extract_grpo_selector_context_cache.py"
CONTEXT_HELPER="${SCRIPT_DIR}/extract_grpo_selector_diagnostic_cache.py"
MATERIALIZE="${SCRIPT_DIR}/materialize_grpo_selector_v3.py"
CHECKPOINT_AUDIT="${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py"
FORMAL_TABLE="${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh"
PLANNING_HEAD="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py"
SCENE_SELECTOR="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py"
PDM_REWARD="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
NAVFORMER="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/detectors/navformer.py"
TEST_SCRIPT="${ALGENGINE_ROOT}/scripts/test.py"
IMPLEMENTATION_FILES=(
    "${CONTEXT_EXTRACTOR}"
    "${CONTEXT_HELPER}"
    "${PDM_REWARD}"
    "${PLANNING_HEAD}"
    "${SCENE_SELECTOR}"
    "${NAVFORMER}"
)

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${MODE}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=bootstrap
CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
printf 'RUNNING mode=%s code_sha=%s started_utc=%s\n'     "${MODE}" "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap 'rc=$?; printf "FAIL mode=%s stage=%s exit=%s\n" "${MODE}" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL rare-original-v1 mode=${MODE} stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

json_get() {
    "${ALGENGINE_PYTHON}" -c '
import json,sys
value=json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value=value[key]
print(value)
' "$1" "$2"
}

json_pass() {
    "${ALGENGINE_PYTHON}" -c '
import json,sys
row=json.load(open(sys.argv[1]))
assert row.get("status")=="PASS"
' "$1"
}

run_preflight() {
    CURRENT_STAGE=preflight_paths
    for path in         "${CONFIG}" "${BASELINE}" "${FULL_FILTER}" "${NAVTEST_FILTER}"         "${ANNOTATION}" "${PREPARE}" "${SPLIT_TOOL}" "${TRAINER}"         "${EVALUATOR}" "${SELECTOR}" "${CERTIFIER}" "${SUMMARIZER}"         "${MATERIALIZE}" "${CHECKPOINT_AUDIT}" "${FORMAL_TABLE}"         "${TEST_SCRIPT}" "${IMPLEMENTATION_FILES[@]}"; do
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
    "${ALGENGINE_PYTHON}"         "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py"         --extension "${WORLDENGINE_MMCV_EXTENSION}"         --expected-capability sm_90 --all-visible

    CURRENT_STAGE=full_navtrain_index
    "${ALGENGINE_PYTHON}" "${PREPARE}" index         --physical-cache-root "${PHYSICAL_CACHE}"         --navtrain-filter "${FULL_FILTER}"         --navtest-filter "${NAVTEST_FILTER}"         --annotation-file "${ANNOTATION}"         --output-dir "${FULL_INDEX}"         --expected-tokens "${EXPECTED_NAVTRAIN_TOKENS}"         --expected-logs "${EXPECTED_NAVTRAIN_LOGS}"         --expected-physical-tokens "${EXPECTED_PHYSICAL_TOKENS}"
    echo "PASS full-navtrain index: ${FULL_INDEX}"
}

build_submission() {
    local seed="$1"
    local seed_root="$2"
    local namespace="selector_context_train_seed${seed}"
    "${ALGENGINE_PYTHON}" "${PREPARE}" submission         --results "${seed_root}/inference_results.pkl"         --cache-index "${FULL_INDEX}"         --checkpoint "${BASELINE}"         --config "${CONFIG}"         --provenance-file "${TEST_SCRIPT}"         --provenance-file "${PDM_REWARD}"         --provenance-file "${PLANNING_HEAD}"         --provenance-file "${SCENE_SELECTOR}"         --provenance-file "${NAVFORMER}"         --expected-checkpoint-sha256 "${BASELINE_SHA256}"         --noise-seed "${seed}"         --noise-namespace "${namespace}"         --output "${seed_root}/navsim_submission.pkl"         --manifest "${seed_root}/submission_manifest.json"
}

run_mining_seed() {
    local seed="$1"
    local seed_root="${MINING_ROOT}/seed${seed}"
    local namespace="selector_context_train_seed${seed}"
    local results="${seed_root}/inference_results.pkl"
    local score_root="${seed_root}/official_scores"
    local score_csv="${score_root}/pdm_scores_merged.csv"
    mkdir -p "${seed_root}"

    CURRENT_STAGE="mining_seed${seed}_inference"
    if [[ ! -f "${results}" ]]; then
        export DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE="${namespace}"
        export NAVSIM_METRIC_CACHE_PATH_TRAIN="${FULL_INDEX}"
        cd "${ALGENGINE_ROOT}"
        "${ALGENGINE_TORCHRUN}"             --nproc_per_node=8             --master_port="$((29700 + seed))"             "${TEST_SCRIPT}" "${CONFIG}" "${BASELINE}"             --launcher pytorch             --seed "${seed}"             --out "${results}"             --tmpdir "${seed_root}/tmp_collect"             --cfg-options                 data.workers_per_gpu=2                 data.test.ann_file="${ANNOTATION}"                 data.test.nav_filter_path="${FULL_FILTER}"                 data.test.metric_cache_path="${FULL_INDEX}"                 data.test.pipeline.0.img_root="${TRAIN_IMAGE_ROOT}"
        cd "${WORLDENGINE_ROOT}"
    else
        echo "REUSE baseline inference seed ${seed}; submission audit follows"
    fi

    CURRENT_STAGE="mining_seed${seed}_submission"
    build_submission "${seed}" "${seed_root}"

    CURRENT_STAGE="mining_seed${seed}_official_score"
    if [[ ! -f "${score_csv}" ]]; then
        NAVSIM_DEVKIT_ROOT="${DIFFUSIONDRIVE_ROOT}"         PYTHON_BIN="${ALGENGINE_PYTHON}"         NAVSIM_RESCORE_SPLIT=navtrain         NAVSIM_RESCORE_SHARDS=8             bash "${ALGENGINE_ROOT}/scripts/e2e_navsim_official_rescore.sh"                 "${seed_root}/navsim_submission.pkl" "${FULL_INDEX}" "${score_root}"
    else
        echo "REUSE official score seed ${seed}; exact audit follows"
    fi

    CURRENT_STAGE="mining_seed${seed}_audit"
    "${ALGENGINE_PYTHON}" "${PREPARE}" audit-score         --score-csv "${score_csv}"         --submission "${seed_root}/navsim_submission.pkl"         --cache-index "${FULL_INDEX}"         --expected-tokens "${EXPECTED_NAVTRAIN_TOKENS}"         --output "${seed_root}/score_audit.json"
    echo "PASS full-navtrain mining seed ${seed}"
}

run_mining() {
    for seed in 0 1 2; do
        run_mining_seed "${seed}"
    done
    CURRENT_STAGE=rare_common_mining
    "${ALGENGINE_PYTHON}" "${PREPARE}" mine         --score "0=${MINING_ROOT}/seed0/official_scores/pdm_scores_merged.csv"         --score "1=${MINING_ROOT}/seed1/official_scores/pdm_scores_merged.csv"         --score "2=${MINING_ROOT}/seed2/official_scores/pdm_scores_merged.csv"         --cache-index "${FULL_INDEX}"         --annotation-file "${ANNOTATION}"         --base-filter "${FULL_FILTER}"         --output-dir "${RARE_DATA_ROOT}"         --expected-tokens "${EXPECTED_NAVTRAIN_TOKENS}"         --ego-progress-percentile 1         --pair-seed 20260818

    CURRENT_STAGE=log_disjoint_tuning_split
    "${ALGENGINE_PYTHON}" "${SPLIT_TOOL}"         --pair-manifest "${RARE_DATA_ROOT}/pairs.jsonl"         --rare-data-audit "${RARE_DATA_ROOT}/rare_data_audit.json"         --base-filter "${FULL_FILTER}"         --output-dir "${TUNING_SPLIT_ROOT}"         --split-seed 20260819         --train-fraction 0.8         --development-fraction 0.1
    echo "PASS rare-original mining and log-disjoint tuning split"
}

validate_context_cache() {
    local output="$1"
    local split="$2"
    local seed="$3"
    local expected_count="$4"
    local filter="$5"
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" "${ALGENGINE_PYTHON}" -c '
import sys
from pathlib import Path
import grpo_selector_v3_cached_common as common
cache,manifest=common.load_cache(sys.argv[1],sys.argv[2])
expected_seed,expected_count=int(sys.argv[3]),int(sys.argv[4])
assert int(manifest["noise_seed"])==expected_seed
assert len(cache["tokens"])==expected_count==int(manifest["num_tokens"])
assert manifest["nav_filter_sha256"]==common.sha256_file(Path(sys.argv[5]))
expected={str(Path(path).resolve()) for path in sys.argv[6:]}
recorded=manifest.get("implementation_files",{})
assert set(recorded)==expected, (set(recorded),expected)
assert all(recorded[path]==common.sha256_file(Path(path)) for path in expected)
assert manifest["cache_sha256"]==common.sha256_file(Path(sys.argv[1]))
print("PASS context cache",sys.argv[2],expected_seed,expected_count)
' "${output}/cache.pt" "${split}" "${seed}" "${expected_count}" "${filter}"         "${IMPLEMENTATION_FILES[@]}"
}

run_cache_one() {
    local label="$1"
    local split="$2"
    local seed="$3"
    local expected_count="$4"
    local filter="$5"
    local port="$6"
    local output="${CACHE_ROOT}/${label}/${split}_seed${seed}"
    CURRENT_STAGE="cache_${label}_${split}_seed${seed}"
    if [[ -f "${output}/cache.pt" && -f "${output}/manifest.json" ]]; then
        validate_context_cache "${output}" "${split}" "${seed}" "${expected_count}" "${filter}"
        return
    fi
    mkdir -p "${output}"
    export NAVSIM_METRIC_CACHE_PATH_TRAIN="${FULL_INDEX}"
    cd "${ALGENGINE_ROOT}"
    "${ALGENGINE_TORCHRUN}"         --nproc_per_node=8         --master_port="${port}"         "${CONTEXT_EXTRACTOR}"         "${CONFIG}" "${BASELINE}"         --expected-checkpoint-sha256 "${BASELINE_SHA256}"         --nav-filter "${filter}"         --split "${split}"         --noise-seed "${seed}"         --expected-num-tokens "${expected_count}"         --output-dir "${output}"         --workers-per-gpu 2         --launcher pytorch
    cd "${WORLDENGINE_ROOT}"
    validate_context_cache "${output}" "${split}" "${seed}" "${expected_count}" "${filter}"
}

run_caches() {
    for lane in 0 1 2; do
        run_cache_lane "${lane}"
    done
    echo "PASS full and log-disjoint tuning cache bundles"
}

run_cache_lane() {
    local lane="$1"
    [[ "${lane}" =~ ^[012]$ ]] || {
        echo "Invalid cache lane: ${lane}" >&2
        return 2
    }
    require_file "${RARE_DATA_ROOT}/rare_data_audit.json"
    require_file "${TUNING_SPLIT_ROOT}/split_audit.json"
    local full_count
    full_count="$(json_get "${RARE_DATA_ROOT}/rare_data_audit.json" union_count)"
    run_cache_one full train "${lane}" "${full_count}" \
        "${RARE_DATA_ROOT}/rare_common_union.yaml" "$((29800 + lane))"

    local train_count development_count certification_count
    train_count="$(json_get "${TUNING_SPLIT_ROOT}/split_audit.json" splits.train.union_tokens)"
    development_count="$(json_get "${TUNING_SPLIT_ROOT}/split_audit.json" splits.development.union_tokens)"
    certification_count="$(json_get "${TUNING_SPLIT_ROOT}/split_audit.json" splits.certification.union_tokens)"
    run_cache_one tuning train "${lane}" "${train_count}" \
        "${TUNING_SPLIT_ROOT}/train/rare_common_union.yaml" "$((29810 + lane))"
    run_cache_one tuning development "$((lane + 3))" "${development_count}" \
        "${TUNING_SPLIT_ROOT}/development/rare_common_union.yaml" "$((29813 + lane))"
    run_cache_one tuning certification "$((lane + 6))" "${certification_count}" \
        "${TUNING_SPLIT_ROOT}/certification/rare_common_union.yaml" "$((29816 + lane))"
    echo "PASS cache lane ${lane}"
}

trial_ready() {
    local trial_root="$1"
    [[ -f "${trial_root}/report.json" && -f "${trial_root}/development_evaluation.json" ]] || return 1
    json_pass "${trial_root}/report.json"
    json_pass "${trial_root}/development_evaluation.json"
}

run_sweep_trial() {
    local gpu="$1"
    local temperature="$2"
    local learning_rate="$3"
    local kl_weight="$4"
    local trial="t${temperature}_lr${learning_rate}_kl${kl_weight}"
    local trial_root="${SWEEP_ROOT}/trials/${trial}"
    mkdir -p "${trial_root}"
    if trial_ready "${trial_root}"; then
        echo "REUSE verified rare-original sweep trial ${trial}"
        return
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${TRAINER}"         --train-cache "${CACHE_ROOT}/tuning/train_seed0/cache.pt"         --train-cache "${CACHE_ROOT}/tuning/train_seed1/cache.pt"         --train-cache "${CACHE_ROOT}/tuning/train_seed2/cache.pt"         --pair-manifest "${TUNING_SPLIT_ROOT}/train/pairs.jsonl"         --rare-data-audit "${TUNING_SPLIT_ROOT}/train/rare_data_audit.json"         --output-dir "${trial_root}"         --temperature "${temperature}"         --learning-rate "${learning_rate}"         --kl-weight "${kl_weight}"         --sampling-mode rare_balanced         --seed 0         --epochs 64         --examples-per-cache-epoch 6339         --checkpoint-epochs 1,2,4,8,16,32,64         --batch-size 64         --device cuda         --ablation full
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${EVALUATOR}"         --training-report "${trial_root}/report.json"         --cache "${CACHE_ROOT}/tuning/development_seed3/cache.pt"         --cache "${CACHE_ROOT}/tuning/development_seed4/cache.pt"         --cache "${CACHE_ROOT}/tuning/development_seed5/cache.pt"         --split development         --pair-manifest "${TUNING_SPLIT_ROOT}/development/pairs.jsonl"         --rare-data-audit "${TUNING_SPLIT_ROOT}/development/rare_data_audit.json"         --output "${trial_root}/development_evaluation.json"         --device cuda         --batch-size 64
    trial_ready "${trial_root}"
}

run_sweep() {
    for path in         "${CACHE_ROOT}/tuning/train_seed0/cache.pt"         "${CACHE_ROOT}/tuning/train_seed1/cache.pt"         "${CACHE_ROOT}/tuning/train_seed2/cache.pt"         "${CACHE_ROOT}/tuning/development_seed3/cache.pt"         "${CACHE_ROOT}/tuning/development_seed4/cache.pt"         "${CACHE_ROOT}/tuning/development_seed5/cache.pt"         "${CACHE_ROOT}/tuning/certification_seed6/cache.pt"         "${CACHE_ROOT}/tuning/certification_seed7/cache.pt"         "${CACHE_ROOT}/tuning/certification_seed8/cache.pt"; do
        require_file "${path}"
    done
    mkdir -p "${SWEEP_ROOT}/trials" "${CERTIFICATION_ROOT}"
    local temperatures=(1 1 1 1 1 1 1 2)
    local learning_rates=(1e-4 1e-4 3e-4 3e-4 3e-4 1e-3 1e-3 3e-4)
    local kl_weights=(0 1e-3 0 1e-4 1e-3 1e-4 1e-3 1e-3)
    local pids=()
    CURRENT_STAGE=rare_development_sweep
    for gpu in 0 1 2 3 4 5 6 7; do
        (
            run_sweep_trial "${gpu}" "${temperatures[${gpu}]}"                 "${learning_rates[${gpu}]}" "${kl_weights[${gpu}]}"
        ) > "${LOG_DIR}/sweep_gpu${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "Rare-original sweep failed; inspect ${LOG_DIR}/sweep_gpu*.log" >&2
        return 1
    fi

    CURRENT_STAGE=rare_development_selection
    "${ALGENGINE_PYTHON}" "${SELECTOR}"         --trial-root "${SWEEP_ROOT}/trials"         --expected-candidates 56         --output "${SWEEP_ROOT}/selection.json"

    CURRENT_STAGE=rare_one_shot_certification
    CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${CERTIFIER}"         --selection "${SWEEP_ROOT}/selection.json"         --cache "${CACHE_ROOT}/tuning/certification_seed6/cache.pt"         --cache "${CACHE_ROOT}/tuning/certification_seed7/cache.pt"         --cache "${CACHE_ROOT}/tuning/certification_seed8/cache.pt"         --pair-manifest "${TUNING_SPLIT_ROOT}/certification/pairs.jsonl"         --rare-data-audit "${TUNING_SPLIT_ROOT}/certification/rare_data_audit.json"         --output "${CERTIFICATION_ROOT}/report.json"         --device cuda         --batch-size 64
    echo "PASS rare-original development selection and one-shot certification"
}

training_ready() {
    local family="$1"
    local seed="$2"
    local method="$3"
    local sampling_mode="$4"
    local expected_epoch="$5"
    local seed_root="${MODEL_ROOT}/${family}/seed${seed}"
    [[ -f "${seed_root}/train/report.json"         && -f "${seed_root}/checkpoint_manifest.json"         && -f "${seed_root}/checkpoint.pth" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import hashlib,json,sys
report=json.load(open(sys.argv[1]))
manifest=json.load(open(sys.argv[2]))
assert report["status"]=="PASS"
assert report["method"]==sys.argv[4]
assert report["sampling_mode"]==sys.argv[5]
assert int(report["epochs"])==int(sys.argv[6])
for path,expected in report["implementation_files"].items():
    assert hashlib.sha256(open(path,"rb").read()).hexdigest()==expected
checkpoint_sha=hashlib.sha256(open(sys.argv[3],"rb").read()).hexdigest()
assert manifest["checkpoint_sha256"]==checkpoint_sha
' "${seed_root}/train/report.json" "${seed_root}/checkpoint_manifest.json"         "${seed_root}/checkpoint.pth" "${method}" "${sampling_mode}" "${expected_epoch}"
}

train_family_seed() {
    local family="$1"
    local seed="$2"
    local gpu="$3"
    local sampling_mode="$4"
    local temperature="$5"
    local learning_rate="$6"
    local kl_weight="$7"
    local epochs="$8"
    local method="$9"
    local seed_root="${MODEL_ROOT}/${family}/seed${seed}"
    local train_root="${seed_root}/train"
    mkdir -p "${train_root}"
    if training_ready "${family}" "${seed}" "${method}" "${sampling_mode}" "${epochs}"; then
        echo "REUSE verified ${family} seed ${seed}"
        return
    fi
    if [[ -e "${seed_root}/checkpoint.pth" || -e "${train_root}/report.json" ]]; then
        echo "Existing ${family} seed ${seed} artifacts failed provenance; use a fresh ROOT" >&2
        return 1
    fi
    local formal_args=()
    if [[ "${family}" == "rare_frozen" || "${family}" == "paired_common" ]]; then
        formal_args+=(--formal-contract)
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${TRAINER}"         --train-cache "${CACHE_ROOT}/full/train_seed0/cache.pt"         --train-cache "${CACHE_ROOT}/full/train_seed1/cache.pt"         --train-cache "${CACHE_ROOT}/full/train_seed2/cache.pt"         --pair-manifest "${RARE_DATA_ROOT}/pairs.jsonl"         --rare-data-audit "${RARE_DATA_ROOT}/rare_data_audit.json"         --output-dir "${train_root}"         --temperature "${temperature}"         --learning-rate "${learning_rate}"         --kl-weight "${kl_weight}"         --sampling-mode "${sampling_mode}"         --method-name "${method}"         --seed "${seed}"         --epochs "${epochs}"         --examples-per-cache-epoch 6339         --checkpoint-epochs "${epochs}"         --batch-size 64         --device cuda         --ablation full         "${formal_args[@]}"
    local selector_state="${train_root}/epoch_${epochs}_scene_selector.pt"
    local selector_sha
    selector_sha="$(sha256sum "${selector_state}" | awk '{print $1}')"
    "${ALGENGINE_PYTHON}" "${MATERIALIZE}"         --baseline "${BASELINE}"         --scene-selector-state "${selector_state}"         --expected-selector-sha256 "${selector_sha}"         --expected-method "${method}"         --release-name "e2e_diffusiondrive_grpo_selector_v3_${family}_s${seed}"         --output "${seed_root}/checkpoint.pth"         --manifest "${seed_root}/checkpoint_manifest.json"
    "${ALGENGINE_PYTHON}" "${CHECKPOINT_AUDIT}"         --baseline "${BASELINE}"         --checkpoint "${seed_root}/checkpoint.pth"         --output "${seed_root}/checkpoint_audit.json"
    training_ready "${family}" "${seed}" "${method}" "${sampling_mode}" "${epochs}"
    echo "PASS trained ${family} seed ${seed}"
}

selected_values() {
    "${ALGENGINE_PYTHON}" -c '
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
selected=row["selected"]
for key in ("temperature","learning_rate","kl_weight","epoch"):
    print(selected[key])
' "${SWEEP_ROOT}/selection.json"
}

run_training() {
    for seed in 0 1 2; do
        require_file "${CACHE_ROOT}/full/train_seed${seed}/cache.pt"
    done
    require_file "${SWEEP_ROOT}/selection.json"
    require_file "${CERTIFICATION_ROOT}/report.json"
    readarray -t tuned < <(selected_values)
    local tuned_temperature="${tuned[0]}"
    local tuned_learning_rate="${tuned[1]}"
    local tuned_kl_weight="${tuned[2]}"
    local tuned_epoch="${tuned[3]}"
    local rare_method="scene_conditioned_exact_group_grpo_v3_rare_original_v1"
    local common_method="scene_conditioned_exact_group_grpo_v3_paired_common_v1"
    local tuned_method="scene_conditioned_exact_group_grpo_v3_rare_original_tuned_v1"

    CURRENT_STAGE=formal_replica_training
    local pids=()
    for seed in 0 1 2; do
        (
            train_family_seed rare_frozen "${seed}" "${seed}" rare_balanced                 1 1e-4 1e-3 16 "${rare_method}"
        ) > "${LOG_DIR}/train_rare_frozen_seed${seed}.log" 2>&1 &
        pids+=("$!")
    done
    for seed in 0 1 2; do
        (
            train_family_seed paired_common "${seed}" "$((seed + 3))" paired_common                 1 1e-4 1e-3 16 "${common_method}"
        ) > "${LOG_DIR}/train_paired_common_seed${seed}.log" 2>&1 &
        pids+=("$!")
    done
    for seed in 0 1; do
        (
            train_family_seed rare_tuned "${seed}" "$((seed + 6))" rare_balanced                 "${tuned_temperature}" "${tuned_learning_rate}"                 "${tuned_kl_weight}" "${tuned_epoch}" "${tuned_method}"
        ) > "${LOG_DIR}/train_rare_tuned_seed${seed}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    [[ "${failed}" -eq 0 ]] || {
        echo "One or more formal replica trainings failed; inspect ${LOG_DIR}/train_*.log" >&2
        return 1
    }
    train_family_seed rare_tuned 2 0 rare_balanced         "${tuned_temperature}" "${tuned_learning_rate}"         "${tuned_kl_weight}" "${tuned_epoch}" "${tuned_method}"         > "${LOG_DIR}/train_rare_tuned_seed2.log" 2>&1

    for seed in 0 1 2; do
        training_ready rare_frozen "${seed}" "${rare_method}" rare_balanced 16
        training_ready paired_common "${seed}" "${common_method}" paired_common 16
        training_ready rare_tuned "${seed}" "${tuned_method}" rare_balanced "${tuned_epoch}"
    done
    echo "PASS frozen rare, paired-common, and tuned rare replicas"
}

evaluation_ready() {
    local family="$1"
    local seed="$2"
    local model="e2e_diffusiondrive_grpo_selector_v3_${family}_s${seed}"
    local summary="${FORMAL_ROOT}/formal_eval/${model}/summary.json"
    local manifest="${MODEL_ROOT}/${family}/seed${seed}/checkpoint_manifest.json"
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
    local family="$1"
    local seed="$2"
    local seed_root="${MODEL_ROOT}/${family}/seed${seed}"
    local checkpoint="${seed_root}/checkpoint.pth"
    local manifest="${seed_root}/checkpoint_manifest.json"
    local report="${seed_root}/train/report.json"
    local model="e2e_diffusiondrive_grpo_selector_v3_${family}_s${seed}"
    if evaluation_ready "${family}" "${seed}"; then
        echo "REUSE verified formal evaluation ${family} seed ${seed}"
        return
    fi
    local checkpoint_sha temperature kl_weight
    checkpoint_sha="$(json_get "${manifest}" checkpoint_sha256)"
    temperature="$(json_get "${report}" temperature)"
    kl_weight="$(json_get "${report}" kl_weight)"
    export DIFFUSIONDRIVE_GRPO_FORMAL_ROOT="${FORMAL_ROOT}"
    export DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE="${temperature}"
    export DIFFUSIONDRIVE_GRPO_KL_WEIGHT="${kl_weight}"
    "${FORMAL_TABLE}"         "${checkpoint}" "${checkpoint_sha}" "${model}" "${seed}"         "V3 rare-original-v1 family=${family}; paired eval seed=${seed}"
    evaluation_ready "${family}" "${seed}"
}

run_evaluation_family() {
    local family="$1"
    case "${family}" in
        paired_common|rare_frozen|rare_tuned) ;;
        *)
            echo "Invalid evaluation family: ${family}" >&2
            return 2
            ;;
    esac
    for seed in 0 1 2; do
        CURRENT_STAGE="formal_eval_${family}_seed${seed}"
        run_evaluation_seed "${family}" "${seed}"
    done
    echo "PASS three formal four-block evaluations family=${family}"
}

run_evaluations() {
    for family in paired_common rare_frozen rare_tuned; do
        run_evaluation_family "${family}"
    done
    echo "PASS nine formal four-block evaluations"
}

run_summary() {
    CURRENT_STAGE=formal_aggregate
    local args=()
    for seed in 0 1 2; do
        args+=(
            --base "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/formal_eval/e2e_diffusiondrive_reference_paired_s${seed}/summary.json"
            --common-v3 "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_progress_fix_v1_s${seed}/summary.json"
            --paired-common "${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_v3_paired_common_s${seed}/summary.json"
            --rare-frozen "${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_frozen_s${seed}/summary.json"
            --rare-tuned "${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_tuned_s${seed}/summary.json"
        )
    done
    "${ALGENGINE_PYTHON}" "${SUMMARIZER}"         "${args[@]}"         --rare-data-audit "${RARE_DATA_ROOT}/rare_data_audit.json"         --tuning-split-audit "${TUNING_SPLIT_ROOT}/split_audit.json"         --certification "${CERTIFICATION_ROOT}/report.json"         --output "${FORMAL_ROOT}/rare_original_comparison.json"
    echo "PASS formal aggregate: ${FORMAL_ROOT}/rare_original_comparison.md"
}

case "${MODE}" in
    preflight)
        run_preflight
        ;;
    mine)
        run_preflight
        run_mining
        ;;
    mine-seed0|mine-seed1|mine-seed2)
        run_preflight
        run_mining_seed "${MODE#mine-seed}"
        ;;
    cache)
        run_preflight
        run_caches
        ;;
    cache-lane0|cache-lane1|cache-lane2)
        run_preflight
        run_cache_lane "${MODE#cache-lane}"
        ;;
    sweep)
        run_preflight
        run_sweep
        ;;
    train)
        run_preflight
        run_training
        ;;
    eval)
        run_preflight
        run_evaluations
        ;;
    eval-paired-common)
        run_preflight
        run_evaluation_family paired_common
        ;;
    eval-rare-frozen)
        run_preflight
        run_evaluation_family rare_frozen
        ;;
    eval-rare-tuned)
        run_preflight
        run_evaluation_family rare_tuned
        ;;
    summarize)
        run_summary
        ;;
    all)
        run_preflight
        run_mining
        run_caches
        run_sweep
        run_training
        run_evaluations
        run_summary
        ;;
esac

CURRENT_STAGE=complete
printf 'PASS mode=%s code_sha=%s completed_utc=%s\n'     "${MODE}" "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive selector V3 rare-original-v1 mode ${MODE}"
echo "status: ${STATUS_FILE}"
echo "persistent_log: ${LOG_FILE}"
