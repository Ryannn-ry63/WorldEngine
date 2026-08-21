#!/usr/bin/env bash
# V2 rare-log: shared V3 data/caches, independent V2 tuning, paired formal seeds.

set -Eeo pipefail

MODE="${1:-preflight}"
SEED="${2:-}"
case "${MODE}" in
    preflight|tune|formal-seed|summarize) ;;
    *) echo "Usage: $0 [preflight|tune|formal-seed SEED|summarize]" >&2; exit 2 ;;
esac
if [[ "${MODE}" == "formal-seed" && ! "${SEED}" =~ ^[012]$ ]]; then
    echo "formal-seed requires seed 0, 1, or 2" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/projects/AlgEngine/scripts/diffusiondrive"
if [[ "${MODE}" == "summarize" ]]; then
    export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
else
    export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
fi
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2_rare.py"
. "${SCRIPT_DIR}/grpo_selector_v2_rare_h100_env.sh"

BASELINE="${DIFFUSIONDRIVE_GRPO_BASELINE}"
BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
SHARED_SOURCE="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1"
RARE_DATA="${SHARED_SOURCE}/rare_data"
TUNING_SPLIT="${SHARED_SOURCE}/tuning_split"
SHARED_CACHE="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2_rare_original_v1"
SWEEP_ROOT="${ROOT}/sweep"
CERT_ROOT="${ROOT}/certification"
MODEL_ROOT="${ROOT}/models"
FORMAL_ROOT="${ROOT}/formal"
LOG_DIR="${ROOT}/logs"
TRAINER="${SCRIPT_DIR}/train_grpo_selector_v2_cached_rare_original.py"
EVALUATOR="${SCRIPT_DIR}/evaluate_grpo_selector_v2_rare_original.py"
SELECTOR="${SCRIPT_DIR}/select_grpo_selector_v2_rare_original.py"
CERTIFIER="${SCRIPT_DIR}/certify_grpo_selector_v2_rare_original.py"
MATERIALIZER="${SCRIPT_DIR}/materialize_grpo_selector_v2_rare.py"
AUDITOR="${SCRIPT_DIR}/audit_grpo_selector_v2_rare_checkpoint.py"
FORMAL_TABLE="${SCRIPT_DIR}/run_grpo_selector_v2_rare_formal_table_h100.sh"
SUMMARIZER="${SCRIPT_DIR}/summarize_grpo_selector_v2_rare.py"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${MODE}${SEED:+_s${SEED}}_$(date -u +%Y%m%dT%H%M%SZ).log"
STATUS="${ROOT}/status_${MODE}${SEED:+_s${SEED}}.txt"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=bootstrap
CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
printf 'RUNNING mode=%s seed=%s code_sha=%s\n' "${MODE}" "${SEED:-none}" "${CODE_SHA}" > "${STATUS}"
trap 'rc=$?; printf "FAIL mode=%s seed=%s stage=%s exit=%s\n" "${MODE}" "${SEED:-none}" "${CURRENT_STAGE}" "${rc}" > "${STATUS}"; echo "FAIL V2 rare-log stage=${CURRENT_STAGE} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR

json_pass() {
    "${ALGENGINE_PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="PASS"' "$1"
}

run_preflight() {
    CURRENT_STAGE=preflight
    for path in "${BASELINE}" "${DIFFUSIONDRIVE_GRPO_CONFIG}" "${TRAINER}" \
        "${EVALUATOR}" "${SELECTOR}" "${CERTIFIER}" "${MATERIALIZER}" \
        "${AUDITOR}" "${FORMAL_TABLE}" "${SUMMARIZER}" \
        "${RARE_DATA}/pairs.jsonl" "${RARE_DATA}/rare_data_audit.json" \
        "${TUNING_SPLIT}/split_audit.json"; do
        [[ -f "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 1; }
    done
    for seed in 0 1 2; do
        for path in \
            "${SHARED_CACHE}/full/train_seed${seed}/cache.pt" \
            "${SHARED_CACHE}/tuning/train_seed${seed}/cache.pt" \
            "${SHARED_CACHE}/tuning/development_seed$((seed + 3))/cache.pt" \
            "${SHARED_CACHE}/tuning/certification_seed$((seed + 6))/cache.pt"; do
            [[ -f "${path}" ]] || { echo "Missing shared cache: ${path}" >&2; exit 1; }
        done
    done
    [[ "$(sha256sum "${BASELINE}" | awk '{print $1}')" == "${BASELINE_SHA256}" ]]
    if ! git -C "${WORLDENGINE_ROOT}" diff --quiet -- projects/AlgEngine projects/SimEngine; then
        echo "Tracked implementation files are dirty" >&2
        exit 1
    fi
    "${ALGENGINE_PYTHON}" "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
        --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible
    echo "PASS V2 rare-log shared-data preflight"
}

trial_ready() {
    [[ -f "$1/report.json" && -f "$1/development_evaluation.json" ]] || return 1
    json_pass "$1/report.json"
    json_pass "$1/development_evaluation.json"
}

run_trial() {
    local gpu="$1" temperature="$2" learning_rate="$3" kl_weight="$4"
    local name="t${temperature}_lr${learning_rate}_kl${kl_weight}"
    local trial="${SWEEP_ROOT}/trials/${name}"
    mkdir -p "${trial}"
    if trial_ready "${trial}"; then
        echo "SKIP verified V2 rare-log trial ${name}"
        return
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --train-cache "${SHARED_CACHE}/tuning/train_seed0/cache.pt" \
        --train-cache "${SHARED_CACHE}/tuning/train_seed1/cache.pt" \
        --train-cache "${SHARED_CACHE}/tuning/train_seed2/cache.pt" \
        --pair-manifest "${TUNING_SPLIT}/train/pairs.jsonl" \
        --rare-data-audit "${TUNING_SPLIT}/train/rare_data_audit.json" \
        --output-dir "${trial}" --temperature "${temperature}" \
        --learning-rate "${learning_rate}" --kl-weight "${kl_weight}" \
        --sampling-mode rare_balanced --seed 0 --epochs 64 \
        --examples-per-cache-epoch 6339 --checkpoint-epochs 1,2,4,8,16,32,64 \
        --batch-size 64 --device cuda
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${EVALUATOR}" \
        --training-report "${trial}/report.json" \
        --cache "${SHARED_CACHE}/tuning/development_seed3/cache.pt" \
        --cache "${SHARED_CACHE}/tuning/development_seed4/cache.pt" \
        --cache "${SHARED_CACHE}/tuning/development_seed5/cache.pt" \
        --split development --pair-manifest "${TUNING_SPLIT}/development/pairs.jsonl" \
        --rare-data-audit "${TUNING_SPLIT}/development/rare_data_audit.json" \
        --output "${trial}/development_evaluation.json" --device cuda --batch-size 64
    trial_ready "${trial}"
}

run_tune() {
    CURRENT_STAGE=development_sweep
    mkdir -p "${SWEEP_ROOT}/trials" "${CERT_ROOT}"
    local temperatures=(1 1 1 1 1 1 1 2)
    local learning_rates=(1e-4 1e-4 3e-4 3e-4 3e-4 1e-3 1e-3 3e-4)
    local kl_weights=(0 1e-3 0 1e-4 1e-3 1e-4 1e-3 1e-3)
    local pids=()
    for gpu in 0 1 2 3 4 5 6 7; do
        run_trial "${gpu}" "${temperatures[${gpu}]}" \
            "${learning_rates[${gpu}]}" "${kl_weights[${gpu}]}" \
            > "${LOG_DIR}/tune_gpu${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    [[ "${failed}" -eq 0 ]] || { echo "V2 sweep lane failed" >&2; return 1; }
    CURRENT_STAGE=development_selection
    "${ALGENGINE_PYTHON}" "${SELECTOR}" --trial-root "${SWEEP_ROOT}/trials" \
        --expected-candidates 56 --output "${SWEEP_ROOT}/selection.json"
    CURRENT_STAGE=one_shot_certification
    CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${CERTIFIER}" \
        --selection "${SWEEP_ROOT}/selection.json" \
        --cache "${SHARED_CACHE}/tuning/certification_seed6/cache.pt" \
        --cache "${SHARED_CACHE}/tuning/certification_seed7/cache.pt" \
        --cache "${SHARED_CACHE}/tuning/certification_seed8/cache.pt" \
        --pair-manifest "${TUNING_SPLIT}/certification/pairs.jsonl" \
        --rare-data-audit "${TUNING_SPLIT}/certification/rare_data_audit.json" \
        --output "${CERT_ROOT}/report.json" --device cuda --batch-size 64
    echo "PASS V2 rare-log independent tuning and certification"
}

selected_values() {
    "${ALGENGINE_PYTHON}" -c '
import json,sys
x=json.load(open(sys.argv[1]))["selected"]
for key in ("temperature","learning_rate","kl_weight","epoch"): print(x[key])
' "${SWEEP_ROOT}/selection.json"
}

tuning_ready() {
    [[ -f "${SWEEP_ROOT}/selection.json" && -f "${CERT_ROOT}/report.json" ]] || return 1
    json_pass "${SWEEP_ROOT}/selection.json" && json_pass "${CERT_ROOT}/report.json"
}

wait_for_tuning() {
    local started now line=""
    local timeout="${DIFFUSIONDRIVE_V2_TUNE_WAIT_TIMEOUT_SECONDS:-86400}"
    local poll="${DIFFUSIONDRIVE_V2_TUNE_WAIT_POLL_SECONDS:-30}"
    local seed0_status="${ROOT}/status_formal-seed_s0.txt"
    started="$(date +%s)"
    while ! tuning_ready; do
        if [[ -f "${seed0_status}" ]]; then
            IFS= read -r line < "${seed0_status}" || true
            if [[ "${line}" == FAIL\ * ]]; then
                echo "Seed 0 tuning/formal job failed: ${line}" >&2
                return 1
            fi
        fi
        now="$(date +%s)"
        if (( now - started >= timeout )); then
            echo "Timed out waiting ${timeout}s for seed 0 tuning certification" >&2
            return 1
        fi
        echo "WAIT seed ${SEED}: fixed result is ready; waiting for seed 0 tuning certification"
        sleep "${poll}"
    done
    echo "PASS seed ${SEED}: verified seed 0 tuning certification"
}

training_ready() {
    local family="$1" seed="$2" method="$3" epoch="$4"
    local root="${MODEL_ROOT}/${family}/seed${seed}"
    [[ -f "${root}/train/report.json" && -f "${root}/checkpoint.pth" \
        && -f "${root}/checkpoint_manifest.json" && -f "${root}/checkpoint_audit.json" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import hashlib,json,sys
r=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2])); a=json.load(open(sys.argv[3]))
assert r["status"]==m["status"]==a["status"]=="PASS"
assert r["architecture"]=="diffusiondrive_plan_cls_branch_v2"
assert r["method"]==sys.argv[5] and int(r["epochs"])==int(sys.argv[6])
if int(sys.argv[6])==16:
 assert r["sampling"]["total_examples"]==304272
 assert r["sampling"]["total_optimizer_steps"]==4800
sha=hashlib.sha256(open(sys.argv[4],"rb").read()).hexdigest()
assert sha==m["checkpoint_sha256"]==a["checkpoint_sha256"]
assert a["changed_tensor_count"]==10 and a["forbidden_changed_tensor_count"]==0
' "${root}/train/report.json" "${root}/checkpoint_manifest.json" \
        "${root}/checkpoint_audit.json" "${root}/checkpoint.pth" "${method}" "${epoch}"
}

train_family() {
    local family="$1" seed="$2" gpu="$3" temperature="$4" learning_rate="$5"
    local kl_weight="$6" epochs="$7" method="$8"
    local root="${MODEL_ROOT}/${family}/seed${seed}"
    if training_ready "${family}" "${seed}" "${method}" "${epochs}"; then
        echo "SKIP verified ${family} seed ${seed}"
        return
    fi
    [[ ! -e "${root}" ]] || { echo "Partial immutable root exists: ${root}" >&2; return 1; }
    mkdir -p "${root}/train"
    local formal_args=()
    [[ "${family}" == "rare_original_fixed" ]] && formal_args+=(--formal-contract)
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --train-cache "${SHARED_CACHE}/full/train_seed0/cache.pt" \
        --train-cache "${SHARED_CACHE}/full/train_seed1/cache.pt" \
        --train-cache "${SHARED_CACHE}/full/train_seed2/cache.pt" \
        --pair-manifest "${RARE_DATA}/pairs.jsonl" \
        --rare-data-audit "${RARE_DATA}/rare_data_audit.json" \
        --output-dir "${root}/train" --temperature "${temperature}" \
        --learning-rate "${learning_rate}" --kl-weight "${kl_weight}" \
        --sampling-mode rare_balanced --method-name "${method}" --seed "${seed}" \
        --epochs "${epochs}" --examples-per-cache-epoch 6339 \
        --checkpoint-epochs "${epochs}" --batch-size 64 --device cuda "${formal_args[@]}"
    local state="${root}/train/epoch_${epochs}_selector.pt"
    local state_sha="$(sha256sum "${state}" | awk '{print $1}')"
    "${ALGENGINE_PYTHON}" "${MATERIALIZER}" --baseline "${BASELINE}" \
        --selector-state "${state}" --expected-selector-sha256 "${state_sha}" \
        --expected-method "${method}" --release-name "e2e_diffusiondrive_grpo_selector_v2_${family}_s${seed}" \
        --output "${root}/checkpoint.pth" --manifest "${root}/checkpoint_manifest.json"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" --baseline "${BASELINE}" \
        --checkpoint "${root}/checkpoint.pth" --output "${root}/checkpoint_audit.json"
    training_ready "${family}" "${seed}" "${method}" "${epochs}"
}

evaluation_ready() {
    local family="$1" seed="$2" model="e2e_diffusiondrive_grpo_selector_v2_${1}_s${2}"
    [[ -f "${FORMAL_ROOT}/formal_eval/${model}/summary.json" ]] || return 1
    json_pass "${FORMAL_ROOT}/formal_eval/${model}/summary.json"
}

evaluate_family() {
    local family="$1" seed="$2"
    if evaluation_ready "${family}" "${seed}"; then
        echo "SKIP verified eval ${family} seed ${seed}"
        return
    fi
    local root="${MODEL_ROOT}/${family}/seed${seed}"
    local checkpoint_sha="$(sha256sum "${root}/checkpoint.pth" | awk '{print $1}')"
    local temperature="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["temperature"])' "${root}/train/report.json")"
    local kl_weight="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["kl_weight"])' "${root}/train/report.json")"
    export DIFFUSIONDRIVE_GRPO_FORMAL_ROOT="${FORMAL_ROOT}"
    export DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE="${temperature}"
    export DIFFUSIONDRIVE_GRPO_KL_WEIGHT="${kl_weight}"
    "${FORMAL_TABLE}" "${root}/checkpoint.pth" "${checkpoint_sha}" \
        "e2e_diffusiondrive_grpo_selector_v2_${family}_s${seed}" "${seed}" \
        "V2 corrected rare-log family=${family}; shared V3 data; eval seed=${seed}"
    evaluation_ready "${family}" "${seed}"
}

run_formal_seed() {
    if [[ "${SEED}" == "0" ]] && ! tuning_ready; then
        CURRENT_STAGE=tune_coordinator
        run_tune
    fi

    if [[ "${SEED}" != "0" ]]; then
        CURRENT_STAGE=train_fixed
        train_family rare_original_fixed "${SEED}" 0 1 1e-4 1e-3 16 \
            exact_group_grpo_v2_rare_original_v1 \
            > "${LOG_DIR}/train_fixed_s${SEED}.log" 2>&1
        CURRENT_STAGE=evaluate_fixed
        evaluate_family rare_original_fixed "${SEED}"
        CURRENT_STAGE=wait_for_seed0_tuning
        wait_for_tuning
        readarray -t tuned < <(selected_values)
        CURRENT_STAGE=train_tuned
        train_family rare_original_tuned "${SEED}" 0 "${tuned[0]}" "${tuned[1]}" \
            "${tuned[2]}" "${tuned[3]}" exact_group_grpo_v2_rare_original_tuned_v1 \
            > "${LOG_DIR}/train_tuned_s${SEED}.log" 2>&1
        CURRENT_STAGE=evaluate_tuned
        evaluate_family rare_original_tuned "${SEED}"
        echo "PASS V2 rare-log coordinated fixed+tuned formal seed ${SEED}"
        return
    fi

    tuning_ready
    readarray -t tuned < <(selected_values)
    CURRENT_STAGE=train_fixed_and_tuned
    train_family rare_original_fixed "${SEED}" 0 1 1e-4 1e-3 16 \
        exact_group_grpo_v2_rare_original_v1 > "${LOG_DIR}/train_fixed_s${SEED}.log" 2>&1 &
    local fixed_pid=$!
    train_family rare_original_tuned "${SEED}" 1 "${tuned[0]}" "${tuned[1]}" \
        "${tuned[2]}" "${tuned[3]}" exact_group_grpo_v2_rare_original_tuned_v1 \
        > "${LOG_DIR}/train_tuned_s${SEED}.log" 2>&1 &
    local tuned_pid=$!
    local failed=0
    wait "${fixed_pid}" || failed=1
    wait "${tuned_pid}" || failed=1
    [[ "${failed}" -eq 0 ]] || { echo "Formal training failed" >&2; return 1; }
    CURRENT_STAGE=evaluate_fixed
    evaluate_family rare_original_fixed "${SEED}"
    CURRENT_STAGE=evaluate_tuned
    evaluate_family rare_original_tuned "${SEED}"
    echo "PASS V2 rare-log coordinated fixed+tuned formal seed ${SEED}"
}

run_summary() {
    CURRENT_STAGE=summarize
    local args=()
    for seed in 0 1 2; do
        args+=(--input "epoch100:${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/formal_eval/e2e_diffusiondrive_reference_paired_s${seed}/summary.json")
        mapfile -t common_matches < <(find "${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2_progress_fix_v1/formal/formal_eval" \
            -mindepth 2 -maxdepth 2 -path "*/e2e_diffusiondrive_grpo_selector_v2_progress_fix_v1_s${seed}_*/summary.json")
        [[ "${#common_matches[@]}" -eq 1 ]] || { echo "Cannot uniquely resolve V2 common seed ${seed}" >&2; return 1; }
        args+=(--input "v2_common_progress_fixed:${seed}=${common_matches[0]}")
        args+=(--input "rare_original_fixed:${seed}=${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_v2_rare_original_fixed_s${seed}/summary.json")
        args+=(--input "rare_original_tuned:${seed}=${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_v2_rare_original_tuned_s${seed}/summary.json")
    done
    "${ALGENGINE_PYTHON}" "${SUMMARIZER}" "${args[@]}" \
        --provenance "${RARE_DATA}/rare_data_audit.json" \
        --provenance "${TUNING_SPLIT}/split_audit.json" \
        --provenance "${SWEEP_ROOT}/selection.json" \
        --provenance "${CERT_ROOT}/report.json" \
        --output "${FORMAL_ROOT}/v2_rare_original_comparison.json"
}

if [[ "${MODE}" != "summarize" ]]; then
    run_preflight
fi
case "${MODE}" in
    preflight) ;;
    tune) run_tune ;;
    formal-seed) run_formal_seed ;;
    summarize) run_summary ;;
esac
CURRENT_STAGE=complete
printf 'PASS mode=%s seed=%s code_sha=%s\n' "${MODE}" "${SEED:-none}" "${CODE_SHA}" > "${STATUS}"
trap - ERR
echo "PASS V2 rare-log mode=${MODE}${SEED:+ seed=${SEED}}"
echo "status: ${STATUS}"
echo "persistent_log: ${LOG_FILE}"
