#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
SEED="${2:-}"
case "${MODE}" in
    preflight|prepare|select|smoke|summarize) ;;
    tune-lane|formal)
        if ! [[ "${SEED}" =~ ^[012]$ ]]; then
            echo "Usage: $0 ${MODE} {0|1|2}" >&2
            exit 2
        fi
        ;;
    *)
        echo "Usage: $0 [preflight|prepare|smoke|tune-lane {0|1|2}|select|formal {0|1|2}|summarize]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORLDENGINE_ROOT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export DIFFUSIONDRIVE_GRPO_CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
export ALGENGINE_ENV="${DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine}"
export ALGENGINE_PYTHON="${ALGENGINE_ENV}/bin/python"
export ALGENGINE_TORCHRUN="${ALGENGINE_ENV}/bin/torchrun"
export SIMENGINE_PYTHON="${SIMENGINE_PYTHON:-/root/miniconda3/envs/simengine/bin/python}"

if [[ "${MODE}" != "prepare" && "${MODE}" != "select" && "${MODE}" != "summarize" ]]; then
    . "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
fi

CONFIG="${DIFFUSIONDRIVE_GRPO_CONFIG}"
BASELINE="${DIFFUSIONDRIVE_GRPO_BASELINE:-/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth}"
BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
SOURCE_SCENARIOS="${WORLDENGINE_ROOT}/data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl"
SOURCE_ASSETS="${WORLDENGINE_ROOT}/data/sim_engine/assets/navtest_failures/assets"
OLD_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1"
OLD_SOURCE_ROOT="${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_clpdms_tuning_v1"
SPLIT_ROOT="${ROOT}/closed_loop_split"
MODEL_ROOT="${ROOT}/models"
METRICS_ROOT="${ROOT}/development_metrics"
FORMAL_ROOT="${ROOT}/formal"
LOG_DIR="${ROOT}/logs"
RECIPE_CONFIG="${SCRIPT_DIR}/grpo_selector_v3_rare_clpdms_recipes.json"
AUDITOR="${SCRIPT_DIR}/grpo_selector_v3_rare_clpdms_tuning.py"
TRAINER="${SCRIPT_DIR}/train_grpo_selector_v3_cached_rare_original.py"
MATERIALIZE="${SCRIPT_DIR}/materialize_grpo_selector_v3.py"
CHECKPOINT_AUDIT="${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py"
RAY_RUNNER="${SCRIPT_DIR}/run_ray_distributed_testing_diffusiondrive_h100.sh"
FORMAL_TABLE="${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh"
PAIR_MANIFEST="${OLD_SOURCE_ROOT}/rare_data/pairs.jsonl"
RARE_DATA_AUDIT="${OLD_SOURCE_ROOT}/rare_data/rare_data_audit.json"
CACHE_ROOT="${OLD_ROOT}/cache/full"
SPLIT_AUDIT="${SPLIT_ROOT}/split_audit.json"
SCREEN_REPORT="${ROOT}/selection/seed0_screen.json"
SELECTION_REPORT="${ROOT}/selection/selection.json"
LANE_REPORT="${ROOT}/selection/lane_seed${SEED:-unset}.json"
FINAL_REPORT="${FORMAL_ROOT}/clpdms_tuning_summary.json"
CHALLENGERS=(early32 early48 lowlr48 lowlr64 anchored64 lowlr_anchored64)
CONTROLS=(rare_tuned rare_frozen)

mkdir -p "${LOG_DIR}" "${ROOT}/selection"
LOG_FILE="${LOG_DIR}/${MODE}${SEED:+_seed${SEED}}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=bootstrap
trap 'rc=$?; echo "FAIL rare-clpdms-v1 mode='"${MODE}"' stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: '"${LOG_FILE}"'" >&2' ERR

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

json_pass() {
    "${ALGENGINE_PYTHON}" -c '
import json,sys
payload=json.load(open(sys.argv[1]))
assert payload.get("status")=="PASS"
' "$1"
}

recipe_value() {
    local name="$1"
    local key="$2"
    "${ALGENGINE_PYTHON}" -c '
import json,sys
payload=json.load(open(sys.argv[1]))
name,key=sys.argv[2:]
rows={**payload["controls"], **{row["name"]:row for row in payload["challengers"]}}
print(rows[name][key])
' "${RECIPE_CONFIG}" "${name}" "${key}"
}

control_manifest() {
    local name="$1"
    local seed="$2"
    local relative
    relative="$(recipe_value "${name}" model_root)"
    echo "${WORLDENGINE_ROOT}/${relative}/seed${seed}/checkpoint_manifest.json"
}

control_summary() {
    local name="$1"
    local seed="$2"
    local template
    template="$(recipe_value "${name}" formal_summary_template)"
    echo "${WORLDENGINE_ROOT}/${template/\{seed\}/${seed}}"
}

checkpoint_manifest() {
    local name="$1"
    local seed="$2"
    if [[ "${name}" == "rare_tuned" || "${name}" == "rare_frozen" ]]; then
        control_manifest "${name}" "${seed}"
    else
        echo "${MODEL_ROOT}/${name}/seed${seed}/checkpoint_manifest.json"
    fi
}

checkpoint_path() {
    "${ALGENGINE_PYTHON}" -c '
import json,sys
print(json.load(open(sys.argv[1]))["checkpoint"])
' "$1"
}

summary_reactive_csv() {
    "${ALGENGINE_PYTHON}" -c '
import json,sys
print(json.load(open(sys.argv[1]))["input_files"]["closedloop_r"]["path"])
' "$1"
}

run_static_preflight() {
    CURRENT_STAGE=static_preflight
    for path in "${CONFIG}" "${BASELINE}" "${SOURCE_SCENARIOS}" "${RECIPE_CONFIG}"         "${AUDITOR}" "${TRAINER}" "${MATERIALIZE}" "${CHECKPOINT_AUDIT}"         "${RAY_RUNNER}" "${FORMAL_TABLE}" "${PAIR_MANIFEST}" "${RARE_DATA_AUDIT}"; do
        require_file "${path}"
    done
    [[ -d "${SOURCE_ASSETS}" ]] || { echo "Missing navtest_failures assets" >&2; exit 1; }
    [[ "$(sha256sum "${BASELINE}" | awk '{print $1}')" == "${BASELINE_SHA256}" ]] || {
        echo "Baseline checkpoint SHA256 drifted" >&2
        exit 1
    }
    for seed in 0 1 2; do
        require_file "${CACHE_ROOT}/train_seed${seed}/cache.pt"
        require_file "${CACHE_ROOT}/train_seed${seed}/manifest.json"
    done
    json_pass "${RARE_DATA_AUDIT}"
    if [[ -n "$(git -C "${WORLDENGINE_ROOT}" status --porcelain --untracked-files=no)" ]]; then
        echo "Tracked worktree is dirty; commit the tuning code before GPU execution" >&2
        exit 1
    fi
}

run_cuda_preflight() {
    CURRENT_STAGE=cuda_preflight
    export H100_SUPPORT_DIR="${SIMENGINE_ROOT}/scripts/diffusiondrive"
    export WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP=1
    export WORLDENGINE_GSPLAT_EXTENSION="${WORLDENGINE_ROOT}/artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so"
    export PYTHONPATH="${DIFFUSIONDRIVE_BOOTSTRAP}:${H100_SUPPORT_DIR}:${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${DIFFUSIONDRIVE_ROOT}:${PYTHONPATH:-}"
    require_file "${WORLDENGINE_MMCV_EXTENSION}"
    require_file "${WORLDENGINE_GSPLAT_EXTENSION}"
    "${ALGENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_mmcv_cuda.py"         --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible
    "${SIMENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_gsplat_cuda.py"         --extension "${WORLDENGINE_GSPLAT_EXTENSION}" --expected-capability sm_90         --ray-workers "${DIFFUSIONDRIVE_EXPECTED_GPU_COUNT:-8}"
}

ensure_split() {
    CURRENT_STAGE=closed_loop_log_disjoint_split
    local code_sha
    code_sha="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" split         --source "${SOURCE_SCENARIOS}"         --output-root "${SPLIT_ROOT}"         --split-seed 20260820         --development-fraction 0.2         --code-commit "${code_sha}"         --expected-source-scenes 288         --expected-source-logs 89         --expected-development-scenes 58         --expected-development-logs 22         --expected-confirmation-scenes 230         --expected-confirmation-logs 67
    json_pass "${SPLIT_AUDIT}"
}

training_ready() {
    local name="$1"
    local seed="$2"
    local manifest="${MODEL_ROOT}/${name}/seed${seed}/checkpoint_manifest.json"
    local report="${MODEL_ROOT}/${name}/seed${seed}/train/report.json"
    local audit="${MODEL_ROOT}/${name}/seed${seed}/checkpoint_audit.json"
    [[ -f "${manifest}" && -f "${report}" && -f "${audit}" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import hashlib,json,sys
manifest=json.load(open(sys.argv[1]))
report=json.load(open(sys.argv[2]))
assert manifest["status"]=="PASS" and report["status"]=="PASS"
assert report["method"]==sys.argv[3]
assert float(report["temperature"])==float(sys.argv[4])
assert float(report["learning_rate"])==float(sys.argv[5])
assert float(report["kl_weight"])==float(sys.argv[6])
assert int(report["epochs"])==int(sys.argv[7])
checkpoint=manifest["checkpoint"]
digest=hashlib.sha256()
with open(checkpoint,"rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
digest=digest.hexdigest()
assert digest==manifest["checkpoint_sha256"]
audit=json.load(open(sys.argv[8]))
assert audit["status"]=="PASS"
assert int(audit["changed_baseline_tensor_count"])==0
assert audit["checkpoint_sha256"]==manifest["checkpoint_sha256"]
' "${manifest}" "${report}"         "scene_conditioned_exact_group_grpo_v3_rare_clpdms_tuning_v1_${name}"         "$(recipe_value "${name}" temperature)"         "$(recipe_value "${name}" learning_rate)"         "$(recipe_value "${name}" kl_weight)"         "$(recipe_value "${name}" epoch)" "${audit}"
}

train_candidate() {
    local name="$1"
    local seed="$2"
    local gpu="$3"
    local temperature learning_rate kl_weight epoch method seed_root train_root
    temperature="$(recipe_value "${name}" temperature)"
    learning_rate="$(recipe_value "${name}" learning_rate)"
    kl_weight="$(recipe_value "${name}" kl_weight)"
    epoch="$(recipe_value "${name}" epoch)"
    method="scene_conditioned_exact_group_grpo_v3_rare_clpdms_tuning_v1_${name}"
    seed_root="${MODEL_ROOT}/${name}/seed${seed}"
    train_root="${seed_root}/train"
    if training_ready "${name}" "${seed}"; then
        echo "REUSE verified candidate ${name} seed ${seed}"
        return
    fi
    if [[ -e "${seed_root}/checkpoint.pth" || -e "${train_root}/report.json" ]]; then
        echo "Existing candidate artifacts failed provenance: ${seed_root}" >&2
        return 1
    fi
    mkdir -p "${train_root}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${TRAINER}"         --train-cache "${CACHE_ROOT}/train_seed0/cache.pt"         --train-cache "${CACHE_ROOT}/train_seed1/cache.pt"         --train-cache "${CACHE_ROOT}/train_seed2/cache.pt"         --pair-manifest "${PAIR_MANIFEST}"         --rare-data-audit "${RARE_DATA_AUDIT}"         --output-dir "${train_root}"         --temperature "${temperature}"         --learning-rate "${learning_rate}"         --kl-weight "${kl_weight}"         --sampling-mode rare_balanced         --method-name "${method}"         --seed "${seed}"         --epochs "${epoch}"         --examples-per-cache-epoch 6339         --checkpoint-epochs "${epoch}"         --batch-size 64         --device cuda         --ablation full
    local selector_state selector_sha
    selector_state="${train_root}/epoch_${epoch}_scene_selector.pt"
    selector_sha="$(sha256sum "${selector_state}" | awk '{print $1}')"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${MATERIALIZE}"         --baseline "${BASELINE}"         --scene-selector-state "${selector_state}"         --expected-selector-sha256 "${selector_sha}"         --expected-method "${method}"         --release-name "e2e_diffusiondrive_grpo_selector_v3_rare_clpdms_v1_${name}_s${seed}"         --output "${seed_root}/checkpoint.pth"         --manifest "${seed_root}/checkpoint_manifest.json"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${CHECKPOINT_AUDIT}"         --baseline "${BASELINE}"         --checkpoint "${seed_root}/checkpoint.pth"         --output "${seed_root}/checkpoint_audit.json"
    training_ready "${name}" "${seed}"
    echo "PASS candidate training ${name} seed ${seed}"
}

run_control_metrics() {
    local name="$1"
    local seed="$2"
    local summary csv manifest output
    summary="$(control_summary "${name}" "${seed}")"
    manifest="$(control_manifest "${name}" "${seed}")"
    require_file "${summary}"
    require_file "${manifest}"
    csv="$(summary_reactive_csv "${summary}")"
    output="${METRICS_ROOT}/${name}/seed${seed}.json"
    mkdir -p "$(dirname "${output}")"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" metrics         --split-audit "${SPLIT_AUDIT}"         --split development         --csv "${csv}"         --model "${name}"         --seed "${seed}"         --checkpoint-manifest "${manifest}"         --allow-superset         --output "${output}"
}

run_candidate_metrics() {
    local name="$1"
    local seed="$2"
    local manifest checkpoint model csv output
    manifest="${MODEL_ROOT}/${name}/seed${seed}/checkpoint_manifest.json"
    checkpoint="$(checkpoint_path "${manifest}")"
    model="e2e_diffusiondrive_grpo_selector_v3_rare_clpdms_v1_${name}_s${seed}"
    export DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE="formal_navtest_seed${seed}"
    export DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE="$(recipe_value "${name}" temperature)"
    export DIFFUSIONDRIVE_GRPO_KL_WEIGHT="$(recipe_value "${name}" kl_weight)"
    CURRENT_STAGE="cl_dev_${name}_seed${seed}"
    "${RAY_RUNNER}"         "${CONFIG}" "${checkpoint}" "${model}" rare_clpdms_dev_v1 R         navtest_failures "${SPLIT_ROOT}/development/all_scenarios.pkl" 8
    csv="${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${model}/rare_clpdms_dev_v1_R/WE_output/openscene_format/all_scenes_pdm_averages_R.csv"
    output="${METRICS_ROOT}/${name}/seed${seed}.json"
    mkdir -p "$(dirname "${output}")"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" metrics         --split-audit "${SPLIT_AUDIT}"         --split development         --csv "${csv}"         --model "${name}"         --seed "${seed}"         --checkpoint-manifest "${manifest}"         --output "${output}"
}

train_candidate_grid() {
    local seed="$1"
    CURRENT_STAGE="train_seed${seed}_grid"
    local pids=()
    local gpu=0
    for name in "${CHALLENGERS[@]}"; do
        (
            train_candidate "${name}" "${seed}" "${gpu}"
        ) > "${LOG_DIR}/train_${name}_seed${seed}.log" 2>&1 &
        pids+=("$!")
        gpu=$((gpu + 1))
    done
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then failed=1; fi
    done
    [[ "${failed}" -eq 0 ]] || {
        echo "Seed ${seed} grid training failed; inspect ${LOG_DIR}/train_*_seed${seed}.log" >&2
        return 1
    }
}

screen_seed0() {
    CURRENT_STAGE=seed0_screen
    "${ALGENGINE_PYTHON}" "${AUDITOR}" screen         --recipe-config "${RECIPE_CONFIG}"         --metrics-root "${METRICS_ROOT}"         --shortlist-size 2         --output "${SCREEN_REPORT}"
}


audit_tune_lane() {
    local seed="$1"
    local report="${ROOT}/selection/lane_seed${seed}.json"
    local code_sha
    code_sha="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
    CURRENT_STAGE="lane_seed${seed}_audit"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" lane \
        --recipe-config "${RECIPE_CONFIG}" \
        --metrics-root "${METRICS_ROOT}" \
        --repo-root "${WORLDENGINE_ROOT}" \
        --model-root "${MODEL_ROOT}" \
        --split-audit "${SPLIT_AUDIT}" \
        --seed "${seed}" \
        --code-commit "${code_sha}" \
        --output "${report}"
    json_pass "${report}"
}

evaluate_tune_lane() {
    local seed="$1"
    local name
    for name in "${CHALLENGERS[@]}"; do
        run_candidate_metrics "${name}" "${seed}"
    done
    for name in "${CONTROLS[@]}"; do
        run_control_metrics "${name}" "${seed}"
    done
    audit_tune_lane "${seed}"
}

require_tune_lanes() {
    local current report seed
    current="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
    for seed in 0 1 2; do
        report="${ROOT}/selection/lane_seed${seed}.json"
        require_file "${report}"
        json_pass "${report}"
        "${ALGENGINE_PYTHON}" -c '
import json,sys
row=json.load(open(sys.argv[1]))
assert int(row["seed"])==int(sys.argv[2])
assert row["code_commit"]==sys.argv[3]
' "${report}" "${seed}" "${current}"
    done
}

select_three_seed_winner() {
    CURRENT_STAGE=three_seed_selection
    local code_sha
    code_sha="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" select \
        --recipe-config "${RECIPE_CONFIG}" \
        --screen "${SCREEN_REPORT}" \
        --metrics-root "${METRICS_ROOT}" \
        --repo-root "${WORLDENGINE_ROOT}" \
        --model-root "${MODEL_ROOT}" \
        --split-audit "${SPLIT_AUDIT}" \
        --lane-report "${ROOT}/selection/lane_seed0.json" \
        --lane-report "${ROOT}/selection/lane_seed1.json" \
        --lane-report "${ROOT}/selection/lane_seed2.json" \
        --minimum-improvement 0.005 \
        --minimum-nonnegative-seeds 2 \
        --code-commit "${code_sha}" \
        --output "${SELECTION_REPORT}"
    json_pass "${SELECTION_REPORT}"
    "${ALGENGINE_PYTHON}" -c '
import json,sys
row=json.load(open(sys.argv[1]))
print("SELECTED",row["selected"])
' "${SELECTION_REPORT}"
}

run_tune_lane() {
    run_static_preflight
    run_cuda_preflight
    ensure_split
    train_candidate_grid "${SEED}"
    evaluate_tune_lane "${SEED}"
    echo "PASS rare CL-PDMS tune lane seed ${SEED}"
    echo "lane report: ${LANE_REPORT}"
}

run_select() {
    run_static_preflight
    ensure_split
    require_tune_lanes
    screen_seed0
    select_three_seed_winner
    echo "PASS rare CL-PDMS tuning selection"
    echo "selection: ${SELECTION_REPORT}"
}

selected_value() {
    local key="$1"
    "${ALGENGINE_PYTHON}" -c '
import json,sys
value=json.load(open(sys.argv[1]))["selected"]
print(value[sys.argv[2]])
' "${SELECTION_REPORT}" "${key}"
}

validate_selection_commit() {
    local current
    current="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
    "${ALGENGINE_PYTHON}" -c '
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert row["code_commit"]==sys.argv[2], (row["code_commit"],sys.argv[2])
' "${SELECTION_REPORT}" "${current}"
}

run_formal_seed() {
    run_static_preflight
    run_cuda_preflight
    require_file "${SELECTION_REPORT}"
    json_pass "${SELECTION_REPORT}"
    validate_selection_commit
    local selected kind manifest checkpoint checkpoint_sha temperature kl_weight model summary
    selected="$(selected_value name)"
    kind="$(selected_value kind)"
    manifest="$(checkpoint_manifest "${selected}" "${SEED}")"
    checkpoint="$(checkpoint_path "${manifest}")"
    checkpoint_sha="$(${ALGENGINE_PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "${manifest}")"
    if [[ "${kind}" == "control" ]]; then
        summary="$(control_summary "${selected}" "${SEED}")"
        require_file "${summary}"
        echo "PASS selected incumbent/control reuses audited formal seed ${SEED}: ${summary}"
        return
    fi
    temperature="$(selected_value temperature)"
    kl_weight="$(selected_value kl_weight)"
    model="e2e_diffusiondrive_grpo_selector_v3_rare_clpdms_v1_${selected}_s${SEED}"
    summary="${FORMAL_ROOT}/formal_eval/${model}/summary.json"
    if [[ -f "${summary}" ]]; then
        "${ALGENGINE_PYTHON}" -c '
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert int(row["eval_seed"])==int(sys.argv[2])
assert row["checkpoint_sha256"]==sys.argv[3]
' "${summary}" "${SEED}" "${checkpoint_sha}"
        echo "REUSE verified formal seed ${SEED}: ${summary}"
        return
    fi
    export DIFFUSIONDRIVE_GRPO_FORMAL_ROOT="${FORMAL_ROOT}"
    export DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE="${temperature}"
    export DIFFUSIONDRIVE_GRPO_KL_WEIGHT="${kl_weight}"
    CURRENT_STAGE="formal_seed${SEED}"
    "${FORMAL_TABLE}"         "${checkpoint}" "${checkpoint_sha}" "${model}" "${SEED}"         "V3 rare CL-PDMS tuning v1 selected=${selected}; locked selection SHA=$(sha256sum "${SELECTION_REPORT}" | awk '{print $1}')"
    echo "PASS formal seed ${SEED}: ${summary}"
}

selected_summary() {
    local seed="$1"
    local selected kind model
    selected="$(selected_value name)"
    kind="$(selected_value kind)"
    if [[ "${kind}" == "control" ]]; then
        control_summary "${selected}" "${seed}"
    else
        model="e2e_diffusiondrive_grpo_selector_v3_rare_clpdms_v1_${selected}_s${seed}"
        echo "${FORMAL_ROOT}/formal_eval/${model}/summary.json"
    fi
}

run_summarize() {
    require_file "${SELECTION_REPORT}"
    require_file "${SPLIT_AUDIT}"
    json_pass "${SELECTION_REPORT}"
    validate_selection_commit
    local args=()
    local seed selected incumbent
    for seed in 0 1 2; do
        selected="$(selected_summary "${seed}")"
        incumbent="$(control_summary rare_tuned "${seed}")"
        require_file "${selected}"
        require_file "${incumbent}"
        args+=(--selected-summary "${seed}=${selected}")
        args+=(--incumbent-summary "${seed}=${incumbent}")
    done
    CURRENT_STAGE=formal_summary
    "${ALGENGINE_PYTHON}" "${AUDITOR}" finalize         --selection "${SELECTION_REPORT}"         --split-audit "${SPLIT_AUDIT}"         "${args[@]}"         --output "${FINAL_REPORT}"
    echo "PASS rare CL-PDMS formal summary"
    echo "report: ${FINAL_REPORT}"
    echo "table: ${FINAL_REPORT%.json}.md"
}

run_smoke() {
    run_static_preflight
    run_cuda_preflight
    ensure_split
    local manifest checkpoint model csv output
    manifest="$(control_manifest rare_tuned 0)"
    checkpoint="$(checkpoint_path "${manifest}")"
    model="e2e_diffusiondrive_grpo_selector_v3_rare_clpdms_v1_smoke"
    export DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE=formal_navtest_seed0
    export DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE=1
    export DIFFUSIONDRIVE_GRPO_KL_WEIGHT=1e-3
    CURRENT_STAGE=one_h100_reactive_smoke
    "${RAY_RUNNER}"         "${CONFIG}" "${checkpoint}" "${model}" rare_clpdms_smoke_v1 R         navtest_failures "${SPLIT_ROOT}/smoke/all_scenarios.pkl" 1
    csv="${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${model}/rare_clpdms_smoke_v1_R/WE_output/openscene_format/all_scenes_pdm_averages_R.csv"
    output="${ROOT}/local_smoke/metrics.json"
    mkdir -p "$(dirname "${output}")"
    "${ALGENGINE_PYTHON}" "${AUDITOR}" metrics         --split-audit "${SPLIT_AUDIT}"         --split smoke         --csv "${csv}"         --model smoke         --seed 0         --checkpoint-manifest "${manifest}"         --output "${output}"
    echo "PASS one-H100 rare CL-PDMS smoke"
    echo "metrics: ${output}"
}

case "${MODE}" in
    preflight)
        run_static_preflight
        run_cuda_preflight
        ;;
    prepare)
        ensure_split
        ;;
    tune-lane)
        run_tune_lane
        ;;
    select)
        run_select
        ;;
    formal)
        run_formal_seed
        ;;
    summarize)
        run_summarize
        ;;
    smoke)
        run_smoke
        ;;
esac

CURRENT_STAGE=complete
trap - ERR
echo "PASS rare-clpdms-v1 mode=${MODE}${SEED:+ seed=${SEED}}"
echo "persistent_log: ${LOG_FILE}"
