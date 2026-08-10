#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

export DIFFUSIONDRIVE_GRPO_CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2.py"
BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_diagnostic_v1/cache"
OUTPUT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2"
TRIAL_ROOT="${OUTPUT_ROOT}/trials"
SELECTION_ROOT="${OUTPUT_ROOT}/selection"
LOG_ROOT="${OUTPUT_ROOT}/logs"
STATUS_FILE="${OUTPUT_ROOT}/status.txt"
mkdir -p "${TRIAL_ROOT}" "${SELECTION_ROOT}" "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
printf 'RUNNING started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
CURRENT_STAGE=preflight
trap 'rc=$?; printf "FAIL stage=%s exit=%s\n" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2' ERR

TRAIN_CACHE="${CACHE_ROOT}/train_seed0/cache.pt"
CALIBRATION_CACHES=(
    "${CACHE_ROOT}/calibration_seed0/cache.pt"
    "${CACHE_ROOT}/calibration_seed1/cache.pt"
    "${CACHE_ROOT}/calibration_seed2/cache.pt"
)
for required in "${DIFFUSIONDRIVE_GRPO_CONFIG}" "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    "${TRAIN_CACHE}" "${CALIBRATION_CACHES[@]}" \
    "${SCRIPT_DIR}/train_grpo_selector_v2_cached.py" \
    "${SCRIPT_DIR}/select_grpo_selector_v2.py"; do
    [[ -f "${required}" ]] || { echo "Missing required file: ${required}" >&2; exit 1; }
done
[[ "$(sha256sum "${DIFFUSIONDRIVE_GRPO_BASELINE}" | awk '{print $1}')" == "${BASELINE_SHA256}" ]] || {
    echo "Baseline SHA256 mismatch" >&2
    exit 1
}
cd "${ALGENGINE_ROOT}"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/preflight_grpo_online_selector.py" \
    --config "${DIFFUSIONDRIVE_GRPO_CONFIG}"

TEMPERATURES=(1 4 8)
LEARNING_RATES=(1e-4 3e-4 1e-3)
KL_WEIGHTS=(0 1e-4 1e-3)
TASKS=()
for temperature in "${TEMPERATURES[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
        for kl_weight in "${KL_WEIGHTS[@]}"; do
            TASKS+=("${temperature}|${learning_rate}|${kl_weight}")
        done
    done
done

CURRENT_STAGE=exact_group_sweep
task_index=0
while [[ "${task_index}" -lt "${#TASKS[@]}" ]]; do
    PIDS=()
    NAMES=()
    for gpu in 0 1 2 3 4 5 6 7; do
        [[ "${task_index}" -lt "${#TASKS[@]}" ]] || break
        IFS='|' read -r temperature learning_rate kl_weight <<< "${TASKS[${task_index}]}"
        run_name="t${temperature}_lr${learning_rate}_kl${kl_weight}"
        output_dir="${TRIAL_ROOT}/${run_name}"
        report="${output_dir}/report.json"
        if [[ -s "${report}" ]] && "${ALGENGINE_PYTHON}" -c \
            'import json,sys; row=json.load(open(sys.argv[1])); raise SystemExit(0 if row.get("status") == "PASS" else 1)' \
            "${report}"; then
            echo "REUSE PASS V2 trial ${run_name}"
        else
            mkdir -p "${output_dir}"
            echo "START GPU=${gpu} ${run_name}"
            CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" \
                "${SCRIPT_DIR}/train_grpo_selector_v2_cached.py" \
                --train-cache "${TRAIN_CACHE}" \
                --calibration-cache "${CALIBRATION_CACHES[0]}" \
                --calibration-cache "${CALIBRATION_CACHES[1]}" \
                --calibration-cache "${CALIBRATION_CACHES[2]}" \
                --output-dir "${output_dir}" \
                --temperature "${temperature}" \
                --learning-rate "${learning_rate}" \
                --kl-weight "${kl_weight}" \
                --seed 0 --epochs 200 \
                --checkpoint-epochs 1,2,4,8,16,32,64,128,200 \
                --device cuda > "${output_dir}/worker.log" 2>&1 &
            PIDS+=("$!")
            NAMES+=("${run_name}")
        fi
        task_index=$((task_index + 1))
    done
    wave_failed=0
    for index in "${!PIDS[@]}"; do
        if wait "${PIDS[${index}]}"; then
            echo "PASS ${NAMES[${index}]}"
        else
            echo "FAIL ${NAMES[${index}]} (see worker.log)" >&2
            wave_failed=1
        fi
    done
    [[ "${wave_failed}" -eq 0 ]] || exit 1
done

CURRENT_STAGE=scene_heldout_gate
REPORTS=("${TRIAL_ROOT}"/*/report.json)
if [[ ! -f "${REPORTS[0]}" || "${#REPORTS[@]}" -ne 27 ]]; then
    echo "Expected 27 complete V2 reports, found ${#REPORTS[@]}" >&2
    exit 1
fi
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/select_grpo_selector_v2.py" \
    --reports "${REPORTS[@]}" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --expected-baseline-sha256 "${BASELINE_SHA256}" \
    --output-dir "${SELECTION_ROOT}" --bootstrap-replicates 10000

STATUS="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${SELECTION_ROOT}/selection.json")"
if [[ "${STATUS}" == "PASS" ]]; then
    CURRENT_STAGE=selected_checkpoint_audit
    "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_checkpoint.py" \
        --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
        --checkpoint "${SELECTION_ROOT}/selected_checkpoint.pth" \
        --output "${SELECTION_ROOT}/selected_checkpoint_audit.json"
fi

CURRENT_STAGE=complete
printf 'PASS gate_status=%s completed_utc=%s\n' "${STATUS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive selector GRPO V2 sweep (scientific gate status: ${STATUS})"
echo "selection: ${SELECTION_ROOT}/selection.json"
echo "persistent_log: ${LOG_FILE}"
