#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT="${DIFFUSIONDRIVE_EXPECTED_GPU_COUNT:-8}"
export DIFFUSIONDRIVE_GRPO_CONFIG="${ALGENGINE_ROOT:-${SCRIPT_DIR}/../..}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
REWARD_CONTRACT="navsim_pairwise_raw_progress_then_candidate_gate_v1"
V2_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v2_progress_fix_v1"
CACHE_ROOT="${V2_ROOT}/cache"
TRIAL_ROOT="${V2_ROOT}/trials"
SELECTION_ROOT="${V2_ROOT}/selection"
LOG_ROOT="${V2_ROOT}/logs"
STATUS_FILE="${V2_ROOT}/sweep_status.txt"
mkdir -p "${TRIAL_ROOT}" "${SELECTION_ROOT}" "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/sweep_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; printf "FAIL stage=%s exit=%s\n" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL V2 progress-fix sweep stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2' ERR

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

REWARD_FILE="${ALGENGINE_ROOT}/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py"
REWARD_SHA="$(sha256sum "${REWARD_FILE}" | awk '{print $1}')"
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

CURRENT_STAGE=corrected_exact_group_sweep
task_index=0
while [[ "${task_index}" -lt "${#TASKS[@]}" ]]; do
    PIDS=()
    NAMES=()
    for ((gpu = 0; gpu < DIFFUSIONDRIVE_EXPECTED_GPU_COUNT; gpu++)); do
        [[ "${task_index}" -lt "${#TASKS[@]}" ]] || break
        IFS='|' read -r temperature learning_rate kl_weight <<< "${TASKS[${task_index}]}"
        run_name="t${temperature}_lr${learning_rate}_kl${kl_weight}"
        output_dir="${TRIAL_ROOT}/${run_name}"
        report="${output_dir}/report.json"
        if [[ -s "${report}" ]] && "${ALGENGINE_PYTHON}" -c '
import json, sys
p=json.load(open(sys.argv[1]))
ok=(p.get("status")=="PASS" and p.get("reward_contract")==sys.argv[2] and p.get("reward_implementation_sha256")==sys.argv[3])
raise SystemExit(0 if ok else 1)
' "${report}" "${REWARD_CONTRACT}" "${REWARD_SHA}"; then
            echo "REUSE PASS corrected V2 trial ${run_name}"
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
    echo "Expected 27 complete corrected V2 reports, found ${#REPORTS[@]}" >&2
    exit 1
fi
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/select_grpo_selector_v2.py" \
    --reports "${REPORTS[@]}" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --expected-baseline-sha256 "${BASELINE_SHA256}" \
    --expected-reward-contract "${REWARD_CONTRACT}" \
    --allow-predeclared-fallback \
    --fallback-temperature 1 --fallback-learning-rate 1e-3 \
    --fallback-kl-weight 1e-3 --fallback-epoch 32 \
    --experiment-method e2e_diffusiondrive_grpo_selector_v2_progress_fix_v1 \
    --output-dir "${SELECTION_ROOT}" --bootstrap-replicates 10000

CURRENT_STAGE=selected_checkpoint_audit
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_checkpoint.py" \
    --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    --checkpoint "${SELECTION_ROOT}/selected_checkpoint.pth" \
    --output "${SELECTION_ROOT}/selected_checkpoint_audit.json"

GATE_STATUS="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate_status"])' "${SELECTION_ROOT}/selection.json")"
SELECTION_MODE="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection_mode"])' "${SELECTION_ROOT}/selection.json")"
CURRENT_STAGE=complete
printf 'PASS gate_status=%s selection_mode=%s completed_utc=%s\n' "${GATE_STATUS}" "${SELECTION_MODE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive corrected selector GRPO V2 sweep gate=${GATE_STATUS} mode=${SELECTION_MODE}"
echo "selection: ${SELECTION_ROOT}/selection.json"
echo "persistent_log: ${LOG_FILE}"
