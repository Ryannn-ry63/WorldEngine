#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
. "${SCRIPT_DIR}/grpo_selector_v3_experiment_env.sh"

CACHE_ROOT="${V3_ROOT}/cache"
OUTPUT_ROOT="${V3_ROOT}/sweep"
TRIAL_ROOT="${OUTPUT_ROOT}/trials"
STATUS_FILE="${OUTPUT_ROOT}/status.txt"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${TRIAL_ROOT}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
printf 'RUNNING started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
CURRENT_STAGE=preflight
trap 'rc=$?; printf "FAIL stage=%s exit=%s\n" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2' ERR

CACHE_ARGS=(
    --train-cache "${CACHE_ROOT}/train_seed0/cache.pt"
    --train-cache "${CACHE_ROOT}/train_seed1/cache.pt"
    --train-cache "${CACHE_ROOT}/train_seed2/cache.pt"
    --development-cache "${CACHE_ROOT}/development_seed3/cache.pt"
    --development-cache "${CACHE_ROOT}/development_seed4/cache.pt"
    --development-cache "${CACHE_ROOT}/development_seed5/cache.pt"
)
for file in \
    "${CACHE_ROOT}/train_seed0/cache.pt" \
    "${CACHE_ROOT}/train_seed1/cache.pt" \
    "${CACHE_ROOT}/train_seed2/cache.pt" \
    "${CACHE_ROOT}/development_seed3/cache.pt" \
    "${CACHE_ROOT}/development_seed4/cache.pt" \
    "${CACHE_ROOT}/development_seed5/cache.pt" \
    "${SCRIPT_DIR}/train_grpo_selector_v3_cached.py" \
    "${SCRIPT_DIR}/select_grpo_selector_v3.py"; do
    [[ -f "${file}" ]] || { echo "Missing V3 sweep input: ${file}" >&2; exit 1; }
done

TEMPERATURES=(1 1 1 1 1 1 1 2)
LEARNING_RATES=(1e-4 1e-4 3e-4 3e-4 3e-4 1e-3 1e-3 3e-4)
KL_WEIGHTS=(0 1e-3 0 1e-4 1e-3 1e-4 1e-3 1e-3)
PIDS=()
CURRENT_STAGE=parallel_exact_group_sweep
for gpu in 0 1 2 3 4 5 6 7; do
    trial="t${TEMPERATURES[$gpu]}_lr${LEARNING_RATES[$gpu]}_kl${KL_WEIGHTS[$gpu]}"
    output="${TRIAL_ROOT}/${trial}"
    mkdir -p "${output}"
    echo "START gpu=${gpu} trial=${trial}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/train_grpo_selector_v3_cached.py" \
        "${CACHE_ARGS[@]}" \
        --output-dir "${output}" \
        --temperature "${TEMPERATURES[$gpu]}" \
        --learning-rate "${LEARNING_RATES[$gpu]}" \
        --kl-weight "${KL_WEIGHTS[$gpu]}" \
        --seed 0 --epochs 64 --checkpoint-epochs 1,2,4,8,16,32,64 \
        --batch-size 64 --device cuda --ablation full \
        > "${output}/stdout.log" 2>&1 &
    PIDS+=("$!")
done

for gpu in 0 1 2 3 4 5 6 7; do
    if ! wait "${PIDS[$gpu]}"; then
        echo "V3 trial failed on GPU ${gpu}; tail follows" >&2
        tail -n 80 -- "${TRIAL_ROOT}"/*/stdout.log >&2
        exit 1
    fi
done

CURRENT_STAGE=development_only_selection
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/select_grpo_selector_v3.py" \
    --trial-root "${TRIAL_ROOT}" --output "${OUTPUT_ROOT}/selection.json"

CURRENT_STAGE=complete
printf 'PASS completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive selector V3 exact-group sweep"
echo "selection: ${OUTPUT_ROOT}/selection.json"
echo "persistent_log: ${LOG_FILE}"
