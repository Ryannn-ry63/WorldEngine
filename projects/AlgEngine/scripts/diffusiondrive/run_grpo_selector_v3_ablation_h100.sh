#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
. "${SCRIPT_DIR}/grpo_selector_v3_experiment_env.sh"
ROOT="${V3_ROOT}"
CACHE_ROOT="${ROOT}/cache"
OUTPUT_ROOT="${ROOT}/ablation"
TRIAL_ROOT="${OUTPUT_ROOT}/trials"
LOG_DIR="${OUTPUT_ROOT}/logs"
STATUS_FILE="${OUTPUT_ROOT}/status.txt"
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
for file in "${CACHE_ROOT}/train_seed0/cache.pt" \
    "${CACHE_ROOT}/train_seed1/cache.pt" "${CACHE_ROOT}/train_seed2/cache.pt" \
    "${CACHE_ROOT}/development_seed3/cache.pt" \
    "${CACHE_ROOT}/development_seed4/cache.pt" \
    "${CACHE_ROOT}/development_seed5/cache.pt"; do
    [[ -f "${file}" ]] || { echo "Missing V3 ablation cache: ${file}" >&2; exit 1; }
done

ABLATIONS=(feature_only feature_geometry feature_geometry_route full)
PIDS=()
CURRENT_STAGE=representation_ablation
for gpu in 0 1 2 3; do
    output="${TRIAL_ROOT}/${ABLATIONS[$gpu]}"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/train_grpo_selector_v3_cached.py" "${CACHE_ARGS[@]}" \
        --output-dir "${output}" --temperature 1 --learning-rate 3e-4 \
        --kl-weight 1e-3 --seed 0 --epochs 16 --checkpoint-epochs 16 \
        --batch-size 64 --device cuda --ablation "${ABLATIONS[$gpu]}" \
        > "${output}/stdout.log" 2>&1 &
    PIDS+=("$!")
done
for gpu in 0 1 2 3; do
    if ! wait "${PIDS[$gpu]}"; then
        tail -n 80 -- "${TRIAL_ROOT}/${ABLATIONS[$gpu]}/stdout.log" >&2
        exit 1
    fi
done

CURRENT_STAGE=summarize
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/summarize_grpo_selector_v3_ablation.py" \
    --root "${TRIAL_ROOT}" --output "${OUTPUT_ROOT}/report.json"
CURRENT_STAGE=complete
printf 'PASS completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive selector V3 representation ablation"
echo "report: ${OUTPUT_ROOT}/report.json"
echo "persistent_log: ${LOG_FILE}"
