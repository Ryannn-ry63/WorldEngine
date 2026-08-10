#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
SPLIT_ROOT="${DIFFUSIONDRIVE_GRPO_SPLIT_ROOT}"
OUTPUT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_diagnostic_v1"
CACHE_ROOT="${OUTPUT_ROOT}/cache"
RESULT_ROOT="${OUTPUT_ROOT}/probes"
STATUS_FILE="${OUTPUT_ROOT}/status.txt"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${CACHE_ROOT}" "${RESULT_ROOT}"
LOG_FILE="${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
printf 'RUNNING started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
CURRENT_STAGE=preflight
trap 'rc=$?; printf "FAIL stage=%s exit=%s\n" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2' ERR

for file in "${DIFFUSIONDRIVE_GRPO_CONFIG}" "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
    "${SPLIT_ROOT}/navtrain_grpo_train.yaml" \
    "${SPLIT_ROOT}/navtrain_grpo_calibration.yaml" \
    "${SCRIPT_DIR}/extract_grpo_selector_diagnostic_cache.py" \
    "${SCRIPT_DIR}/run_grpo_selector_diagnostic_probes.py"; do
    [[ -f "${file}" ]] || { echo "Missing required file: ${file}" >&2; exit 1; }
done
[[ "$(sha256sum "${DIFFUSIONDRIVE_GRPO_BASELINE}" | awk '{print $1}')" == "${BASELINE_SHA256}" ]] || {
    echo "Baseline SHA256 mismatch" >&2; exit 1;
}
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/preflight_grpo_online_selector.py" \
    --config "${DIFFUSIONDRIVE_GRPO_CONFIG}"
"${ALGENGINE_PYTHON}" "${SIMENGINE_ROOT}/scripts/diffusiondrive/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" --expected-capability sm_90 --all-visible

run_extract() {
    local split="$1"
    local seed="$2"
    local count="$3"
    local filter="$4"
    local output="${CACHE_ROOT}/${split}_seed${seed}"
    local port_base=29420
    if [[ "${split}" == "train" ]]; then
        port_base=29410
    fi
    CURRENT_STAGE="cache_${split}_seed${seed}"
    if [[ -f "${output}/cache.pt" && -f "${output}/manifest.json" ]]; then
        local expected_cache_sha
        local actual_cache_sha
        expected_cache_sha="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["cache_sha256"])' "${output}/manifest.json")"
        actual_cache_sha="$(sha256sum "${output}/cache.pt" | awk '{print $1}')"
        if [[ "${expected_cache_sha}" == "${actual_cache_sha}" ]]; then
            echo "REUSE verified diagnostic cache: ${output}/cache.pt"
            return
        fi
        echo "Existing diagnostic cache failed SHA256 validation; regenerating ${output}"
    fi
    mkdir -p "${output}"
    cd "${ALGENGINE_ROOT}"
    "${ALGENGINE_TORCHRUN}" --nproc_per_node=8 --master_port="$((port_base + seed))" \
        "${SCRIPT_DIR}/extract_grpo_selector_diagnostic_cache.py" \
        "${DIFFUSIONDRIVE_GRPO_CONFIG}" "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
        --expected-checkpoint-sha256 "${BASELINE_SHA256}" \
        --nav-filter "${filter}" --split "${split}" --noise-seed "${seed}" \
        --expected-num-tokens "${count}" --output-dir "${output}" --launcher pytorch
}

# One frozen-generator training cache and three paired held-out noise caches.
run_extract train 0 6339 "${SPLIT_ROOT}/navtrain_grpo_train.yaml"
run_extract calibration 0 1118 "${SPLIT_ROOT}/navtrain_grpo_calibration.yaml"
run_extract calibration 1 1118 "${SPLIT_ROOT}/navtrain_grpo_calibration.yaml"
run_extract calibration 2 1118 "${SPLIT_ROOT}/navtrain_grpo_calibration.yaml"

CURRENT_STAGE=offline_probe_matrix
cd "${ALGENGINE_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/run_grpo_selector_diagnostic_probes.py" \
    --train-cache "${CACHE_ROOT}/train_seed0/cache.pt" \
    --calibration-cache "${CACHE_ROOT}/calibration_seed0/cache.pt" \
    --calibration-cache "${CACHE_ROOT}/calibration_seed1/cache.pt" \
    --calibration-cache "${CACHE_ROOT}/calibration_seed2/cache.pt" \
    --output-dir "${RESULT_ROOT}" --device cuda --bootstrap-replicates 10000

CURRENT_STAGE=complete
printf 'PASS completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive selector diagnostic matrix"
echo "report: ${RESULT_ROOT}/diagnostic_report.md"
echo "persistent_log: ${LOG_FILE}"
