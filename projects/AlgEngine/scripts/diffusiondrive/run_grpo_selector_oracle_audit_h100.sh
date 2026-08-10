#!/usr/bin/env bash
set -Eeo pipefail

SEED="${1:?usage: run_grpo_selector_oracle_audit_h100.sh SEED(0|1|2)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

FORMAL_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal"
case "${SEED}" in
    0)
        TRAIN_DIR="lr1e-6_seed0"
        EXPECTED_SHA256="0db90b1e97dac9aeb9defdfc8c8149dc32f00af805548ebe1a37e2d4519433f5"
        ;;
    1)
        TRAIN_DIR="selected_lr1e-06_seed1"
        EXPECTED_SHA256="34bd2a6828d264c7e74c11d7b4ddd033cf5b33902d5a65181eee9984bb147da9"
        ;;
    2)
        TRAIN_DIR="selected_lr1e-06_seed2"
        EXPECTED_SHA256="494b0172ae878077ec89238673e5f20c39fdd54b79b8c4fb01cb831cf523a45a"
        ;;
    *)
        echo "SEED must be 0, 1, or 2" >&2
        exit 2
        ;;
esac

CHECKPOINT="${FORMAL_ROOT}/train/${TRAIN_DIR}/epoch_1.pth"
BASELINE="${DIFFUSIONDRIVE_GRPO_BASELINE}"
BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
CONFIG="${DIFFUSIONDRIVE_GRPO_CONFIG}"
ANNOTATION="${WORLDENGINE_ROOT}/data/alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl"
IMAGE_ROOT="${WORLDENGINE_ROOT}/data/raw/openscene-v1.1/sensor_blobs/test"
NAV_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtest_split/navtest.yaml"
FAILURES_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtest_split/navtest_failures_filtered.yaml"
METRIC_CACHE="${NAVSIM_METRIC_CACHE_PATH_EVAL}"
OUTPUT_DIR="${FORMAL_ROOT}/oracle_audit/seed${SEED}"
LOG_DIR="${OUTPUT_DIR}/logs"
STATUS_FILE="${OUTPUT_DIR}/status.txt"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
printf 'RUNNING seed=%s started_utc=%s\n' \
    "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"

CURRENT_STAGE=static_preflight
trap 'rc=$?; failed_command=${BASH_COMMAND}; printf "FAIL seed=%s stage=%s exit=%s\n" "${SEED}" "${CURRENT_STAGE}" "${rc}" > "${STATUS_FILE}"; echo "FAIL stage=${CURRENT_STAGE} line=${LINENO} command=${failed_command} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
for path in "${CHECKPOINT}" "${BASELINE}" "${CONFIG}" "${ANNOTATION}" \
    "${NAV_FILTER}" "${FAILURES_FILTER}" "${WORLDENGINE_MMCV_EXTENSION}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Missing required file: ${path}" >&2
        exit 1
    fi
done
for path in "${IMAGE_ROOT}" "${METRIC_CACHE}"; do
    if [[ ! -d "${path}" ]]; then
        echo "Missing required directory: ${path}" >&2
        exit 1
    fi
done
ACTUAL_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]]; then
    echo "Checkpoint SHA256 mismatch: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}" >&2
    exit 1
fi
ACTUAL_BASELINE_SHA256="$(sha256sum "${BASELINE}" | awk '{print $1}')"
if [[ "${ACTUAL_BASELINE_SHA256}" != "${BASELINE_SHA256}" ]]; then
    echo "Baseline SHA256 mismatch" >&2
    exit 1
fi

CURRENT_STAGE=checkpoint_audit
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_checkpoint.py" \
    --baseline "${BASELINE}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT_DIR}/checkpoint_audit.json"

CURRENT_STAGE=cuda_preflight
H100_SUPPORT_DIR="${SIMENGINE_ROOT}/scripts/diffusiondrive"
EXPECTED_CUDA_CAPABILITY="${EXPECTED_CUDA_CAPABILITY:-sm_90}"
"${ALGENGINE_PYTHON}" "${H100_SUPPORT_DIR}/preflight_mmcv_cuda.py" \
    --extension "${WORLDENGINE_MMCV_EXTENSION}" \
    --expected-capability "${EXPECTED_CUDA_CAPABILITY}" \
    --all-visible
PDM_SIMULATOR_ARGS=()
if [[ "${EXPECTED_CUDA_CAPABILITY}" != "sm_90" ]]; then
    PDM_SIMULATOR_ARGS+=(--cpu-pdm-simulator)
fi

CURRENT_STAGE=best_of_20_audit
cd "${ALGENGINE_ROOT}"
MASTER_PORT_VALUE="${MASTER_PORT:-$((28920 + SEED))}"
"${ALGENGINE_TORCHRUN}" \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT_VALUE}" \
    "${SCRIPT_DIR}/audit_grpo_selector_oracle.py" \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    --expected-checkpoint-sha256 "${EXPECTED_SHA256}" \
    --baseline-checkpoint "${BASELINE}" \
    --expected-baseline-sha256 "${BASELINE_SHA256}" \
    --annotation-file "${ANNOTATION}" \
    --image-root "${IMAGE_ROOT}" \
    --nav-filter "${NAV_FILTER}" \
    --failures-filter "${FAILURES_FILTER}" \
    --metric-cache "${METRIC_CACHE}" \
    --output-dir "${OUTPUT_DIR}" \
    --noise-seed "${SEED}" \
    --train-seed "${SEED}" \
    "${PDM_SIMULATOR_ARGS[@]}" \
    --launcher pytorch

CURRENT_STAGE=official_rescore
export NAVSIM_DEVKIT_ROOT="${DIFFUSIONDRIVE_ROOT}"
export PYTHON_BIN="${ALGENGINE_PYTHON}"
for selection in reference current oracle; do
    submission="${OUTPUT_DIR}/${selection}_navsim_submission.pkl"
    result_dir="${OUTPUT_DIR}/${selection}_official_pdms"
    bash "${ALGENGINE_ROOT}/scripts/e2e_navsim_official_rescore.sh" \
        "${submission}" "${METRIC_CACHE}" "${result_dir}"
    result_csv="${result_dir}/pdm_scores_merged.csv"
    if [[ ! -f "${result_csv}" ]]; then
        echo "Missing official PDM result: ${result_csv}" >&2
        exit 1
    fi
    # The official merger appends an aggregate "average" row after token rows.
    row_count="$(awk -F, 'NR > 1 && $1 != "average" { count += 1 } END { print count + 0 }' "${result_csv}")"
    if [[ "${row_count}" -ne 12146 ]]; then
        echo "Official ${selection} row count mismatch: ${row_count}" >&2
        exit 1
    fi
done

CURRENT_STAGE=complete
printf 'PASS seed=%s checkpoint_sha256=%s completed_utc=%s\n' \
    "${SEED}" "${EXPECTED_SHA256}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS_FILE}"
trap - ERR
echo "PASS DiffusionDrive best-of-20 oracle audit seed=${SEED}"
echo "output: ${OUTPUT_DIR}"
echo "persistent_log: ${LOG_FILE}"
