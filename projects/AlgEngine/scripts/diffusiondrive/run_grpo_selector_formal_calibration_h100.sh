#!/usr/bin/env bash
set -eo pipefail

LR="${1:?usage: run_grpo_selector_formal_calibration_h100.sh LR TRAIN_SEED RUN_NAME}"
TRAIN_SEED="${2:?usage: run_grpo_selector_formal_calibration_h100.sh LR TRAIN_SEED RUN_NAME}"
RUN_NAME="${3:?usage: run_grpo_selector_formal_calibration_h100.sh LR TRAIN_SEED RUN_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

export MASTER_PORT="${MASTER_PORT:-23456}"
TRAIN_DIR="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/train/${RUN_NAME}"
OUTPUT_DIR="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/calibration/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
cd "${ALGENGINE_ROOT}"

for EPOCH in 1 2 3 4 5 6 7 8; do
    CHECKPOINT="${TRAIN_DIR}/epoch_${EPOCH}.pth"
    AUDIT="${TRAIN_DIR}/checkpoint_audits/epoch_${EPOCH}.json"
    if [[ ! -f "${CHECKPOINT}" || ! -f "${AUDIT}" ]]; then
        echo "Missing checkpoint or frozen-parameter audit for epoch ${EPOCH}"
        exit 1
    fi
    for NOISE_SEED in 0 1 2; do
        OUTPUT="${OUTPUT_DIR}/epoch_${EPOCH}_noise_${NOISE_SEED}.json"
        if [[ -s "${OUTPUT}" ]] && "${ALGENGINE_PYTHON}" -c \
            'import json, sys; report=json.load(open(sys.argv[1])); raise SystemExit(0 if report.get("status") == "PASS" else 1)' \
            "${OUTPUT}"; then
            echo "SKIP existing PASS calibration LR=${LR} train_seed=${TRAIN_SEED} epoch=${EPOCH} noise_seed=${NOISE_SEED}"
            continue
        fi
        echo "Calibration LR=${LR} train_seed=${TRAIN_SEED} epoch=${EPOCH} noise_seed=${NOISE_SEED}"
        "${ALGENGINE_TORCHRUN}" \
            --nproc_per_node=8 \
            --master_port="${MASTER_PORT}" \
            scripts/diffusiondrive/evaluate_grpo_selector_calibration.py \
            "${DIFFUSIONDRIVE_GRPO_CONFIG}" \
            "${CHECKPOINT}" \
            --calibration-filter "${DIFFUSIONDRIVE_GRPO_CALIBRATION_FILTER_PATH}" \
            --output "${OUTPUT}" \
            --noise-seed "${NOISE_SEED}" \
            --learning-rate "${LR}" \
            --train-seed "${TRAIN_SEED}" \
            --epoch "${EPOCH}"
    done
done

echo "PASS formal calibration grid"
echo "reports: ${OUTPUT_DIR}"
