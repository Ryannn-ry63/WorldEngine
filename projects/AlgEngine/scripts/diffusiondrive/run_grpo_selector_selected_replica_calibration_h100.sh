#!/usr/bin/env bash
set -eo pipefail

SEED="${1:?usage: run_grpo_selector_selected_replica_calibration_h100.sh SEED}"
if [[ "${SEED}" != 1 && "${SEED}" != 2 ]]; then
    echo "Replica seed must be 1 or 2"
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal"
MANIFEST="${ROOT}/selection/selected_seed0.json"
LR="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["learning_rate"])' "${MANIFEST}")"
EPOCH="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["epoch"])' "${MANIFEST}")"
RUN_NAME="selected_lr${LR}_seed${SEED}"
TRAIN_DIR="${ROOT}/train/${RUN_NAME}"
OUTPUT_DIR="${ROOT}/calibration/${RUN_NAME}"
CHECKPOINT="${TRAIN_DIR}/epoch_${EPOCH}.pth"
mkdir -p "${OUTPUT_DIR}"
cd "${ALGENGINE_ROOT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing selected-epoch replica checkpoint: ${CHECKPOINT}"
    exit 1
fi
for NOISE_SEED in 0 1 2; do
    OUTPUT="${OUTPUT_DIR}/epoch_${EPOCH}_noise_${NOISE_SEED}.json"
    "${ALGENGINE_TORCHRUN}" \
        --nproc_per_node=8 \
        --master_port="${MASTER_PORT:-23456}" \
        scripts/diffusiondrive/evaluate_grpo_selector_calibration.py \
        "${DIFFUSIONDRIVE_GRPO_CONFIG}" \
        "${CHECKPOINT}" \
        --calibration-filter "${DIFFUSIONDRIVE_GRPO_CALIBRATION_FILTER_PATH}" \
        --output "${OUTPUT}" \
        --noise-seed "${NOISE_SEED}" \
        --learning-rate "${LR}" \
        --train-seed "${SEED}" \
        --epoch "${EPOCH}"
done
echo "PASS selected replica calibration seed ${SEED}"
echo "reports: ${OUTPUT_DIR}"
