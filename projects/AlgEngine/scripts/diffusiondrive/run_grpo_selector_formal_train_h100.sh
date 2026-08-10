#!/usr/bin/env bash
set -eo pipefail

LR="${1:?usage: run_grpo_selector_formal_train_h100.sh LR SEED [RUN_NAME]}"
SEED="${2:?usage: run_grpo_selector_formal_train_h100.sh LR SEED [RUN_NAME]}"
RUN_NAME="${3:-lr${LR}_seed${SEED}}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash"
    exit 2
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer"
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

export MASTER_PORT="${MASTER_PORT:-23456}"
export DIFFUSIONDRIVE_GRPO_LR="${LR}"
WORK_DIR="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/train/${RUN_NAME}"
GATE="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/preflight/real_candidate_parity_seed0.json"
if [[ ! -f "${GATE}" ]]; then
    echo "Missing real-candidate parity gate: ${GATE}"
    exit 1
fi
if [[ ! -f "${DIFFUSIONDRIVE_GRPO_SPLIT_ROOT}/split_audit.json" ]]; then
    echo "Missing immutable navtrain split audit"
    exit 1
fi
mkdir -p "${WORK_DIR}/checkpoint_audits"
cd "${ALGENGINE_ROOT}"

echo "[1/3] Formal contract preflight"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/preflight_grpo_online_selector.py \
    --config "${DIFFUSIONDRIVE_GRPO_CONFIG}"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/audit_grpo_online_dataset.py \
    --config "${DIFFUSIONDRIVE_GRPO_CONFIG}"

echo "[2/3] 8xH100, 8 full epochs, LR=${LR}, seed=${SEED}"
"${ALGENGINE_TORCHRUN}" \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT}" \
    scripts/train.py \
    "${DIFFUSIONDRIVE_GRPO_CONFIG}" \
    --launcher pytorch \
    --seed "${SEED}" \
    --work-dir "${WORK_DIR}" \
    --no-validate

echo "[3/3] Audit every checkpoint: only final selector may change"
for EPOCH in 1 2 3 4 5 6 7 8; do
    CHECKPOINT="${WORK_DIR}/epoch_${EPOCH}.pth"
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Missing epoch checkpoint: ${CHECKPOINT}"
        exit 1
    fi
    "${ALGENGINE_PYTHON}" scripts/diffusiondrive/audit_grpo_selector_checkpoint.py \
        --baseline "${DIFFUSIONDRIVE_GRPO_BASELINE}" \
        --checkpoint "${CHECKPOINT}" \
        --output "${WORK_DIR}/checkpoint_audits/epoch_${EPOCH}.json"
done

echo "PASS formal DiffusionDrive selector-GRPO training"
echo "work_dir: ${WORK_DIR}"
echo "final_checkpoint: ${WORK_DIR}/epoch_8.pth"
