#!/usr/bin/env bash
set -eo pipefail

EVAL_SEED="${1:?usage: run_grpo_selector_formal_paired_eval_h100.sh EVAL_SEED}"
if [[ ! "${EVAL_SEED}" =~ ^[012]$ ]]; then
    echo "EVAL_SEED must be 0, 1, or 2"
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal"
MANIFEST="${ROOT}/selection/selected_seed0.json"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing selection manifest: ${MANIFEST}"
    exit 1
fi
LR="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["learning_rate"])' "${MANIFEST}")"
EPOCH="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["epoch"])' "${MANIFEST}")"

REFERENCE_DIR="${ROOT}/reference_checkpoint"
REFERENCE_COPY="${REFERENCE_DIR}/epoch_100.pth"
mkdir -p "${REFERENCE_DIR}"
if [[ ! -f "${REFERENCE_COPY}" ]]; then
    REFERENCE_TMP="${REFERENCE_COPY}.tmp.${BASHPID}"
    cp "${DIFFUSIONDRIVE_GRPO_BASELINE}" "${REFERENCE_TMP}"
    # Publish atomically. A concurrent evaluator may win this race, in which
    # case its byte-identical copy is retained.
    ln "${REFERENCE_TMP}" "${REFERENCE_COPY}" 2>/dev/null || true
    rm -f "${REFERENCE_TMP}"
fi
REFERENCE_SHA="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
if [[ "$(sha256sum "${REFERENCE_COPY}" | awk '{print $1}')" != "${REFERENCE_SHA}" ]]; then
    echo "Local reference copy SHA mismatch"
    exit 1
fi

# Each paired-eval seed must have a private reference checkpoint directory.
# The evaluator writes NAVSIM outputs below dirname(CHECKPOINT)/test. Sharing
# reference_checkpoint/epoch_100.pth across concurrent seeds makes their
# *_official_pdms directories collide and breaks the one-directory gate.
REFERENCE_EVAL_DIR="${ROOT}/formal_eval/e2e_diffusiondrive_reference_paired_s${EVAL_SEED}/checkpoint"
REFERENCE_EVAL_COPY="${REFERENCE_EVAL_DIR}/epoch_100.pth"
mkdir -p "${REFERENCE_EVAL_DIR}"
if [[ ! -f "${REFERENCE_EVAL_COPY}" ]]; then
    REFERENCE_EVAL_TMP="${REFERENCE_EVAL_COPY}.tmp.${BASHPID}"
    cp "${REFERENCE_COPY}" "${REFERENCE_EVAL_TMP}"
    ln "${REFERENCE_EVAL_TMP}" "${REFERENCE_EVAL_COPY}" 2>/dev/null || true
    rm -f "${REFERENCE_EVAL_TMP}"
fi
if [[ "$(sha256sum "${REFERENCE_EVAL_COPY}" | awk '{print $1}')" != "${REFERENCE_SHA}" ]]; then
    echo "Private reference copy SHA mismatch"
    exit 1
fi

if [[ "${EVAL_SEED}" == 0 ]]; then
    CURRENT_CHECKPOINT="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' "${MANIFEST}")"
    CURRENT_SHA="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint_sha256"])' "${MANIFEST}")"
else
    RUN_NAME="selected_lr${LR}_seed${EVAL_SEED}"
    CURRENT_CHECKPOINT="${ROOT}/train/${RUN_NAME}/epoch_${EPOCH}.pth"
    AUDIT="${ROOT}/train/${RUN_NAME}/checkpoint_audits/epoch_${EPOCH}.json"
    CURRENT_SHA="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_sha256"])' "${AUDIT}")"
fi

"${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh" \
    "${REFERENCE_EVAL_COPY}" "${REFERENCE_SHA}" \
    "e2e_diffusiondrive_reference_paired_s${EVAL_SEED}" "${EVAL_SEED}" \
    "epoch100 reference; paired diffusion seed ${EVAL_SEED}"

"${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh" \
    "${CURRENT_CHECKPOINT}" "${CURRENT_SHA}" \
    "e2e_diffusiondrive_grpo_selector_s${EVAL_SEED}" "${EVAL_SEED}" \
    "selector-only GRPO; lr=${LR}; epoch=${EPOCH}; train/eval seed=${EVAL_SEED}"

echo "PASS paired reference/current four-block evaluation seed ${EVAL_SEED}"
