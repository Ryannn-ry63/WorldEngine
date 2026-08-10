#!/usr/bin/env bash
set -eo pipefail

SEED="${1:?usage: run_grpo_selector_selected_replica_h100.sh SEED}"
if [[ "${SEED}" != 1 && "${SEED}" != 2 ]]; then
    echo "Replica seed must be 1 or 2"
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
MANIFEST="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/selection/selected_seed0.json"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing selected seed-0 manifest: ${MANIFEST}"
    exit 1
fi
LR="$("${ALGENGINE_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["learning_rate"])' "${MANIFEST}")"
RUN_NAME="selected_lr${LR}_seed${SEED}"

"${SCRIPT_DIR}/run_grpo_selector_formal_train_h100.sh" "${LR}" "${SEED}" "${RUN_NAME}"

echo "PASS selected-LR replica seed ${SEED}"
echo "run_name: ${RUN_NAME}"
