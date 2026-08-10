#!/usr/bin/env bash
set -eo pipefail

LR="${1:?usage: run_grpo_selector_formal_seed0_bundle_h100.sh LR}"
case "${LR}" in
    1e-6|3e-6|1e-5) ;;
    *) echo "LR must be one of: 1e-6, 3e-6, 1e-5"; exit 2 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="lr${LR}_seed0"
"${SCRIPT_DIR}/run_grpo_selector_formal_train_h100.sh" "${LR}" 0 "${RUN_NAME}"
"${SCRIPT_DIR}/run_grpo_selector_formal_calibration_h100.sh" "${LR}" 0 "${RUN_NAME}"
echo "PASS seed-0 formal LR bundle ${LR}"
