#!/usr/bin/env bash
set -eo pipefail

SEED="${1:?usage: run_grpo_selector_selected_replica_bundle_h100.sh SEED}"
if [[ "${SEED}" != 1 && "${SEED}" != 2 ]]; then
    echo "Replica seed must be 1 or 2"
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_grpo_selector_selected_replica_h100.sh" "${SEED}"
"${SCRIPT_DIR}/run_grpo_selector_selected_replica_calibration_h100.sh" "${SEED}"
echo "PASS selected-LR train+calibration bundle seed ${SEED}"
