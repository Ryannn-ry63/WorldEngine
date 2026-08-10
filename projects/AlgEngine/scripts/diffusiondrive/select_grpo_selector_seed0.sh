#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal"
OUTPUT="${ROOT}/selection/selected_seed0.json"
mkdir -p "${ROOT}/selection"
cd "${ALGENGINE_ROOT}"

REPORTS=(
    "${ROOT}/calibration/lr1e-6_seed0/"epoch_*_noise_*.json
    "${ROOT}/calibration/lr3e-6_seed0/"epoch_*_noise_*.json
    "${ROOT}/calibration/lr1e-5_seed0/"epoch_*_noise_*.json
)
if [[ "${#REPORTS[@]}" -ne 72 ]]; then
    echo "Expected 72 seed-0 calibration reports, found ${#REPORTS[@]}"
    exit 1
fi
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/select_grpo_selector_checkpoint.py \
    --reports "${REPORTS[@]}" \
    --split-audit "${DIFFUSIONDRIVE_GRPO_SPLIT_ROOT}/split_audit.json" \
    --output "${OUTPUT}"
echo "PASS seed-0 LR/epoch selection"
echo "manifest: ${OUTPUT}"
