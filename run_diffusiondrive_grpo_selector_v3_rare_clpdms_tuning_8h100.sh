#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=8
if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 tune-lane {0|1|2} | select | formal {0|1|2} | summarize" >&2
    exit 2
fi
exec "${ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_rare_clpdms_tuning_h100.sh" "$@"
