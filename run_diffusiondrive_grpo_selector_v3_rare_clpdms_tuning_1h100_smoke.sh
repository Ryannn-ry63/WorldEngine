#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_EXPECTED_GPU_COUNT=1
exec "${ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_rare_clpdms_tuning_h100.sh" smoke
