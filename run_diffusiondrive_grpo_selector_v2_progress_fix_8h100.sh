#!/usr/bin/env bash
set -Eeo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /usr/bin/bash \
    "${ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v2_progress_fix_h100.sh" \
    "$@"
