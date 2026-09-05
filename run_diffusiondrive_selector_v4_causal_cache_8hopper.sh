#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/projects/AlgEngine/scripts/diffusiondrive/run_selector_v4_causal_cache_8hopper.sh" "$@"
