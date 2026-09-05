#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_interaction_selector_formal_eval_8hopper.sh" "$@"
