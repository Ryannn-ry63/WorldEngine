#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_ivps_search_8hopper.sh" "$@"
