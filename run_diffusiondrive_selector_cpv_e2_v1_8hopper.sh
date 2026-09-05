#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_cpv_e2_v1_8hopper.sh" "$@"
