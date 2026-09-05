#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT}/projects/AlgEngine/scripts/diffusiondrive/run_hard_pair_audit_1gpu.sh" "$@"
