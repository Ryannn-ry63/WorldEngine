#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT_DIR}/projects/AlgEngine/scripts/diffusiondrive/run_diffusion_robust_group_search_8hopper.sh" "$@"
