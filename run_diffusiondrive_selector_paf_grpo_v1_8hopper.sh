#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT_DIR}/projects/AlgEngine/scripts/diffusiondrive/run_proposal_aware_full_feedback_v1_8hopper.sh" "$@"
