#!/usr/bin/env bash
# Queue entrypoint for the single authorized CPV E2 development evaluation.
# The delegated development mode runs D1/D2 in parallel, then applies the gate.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_ROOT="${ROOT}/experiments/diffusiondrive/selector_cpv_e2_v1/search/search_20260831T065356Z"

exec "${ROOT}/run_diffusiondrive_selector_cpv_e2_v1_8hopper.sh" \
    development "${SEARCH_ROOT}"
