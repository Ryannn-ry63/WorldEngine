#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal"
OUTPUT="${ROOT}/final/formal_three_seed_result.json"
mkdir -p "${ROOT}/final"
cd "${ALGENGINE_ROOT}"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/finalize_grpo_selector_formal.py \
    --root "${ROOT}" \
    --output "${OUTPUT}"
echo "PASS formal three-seed acceptance"
echo "result: ${OUTPUT}"
echo "table rows: ${OUTPUT}.tsv"
