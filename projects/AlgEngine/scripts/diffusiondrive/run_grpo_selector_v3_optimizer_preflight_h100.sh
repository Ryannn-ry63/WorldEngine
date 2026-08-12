#!/usr/bin/env bash
set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

FAILED=0
for ablation in feature_only feature_geometry feature_geometry_route full; do
    echo "START V3 H100 optimizer preflight ablation=${ablation}"
    if ! CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 "${ALGENGINE_PYTHON}" \
        "${SCRIPT_DIR}/preflight_grpo_selector_v3_h100_optimizer.py" \
        --ablation "${ablation}"; then
        echo "FAIL V3 H100 optimizer preflight ablation=${ablation}" >&2
        FAILED=1
    fi
done
if [[ "${FAILED}" -ne 0 ]]; then
    exit 1
fi
echo "PASS DiffusionDrive selector V3 H100 optimizer preflight"
