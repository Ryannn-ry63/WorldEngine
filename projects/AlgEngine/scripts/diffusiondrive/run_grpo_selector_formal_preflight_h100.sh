#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

export MASTER_PORT="${MASTER_PORT:-23456}"
PREFLIGHT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/preflight"
mkdir -p "${PREFLIGHT_ROOT}"
cd "${ALGENGINE_ROOT}"

echo "[1/5] Create/reuse immutable scene-disjoint navtrain split"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/prepare_grpo_selector_navtrain_split.py \
    --config "${DIFFUSIONDRIVE_GRPO_CONFIG}" \
    --output-dir "${DIFFUSIONDRIVE_GRPO_SPLIT_ROOT}"

echo "[2/5] Audit cache coverage and formal model/checkpoint contract"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/audit_grpo_online_dataset.py \
    --config "${DIFFUSIONDRIVE_GRPO_CONFIG}"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/preflight_grpo_online_selector.py \
    --config "${DIFFUSIONDRIVE_GRPO_CONFIG}"

echo "[3/5] Synthetic trajectory official-PDM parity"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/validate_online_pdm_reward.py \
    --metric-cache "${NAVSIM_METRIC_CACHE_PATH_TRAIN}" \
    --device cuda \
    --num-candidates 20 \
    --atol 1e-5

echo "[4/5] Real generated candidates: 32 tokens x 20 trajectories"
"${ALGENGINE_PYTHON}" scripts/diffusiondrive/validate_grpo_selector_real_candidates.py \
    "${DIFFUSIONDRIVE_GRPO_CONFIG}" \
    --nav-filter "${DIFFUSIONDRIVE_GRPO_CALIBRATION_FILTER_PATH}" \
    --num-tokens 32 \
    --noise-seed 0 \
    --atol 1e-5 \
    --output "${PREFLIGHT_ROOT}/real_candidate_parity_seed0.json"

echo "[5/5] Dedicated selector unit/contract tests"
"${ALGENGINE_PYTHON}" -m pytest -q \
    tests/test_diffusion_grpo_online_selector.py \
    tests/test_diffusion_grpo_selector_formal.py

echo "PASS formal DiffusionDrive selector-GRPO preflight"
echo "gate: ${PREFLIGHT_ROOT}/real_candidate_parity_seed0.json"
