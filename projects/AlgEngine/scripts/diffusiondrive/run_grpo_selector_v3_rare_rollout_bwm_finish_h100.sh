#!/usr/bin/env bash
# Train/evaluate V3 on the audited multi-source BWM mixture.

set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export DIFFUSIONDRIVE_RARE_ROLLOUT_METHOD="scene_conditioned_exact_group_grpo_v3_rare_rollout_bwm_v1"
export DIFFUSIONDRIVE_RARE_ROLLOUT_DATA_METHOD="diffusiondrive_v3_rare_rollout_bwm_mixture_v1"
export DIFFUSIONDRIVE_RARE_ROLLOUT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_bwm_v1"
export DIFFUSIONDRIVE_RARE_ROLLOUT_DATA_ROOT="${DIFFUSIONDRIVE_RARE_ROLLOUT_ROOT}/data"
export DIFFUSIONDRIVE_RARE_ROLLOUT_MODEL_PREFIX="e2e_diffusiondrive_grpo_selector_v3_rare_rollout_bwm_v1"
export DIFFUSIONDRIVE_RARE_ROLLOUT_FORMAL_NOTE="V3 rare-rollout BWM; senior collision/EP/offroad worlds; fixed compute"
export DIFFUSIONDRIVE_RARE_ROLLOUT_LEGACY_AGGREGATE=0

exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_finish_h100.sh" "$@"
