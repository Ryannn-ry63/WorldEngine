#!/usr/bin/env bash
# Reuse the frozen rare-rollout data with gate-conditioned GRPO training.

set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export DIFFUSIONDRIVE_RARE_ROLLOUT_METHOD="scene_conditioned_gate_conditioned_group_grpo_v3_rare_rollout_v1"
export DIFFUSIONDRIVE_RARE_ROLLOUT_OBJECTIVE="gate_conditioned_pdm"
export DIFFUSIONDRIVE_RARE_ROLLOUT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_gate_conditioned_v1"
export DIFFUSIONDRIVE_RARE_ROLLOUT_DATA_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data"
export DIFFUSIONDRIVE_RARE_ROLLOUT_MODEL_PREFIX="e2e_diffusiondrive_grpo_selector_v3_gate_conditioned_v1"
export DIFFUSIONDRIVE_RARE_ROLLOUT_FORMAL_NOTE="V3 gate-conditioned GRPO; frozen rare-rollout-v1 data; fixed compute"
export DIFFUSIONDRIVE_RARE_ROLLOUT_LEGACY_AGGREGATE=0

exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_finish_h100.sh" "$@"
