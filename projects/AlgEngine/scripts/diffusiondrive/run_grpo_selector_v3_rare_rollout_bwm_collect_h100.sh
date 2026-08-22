#!/usr/bin/env bash
# Collect one BWM generalization lane with the immutable epoch-100 policy.

set -Eeo pipefail

LANE="${1:?usage: $0 LANE [MAX_SCENARIOS] [RUN_KIND]}"
MAX_SCENARIOS="${2:--1}"
RUN_KIND="${3:-formal}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export DIFFUSIONDRIVE_RARE_ROLLOUT_SOURCE_ROOT="${DIFFUSIONDRIVE_RARE_ROLLOUT_SOURCE_ROOT:-${WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_rollout_bwm_v1}"
export DIFFUSIONDRIVE_RARE_ROLLOUT_SCENARIO_AUDIT="${DIFFUSIONDRIVE_RARE_ROLLOUT_SCENARIO_AUDIT:-${DIFFUSIONDRIVE_RARE_ROLLOUT_SOURCE_ROOT}/scenarios/bwm_scenario_audit.json}"
export DIFFUSIONDRIVE_RARE_ROLLOUT_COLLECTION_ROOT="${DIFFUSIONDRIVE_RARE_ROLLOUT_COLLECTION_ROOT:-${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_bwm_v1}"
export DIFFUSIONDRIVE_RARE_ROLLOUT_NOISE_NAMESPACE="${DIFFUSIONDRIVE_RARE_ROLLOUT_NOISE_NAMESPACE:-diffusiondrive_v3_rare_rollout_bwm_v1}"
export DIFFUSIONDRIVE_RARE_ROLLOUT_RECORDS_PER_SCENE="${DIFFUSIONDRIVE_RARE_ROLLOUT_RECORDS_PER_SCENE:-9}"

exec "${SCRIPT_DIR}/run_grpo_selector_v3_rare_rollout_collect_h100.sh" \
    "${LANE}" "${MAX_SCENARIOS}" "${RUN_KIND}"
