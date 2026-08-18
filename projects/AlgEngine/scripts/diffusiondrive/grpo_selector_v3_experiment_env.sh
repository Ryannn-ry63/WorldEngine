#!/usr/bin/env bash
# Resolve an isolated V3 experiment namespace without changing legacy defaults.

if [[ -z "${WORLDENGINE_ROOT:-}" ]]; then
    echo "WORLDENGINE_ROOT must be set before sourcing V3 experiment env" >&2
    return 1
fi

export DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME="${DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME:-grpo_selector_v3}"
if ! [[ "${DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME" >&2
    return 1
fi

export DIFFUSIONDRIVE_GRPO_V3_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/${DIFFUSIONDRIVE_GRPO_V3_EXPERIMENT_NAME}"
export V3_ROOT="${DIFFUSIONDRIVE_GRPO_V3_ROOT}"
