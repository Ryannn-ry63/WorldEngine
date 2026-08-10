#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="${1:-ddgrpo_online_pi_ref_smoke_lr3e6_s0}"

# Short attribution run for the exact full-action expectation under
# pi_old == pi_ref. Start from the immutable baseline, never from the failed
# uniform-candidate pilot.
export DIFFUSIONDRIVE_GRPO_LR="${DIFFUSIONDRIVE_GRPO_LR:-3e-6}"
export DIFFUSIONDRIVE_GRPO_MAX_ITERS="${DIFFUSIONDRIVE_GRPO_MAX_ITERS:-64}"
export DIFFUSIONDRIVE_GRPO_CHECKPOINT_INTERVAL="${DIFFUSIONDRIVE_GRPO_CHECKPOINT_INTERVAL:-32}"
export DIFFUSIONDRIVE_GRPO_MAX_KEEP_CKPTS="${DIFFUSIONDRIVE_GRPO_MAX_KEEP_CKPTS:-3}"
exec "${SCRIPT_DIR}/run_grpo_online_selector_h100.sh" smoke "${RUN_NAME}"
