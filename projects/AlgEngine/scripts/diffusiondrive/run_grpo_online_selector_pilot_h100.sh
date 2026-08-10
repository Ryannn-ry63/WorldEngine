#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="${1:-ddgrpo_online_pilot_lr3e6_s0}"

# The 1e-5 smoke reached ~0.26 clip fraction by iteration 30.  Start the
# attribution pilot from the immutable baseline with a conservative LR and
# preserve 50-iteration snapshots for paired selector evaluation.
export DIFFUSIONDRIVE_GRPO_LR="${DIFFUSIONDRIVE_GRPO_LR:-3e-6}"
export DIFFUSIONDRIVE_GRPO_MAX_ITERS="${DIFFUSIONDRIVE_GRPO_MAX_ITERS:-250}"
export DIFFUSIONDRIVE_GRPO_CHECKPOINT_INTERVAL="${DIFFUSIONDRIVE_GRPO_CHECKPOINT_INTERVAL:-50}"
export DIFFUSIONDRIVE_GRPO_MAX_KEEP_CKPTS="${DIFFUSIONDRIVE_GRPO_MAX_KEEP_CKPTS:-6}"

exec "${SCRIPT_DIR}/run_grpo_online_selector_h100.sh" pilot "${RUN_NAME}"
