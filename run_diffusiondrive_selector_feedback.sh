#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0
FEEDBACK_WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${FEEDBACK_WORKTREE}/run_diffusiondrive_selector_feedback_8h100.sh" "$@"
