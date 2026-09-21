#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SELECTOR_LAUNCH_PYTHON:-python3}" "$ROOT/projects/AlgEngine/scripts/diffusiondrive/selector_runtime.py" "$@"
