#!/usr/bin/env bash
set -Eeo pipefail

cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
exec ./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_progress_fix_smoke_h100.sh "$@"
