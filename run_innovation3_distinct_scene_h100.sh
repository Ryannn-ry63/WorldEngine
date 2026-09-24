#!/usr/bin/env bash
# Engineering acceptance only. Refuses reused outputs; does not train formal seeds.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
I3_CODE_ROOT="$PWD"
I3_WORKSPACE="$(dirname -- "$I3_CODE_ROOT")"
I3_MODE="${1:-all}"
I3_TAG="${2:-v1}"
I3_BRANCH_WORKERS="${3:-0}"
case "$I3_MODE" in all|2|8) ;; *) echo 'Usage: bash run_innovation3_distinct_scene_h100.sh [all|2|8] [fresh-tag] [branch-workers]' >&2; exit 2;; esac
[[ "$I3_BRANCH_WORKERS" =~ ^[0-9]+$ ]] || { echo 'Invalid branch-worker count' >&2; exit 2; }
(( I3_BRANCH_WORKERS <= 20 )) || { echo 'branch-workers must be <= 20' >&2; exit 2; }
[[ "$I3_TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid output tag' >&2; exit 2; }
I3_SETTINGS="${ONLINE_SETTINGS:-$I3_WORKSPACE/registry/online_runtime_owned_subset_20260923.json}"
I3_MANIFEST="${ONLINE_MANIFEST:-$I3_WORKSPACE/registry/distinct_scene_pilot_20260923_v1/scene_manifest.json}"
I3_PYTHON="/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python"
I3_RUNTIME='projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py'
I3_BRANCH_ARGS=()
if (( I3_BRANCH_WORKERS > 0 )); then I3_BRANCH_ARGS=(--branch-workers "$I3_BRANCH_WORKERS"); fi
I3_TWO="$I3_WORKSPACE/registry/ddp_distinct_h100_2rank_${I3_TAG}.json"
I3_EIGHT="$I3_WORKSPACE/registry/ddp_distinct_h100_8rank_${I3_TAG}.json"
check_result() {
  "$I3_PYTHON" - "$1" "$2" "$I3_MANIFEST" "$I3_CODE_ROOT" "$I3_SETTINGS" <<'PY'
import hashlib,json,sys
from pathlib import Path
report=Path(sys.argv[1]); world=int(sys.argv[2]); manifest=Path(sys.argv[3]); code=Path(sys.argv[4])
x=json.loads(report.read_text())
assert x['status']=='PASS_DDP_REAL_ONLINE_THROUGHPUT_PILOT',x['status']
assert x['distinct_scene_verified'] is True and x['unique_scene_count']==world and x['world_size']==world
assert x['formal_ready'] is False and x['used_for_formal_training'] is False
settings=json.loads(Path(sys.argv[5]).read_text()) if len(sys.argv)>5 else None
if settings is not None: assert x.get('parameterization') == settings.get('online_parameterization', 'frozen_v3_plus_zero_residual')
assert x['scene_manifest']['sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
for name,h in x['source_sha256'].items():
 assert hashlib.sha256((code/name).read_bytes()).hexdigest()==h, 'Code changed since accepted pilot: '+name
print('Accepted distinct-scene',world,'rank pilot:',report)
PY
}
if [[ "$I3_MODE" == 8 ]]; then
  check_result "$I3_TWO" 2
fi
I3_DEVICES='0,1,2,3,4,5,6,7'
[[ "$I3_MODE" == 2 ]] && I3_DEVICES='0,1'
"$I3_PYTHON" "$I3_RUNTIME" --settings "$I3_SETTINGS" \
  --output "$I3_WORKSPACE/registry/ddp_distinct_h100_preflight_${I3_MODE}_${I3_TAG}.json" \
  --devices "$I3_DEVICES" preflight
if [[ "$I3_MODE" != 8 ]]; then
  "$I3_PYTHON" "$I3_RUNTIME" --settings "$I3_SETTINGS" --scene-manifest "$I3_MANIFEST" \
    --output "$I3_TWO" --devices 0,1 --steps 8 --seed 0 "${I3_BRANCH_ARGS[@]}" ddp-online-throughput-probe
  check_result "$I3_TWO" 2
fi
if [[ "$I3_MODE" != 2 ]]; then
  "$I3_PYTHON" "$I3_RUNTIME" --settings "$I3_SETTINGS" --scene-manifest "$I3_MANIFEST" \
    --output "$I3_EIGHT" --devices 0,1,2,3,4,5,6,7 --steps 8 --seed 0 "${I3_BRANCH_ARGS[@]}" ddp-online-throughput-probe
  check_result "$I3_EIGHT" 8
fi
