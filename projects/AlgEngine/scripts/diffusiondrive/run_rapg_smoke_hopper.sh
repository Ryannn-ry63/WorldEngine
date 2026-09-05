#!/usr/bin/env bash
# Unit, real-cache training, materialization and config-load smoke test.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 1

RUN_ID="${RAPG_SMOKE_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${RAPG_EXPERIMENT_ROOT}/smoke/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "smoke root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}/train"
LOG="${ROOT}/smoke.log"
exec > >(tee -a "${LOG}") 2>&1

"${ALGENGINE_PYTHON}" -m pytest -q \
    "${ALGENGINE_ROOT}/tests/test_reference_anchored_preference_graph.py" \
    "${ALGENGINE_ROOT}/tests/test_rapg_scalar_preference_loss.py" \
    "${ALGENGINE_ROOT}/tests/test_trajectory_set_reasoner_selector.py" \
    "${ALGENGINE_ROOT}/tests/test_grpo_selector_v3_cached_common.py"

TRAINER="${SCRIPT_DIR}/train_reference_anchored_preference_graph.py"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${TRAINER}" \
    "${CACHE_ARGS[@]}" \
    --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
    --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
    --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
    --output-dir "${ROOT}/train" --architecture reference_anchored_preference_graph \
    --ablation relational_only --objective official_pdm_plus_scalar_preference \
    --preference-weight 0.5 --data-split train --seed 0 --epochs 1 \
    --examples-per-cache-epoch 16 --checkpoint-epochs 1 --batch-size 8 \
    --smoke-limit-hard-pool 2 --device cuda

STATE="${ROOT}/train/epoch_1_scene_selector.pt"
STATE_SHA="$(sha256sum "${STATE}" | awk '{print $1}')"
METHOD="rapg_v1_reference_anchored_preference_graph_relational_only_exact_group_grpo_scalar_preference"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_rapg_selector.py" \
    --baseline "${RAPG_BASELINE}" --scene-selector-state "${STATE}" \
    --expected-selector-sha256 "${STATE_SHA}" --expected-method "${METHOD}" \
    --output "${ROOT}/checkpoint.pth" --manifest "${ROOT}/checkpoint_manifest.json"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    --baseline "${RAPG_BASELINE}" --checkpoint "${ROOT}/checkpoint.pth" \
    --output "${ROOT}/checkpoint_audit.json"

cd "${ALGENGINE_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" - "${RAPG_CONFIG}" "${ROOT}/checkpoint.pth" <<'PY'
import json,sys,torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
config=Config.fromfile(sys.argv[1])
model=build_model(config.model)
load_checkpoint(model,sys.argv[2],map_location="cpu",strict=False)
selector=model.planning_head.scene_selector
assert selector.__class__.__name__=="ReferenceAnchoredPreferenceGraphSelector"
assert all(torch.isfinite(value).all() for value in selector.state_dict().values())
print(json.dumps({"status":"PASS","selector":selector.__class__.__name__},sort_keys=True))
PY

echo "PASS RAPG real-cache smoke: ${ROOT}"
