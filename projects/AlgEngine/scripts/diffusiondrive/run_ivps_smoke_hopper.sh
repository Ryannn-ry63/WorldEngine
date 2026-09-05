#!/usr/bin/env bash
# Unit, two-stage real-cache training, materialization and config-load smoke.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 1

IVPS_EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1"
RUN_ID="${IVPS_SMOKE_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${IVPS_EXPERIMENT_ROOT}/smoke/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "IVPS smoke root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}/proposal" "${ROOT}/verifier"
LOG="${ROOT}/smoke.log"
exec > >(tee -a "${LOG}") 2>&1

"${ALGENGINE_PYTHON}" -m pytest -q \
    "${ALGENGINE_ROOT}/tests/test_incumbent_verified_preference_selector.py" \
    "${ALGENGINE_ROOT}/tests/test_rapg_scalar_preference_loss.py" \
    "${ALGENGINE_ROOT}/tests/test_rapg_experiment_contracts.py"

mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/train_reference_anchored_preference_graph.py" \
    "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
    --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
    --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
    --output-dir "${ROOT}/proposal" --architecture trajectory_set_reasoner \
    --ablation relational_only --objective official_pdm_plus_scalar_preference \
    --preference-weight 1.0 --data-split train --seed 0 --epochs 1 \
    --examples-per-cache-epoch 16 --checkpoint-epochs 1 --batch-size 8 \
    --smoke-limit-hard-pool 2 --device cuda

CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/train_incumbent_verified_preference_selector.py" \
    "${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
    --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
    --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
    --proposal-checkpoint "${ROOT}/proposal/epoch_1_scene_selector.pt" \
    --output-dir "${ROOT}/verifier" --verifier-reward-margin 0.03 \
    --verifier-reward-temperature 0.05 --override-threshold 0.0 \
    --data-split train --seed 0 --epochs 1 --examples-per-cache-epoch 16 \
    --checkpoint-epochs 1 --batch-size 8 --smoke-limit-hard-pool 2 --device cuda

STATE="${ROOT}/verifier/epoch_1_scene_selector.pt"
STATE_SHA="$(sha256sum "${STATE}" | awk '{print $1}')"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/materialize_rapg_selector.py" \
    --baseline "${RAPG_BASELINE}" --scene-selector-state "${STATE}" \
    --expected-selector-sha256 "${STATE_SHA}" \
    --expected-method ivps_v1_incumbent_verified_preference_m030 \
    --output "${ROOT}/checkpoint.pth" \
    --manifest "${ROOT}/checkpoint_manifest.json" \
    --release-name diffusiondrive_selector_ivps_v1
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    --baseline "${RAPG_BASELINE}" --checkpoint "${ROOT}/checkpoint.pth" \
    --output "${ROOT}/checkpoint_audit.json"

cd "${ALGENGINE_ROOT}"
DIFFUSIONDRIVE_RAPG_ARCHITECTURE=incumbent_verified_preference \
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" - "${RAPG_CONFIG}" \
    "${ROOT}/checkpoint.pth" <<'PY'
import json,sys,torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
config=Config.fromfile(sys.argv[1])
model=build_model(config.model)
load_checkpoint(model,sys.argv[2],map_location="cpu",strict=False)
selector=model.planning_head.scene_selector
assert selector.__class__.__name__=="IncumbentVerifiedPreferenceSelector"
assert all(torch.isfinite(value).all() for value in selector.state_dict().values())
print(json.dumps({"status":"PASS","selector":selector.__class__.__name__},sort_keys=True))
PY

echo "PASS IVPS real-cache smoke: ${ROOT}"
