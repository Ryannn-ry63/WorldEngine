#!/usr/bin/env bash
# Real-cache smoke on one of eight visible homogeneous H100/H200 GPUs.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"
reasoner_hopper_preflight 8

RUN_ID="${REASONER_SMOKE_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${REASONER_EXPERIMENT_ROOT}/smoke/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "smoke root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}/train"
LOG="${ROOT}/smoke.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=unit_tests
trap 'rc=$?; echo "FAIL reasoner smoke stage=${CURRENT_STAGE} exit=${rc} log=${LOG}" >&2' ERR

"${ALGENGINE_PYTHON}" -m pytest -q \
    "${ALGENGINE_ROOT}/tests/test_diffusion_grpo_scene_selector.py" \
    "${ALGENGINE_ROOT}/tests/test_trajectory_set_reasoner_selector.py" \
    "${ALGENGINE_ROOT}/tests/test_grpo_selector_v3_cached_common.py" \
    "${ALGENGINE_ROOT}/tests/test_trajectory_set_reasoner_promotion.py"

TRAINER="${SCRIPT_DIR}/train_trajectory_set_reasoner_grpo.py"
MATERIALIZER="${SCRIPT_DIR}/materialize_trajectory_set_reasoner.py"
METHOD="trajectory_set_reasoner_exact_group_grpo_v1_trajectory_set_reasoner_full_rare_balanced"
CURRENT_STAGE=real_cache_train
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${TRAINER}" \
    --train-cache "${REASONER_CACHE_ROOT}/full/train_seed0/cache.pt" \
    --train-cache "${REASONER_CACHE_ROOT}/full/train_seed1/cache.pt" \
    --train-cache "${REASONER_CACHE_ROOT}/full/train_seed2/cache.pt" \
    --pair-manifest "${REASONER_PAIR_ROOT}/pairs.jsonl" \
    --rare-data-audit "${REASONER_PAIR_ROOT}/rare_data_audit.json" \
    --output-dir "${ROOT}/train" \
    --sampling-mode rare_balanced --architecture trajectory_set_reasoner \
    --ablation full --seed 0 --epochs 1 --examples-per-cache-epoch 64 \
    --checkpoint-epochs 1 --batch-size 16 --device cuda \
    --allow-partial-coverage

STATE="${ROOT}/train/epoch_1_scene_selector.pt"
STATE_SHA="$(sha256sum "${STATE}" | awk '{print $1}')"
CURRENT_STAGE=materialize
"${ALGENGINE_PYTHON}" "${MATERIALIZER}" \
    --baseline "${REASONER_BASELINE}" --scene-selector-state "${STATE}" \
    --expected-selector-sha256 "${STATE_SHA}" --expected-method "${METHOD}" \
    --output "${ROOT}/checkpoint.pth" --manifest "${ROOT}/checkpoint_manifest.json" \
    --release-name trajectory_set_reasoner_grpo_v1_smoke
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
    --baseline "${REASONER_BASELINE}" --checkpoint "${ROOT}/checkpoint.pth" \
    --output "${ROOT}/checkpoint_audit.json"

CURRENT_STAGE=original_config_load
cd "${ALGENGINE_ROOT}"
CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" - \
    "${REASONER_CONFIG}" "${ROOT}/checkpoint.pth" <<'PY'
import json,sys,torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

config=Config.fromfile(sys.argv[1])
model=build_model(config.model)
load_checkpoint(model,sys.argv[2],map_location="cpu",strict=False)
selector=model.planning_head.scene_selector
assert selector.__class__.__name__=="TrajectorySetReasoningResidualSelector"
assert selector.config_dict()["architecture"]=="trajectory_set_reasoner"
assert all(torch.isfinite(value).all() for value in selector.state_dict().values())
print(json.dumps({"status":"PASS","selector":selector.__class__.__name__,"tensor_count":len(selector.state_dict())},sort_keys=True))
PY
cd "${WORLDENGINE_ROOT}"

CURRENT_STAGE=audit_training
"${ALGENGINE_PYTHON}" - "${ROOT}/train/report.json" <<'PY'
import json,math,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS" and row["schema_version"]==4
assert row["selector_architecture"]=="trajectory_set_reasoner"
assert row["sampling"]["total_optimizer_steps"]==12
for key in ("mean_training_loss","mean_training_policy","mean_training_kl"):
    assert math.isfinite(float(row[key]))
assert row["sampling"]["class_examples"]=={"rare":96,"common":96}
print("PASS finite real-cache exact-GRPO training and balanced sampling")
PY

CURRENT_STAGE=complete
trap - ERR
echo "PASS 8-Hopper trajectory-set reasoner smoke"
echo "output: ${ROOT}"
echo "log: ${LOG}"
