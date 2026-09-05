#!/usr/bin/env bash
# Zero-new-label proposal-history information audit (backup path only).
set -Eeuo pipefail

RUN_ID="${1:-audit_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "Invalid run id: ${RUN_ID}" >&2
    exit 2
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"
reasoner_setup_assets

SOURCE_WORLDENGINE="${REASONER_SOURCE_WORLDENGINE_ROOT}"
V3_CHECKPOINT="${SOURCE_WORLDENGINE}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/sweep/trials/t1_lr1e-4_kl1e-3/epoch_64_scene_selector.pt"
V3_SHA256=d19c9d825ab452c96891d5b6abf0f35186460e9538cf37b329fea2c1bd86d133
CACHE_ROOT="${SOURCE_WORLDENGINE}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/tuning"
PAIR_ROOT="${SOURCE_WORLDENGINE}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1/tuning_split/train"
ANNOTATION=/inspire/hdd2/project/roboticsystem2/public/worldengine/data/WE_release_data/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl
AUDITOR="${SCRIPT_DIR}/audit_proposal_history_information.py"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_proposal_history_audit_v1/${RUN_ID}"

for path in \
    "${ALGENGINE_PYTHON}" "${AUDITOR}" "${V3_CHECKPOINT}" "${ANNOTATION}" \
    "${CACHE_ROOT}/train_seed0/cache.pt" "${CACHE_ROOT}/train_seed1/cache.pt" \
    "${CACHE_ROOT}/train_seed2/cache.pt" "${PAIR_ROOT}/pairs.jsonl" \
    "${PAIR_ROOT}/rare_data_audit.json"
do
    [[ -e "${path}" ]] || { echo "Missing history-audit dependency: ${path}" >&2; exit 1; }
done
[[ "$(sha256sum "${V3_CHECKPOINT}" | awk '{print $1}')" == "${V3_SHA256}" ]] || {
    echo "V3 checkpoint SHA256 mismatch" >&2
    exit 1
}

"${ALGENGINE_PYTHON}" - <<'PY'
import json,torch
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if not names:
    raise RuntimeError("history audit requires at least one visible GPU")
if any(not any(family in name.upper() for family in ("H100","H200")) for name in names):
    raise RuntimeError(f"history audit requires Hopper, got {names}")
if any(torch.cuda.get_device_capability(i)!=(9,0) for i in range(len(names))):
    raise RuntimeError("history audit requires sm_90")
if torch.version.cuda!="11.8":
    raise RuntimeError(f"history audit requires CUDA runtime 11.8, got {torch.version.cuda}")
if not str(torch.__version__).startswith("2.0.1+cu118"):
    raise RuntimeError(f"history audit requires torch 2.0.1+cu118, got {torch.__version__}")
print(json.dumps({"status":"PASS","hardware_contract":"hopper_at_least_one","devices":names,"torch":torch.__version__},sort_keys=True))
PY

[[ ! -e "${ROOT}" ]] || { echo "Refusing to reuse audit root: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
"${ALGENGINE_PYTHON}" "${AUDITOR}" \
    --train-cache "${CACHE_ROOT}/train_seed0/cache.pt" \
    --train-cache "${CACHE_ROOT}/train_seed1/cache.pt" \
    --train-cache "${CACHE_ROOT}/train_seed2/cache.pt" \
    --pair-manifest "${PAIR_ROOT}/pairs.jsonl" \
    --rare-data-audit "${PAIR_ROOT}/rare_data_audit.json" \
    --annotation "${ANNOTATION}" \
    --v3-checkpoint "${V3_CHECKPOINT}" \
    --v3-checkpoint-sha256 "${V3_SHA256}" \
    --output "${ROOT}/history_information_gate.json" \
    --pairs-per-token 16 \
    --incumbent-top-k 5 \
    --epochs 8 \
    --batch-size 1024 \
    --feature-batch-size 64 \
    --bootstrap-repetitions 10000 \
    --expected-predecessor-pairs 2424 \
    --seed 20260902 \
    --device cuda | tee "${ROOT}/history_audit.log"

echo "PASS proposal-history information audit: ${ROOT}"
echo "gate: ${ROOT}/history_information_gate.json"
