#!/usr/bin/env bash
# Final train-only current-frame audit: decision-consistent top-1 reranking.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"

GPU_INDEX="${DCSR_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

V3_CHECKPOINT="${REASONER_SOURCE_WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/sweep/trials/t1_lr1e-4_kl1e-3/epoch_64_scene_selector.pt"
V3_SHA256=d19c9d825ab452c96891d5b6abf0f35186460e9538cf37b329fea2c1bd86d133
CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/interaction_selector_v1/cache/tuning"
PAIR_ROOT="${REASONER_TUNING_ROOT}/train"
PRIOR_GATE="${WORLDENGINE_ROOT}/experiments/diffusiondrive/hard_pair_audit_v1/audit/audit_20260901T141121Z/hard_pair_gate.json"
EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_dcsr_audit_v1"
RUN_ID="${DCSR_AUDIT_ID:-audit_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${EXPERIMENT_ROOT}/audit/${RUN_ID}"

required=(
    "${ALGENGINE_PYTHON}"
    "${V3_CHECKPOINT}"
    "${PAIR_ROOT}/pairs.jsonl"
    "${PAIR_ROOT}/rare_data_audit.json"
    "${PRIOR_GATE}"
    "${CACHE_ROOT}/train_seed0/cache.pt"
    "${CACHE_ROOT}/train_seed1/cache.pt"
    "${CACHE_ROOT}/train_seed2/cache.pt"
)
for path in "${required[@]}"; do
    [[ -e "${path}" ]] || { echo "Missing DCSR dependency: ${path}" >&2; exit 1; }
done
[[ "$(sha256sum "${V3_CHECKPOINT}" | awk '{print $1}')" == "${V3_SHA256}" ]] || {
    echo "Rare-tuned V3 checkpoint SHA256 mismatch" >&2
    exit 1
}
[[ ! -e "${RUN_ROOT}" ]] || { echo "DCSR audit root already exists: ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"
LOG="${RUN_ROOT}/audit.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=preflight
trap 'rc=$?; echo "FAIL DCSR audit stage=${CURRENT_STAGE} exit=${rc} log=${LOG}" >&2' ERR

"${ALGENGINE_PYTHON}" - <<'PY'
import json, torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(names) != 1:
    raise RuntimeError(f"expected exactly 1 visible GPU after isolation, got {len(names)}")
if not any(name in names[0].upper() for name in ("H100", "H200")):
    raise RuntimeError(f"DCSR audit requires Hopper, got {names[0]!r}")
if torch.cuda.get_device_capability(0) != (9, 0):
    raise RuntimeError(f"expected sm_90, got {torch.cuda.get_device_capability(0)}")
if torch.version.cuda != "11.8" or not str(torch.__version__).startswith("2.0.1+cu118"):
    raise RuntimeError(f"environment drift: torch={torch.__version__}, runtime={torch.version.cuda}")
print(json.dumps({"status": "PASS", "visible_devices": names, "hardware_contract": "one_hopper", "torch": torch.__version__}, sort_keys=True))
PY

CURRENT_STAGE=dcsr_locked_top1_audit
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/probe_dcsr_top1_objective.py" \
    --train-cache "${CACHE_ROOT}/train_seed0/cache.pt" \
    --train-cache "${CACHE_ROOT}/train_seed1/cache.pt" \
    --train-cache "${CACHE_ROOT}/train_seed2/cache.pt" \
    --pair-manifest "${PAIR_ROOT}/pairs.jsonl" \
    --rare-data-audit "${PAIR_ROOT}/rare_data_audit.json" \
    --v3-checkpoint "${V3_CHECKPOINT}" \
    --v3-checkpoint-sha256 "${V3_SHA256}" \
    --prior-hard-pair-gate "${PRIOR_GATE}" \
    --output "${RUN_ROOT}/locked_top1_gate.json" \
    --device cuda

CURRENT_STAGE=complete
trap - ERR
echo "PASS DCSR locked-top1 train-only audit: ${RUN_ROOT}"
echo "gate: ${RUN_ROOT}/locked_top1_gate.json"
echo "log: ${LOG}"

