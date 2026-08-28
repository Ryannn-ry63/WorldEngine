#!/usr/bin/env bash
# Independently train seeds 0/1/2 after architecture+epoch are development-locked.
set -Eeuo pipefail

ARCHITECTURE_ROOT="${1:?usage: $0 ARCHITECTURE_ROOT LABEL EPOCH}"
LABEL="${2:?usage: $0 ARCHITECTURE_ROOT LABEL EPOCH}"
EPOCH="${3:?usage: $0 ARCHITECTURE_ROOT LABEL EPOCH}"
[[ "${LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid label" >&2; exit 2; }
[[ "${EPOCH}" =~ ^(16|32|64)$ ]] || { echo "epoch must be 16, 32, or 64" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"
reasoner_hopper_preflight 8

SHORTLIST="${ARCHITECTURE_ROOT}/shortlist.json"
[[ -f "${SHORTLIST}" ]] || { echo "missing development shortlist: ${SHORTLIST}" >&2; exit 1; }
"${ALGENGINE_PYTHON}" - "${SHORTLIST}" "${LABEL}" "${EPOCH}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1])); label=sys.argv[2]; epoch=int(sys.argv[3])
assert row["status"]=="PASS" and not row["certification_consumed"]
matches=[value for value in row["shortlist"] if value["trial"]==label and int(value["epoch"])==epoch]
if len(matches)!=1:
    raise RuntimeError(f"label/epoch was not uniquely development-shortlisted: {label}/{epoch}")
print("PASS immutable development shortlist lock",label,epoch)
PY

case "${LABEL}" in
    capacity_v3) ARCHITECTURE=scene_conditioned_v3; ABLATION=full; NUM_SET_LAYERS=4 ;;
    temporal_only) ARCHITECTURE=trajectory_set_reasoner; ABLATION=temporal_only; NUM_SET_LAYERS=1 ;;
    relational_only) ARCHITECTURE=trajectory_set_reasoner; ABLATION=relational_only; NUM_SET_LAYERS=1 ;;
    no_scene_context) ARCHITECTURE=trajectory_set_reasoner; ABLATION=no_scene_context; NUM_SET_LAYERS=1 ;;
    full) ARCHITECTURE=trajectory_set_reasoner; ABLATION=full; NUM_SET_LAYERS=1 ;;
    *) echo "unsupported locked architecture label: ${LABEL}" >&2; exit 2 ;;
esac

RUN_ID="${REASONER_FORMAL_TRAIN_ID:-${LABEL}_epoch${EPOCH}}"
ROOT="${REASONER_EXPERIMENT_ROOT}/formal/train/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "formal train root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/formal_train.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=train
trap 'rc=$?; echo "FAIL reasoner formal train stage=${CURRENT_STAGE} exit=${rc} log=${LOG}" >&2' ERR

TRAINER="${SCRIPT_DIR}/train_trajectory_set_reasoner_grpo.py"
MATERIALIZER="${SCRIPT_DIR}/materialize_trajectory_set_reasoner.py"
EXPECTED_EXAMPLES="$((EPOCH * 6339 * 3))"
EXPECTED_STEPS="$((EPOCH * 300))"

train_seed() {
    local seed="$1" gpu="$2" seed_root="${ROOT}/seed${seed}"
    mkdir -p "${seed_root}/train"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --train-cache "${REASONER_CACHE_ROOT}/full/train_seed0/cache.pt" \
        --train-cache "${REASONER_CACHE_ROOT}/full/train_seed1/cache.pt" \
        --train-cache "${REASONER_CACHE_ROOT}/full/train_seed2/cache.pt" \
        --pair-manifest "${REASONER_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${REASONER_PAIR_ROOT}/rare_data_audit.json" \
        --output-dir "${seed_root}/train" --sampling-mode rare_balanced \
        --architecture "${ARCHITECTURE}" --ablation "${ABLATION}" \
        --num-set-layers "${NUM_SET_LAYERS}" --seed "${seed}" \
        --temperature 1 --learning-rate 1e-4 --kl-weight 1e-3 \
        --epochs "${EPOCH}" --examples-per-cache-epoch 6339 \
        --checkpoint-epochs "${EPOCH}" --batch-size 64 --device cuda
    local state="${seed_root}/train/epoch_${EPOCH}_scene_selector.pt"
    local state_sha method
    state_sha="$(sha256sum "${state}" | awk '{print $1}')"
    method="$(${ALGENGINE_PYTHON} - "${state}" <<'PY'
import sys,torch
print(torch.load(sys.argv[1],map_location="cpu")["method"])
PY
)"
    "${ALGENGINE_PYTHON}" "${MATERIALIZER}" \
        --baseline "${REASONER_BASELINE}" --scene-selector-state "${state}" \
        --expected-selector-sha256 "${state_sha}" --expected-method "${method}" \
        --release-name "trajectory_set_reasoner_${LABEL}_epoch${EPOCH}_seed${seed}" \
        --output "${seed_root}/checkpoint.pth" \
        --manifest "${seed_root}/checkpoint_manifest.json"
    "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
        --baseline "${REASONER_BASELINE}" --checkpoint "${seed_root}/checkpoint.pth" \
        --output "${seed_root}/checkpoint_audit.json"
}

pids=()
for seed in 0 1 2; do
    (train_seed "${seed}" "${seed}") > "${ROOT}/logs/seed${seed}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
[[ "${failed}" -eq 0 ]] || { echo "formal training seed failed; inspect ${ROOT}/logs" >&2; exit 1; }

CURRENT_STAGE=audit
"${ALGENGINE_PYTHON}" - "${ROOT}" "${LABEL}" "${EPOCH}" \
    "${EXPECTED_EXAMPLES}" "${EXPECTED_STEPS}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); label=sys.argv[2]; epoch=int(sys.argv[3])
expected_examples=int(sys.argv[4]); expected_steps=int(sys.argv[5]); rows={}
for seed in (0,1,2):
    seed_root=root/f"seed{seed}"
    report=json.load(open(seed_root/"train"/"report.json"))
    manifest=json.load(open(seed_root/"checkpoint_manifest.json"))
    audit=json.load(open(seed_root/"checkpoint_audit.json"))
    assert report["status"]==manifest["status"]==audit["status"]=="PASS"
    assert report["train_seed"]==seed and report["epochs"]==epoch
    assert report["sampling"]["total_examples"]==expected_examples
    assert report["sampling"]["total_optimizer_steps"]==expected_steps
    assert audit["changed_baseline_tensor_count"]==0
    checkpoint=seed_root/"checkpoint.pth"
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sha==manifest["checkpoint_sha256"]
    rows[str(seed)]={"checkpoint":str(checkpoint),"checkpoint_sha256":sha}
output={"schema_version":1,"status":"PASS","label":label,"epoch":epoch,"independent_training_seeds":[0,1,2],"models":rows}
(root/"formal_train_manifest.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
print(json.dumps(output,sort_keys=True))
PY

CURRENT_STAGE=complete
trap - ERR
echo "PASS independent three-seed trajectory-set formal training"
echo "manifest: ${ROOT}/formal_train_manifest.json"
echo "log: ${LOG}"
