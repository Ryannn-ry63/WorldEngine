#!/usr/bin/env bash
# Stage III: full-data independent train seeds after the development gate locks A1/A2.
set -Eeuo pipefail

DEVELOPMENT_GATE="${1:?usage: $0 DEVELOPMENT_GATE_JSON}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/interaction_selector_env.sh"
interaction_preflight

read -r SELECTED_ARM ARCHITECTURE < <("${ALGENGINE_PYTHON}" - "${DEVELOPMENT_GATE}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS" and row["development_only"]
assert row["certification_consumed"] is False
arm=row["selected_arm"]
assert row["decision"]==f"AUTHORIZE_{arm}_CLOSED_LOOP_DEVELOPMENT"
architecture={"A1":"interaction_generic","A2":"interaction_relation"}[arm]
assert row["arms"][arm]["selector_architecture"]==architecture
print(arm,architecture)
PY
)

RUN_ID="${INTERACTION_FORMAL_TRAIN_ID:-${SELECTED_ARM}_16epoch_3seed_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${INTERACTION_EXPERIMENT_ROOT}/formal/train/${RUN_ID}"
[[ ! -e "${ROOT}" ]] || { echo "formal train root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}/logs"
LOG="${ROOT}/formal_train.log"
exec > >(tee -a "${LOG}") 2>&1
CURRENT_STAGE=full_cache
trap 'rc=$?; echo "FAIL interaction formal train stage=${CURRENT_STAGE} exit=${rc} log=${LOG}" >&2' ERR

for seed in 0 1 2; do
    interaction_extract_cache train "${seed}" 12540 \
        "${REASONER_PAIR_ROOT}/rare_common_union.yaml" full
done

TRAINER="${SCRIPT_DIR}/train_trajectory_set_reasoner_grpo.py"
MATERIALIZER="${SCRIPT_DIR}/materialize_trajectory_set_reasoner.py"
train_seed() {
    local seed="$1" gpu="$2" seed_root="${ROOT}/seed${seed}"
    mkdir -p "${seed_root}/train"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" \
        "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --train-cache "${INTERACTION_CACHE_ROOT}/full/train_seed0/cache.pt" \
        --train-cache "${INTERACTION_CACHE_ROOT}/full/train_seed1/cache.pt" \
        --train-cache "${INTERACTION_CACHE_ROOT}/full/train_seed2/cache.pt" \
        --pair-manifest "${REASONER_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${REASONER_PAIR_ROOT}/rare_data_audit.json" \
        --output-dir "${seed_root}/train" --sampling-mode rare_balanced \
        --architecture "${ARCHITECTURE}" --ablation full \
        --method-name "interaction_selector_v1_${SELECTED_ARM}_formal_seed${seed}" \
        --seed "${seed}" --temperature 1 --learning-rate 1e-4 --kl-weight 1e-3 \
        --epochs 16 --examples-per-cache-epoch 6339 \
        --checkpoint-epochs 16 --batch-size 64 --device cuda
    local state="${seed_root}/train/epoch_16_scene_selector.pt"
    local state_sha method
    state_sha="$(sha256sum "${state}" | awk '{print $1}')"
    method="$("${ALGENGINE_PYTHON}" - "${state}" <<'PY'
import sys,torch
print(torch.load(sys.argv[1],map_location="cpu")["method"])
PY
)"
    "${ALGENGINE_PYTHON}" "${MATERIALIZER}" \
        --baseline "${REASONER_BASELINE}" --scene-selector-state "${state}" \
        --expected-selector-sha256 "${state_sha}" --expected-method "${method}" \
        --release-name "interaction_selector_v1_${SELECTED_ARM}_seed${seed}" \
        --output "${seed_root}/checkpoint.pth" \
        --manifest "${seed_root}/checkpoint_manifest.json"
    "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py" \
        --baseline "${REASONER_BASELINE}" --checkpoint "${seed_root}/checkpoint.pth" \
        --output "${seed_root}/checkpoint_audit.json"
}

CURRENT_STAGE=independent_training
pids=()
for seed in 0 1 2; do
    (train_seed "${seed}" "${seed}") > "${ROOT}/logs/seed${seed}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
[[ "${failed}" -eq 0 ]] || { echo "formal seed failed; inspect ${ROOT}/logs" >&2; exit 1; }

CURRENT_STAGE=audit
"${ALGENGINE_PYTHON}" - "${ROOT}" "${DEVELOPMENT_GATE}" "${SELECTED_ARM}" "${ARCHITECTURE}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); gate=Path(sys.argv[2]).resolve(); arm=sys.argv[3]; architecture=sys.argv[4]
models={}
for seed in (0,1,2):
    seed_root=root/f"seed{seed}"
    report=json.load(open(seed_root/"train"/"report.json"))
    material=json.load(open(seed_root/"checkpoint_manifest.json"))
    audit=json.load(open(seed_root/"checkpoint_audit.json"))
    assert report["status"]==material["status"]==audit["status"]=="PASS"
    assert report["selector_architecture"]==architecture and report["train_seed"]==seed
    assert report["epochs"]==16 and report["sampling"]["total_examples"]==304272
    assert report["sampling"]["total_optimizer_steps"]==4800
    assert audit["changed_baseline_tensor_count"]==0
    checkpoint=seed_root/"checkpoint.pth"
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert sha==material["checkpoint_sha256"]
    models[str(seed)]={"checkpoint":str(checkpoint),"checkpoint_sha256":sha}
output={
    "schema_version":1,"status":"PASS",
    "method":"interaction_selector_v1_three_seed_formal_train",
    "development_gate":str(gate),"development_gate_sha256":hashlib.sha256(gate.read_bytes()).hexdigest(),
    "selected_arm":arm,"selector_architecture":architecture,
    "independent_training_seeds":[0,1,2],"models":models,
    "next_action":"submit one formal evaluation per seed with the locked architecture",
}
(root/"formal_train_manifest.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
print(json.dumps(output,sort_keys=True))
PY

CURRENT_STAGE=complete
trap - ERR
echo "PASS interaction selector three-seed formal training"
echo "manifest: ${ROOT}/formal_train_manifest.json"
echo "log: ${LOG}"

