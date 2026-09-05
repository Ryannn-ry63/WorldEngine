#!/usr/bin/env bash
# Run one locked formal model; submit this script once for each seed 0/1/2.
set -Eeuo pipefail

MANIFEST="${1:?usage: $0 FORMAL_TRAIN_MANIFEST SEED}"
SEED="${2:?usage: $0 FORMAL_TRAIN_MANIFEST SEED}"
[[ "${SEED}" =~ ^[012]$ ]] || { echo "seed must be 0, 1, or 2" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t VALUES < <(/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python - "${MANIFEST}" "${SEED}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1])); seed=sys.argv[2]
assert row["status"]=="PASS" and row["independent_training_seeds"]==[0,1,2]
assert row["selector_architecture"] in {"interaction_generic","interaction_relation"}
print(row["models"][seed]["checkpoint"])
print(row["models"][seed]["checkpoint_sha256"])
print(row["selected_arm"])
print(row["selector_architecture"])
PY
)
[[ "${#VALUES[@]}" -eq 4 ]] || { echo "invalid formal train manifest" >&2; exit 1; }
CHECKPOINT="${VALUES[0]}"
CHECKPOINT_SHA="${VALUES[1]}"
ARM="${VALUES[2]}"
ARCHITECTURE="${VALUES[3]}"

export DIFFUSIONDRIVE_INTERACTION_ARCHITECTURE="${ARCHITECTURE}"
export REASONER_CONFIG_OVERRIDE="$(cd "${SCRIPT_DIR}/../../configs/diffusiondrive" && pwd)/e2e_diffusiondrive_grpo_selector_interaction_v1.py"
export REASONER_EXPERIMENT_ROOT_OVERRIDE="$(cd "${SCRIPT_DIR}/../../../.." && pwd)/experiments/diffusiondrive/interaction_selector_v1"
MODEL_NAME="interaction_selector_v1_${ARM}_trainseed${SEED}"
NOTE="locked ${ARM} explicit frozen-track interaction selector; independent train seed ${SEED}"
exec "${SCRIPT_DIR}/run_trajectory_set_reasoner_formal_eval_8hopper.sh" \
    "${CHECKPOINT}" "${CHECKPOINT_SHA}" "${MODEL_NAME}" "${SEED}" "${NOTE}"

