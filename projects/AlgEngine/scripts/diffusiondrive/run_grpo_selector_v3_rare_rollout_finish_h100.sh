#!/usr/bin/env bash
# One 8-H100 allocation: three fresh selector replicas, then four-block evals.

set -Eeo pipefail

MODE="${1:-all}"
TARGET_SEED=""
case "${MODE}" in
    all)
        [[ "${#}" -eq 0 ]] || {
            echo "usage: $0 [seed {0|1|2}]" >&2
            exit 2
        }
        ;;
    seed)
        TARGET_SEED="${2:-}"
        if [[ "${#}" -ne 2 || ! "${TARGET_SEED}" =~ ^[0-2]$ ]]; then
            echo "usage: $0 seed {0|1|2}" >&2
            exit 2
        fi
        ;;
    *)
        echo "usage: $0 [seed {0|1|2}]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DIFFUSIONDRIVE_GRPO_CONFIG="${SCRIPT_DIR}/../../configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
. "${SCRIPT_DIR}/grpo_selector_h100_env.sh"

BASELINE="${DIFFUSIONDRIVE_GRPO_BASELINE}"
BASELINE_SHA256="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
METHOD="scene_conditioned_exact_group_grpo_v3_rare_rollout_v1"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1"
DATA_ROOT="${ROOT}/data"
REAL_CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/full"
MODEL_ROOT="${ROOT}/models"
FORMAL_ROOT="${ROOT}/formal"
TRAINER="${SCRIPT_DIR}/train_grpo_selector_v3_cached_rare_rollout.py"
MATERIALIZER="${SCRIPT_DIR}/materialize_grpo_selector_v3.py"
CHECKPOINT_AUDIT="${SCRIPT_DIR}/audit_grpo_selector_v3_checkpoint.py"
FORMAL_TABLE="${SCRIPT_DIR}/run_grpo_selector_formal_table_h100.sh"
SUMMARIZER="${SCRIPT_DIR}/summarize_grpo_selector_v3_rare_rollout.py"
if [[ "${MODE}" == "seed" ]]; then
    RUN_LABEL="seed${TARGET_SEED}"
    STATUS="${ROOT}/finish_${RUN_LABEL}_status.txt"
    LOG_BASENAME="finish_${RUN_LABEL}"
else
    RUN_LABEL="all"
    # Preserve the original combined-pipeline paths for backward compatibility.
    STATUS="${ROOT}/finish_status.txt"
    LOG_BASENAME="finish"
fi
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${LOG_BASENAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CURRENT_STAGE=static_preflight
CODE_SHA="$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD)"
printf 'RUNNING code_sha=%s stage=%s started_utc=%s\n' \
    "${CODE_SHA}" "${CURRENT_STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS}"
trap 'rc=$?; printf "FAIL code_sha=%s stage=%s exit=%s\n" "${CODE_SHA}" "${CURRENT_STAGE}" "${rc}" > "${STATUS}"; echo "FAIL rare-rollout finish stage=${CURRENT_STAGE} line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; echo "persistent_log: ${LOG_FILE}" >&2' ERR
for path in "${BASELINE}" "${DATA_ROOT}/manifest.json" "${DATA_ROOT}/hard_pool.jsonl" \
    "${DATA_ROOT}/synthetic_cache.pt" "${TRAINER}" "${MATERIALIZER}" \
    "${CHECKPOINT_AUDIT}" "${FORMAL_TABLE}" "${SUMMARIZER}"; do
    [[ -f "${path}" ]] || { echo "Missing finish input: ${path}" >&2; exit 1; }
done
for seed in 0 1 2; do
    [[ -f "${REAL_CACHE_ROOT}/train_seed${seed}/cache.pt" ]] || {
        echo "Missing real cache seed ${seed}" >&2
        exit 1
    }
done
ACTUAL_BASELINE_SHA="$(sha256sum "${BASELINE}" | awk '{print $1}')"
[[ "${ACTUAL_BASELINE_SHA}" == "${BASELINE_SHA256}" ]] || {
    echo "Baseline SHA256 mismatch" >&2
    exit 1
}
if ! git -C "${WORLDENGINE_ROOT}" diff --quiet -- projects/AlgEngine projects/SimEngine; then
    echo "Tracked implementation files are dirty; refusing formal finish" >&2
    exit 1
fi
"${ALGENGINE_PYTHON}" -c '
import hashlib,json,pathlib,sys
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert row["source_policy"]=="immutable_epoch100_diffusiondrive"
assert row["behavior_policy_is_trained_v3"] is False
assert row["mixture_contract"]["overall_common_fraction"]==0.5
assert row["mixture_contract"]["overall_hard_fraction"]==0.5
assert sha(pathlib.Path(row["synthetic_cache"]))==row["synthetic_cache_sha256"]
assert sha(pathlib.Path(row["hard_pool"]))==row["hard_pool_sha256"]
' "${DATA_ROOT}/manifest.json"

CURRENT_STAGE=cuda_optimizer_preflight
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 "${ALGENGINE_PYTHON}" \
    "${SCRIPT_DIR}/preflight_grpo_selector_v3_h100_optimizer.py" --ablation full

training_ready() {
    local seed="$1"
    local seed_root="${MODEL_ROOT}/seed${seed}"
    [[ -f "${seed_root}/train/report.json" \
        && -f "${seed_root}/checkpoint.pth" \
        && -f "${seed_root}/checkpoint_manifest.json" \
        && -f "${seed_root}/checkpoint_audit.json" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import hashlib,json,pathlib,sys
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()
report=json.load(open(sys.argv[1]))
manifest=json.load(open(sys.argv[2]))
audit=json.load(open(sys.argv[3]))
checkpoint=pathlib.Path(sys.argv[4])
assert report["status"]==manifest["status"]==audit["status"]=="PASS"
assert report["method"]==sys.argv[5]
assert report["source_policy"]=="immutable_epoch100_diffusiondrive"
assert report["fresh_selector_initialization"]=="exact_zero"
assert report["sampling"]["total_examples"]==304272
assert report["sampling"]["total_optimizer_steps"]==4800
assert report["sampling"]["class_examples"]["common"]==152136
assert report["sampling"]["class_examples"]["hard"]==152136
assert sha(checkpoint)==manifest["checkpoint_sha256"]==audit["checkpoint_sha256"]
assert audit["changed_baseline_tensor_count"]==0
assert audit["scene_selector_tensor_count"]==54
' "${seed_root}/train/report.json" "${seed_root}/checkpoint_manifest.json" \
        "${seed_root}/checkpoint_audit.json" "${seed_root}/checkpoint.pth" "${METHOD}"
}

train_seed() {
    local seed="$1"
    local gpu="$2"
    local seed_root="${MODEL_ROOT}/seed${seed}"
    if training_ready "${seed}"; then
        echo "SKIP verified rare-rollout training seed ${seed}"
        return
    fi
    if [[ -e "${seed_root}" ]]; then
        echo "Partial immutable model root exists: ${seed_root}" >&2
        return 1
    fi
    mkdir -p "${seed_root}/train"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --real-cache "0=${REAL_CACHE_ROOT}/train_seed0/cache.pt" \
        --real-cache "1=${REAL_CACHE_ROOT}/train_seed1/cache.pt" \
        --real-cache "2=${REAL_CACHE_ROOT}/train_seed2/cache.pt" \
        --synthetic-cache "${DATA_ROOT}/synthetic_cache.pt" \
        --data-manifest "${DATA_ROOT}/manifest.json" \
        --hard-pool "${DATA_ROOT}/hard_pool.jsonl" \
        --output-dir "${seed_root}/train" \
        --temperature 1.0 --learning-rate 1e-4 --kl-weight 1e-3 \
        --seed "${seed}" --epochs 16 --examples-per-cache-epoch 6339 \
        --checkpoint-epochs 16 --batch-size 64 --device cuda \
        --method-name "${METHOD}" --formal-contract
    local state="${seed_root}/train/epoch_16_scene_selector.pt"
    local state_sha
    state_sha="$(sha256sum "${state}" | awk '{print $1}')"
    "${ALGENGINE_PYTHON}" "${MATERIALIZER}" \
        --baseline "${BASELINE}" \
        --scene-selector-state "${state}" \
        --expected-selector-sha256 "${state_sha}" \
        --expected-method "${METHOD}" \
        --release-name "e2e_diffusiondrive_grpo_selector_v3_rare_rollout_v1_s${seed}" \
        --output "${seed_root}/checkpoint.pth" \
        --manifest "${seed_root}/checkpoint_manifest.json"
    "${ALGENGINE_PYTHON}" "${CHECKPOINT_AUDIT}" \
        --baseline "${BASELINE}" \
        --checkpoint "${seed_root}/checkpoint.pth" \
        --output "${seed_root}/checkpoint_audit.json"
    training_ready "${seed}"
    echo "PASS rare-rollout model seed ${seed}"
}

CURRENT_STAGE="train_${RUN_LABEL}"
train_pids=()
if [[ "${MODE}" == "seed" ]]; then
    TRAIN_SEEDS=("${TARGET_SEED}")
else
    TRAIN_SEEDS=(0 1 2)
fi
for seed in "${TRAIN_SEEDS[@]}"; do
    if [[ "${MODE}" == "seed" ]]; then
        gpu=0
    else
        gpu="${seed}"
    fi
    train_seed "${seed}" "${gpu}" > "${LOG_DIR}/train_seed${seed}.log" 2>&1 &
    train_pids+=("$!")
done
training_failure=0
for pid in "${train_pids[@]}"; do
    if ! wait "${pid}"; then
        training_failure=1
    fi
done
[[ "${training_failure}" -eq 0 ]] || {
    echo "One or more training replicas failed; inspect ${LOG_DIR}/train_seed*.log" >&2
    exit 1
}
for seed in "${TRAIN_SEEDS[@]}"; do
    training_ready "${seed}"
done

evaluation_ready() {
    local seed="$1"
    local model="e2e_diffusiondrive_grpo_selector_v3_rare_rollout_v1_s${seed}"
    local summary="${FORMAL_ROOT}/formal_eval/${model}/summary.json"
    local manifest="${MODEL_ROOT}/seed${seed}/checkpoint_manifest.json"
    [[ -f "${summary}" && -f "${manifest}" ]] || return 1
    "${ALGENGINE_PYTHON}" -c '
import json,sys
summary=json.load(open(sys.argv[1])); manifest=json.load(open(sys.argv[2]))
assert summary["status"]=="PASS"
assert int(summary["eval_seed"])==int(sys.argv[3])
assert summary["checkpoint_sha256"]==manifest["checkpoint_sha256"]
' "${summary}" "${manifest}" "${seed}"
}

export DIFFUSIONDRIVE_GRPO_FORMAL_ROOT="${FORMAL_ROOT}"
export DIFFUSIONDRIVE_GRPO_POLICY_TEMPERATURE=1.0
export DIFFUSIONDRIVE_GRPO_KL_WEIGHT=0.001
for seed in "${TRAIN_SEEDS[@]}"; do
    CURRENT_STAGE="formal_eval_seed${seed}"
    if evaluation_ready "${seed}"; then
        echo "SKIP verified rare-rollout formal evaluation seed ${seed}"
        continue
    fi
    checkpoint="${MODEL_ROOT}/seed${seed}/checkpoint.pth"
    checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
    model="e2e_diffusiondrive_grpo_selector_v3_rare_rollout_v1_s${seed}"
    "${FORMAL_TABLE}" "${checkpoint}" "${checkpoint_sha}" "${model}" "${seed}" \
        "V3 rare-rollout-v1; epoch100 behavior; fresh selector; fixed compute"
    evaluation_ready "${seed}"
done

if [[ "${MODE}" == "all" ]]; then
    CURRENT_STAGE=aggregate_formal_results
    summary_args=()
    for seed in 0 1 2; do
        summary_args+=(
            --base "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/formal_eval/e2e_diffusiondrive_reference_paired_s${seed}/summary.json"
            --common-v3 "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_progress_fix_v1_s${seed}/summary.json"
            --rare-frozen "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_frozen_s${seed}/summary.json"
            --rare-tuned "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_tuned_s${seed}/summary.json"
            --rare-rollout "${seed}=${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_rollout_v1_s${seed}/summary.json"
        )
    done
    "${ALGENGINE_PYTHON}" "${SUMMARIZER}" "${summary_args[@]}" \
        --data-manifest "${DATA_ROOT}/manifest.json" \
        --output "${FORMAL_ROOT}/rare_rollout_comparison.json"
fi

CURRENT_STAGE=complete
printf 'PASS code_sha=%s completed_utc=%s\n' \
    "${CODE_SHA}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATUS}"
trap - ERR
if [[ "${MODE}" == "seed" ]]; then
    echo "PASS DiffusionDrive V3 rare-rollout finish seed ${TARGET_SEED}"
    echo "summary: ${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_rollout_v1_s${TARGET_SEED}/summary.json"
else
    echo "PASS DiffusionDrive V3 rare-rollout finish pipeline"
    echo "comparison: ${FORMAL_ROOT}/rare_rollout_comparison.md"
fi
echo "persistent_log: ${LOG_FILE}"
