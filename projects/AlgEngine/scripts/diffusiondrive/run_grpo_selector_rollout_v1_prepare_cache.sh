#!/usr/bin/env bash
set -Eeo pipefail

DATA_TYPE="${1:?usage: run_grpo_selector_rollout_v1_prepare_cache.sh DATA_TYPE RUN_ID}"
RUN_ID="${2:?usage: run_grpo_selector_rollout_v1_prepare_cache.sh DATA_TYPE RUN_ID}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORLDENGINE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
ALGENGINE_PYTHON="${ALGENGINE_PYTHON:-/root/miniconda3/envs/algengine/bin/python}"
export PYTHONPATH="${ALGENGINE_ROOT}:${SIMENGINE_ROOT}:${PYTHONPATH:-}"
BASELINE=/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth
BASELINE_SHA=1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_rollout_v1"
ROLLOUT_BASE="${ROOT}/rollouts/${DATA_TYPE}"
CACHE_ROOT="${ROOT}/cache"
SPLIT_MANIFEST="${ROOT}/splits/${DATA_TYPE}_${RUN_ID}.json"
SELECTOR_CONFIG="${WORLDENGINE_ROOT}/projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3_rollout_base.py"

[[ ! -e "${SPLIT_MANIFEST}" ]] || {
    echo "Immutable split manifest exists: ${SPLIT_MANIFEST}" >&2
    exit 1
}

for seed in {0..8}; do
    audit="${ROLLOUT_BASE}/seed${seed}/${RUN_ID}/rollout_audit.json"
    [[ -f "${audit}" ]] || { echo "Missing rollout audit: ${audit}" >&2; exit 1; }
done
for seed in {0..8}; do
    if (( seed <= 2 )); then
        split=train
    elif (( seed <= 5 )); then
        split=development
    else
        split=certification
    fi
    output="${CACHE_ROOT}/${split}_seed${seed}"
    [[ ! -e "${output}" ]] || {
        echo "Immutable cache target exists: ${output}" >&2
        exit 1
    }
done
[[ -f "${SELECTOR_CONFIG}" ]] || { echo "Missing rollout selector config: ${SELECTOR_CONFIG}" >&2; exit 1; }

split_args=()
for seed in {0..8}; do
    split_args+=(--record-root "${seed}=${ROLLOUT_BASE}/seed${seed}/${RUN_ID}")
done
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/prepare_grpo_selector_rollout_v1_splits.py" \
    "${split_args[@]}" --minimum-scenes 20 --output "${SPLIT_MANIFEST}"

for seed in {0..8}; do
    if (( seed <= 2 )); then
        split=train
    elif (( seed <= 5 )); then
        split=development
    else
        split=certification
    fi
    expected_records="$(
        "${ALGENGINE_PYTHON}" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["split_record_counts_per_seed"][sys.argv[2]])' \
            "${SPLIT_MANIFEST}" "${split}"
    )"
    output="${CACHE_ROOT}/${split}_seed${seed}"
    [[ ! -e "${output}" ]] || { echo "Immutable cache target exists: ${output}" >&2; exit 1; }
    "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/build_grpo_selector_rollout_v1_cache.py" \
        --record-root "${ROLLOUT_BASE}/seed${seed}/${RUN_ID}" \
        --split-manifest "${SPLIT_MANIFEST}" --split "${split}" \
        --rollout-seed "${seed}" \
        --noise-namespace "diffusiondrive_rollout_v1_seed${seed}" \
        --selector-config "${SELECTOR_CONFIG}" \
        --baseline-checkpoint "${BASELINE}" \
        --expected-baseline-sha256 "${BASELINE_SHA}" \
        --expected-num-records "${expected_records}" \
        --output-dir "${output}"
done
echo "PASS DiffusionDrive rollout-v1 immutable split and nine caches"
echo "split_manifest: ${SPLIT_MANIFEST}"
echo "cache_root: ${CACHE_ROOT}"
