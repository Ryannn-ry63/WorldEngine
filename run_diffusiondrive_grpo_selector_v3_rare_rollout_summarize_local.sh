#!/usr/bin/env bash
# CPU-only aggregation after finish-seed 0/1/2 have all passed.

set -Eeo pipefail

WORLDENGINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGENGINE_PYTHON="/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python"
SCRIPT_DIR="${WORLDENGINE_ROOT}/projects/AlgEngine/scripts/diffusiondrive"
ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1"
FORMAL_ROOT="${ROOT}/formal"
DATA_MANIFEST="${ROOT}/data/manifest.json"
SUMMARIZER="${SCRIPT_DIR}/summarize_grpo_selector_v3_rare_rollout.py"
OUTPUT="${FORMAL_ROOT}/rare_rollout_comparison.json"

for path in "${ALGENGINE_PYTHON}" "${SUMMARIZER}" "${DATA_MANIFEST}"; do
    [[ -e "${path}" ]] || {
        echo "Missing summarize input: ${path}" >&2
        exit 1
    }
done

summary_args=()
for seed in 0 1 2; do
    rollout_summary="${FORMAL_ROOT}/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_rollout_v1_s${seed}/summary.json"
    [[ -f "${rollout_summary}" ]] || {
        echo "Seed ${seed} has not completed: ${rollout_summary}" >&2
        exit 1
    }
    summary_args+=(
        --base "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_formal/formal_eval/e2e_diffusiondrive_reference_paired_s${seed}/summary.json"
        --common-v3 "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_progress_fix_v1_s${seed}/summary.json"
        --rare-frozen "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_frozen_s${seed}/summary.json"
        --rare-tuned "${seed}=${WORLDENGINE_ROOT}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/formal/formal_eval/e2e_diffusiondrive_grpo_selector_v3_rare_tuned_s${seed}/summary.json"
        --rare-rollout "${seed}=${rollout_summary}"
    )
done

"${ALGENGINE_PYTHON}" "${SUMMARIZER}" "${summary_args[@]}" \
    --data-manifest "${DATA_MANIFEST}" \
    --output "${OUTPUT}"

echo "PASS DiffusionDrive V3 rare-rollout local summary"
echo "comparison: ${FORMAL_ROOT}/rare_rollout_comparison.md"
