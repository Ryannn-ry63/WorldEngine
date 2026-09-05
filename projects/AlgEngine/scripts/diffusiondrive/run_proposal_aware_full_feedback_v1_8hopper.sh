#!/usr/bin/env bash
# Staged, train-only mechanism audit and fixed-budget four-arm PAF search.
set -Eeuo pipefail

MODE="${1:-}"
shift || true
case "${MODE}" in
    audit-existing|cache-fresh-seed|audit-fresh|search|fresh-and-search|all) ;;
    *)
        echo "Usage:" >&2
        echo "  $0 audit-existing [RUN_ID]" >&2
        echo "  $0 cache-fresh-seed {9|10|11} EXISTING_GATE" >&2
        echo "  $0 audit-fresh EXISTING_GATE [RUN_ID]" >&2
        echo "  $0 search FRESH_GATE [SEARCH_ID]" >&2
        echo "  $0 fresh-and-search EXISTING_GATE [RUN_ID]" >&2
        echo "  $0 all [RUN_ID]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"
reasoner_hopper_preflight 8

SOURCE_WORLDENGINE="${REASONER_SOURCE_WORLDENGINE_ROOT}"
V3_CHECKPOINT="${SOURCE_WORLDENGINE}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/sweep/trials/t1_lr1e-4_kl1e-3/epoch_64_scene_selector.pt"
V3_SHA256=d19c9d825ab452c96891d5b6abf0f35186460e9538cf37b329fea2c1bd86d133
OLD_CACHE_ROOT="${REASONER_CACHE_ROOT}/tuning"
PAIR_ROOT="${REASONER_TUNING_ROOT}/train"
V3_CONFIG="${SOURCE_WORLDENGINE}/projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
NAV_FILTER="${PAIR_ROOT}/rare_common_union.yaml"
EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_paf_grpo_v1"
FRESH_CACHE_ROOT="${EXPERIMENT_ROOT}/cache"
AUDITOR="${SCRIPT_DIR}/audit_selector_policy_suppression.py"
EXTRACTOR="${SCRIPT_DIR}/extract_grpo_selector_context_cache.py"
TRAINER="${SCRIPT_DIR}/train_proposal_aware_full_feedback_grpo.py"
EVALUATOR="${SCRIPT_DIR}/evaluate_proposal_aware_full_feedback_grpo.py"
SELECTOR="${SCRIPT_DIR}/select_proposal_aware_full_feedback.py"

required=(
    "${ALGENGINE_PYTHON}" "${ALGENGINE_TORCHRUN}" "${V3_CHECKPOINT}" "${V3_CONFIG}"
    "${PAIR_ROOT}/pairs.jsonl" "${PAIR_ROOT}/rare_data_audit.json" "${NAV_FILTER}"
    "${OLD_CACHE_ROOT}/train_seed0/cache.pt" "${OLD_CACHE_ROOT}/train_seed1/cache.pt"
    "${OLD_CACHE_ROOT}/train_seed2/cache.pt" "${AUDITOR}" "${EXTRACTOR}"
    "${TRAINER}" "${EVALUATOR}" "${SELECTOR}"
)
for path in "${required[@]}"; do
    [[ -e "${path}" ]] || { echo "Missing PAF dependency: ${path}" >&2; exit 1; }
done
[[ "$(sha256sum "${V3_CHECKPOINT}" | awk '{print $1}')" == "${V3_SHA256}" ]] || {
    echo "V3 checkpoint SHA256 mismatch" >&2
    exit 1
}
mkdir -p "${EXPERIMENT_ROOT}/audit" "${EXPERIMENT_ROOT}/search" "${FRESH_CACHE_ROOT}"

validate_id() {
    [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid run id: $1" >&2; exit 2; }
}

verify_existing_gate() {
    local gate="$1"
    "${ALGENGINE_PYTHON}" - "${gate}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert row["method"]=="selector_policy_suppression_audit_v1"
assert row["stage"]=="existing"
assert row["decision"]=="AUTHORIZE_FRESH_NOISE_CONFIRMATION"
assert row["noise_seeds"]==[0,1,2]
assert row["scientific_contract"]["development_consumed"] is False
PY
}

verify_fresh_gate() {
    local gate="$1"
    "${ALGENGINE_PYTHON}" - "${gate}" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS"
assert row["method"]=="selector_policy_suppression_audit_v1"
assert row["stage"]=="fresh"
assert row["decision"]=="AUTHORIZE_PROPOSAL_AWARE_FULL_FEEDBACK_GRPO"
assert row["noise_seeds"]==[9,10,11]
assert row["scientific_contract"]["development_consumed"] is False
PY
}

run_existing_audit() {
    local run_id="${1:-audit_$(date -u +%Y%m%dT%H%M%SZ)}"
    validate_id "${run_id}"
    local root="${EXPERIMENT_ROOT}/audit/${run_id}"
    [[ ! -e "${root}" ]] || { echo "Refusing to reuse audit root: ${root}" >&2; exit 1; }
    mkdir -p "${root}"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${AUDITOR}" \
        --cache "${OLD_CACHE_ROOT}/train_seed0/cache.pt" \
        --cache "${OLD_CACHE_ROOT}/train_seed1/cache.pt" \
        --cache "${OLD_CACHE_ROOT}/train_seed2/cache.pt" \
        --expected-noise-seeds 0,1,2 \
        --pair-manifest "${PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${PAIR_ROOT}/rare_data_audit.json" \
        --v3-checkpoint "${V3_CHECKPOINT}" \
        --v3-checkpoint-sha256 "${V3_SHA256}" \
        --stage existing \
        --output "${root}/mechanism_gate.json" \
        --bootstrap-repetitions 10000 \
        --device cuda | tee "${root}/audit.log"
    echo "PASS existing-cache mechanism audit: ${root}"
    echo "gate: ${root}/mechanism_gate.json"
}

fresh_count() {
    "${ALGENGINE_PYTHON}" - "${PAIR_ROOT}/rare_data_audit.json" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
print(int(row["union_count"]))
PY
}

validate_fresh_cache() {
    local seed="$1"
    local output="${FRESH_CACHE_ROOT}/train_seed${seed}"
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${ALGENGINE_PYTHON}" - \
        "${output}/cache.pt" "${seed}" "$(fresh_count)" "${NAV_FILTER}" <<'PY'
import sys
from pathlib import Path
import grpo_selector_v3_cached_common as common
cache,manifest=common.load_cache(Path(sys.argv[1]),"train")
assert int(manifest["noise_seed"])==int(sys.argv[2])
assert len(cache["tokens"])==int(sys.argv[3])==int(manifest["num_tokens"])
assert manifest["nav_filter_sha256"]==common.sha256_file(Path(sys.argv[4]))
assert manifest["cache_sha256"]==common.sha256_file(Path(sys.argv[1]))
assert manifest["checkpoint_sha256"]=="1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514"
print("PASS fresh cache",manifest["noise_seed"],manifest["num_tokens"])
PY
}

run_fresh_cache() {
    local seed="$1"
    local existing_gate="$2"
    [[ "${seed}" =~ ^(9|10|11)$ ]] || { echo "Fresh seed must be 9, 10, or 11" >&2; exit 2; }
    verify_existing_gate "${existing_gate}"
    local output="${FRESH_CACHE_ROOT}/train_seed${seed}"
    if [[ -f "${output}/cache.pt" && -f "${output}/manifest.json" ]]; then
        validate_fresh_cache "${seed}"
        return
    fi
    [[ ! -e "${output}" ]] || { echo "Incomplete fresh cache root exists: ${output}" >&2; exit 1; }
    mkdir -p "${output}"
    export NAVSIM_METRIC_CACHE_PATH_TRAIN="${REASONER_METRIC_CACHE_INDEX}"
    cd "${ALGENGINE_ROOT}"
    "${ALGENGINE_TORCHRUN}" \
        --nproc_per_node=8 \
        --master_port="$((29900 + seed))" \
        "${EXTRACTOR}" "${V3_CONFIG}" "${REASONER_BASELINE}" \
        --expected-checkpoint-sha256 "${REASONER_BASELINE_SHA256}" \
        --nav-filter "${NAV_FILTER}" \
        --split train \
        --noise-seed "${seed}" \
        --expected-num-tokens "$(fresh_count)" \
        --output-dir "${output}" \
        --workers-per-gpu 2 \
        --launcher pytorch
    cd "${WORLDENGINE_ROOT}"
    validate_fresh_cache "${seed}"
}

run_fresh_audit() {
    local existing_gate="$1"
    local run_id="${2:-fresh_$(date -u +%Y%m%dT%H%M%SZ)}"
    verify_existing_gate "${existing_gate}"
    validate_id "${run_id}"
    local root="${EXPERIMENT_ROOT}/audit/${run_id}"
    [[ ! -e "${root}" ]] || { echo "Refusing to reuse audit root: ${root}" >&2; exit 1; }
    for seed in 9 10 11; do validate_fresh_cache "${seed}"; done
    mkdir -p "${root}"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${AUDITOR}" \
        --cache "${FRESH_CACHE_ROOT}/train_seed9/cache.pt" \
        --cache "${FRESH_CACHE_ROOT}/train_seed10/cache.pt" \
        --cache "${FRESH_CACHE_ROOT}/train_seed11/cache.pt" \
        --expected-noise-seeds 9,10,11 \
        --pair-manifest "${PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${PAIR_ROOT}/rare_data_audit.json" \
        --v3-checkpoint "${V3_CHECKPOINT}" \
        --v3-checkpoint-sha256 "${V3_SHA256}" \
        --stage fresh \
        --required-prior-gate "${existing_gate}" \
        --output "${root}/mechanism_gate.json" \
        --bootstrap-repetitions 10000 \
        --device cuda | tee "${root}/audit.log"
    echo "PASS fresh-noise mechanism audit: ${root}"
    echo "gate: ${root}/mechanism_gate.json"
}

run_search() {
    local fresh_gate="$1"
    local search_id="${2:-search_$(date -u +%Y%m%dT%H%M%SZ)}"
    verify_fresh_gate "${fresh_gate}"
    validate_id "${search_id}"
    local root="${EXPERIMENT_ROOT}/search/${search_id}"
    [[ ! -e "${root}" ]] || { echo "Refusing to reuse search root: ${root}" >&2; exit 1; }
    for seed in 9 10 11; do validate_fresh_cache "${seed}"; done
    mkdir -p "${root}/trials" "${root}/logs"
    local arms=(direct_grpo full_feedback opportunity_grpo paf_grpo)

    run_arm() {
        local gpu="$1"
        local arm="$2"
        local trial="${root}/trials/${arm}"
        mkdir -p "${trial}"
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
        "${ALGENGINE_PYTHON}" "${TRAINER}" \
            --train-cache "${OLD_CACHE_ROOT}/train_seed0/cache.pt" \
            --train-cache "${OLD_CACHE_ROOT}/train_seed1/cache.pt" \
            --train-cache "${OLD_CACHE_ROOT}/train_seed2/cache.pt" \
            --pair-manifest "${PAIR_ROOT}/pairs.jsonl" \
            --rare-data-audit "${PAIR_ROOT}/rare_data_audit.json" \
            --v3-checkpoint "${V3_CHECKPOINT}" \
            --v3-checkpoint-sha256 "${V3_SHA256}" \
            --mechanism-gate "${fresh_gate}" \
            --output-dir "${trial}" \
            --arm "${arm}" \
            --temperature 1 \
            --learning-rate 3e-5 \
            --kl-weight 1e-3 \
            --opportunity-lower 0.005 \
            --opportunity-upper 0.1 \
            --epochs 8 \
            --examples-per-epoch 6339 \
            --batch-size 64 \
            --checkpoint-epochs 1,2,4,8 \
            --require-full-coverage \
            --seed 0 \
            --device cuda

        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
        "${ALGENGINE_PYTHON}" "${EVALUATOR}" \
            --checkpoint "${trial}/epoch_8_scene_selector.pt" \
            --cache "${FRESH_CACHE_ROOT}/train_seed9/cache.pt" \
            --cache "${FRESH_CACHE_ROOT}/train_seed10/cache.pt" \
            --cache "${FRESH_CACHE_ROOT}/train_seed11/cache.pt" \
            --pair-manifest "${PAIR_ROOT}/pairs.jsonl" \
            --rare-data-audit "${PAIR_ROOT}/rare_data_audit.json" \
            --mechanism-gate "${fresh_gate}" \
            --output "${trial}/fresh_noise_evaluation.json" \
            --bootstrap-repetitions 5000 \
            --device cuda
    }

    local pids=()
    for gpu in 0 1 2 3; do
        (run_arm "${gpu}" "${arms[${gpu}]}") > "${root}/logs/${arms[${gpu}]}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then failed=1; fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "One or more PAF arms failed; inspect ${root}/logs" >&2
        exit 1
    fi
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" "${ALGENGINE_PYTHON}" "${SELECTOR}" \
        --trial-root "${root}/trials" \
        --output "${root}/search_gate.json" \
        --bootstrap-repetitions 10000
    echo "PASS PAF fixed-budget search: ${root}"
    echo "gate: ${root}/search_gate.json"
}

case "${MODE}" in
    audit-existing)
        run_existing_audit "${1:-}"
        ;;
    cache-fresh-seed)
        [[ $# -eq 2 ]] || { echo "cache-fresh-seed needs SEED EXISTING_GATE" >&2; exit 2; }
        run_fresh_cache "$1" "$2"
        ;;
    audit-fresh)
        [[ $# -ge 1 ]] || { echo "audit-fresh needs EXISTING_GATE" >&2; exit 2; }
        run_fresh_audit "$1" "${2:-}"
        ;;
    search)
        [[ $# -ge 1 ]] || { echo "search needs FRESH_GATE" >&2; exit 2; }
        run_search "$1" "${2:-}"
        ;;
    fresh-and-search)
        [[ $# -ge 1 ]] || { echo "fresh-and-search needs EXISTING_GATE" >&2; exit 2; }
        existing_gate="$1"
        run_id="${2:-run_$(date -u +%Y%m%dT%H%M%SZ)}"
        for seed in 9 10 11; do run_fresh_cache "${seed}" "${existing_gate}"; done
        run_fresh_audit "${existing_gate}" "${run_id}_fresh"
        fresh_gate="${EXPERIMENT_ROOT}/audit/${run_id}_fresh/mechanism_gate.json"
        run_search "${fresh_gate}" "${run_id}_search"
        ;;
    all)
        run_id="${1:-run_$(date -u +%Y%m%dT%H%M%SZ)}"
        run_existing_audit "${run_id}_existing"
        existing_gate="${EXPERIMENT_ROOT}/audit/${run_id}_existing/mechanism_gate.json"
        verify_existing_gate "${existing_gate}"
        for seed in 9 10 11; do run_fresh_cache "${seed}" "${existing_gate}"; done
        run_fresh_audit "${existing_gate}" "${run_id}_fresh"
        fresh_gate="${EXPERIMENT_ROOT}/audit/${run_id}_fresh/mechanism_gate.json"
        verify_fresh_gate "${fresh_gate}"
        run_search "${fresh_gate}" "${run_id}_search"
        ;;
esac
