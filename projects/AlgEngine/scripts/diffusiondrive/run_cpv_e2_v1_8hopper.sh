#!/usr/bin/env bash
# Frozen CPV E2: existing-data common coverage only. Development is a separate mode.
set -Eeuo pipefail

MODE="${1:-}"
case "${MODE}" in
    prepare|cache-seed0|cache-seed1|cache-seed2|search|development) ;;
    *)
        echo "usage: $0 {prepare|cache-seed0|cache-seed1|cache-seed2|search|development} [SEARCH_ROOT]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8

E2_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_cpv_e2_v1"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
DATA_ROOT="${E2_ROOT}/data"
CACHE_ROOT="${E2_ROOT}/cache"
SOURCE_ROOT="${RAPG_SOURCE_WORLDENGINE_ROOT}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1"
RARE_DATA_ROOT="${SOURCE_ROOT}/rare_data"
FULL_INDEX="${SOURCE_ROOT}/metric_cache_navtrain_full"
METADATA_ROOT="${RAPG_SOURCE_WORLDENGINE_ROOT}/data/raw/openscene-v1.1/meta_datas_navformer/trainval"
BASE_FILTER="${ALGENGINE_ROOT}/configs/navsim_splits/navtrain_split/navtrain.yaml"
V3_CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py"
PREPARE="${SCRIPT_DIR}/prepare_cpv_e2_common_coverage.py"
EXTRACT="${SCRIPT_DIR}/extract_grpo_selector_context_cache.py"
SUBSET="${SCRIPT_DIR}/subset_cpv_e2_common_cache.py"
TRAIN="${SCRIPT_DIR}/train_cpv_e2_common_coverage.py"
EVALUATE="${SCRIPT_DIR}/evaluate_rapg_offline.py"
GATE="${SCRIPT_DIR}/select_cpv_e2_offline_development.py"
TORCHRUN="${ALGENGINE_ENV}/bin/torchrun"
E1_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_cpv_v1/search/search_20260830T092003Z"
PROPOSAL32="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt"
PROPOSAL32_SHA256=562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6

require_file() {
    [[ -f "$1" ]] || { echo "missing CPV E2 dependency: $1" >&2; exit 1; }
}

prepare_data() {
    [[ ! -e "${DATA_ROOT}" ]] || {
        "${ALGENGINE_PYTHON}" - "${DATA_ROOT}/manifest.json" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS" and row["method"]=="cpv_e2_existing_navtrain_common_coverage_v1"
assert row["arms"]["D0_paired_common"]["unique_tokens"]==5267
assert row["arms"]["D1_diverse_matched"]["unique_tokens"]==5267
assert row["arms"]["D2_diverse_20k"]["unique_tokens"]==20000
print("PASS existing CPV E2 data manifest",sys.argv[1])
PY
        return
    }
    "${ALGENGINE_PYTHON}" "${PREPARE}" \
        --metric-cache-index "${FULL_INDEX}" \
        --rare-pairs "${RARE_DATA_ROOT}/pairs.jsonl" \
        --rare-data-audit "${RARE_DATA_ROOT}/rare_data_audit.json" \
        --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
        --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
        --base-filter "${BASE_FILTER}" --metadata-root "${METADATA_ROOT}" \
        --output-dir "${DATA_ROOT}"
}

validate_cache() {
    local arm="$1"
    local seed="$2"
    local expected="$3"
    local path="${CACHE_ROOT}/${arm}/train_seed${seed}/cache.pt"
    "${ALGENGINE_PYTHON}" - "${path}" "${DATA_ROOT}/${arm}/tokens.jsonl" "${expected}" <<'PY'
import json,sys
from pathlib import Path
import grpo_selector_v3_cached_common as common
cache,manifest=common.load_cache(Path(sys.argv[1]),"train")
expected=[json.loads(x)["token"] for x in open(sys.argv[2]) if x.strip()]
assert len(expected)==int(sys.argv[3])
assert cache["tokens"]==sorted(expected)
print("PASS CPV E2 cache",manifest.get("arm",sys.argv[1]),len(expected))
PY
}

cache_seed() {
    local seed="$1"
    local d2_output="${CACHE_ROOT}/D2_diverse_20k/train_seed${seed}"
    prepare_data
    if [[ ! -f "${d2_output}/cache.pt" || ! -f "${d2_output}/manifest.json" ]]; then
        [[ ! -e "${d2_output}" ]] || { echo "partial D2 cache exists: ${d2_output}" >&2; exit 1; }
        mkdir -p "${d2_output}"
        export NAVSIM_METRIC_CACHE_PATH_TRAIN="${FULL_INDEX}"
        (
            cd "${ALGENGINE_ROOT}"
            "${TORCHRUN}" --nproc_per_node=8 --master_port="$((29920 + seed))" \
                "${EXTRACT}" "${V3_CONFIG}" "${RAPG_BASELINE}" \
                --expected-checkpoint-sha256 "${RAPG_BASELINE_SHA256}" \
                --nav-filter "${DATA_ROOT}/D2_diverse_20k/common_tokens.yaml" \
                --split train --noise-seed "${seed}" --expected-num-tokens 20000 \
                --output-dir "${d2_output}" --workers-per-gpu 2 --launcher pytorch
        )
    fi
    validate_cache D2_diverse_20k "${seed}" 20000

    local d1_output="${CACHE_ROOT}/D1_diverse_matched/train_seed${seed}"
    if [[ ! -f "${d1_output}/cache.pt" || ! -f "${d1_output}/manifest.json" ]]; then
        [[ ! -e "${d1_output}" ]] || { echo "partial D1 cache exists: ${d1_output}" >&2; exit 1; }
        "${ALGENGINE_PYTHON}" "${SUBSET}" \
            --source-cache "${d2_output}/cache.pt" \
            --token-records "${DATA_ROOT}/D1_diverse_matched/tokens.jsonl" \
            --nav-filter "${DATA_ROOT}/D1_diverse_matched/common_tokens.yaml" \
            --output-dir "${d1_output}" --arm D1_diverse_matched
    fi
    validate_cache D1_diverse_matched "${seed}" 5267
}

common_cache_args() {
    local arm="$1"
    printf '%s\n' \
        --common-cache "0=${CACHE_ROOT}/${arm}/train_seed0/cache.pt" \
        --common-cache "1=${CACHE_ROOT}/${arm}/train_seed1/cache.pt" \
        --common-cache "2=${CACHE_ROOT}/${arm}/train_seed2/cache.pt"
}

validate_training() {
    local root="$1" arm="$2" expected="$3"
    "${ALGENGINE_PYTHON}" - "${root}/report.json" "${root}/epoch_16_scene_selector.pt" "${arm}" "${expected}" <<'PY'
import json,sys,torch
report=json.load(open(sys.argv[1])); checkpoint=torch.load(sys.argv[2],map_location="cpu")
arm=sys.argv[3]; expected=int(sys.argv[4])
for row in (report,checkpoint):
    assert row["common_arm"]==arm
    assert row["method"]==f"cpv_e2_{arm.lower()}_source_sign_pair_regret"
    assert row["sampling_mode"]=="cpv_decision_source_sign"
    assert row["formal_contract"] is True
assert report["common_unique_tokens"]==expected
assert report["sampling"]["total_examples"]==304272
assert report["sampling"]["total_optimizer_steps"]==4800
assert set(report["sampling"]["risk_group_examples"].values())=={50712}
assert report["frozen_parameter_max_abs_delta"]==0.0
print("PASS CPV E2 training",arm,expected)
PY
}

train_arm() {
    local gpu="$1" arm="$2" expected="$3" root="$4"
    mapfile -t REAL_ARGS < <(rapg_real_cache_args)
    mapfile -t COMMON_ARGS < <(common_cache_args "${arm}")
    mkdir -p "${root}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${TRAIN}" \
        "${REAL_ARGS[@]}" "${COMMON_ARGS[@]}" \
        --common-data-manifest "${DATA_ROOT}/manifest.json" --common-arm "${arm}" \
        --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
        --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
        --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
        --proposal-checkpoint "${PROPOSAL32}" --output-dir "${root}" \
        --formal-contract >"${root}/train.log" 2>&1
    validate_training "${root}" "${arm}" "${expected}"
}

run_search() {
    prepare_data
    for seed in 0 1 2; do
        validate_cache D1_diverse_matched "${seed}" 5267
        validate_cache D2_diverse_20k "${seed}" 20000
    done
    [[ "$(sha256sum "${PROPOSAL32}" | awk '{print $1}')" == "${PROPOSAL32_SHA256}" ]] || exit 1
    local run_id="${CPV_E2_SEARCH_ID:-search_$(date -u +%Y%m%dT%H%M%SZ)}"
    local root="${E2_ROOT}/search/${run_id}"
    [[ ! -e "${root}" ]] || { echo "CPV E2 search exists: ${root}" >&2; exit 1; }
    mkdir -p "${root}"
    pids=()
    train_arm 0 D1_diverse_matched 5267 "${root}/D1_diverse_matched" & pids+=("$!")
    train_arm 1 D2_diverse_20k 20000 "${root}/D2_diverse_20k" & pids+=("$!")
    for pid in "${pids[@]}"; do wait "${pid}"; done
    printf '%s\n' "${DATA_ROOT}/manifest.json" >"${root}/data_manifest.path"
    printf '%s  %s\n' "${PROPOSAL32_SHA256}" "${PROPOSAL32}" >"${root}/proposal32.sha256"
    echo "PASS CPV E2 fixed search: ${root}"
}

evaluate_arm() {
    local gpu="$1" arm="$2" checkpoint="$3" output="$4"
    mapfile -t REAL_ARGS < <(rapg_real_cache_args)
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${EVALUATE}" \
        "${REAL_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" \
        --data-manifest "${RAPG_DATA_ROOT}/manifest.json" \
        --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl" --checkpoint "${checkpoint}" \
        --split development --output "${output}" --batch-size 64 --device cuda \
        >"${output%.json}.log" 2>&1
}

run_development() {
    local root="${2:-}"
    [[ -n "${root}" && -d "${root}" ]] || { echo "development requires an existing SEARCH_ROOT" >&2; exit 2; }
    root="$(cd "${root}" && pwd)"
    [[ ! -e "${root}/development_gate.json" ]] || {
        echo "E2 development gate exists; refusing to reread development" >&2; exit 2;
    }
    local eval_root="${root}/offline_development"
    mkdir -p "${eval_root}"
    pids=()
    evaluate_arm 0 D1 "${root}/D1_diverse_matched/epoch_16_scene_selector.pt" "${eval_root}/D1.json" & pids+=("$!")
    evaluate_arm 1 D2 "${root}/D2_diverse_20k/epoch_16_scene_selector.pt" "${eval_root}/D2.json" & pids+=("$!")
    for pid in "${pids[@]}"; do wait "${pid}"; done
    "${ALGENGINE_PYTHON}" "${GATE}" \
        --proposal-32 "${E1_ROOT}/offline_development/proposal32.json" \
        --proposal-48 "${E1_ROOT}/offline_development/proposal48.json" \
        --d0 "${E1_ROOT}/offline_development/A3_source_sign_regret.json" \
        --d1 "${eval_root}/D1.json" --d2 "${eval_root}/D2.json" \
        --data-manifest "${DATA_ROOT}/manifest.json" \
        --output "${root}/development_gate.json" \
        --decision-ledger "${RAPG_SOURCE_WORLDENGINE_ROOT}/DIFFUSIONDRIVE_SELECTOR_DECISION_LEDGER.json"
}

for path in "${PREPARE}" "${EXTRACT}" "${SUBSET}" "${TRAIN}" "${EVALUATE}" "${GATE}" \
    "${RAPG_DATA_ROOT}/manifest.json" "${RAPG_DATA_ROOT}/hard_pool.jsonl" \
    "${RARE_DATA_ROOT}/pairs.jsonl" "${RARE_DATA_ROOT}/rare_data_audit.json"; do
    require_file "${path}"
done

case "${MODE}" in
    prepare) prepare_data ;;
    cache-seed0) cache_seed 0 ;;
    cache-seed1) cache_seed 1 ;;
    cache-seed2) cache_seed 2 ;;
    search) run_search ;;
    development) run_development "$@" ;;
esac
