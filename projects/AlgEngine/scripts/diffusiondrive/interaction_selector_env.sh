#!/usr/bin/env bash
# Shared frozen contracts for interaction-aware selector experiments.
set +u
INTERACTION_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${INTERACTION_SCRIPT_DIR}/trajectory_set_reasoner_env.sh"
set -u

export INTERACTION_CONFIG="${ALGENGINE_ROOT}/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_interaction_v1.py"
export INTERACTION_CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/interaction_selector_v1/cache"
export INTERACTION_EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/interaction_selector_v1"
export INTERACTION_TUNING_ROOT="${REASONER_TUNING_ROOT}"

interaction_preflight() {
    reasoner_hopper_preflight 8
    local required=(
        "${INTERACTION_CONFIG}"
        "${INTERACTION_TUNING_ROOT}/train/rare_common_union.yaml"
        "${INTERACTION_TUNING_ROOT}/train/pairs.jsonl"
        "${INTERACTION_TUNING_ROOT}/train/rare_data_audit.json"
        "${INTERACTION_TUNING_ROOT}/development/rare_common_union.yaml"
        "${INTERACTION_TUNING_ROOT}/development/pairs.jsonl"
        "${INTERACTION_TUNING_ROOT}/development/rare_data_audit.json"
    )
    for path in "${required[@]}"; do
        [[ -f "${path}" ]] || { echo "Missing interaction dependency: ${path}" >&2; return 1; }
    done
    mkdir -p "${INTERACTION_CACHE_ROOT}" "${INTERACTION_EXPERIMENT_ROOT}"
}

interaction_extract_cache() {
    local split="$1" seed="$2" count="$3" filter="$4"
    local scope="${5:-tuning}"
    [[ "${scope}" == "tuning" || "${scope}" == "full" ]] || {
        echo "interaction cache scope must be tuning or full" >&2
        return 2
    }
    local output="${INTERACTION_CACHE_ROOT}/${scope}/${split}_seed${seed}"
    if [[ -f "${output}/cache.pt" && -f "${output}/manifest.json" ]]; then
        PYTHONPATH="${INTERACTION_SCRIPT_DIR}:${PYTHONPATH}" "${ALGENGINE_PYTHON}" - \
            "${output}/cache.pt" "${split}" "${seed}" "${count}" \
            "${INTERACTION_CONFIG}" "${filter}" "${REASONER_BASELINE_SHA256}" <<'PY'
import hashlib,sys
import grpo_selector_v3_cached_common as common
cache,manifest=common.load_cache(sys.argv[1],sys.argv[2])
assert cache["schema_version"]==4
assert manifest["source_kind"]=="frozen_track_interaction"
assert int(manifest["noise_seed"])==int(sys.argv[3])
assert int(manifest["num_tokens"])==int(sys.argv[4])
digest=lambda path: hashlib.sha256(open(path,"rb").read()).hexdigest()
assert manifest["config_sha256"]==digest(sys.argv[5])
assert manifest["nav_filter_sha256"]==digest(sys.argv[6])
assert manifest["checkpoint_sha256"]==sys.argv[7]
for path,expected in manifest["implementation_files"].items():
    assert digest(path)==expected, f"interaction cache implementation drifted: {path}"
assert manifest["track_contract"]["ground_truth_matching"] is False
print("REUSE verified interaction cache",sys.argv[1])
PY
        return
    fi
    if [[ -e "${output}" ]]; then
        if [[ -d "${output}" && -z "$(find "${output}" -mindepth 1 -print -quit)" ]]; then
            echo "RETRY empty interaction cache directory: ${output}"
        else
            echo "Refusing to overwrite non-empty incomplete interaction cache: ${output}" >&2
            return 1
        fi
    fi
    mkdir -p "${output}"
    (
        cd "${ALGENGINE_ROOT}"
        # The rare/common manifests were constructed against the audited full
        # navtrain metric index.  The default 47,950-token trainval cache only
        # covers 727 of these tuning tokens and must not silently redefine the
        # experiment population.
        export NAVSIM_METRIC_CACHE_PATH_TRAIN="${REASONER_METRIC_CACHE_INDEX}"
        DIFFUSIONDRIVE_INTERACTION_ARCHITECTURE=interaction_generic \
            "${ALGENGINE_TORCHRUN}" --nproc_per_node=8 \
            --master_port="$((29700 + seed))" \
            "${INTERACTION_SCRIPT_DIR}/extract_grpo_selector_context_cache.py" \
            "${INTERACTION_CONFIG}" "${REASONER_BASELINE}" \
            --expected-checkpoint-sha256 "${REASONER_BASELINE_SHA256}" \
            --nav-filter "${filter}" --split "${split}" --noise-seed "${seed}" \
            --expected-num-tokens "${count}" --output-dir "${output}" \
            --workers-per-gpu 2 --launcher pytorch
    )
}
