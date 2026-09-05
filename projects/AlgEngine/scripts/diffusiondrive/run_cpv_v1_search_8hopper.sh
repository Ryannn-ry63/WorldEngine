#!/usr/bin/env bash
# Frozen CPV E0/E1 run: one diagnostic and exactly A1/A2/A3.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 8

CPV_EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_cpv_v1"
RUN_ID="${CPV_SEARCH_ID:-search_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${CPV_EXPERIMENT_ROOT}/search/${RUN_ID}"
PCRA_V1_SEARCH="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_pcra_v1/search/search_20260829T171222Z"
PROPOSAL32="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt"
PROPOSAL48="${PCRA_V1_SEARCH}/proposal_compute48/epoch_48_scene_selector.pt"
A0="${PCRA_V1_SEARCH}/pair_regret/epoch_16_scene_selector.pt"
PROPOSAL32_SHA256=562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6
for path in "${PROPOSAL32}" "${PROPOSAL48}" "${A0}"; do
    [[ -f "${path}" ]] || { echo "missing frozen CPV input: ${path}" >&2; exit 1; }
done
[[ "$(sha256sum "${PROPOSAL32}" | awk '{print $1}')" == "${PROPOSAL32_SHA256}" ]] || {
    echo "frozen proposal32 SHA256 mismatch" >&2
    exit 1
}
[[ ! -e "${ROOT}" ]] || { echo "CPV search root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
COMMON_ARGS=("${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" --data-manifest "${RAPG_DATA_ROOT}/manifest.json" --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl")

run_e0() {
    local output="${ROOT}/E0_regime_probe"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES=0 "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/probe_cpv_regime_predictability.py" \
        "${COMMON_ARGS[@]}" --proposal-checkpoint "${PROPOSAL32}" \
        --output-dir "${output}" --device cuda --batch-size 64 \
        --probe-epochs 50 --seed 0 >"${output}/probe.log" 2>&1
}

run_e1() {
    local gpu="$1" label="$2" risk="$3"
    local output="${ROOT}/${label}"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_counterfactual_proposal_verifier.py" \
        "${COMMON_ARGS[@]}" --proposal-checkpoint "${PROPOSAL32}" \
        --output-dir "${output}" --risk-aggregation "${risk}" \
        --override-threshold 0.0 --temperature 1.0 --learning-rate 1e-4 \
        --kl-weight 1e-3 --seed 0 --epochs 16 --examples-per-cache-epoch 6339 \
        --checkpoint-epochs 16 --batch-size 64 --device cuda \
        --data-split train --formal-contract >"${output}/train.log" 2>&1
}

pids=()
run_e0 & pids+=("$!")
run_e1 1 A1_source_regret source & pids+=("$!")
run_e1 2 A2_sign_regret sign & pids+=("$!")
run_e1 3 A3_source_sign_regret source_sign & pids+=("$!")
for pid in "${pids[@]}"; do
    wait "${pid}"
done

"${ALGENGINE_PYTHON}" - "${ROOT}" "${PROPOSAL32_SHA256}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
proposal_sha = sys.argv[2]
e0 = json.loads((root / "E0_regime_probe" / "report.json").read_text())
assert e0["status"] == "PASS"
assert e0["promotable"] is False
assert e0["labels_are_model_inputs"] is False
assert e0["reward_consumed"] is False
assert e0["proposal_checkpoint_sha256"] == proposal_sha

expected = {
    "A1_source_regret": ("source", 3),
    "A2_sign_regret": ("sign", 2),
    "A3_source_sign_regret": ("source_sign", 6),
}
for label, (risk, groups) in expected.items():
    report = json.loads((root / label / "report.json").read_text())
    checkpoint = root / label / "epoch_16_scene_selector.pt"
    assert checkpoint.is_file()
    assert report["status"] == "PASS"
    assert report["training_data_split"] == "train"
    assert report["training_epochs"] == 16
    assert report["training_examples_per_cache_epoch"] == 6339
    assert report["training_batch_size"] == 64
    assert report["formal_contract"] is True
    assert report["arbiter_risk"] == risk
    assert report["arbiter_loss"] == "regret"
    assert report["use_decision_context"] is False
    assert report["override_threshold"] == 0.0
    assert report["proposal_checkpoint_sha256"] == proposal_sha
    assert report["frozen_parameter_max_abs_delta"] == 0.0
    assert report["sampling"]["total_examples"] == 304272
    assert report["sampling"]["total_optimizer_steps"] == 4800
    risk_counts = report["sampling"]["risk_group_examples"]
    assert len(risk_counts) == groups
    assert len(set(risk_counts.values())) == 1
    diagnostics = report["proposal_regret_arbitration_diagnostics"]
    assert diagnostics["risk_aggregation"] == risk
    assert diagnostics["decision_pool_mined"] is True
    assert diagnostics["reward_components_consumed"] is False
    assert diagnostics["proposal_frozen"] is True
PY

printf '%s  %s\n' "${PROPOSAL32_SHA256}" "${PROPOSAL32}" >"${ROOT}/proposal32.sha256"
printf '%s\n' "${PROPOSAL48}" >"${ROOT}/proposal48.path"
printf '%s\n' "${A0}" >"${ROOT}/A0_global_regret.path"
echo "PASS CPV E0/E1 fixed search: ${ROOT}"
