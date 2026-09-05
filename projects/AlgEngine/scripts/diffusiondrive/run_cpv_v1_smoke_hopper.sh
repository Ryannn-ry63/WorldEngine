#!/usr/bin/env bash
# Single-visible-GPU smoke for CPV E0 and every E1 risk contract.
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES="${CPV_SMOKE_GPU:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/rapg_env.sh"
rapg_preflight 1 smoke

CPV_EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_cpv_v1"
RUN_ID="${CPV_SMOKE_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="${CPV_EXPERIMENT_ROOT}/smoke/${RUN_ID}"
PROPOSAL32="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_ivps_v1/search/search_20260829T150702Z/proposal_compute32/epoch_32_scene_selector.pt"
PROPOSAL32_SHA256=562b01f30a9999f44265e224b9b99a28507677ebf3530b03c6cc9c167380b5b6
[[ -f "${PROPOSAL32}" ]] || { echo "missing frozen proposal32: ${PROPOSAL32}" >&2; exit 1; }
[[ "$(sha256sum "${PROPOSAL32}" | awk '{print $1}')" == "${PROPOSAL32_SHA256}" ]] || exit 1
[[ ! -e "${ROOT}" ]] || { echo "CPV smoke root already exists: ${ROOT}" >&2; exit 1; }
mkdir -p "${ROOT}"
mapfile -t CACHE_ARGS < <(rapg_real_cache_args)
COMMON_ARGS=("${CACHE_ARGS[@]}" --synthetic-cache "${RAPG_DATA_ROOT}/synthetic_cache.pt" --data-manifest "${RAPG_DATA_ROOT}/manifest.json" --hard-pool "${RAPG_DATA_ROOT}/hard_pool.jsonl")

mkdir -p "${ROOT}/E0_regime_probe"
"${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/probe_cpv_regime_predictability.py" \
    "${COMMON_ARGS[@]}" --proposal-checkpoint "${PROPOSAL32}" \
    --output-dir "${ROOT}/E0_regime_probe" --device cuda --batch-size 64 \
    --probe-epochs 2 --seed 0 --smoke-limit-pairs 32 \
    >"${ROOT}/E0_regime_probe/probe.log" 2>&1

run_e1() {
    local label="$1" risk="$2"
    local output="${ROOT}/${label}"
    mkdir -p "${output}"
    "${ALGENGINE_PYTHON}" "${SCRIPT_DIR}/train_counterfactual_proposal_verifier.py" \
        "${COMMON_ARGS[@]}" --proposal-checkpoint "${PROPOSAL32}" \
        --output-dir "${output}" --risk-aggregation "${risk}" \
        --override-threshold 0.0 --temperature 1.0 --learning-rate 1e-4 \
        --kl-weight 1e-3 --seed 0 --epochs 1 --examples-per-cache-epoch 60 \
        --checkpoint-epochs 1 --batch-size 64 --device cuda --data-split train \
        --smoke-limit-hard-pool 256 >"${output}/train.log" 2>&1
}

run_e1 A1_source_regret source
run_e1 A2_sign_regret sign
run_e1 A3_source_sign_regret source_sign

"${ALGENGINE_PYTHON}" - "${ROOT}" "${PROPOSAL32_SHA256}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
proposal_sha = sys.argv[2]
e0 = json.loads((root / "E0_regime_probe" / "report.json").read_text())
assert e0["status"] == "PASS" and e0["promotable"] is False
assert e0["labels_are_model_inputs"] is False and e0["reward_consumed"] is False
assert e0["proposal_checkpoint_sha256"] == proposal_sha
expected = {
    "A1_source_regret": ("source", 3),
    "A2_sign_regret": ("sign", 2),
    "A3_source_sign_regret": ("source_sign", 6),
}
for label, (risk, group_count) in expected.items():
    report = json.loads((root / label / "report.json").read_text())
    assert (root / label / "epoch_1_scene_selector.pt").is_file()
    assert report["status"] == "PASS" and report["arbiter_risk"] == risk
    assert report["training_data_split"] == "train"
    assert report["frozen_parameter_max_abs_delta"] == 0.0
    assert report["proposal_checkpoint_sha256"] == proposal_sha
    counts = report["sampling"]["risk_group_examples"]
    assert len(counts) == group_count and len(set(counts.values())) == 1
    assert report["sampling"]["total_examples"] == 180
    assert report["sampling"]["total_optimizer_steps"] == 3
    diagnostics = report["proposal_regret_arbitration_diagnostics"]
    assert set(diagnostics["mean_group_losses"]) == set(counts)
    assert all(value is not None for value in diagnostics["mean_group_losses"].values())
    assert all(value > 0 for value in diagnostics["group_active_examples"].values())
    assert all(value > 0.0 for value in diagnostics["group_regret_mass"].values())
PY

echo "PASS CPV E0/E1 smoke: ${ROOT}"
