#!/usr/bin/env bash
# Immutable LC-PGRPO mechanism audit, log-CV, controls, development, and certification.
set -Eeuo pipefail

MODE="${1:-}"
shift || true
case "${MODE}" in
    mechanism|smoke|cv|control|development|certification|all) ;;
    *)
        echo "Usage:" >&2
        echo "  $0 mechanism [RUN_ID]" >&2
        echo "  $0 smoke MECHANISM_GATE [RUN_ID]" >&2
        echo "  $0 cv MECHANISM_GATE [RUN_ID]" >&2
        echo "  $0 control CV_GATE [RUN_ID]" >&2
        echo "  $0 development CONTROL_GATE [RUN_ID]" >&2
        echo "  $0 certification DEVELOPMENT_GATE [RUN_ID]" >&2
        echo "  $0 all [RUN_ID]" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/trajectory_set_reasoner_env.sh"
reasoner_setup_assets

SOURCE_WORLDENGINE="${REASONER_SOURCE_WORLDENGINE_ROOT}"
V3_CHECKPOINT="${SOURCE_WORLDENGINE}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/sweep/trials/t1_lr1e-4_kl1e-3/epoch_64_scene_selector.pt"
V3_SHA256=d19c9d825ab452c96891d5b6abf0f35186460e9538cf37b329fea2c1bd86d133
TRAIN_CACHE_ROOT="${SOURCE_WORLDENGINE}/experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/cache/tuning"
FRESH_CACHE_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_paf_grpo_v1/cache"
PAIR_SPLIT_ROOT="${SOURCE_WORLDENGINE}/experiments/grpo_sources/diffusiondrive_selector_rare_original_v1/tuning_split"
TRAIN_PAIR_ROOT="${PAIR_SPLIT_ROOT}/train"
DEVELOPMENT_PAIR_ROOT="${PAIR_SPLIT_ROOT}/development"
CERTIFICATION_PAIR_ROOT="${PAIR_SPLIT_ROOT}/certification"

HISTORICAL_DIRECT_CHECKPOINT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/diffusion_set_selector_v1/search/search_20260902T024312Z/trials/repeat_grpo/epoch_8_scene_selector.pt"
REPRODUCED_DIRECT_CHECKPOINT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_paf_grpo_v1/search/run_20260902T034547Z_search/trials/direct_grpo/epoch_8_scene_selector.pt"
DIRECT_FRESH_EVALUATION="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_paf_grpo_v1/search/run_20260902T034547Z_search/trials/direct_grpo/fresh_noise_evaluation.json"
DIRECT_DEVELOPMENT_EVALUATION="${WORLDENGINE_ROOT}/experiments/diffusiondrive/diffusion_set_selector_v1/search/search_20260902T024312Z/trials/repeat_grpo/development_evaluation.json"

EXPERIMENT_ROOT="${WORLDENGINE_ROOT}/experiments/diffusiondrive/selector_lcpgrpo_v1"
AUDITOR="${SCRIPT_DIR}/audit_lineage_consistent_mechanism.py"
TRAINER="${SCRIPT_DIR}/train_lineage_consistent_proximal_grpo.py"
EVALUATOR="${SCRIPT_DIR}/evaluate_lineage_consistent_proximal_grpo.py"
CV_SELECTOR="${SCRIPT_DIR}/select_lcpgrpo_log_cv.py"
CONTROL_SELECTOR="${SCRIPT_DIR}/select_lcpgrpo_lineage_control.py"
DEVELOPMENT_SELECTOR="${SCRIPT_DIR}/select_lcpgrpo_development.py"
CERTIFICATION_SUMMARIZER="${SCRIPT_DIR}/summarize_lcpgrpo_certification.py"

validate_id() {
    [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || {
        echo "Invalid run id: $1" >&2
        exit 2
    }
}

require_path() {
    [[ -e "$1" ]] || {
        echo "Missing LC-PGRPO dependency: $1" >&2
        exit 1
    }
}

for path in \
    "${ALGENGINE_PYTHON}" "${V3_CHECKPOINT}" \
    "${TRAIN_PAIR_ROOT}/pairs.jsonl" "${TRAIN_PAIR_ROOT}/rare_data_audit.json" \
    "${DEVELOPMENT_PAIR_ROOT}/pairs.jsonl" "${DEVELOPMENT_PAIR_ROOT}/rare_data_audit.json" \
    "${CERTIFICATION_PAIR_ROOT}/pairs.jsonl" "${CERTIFICATION_PAIR_ROOT}/rare_data_audit.json" \
    "${HISTORICAL_DIRECT_CHECKPOINT}" "${REPRODUCED_DIRECT_CHECKPOINT}" \
    "${DIRECT_FRESH_EVALUATION}" "${DIRECT_DEVELOPMENT_EVALUATION}" \
    "${AUDITOR}" "${TRAINER}" "${EVALUATOR}" "${CV_SELECTOR}" \
    "${CONTROL_SELECTOR}" "${DEVELOPMENT_SELECTOR}" "${CERTIFICATION_SUMMARIZER}"
do
    require_path "${path}"
done
for seed in 0 1 2 3 4 5 6 7 8; do
    case "${seed}" in
        0|1|2) split=train ;;
        3|4|5) split=development ;;
        6|7|8) split=certification ;;
    esac
    require_path "${TRAIN_CACHE_ROOT}/${split}_seed${seed}/cache.pt"
done
for seed in 9 10 11; do
    require_path "${FRESH_CACHE_ROOT}/train_seed${seed}/cache.pt"
done
[[ "$(sha256sum "${V3_CHECKPOINT}" | awk '{print $1}')" == "${V3_SHA256}" ]] || {
    echo "V3 anchor SHA256 mismatch" >&2
    exit 1
}
mkdir -p "${EXPERIMENT_ROOT}"

cached_hopper_preflight() {
    "${ALGENGINE_PYTHON}" - <<'PY'
import json,torch
names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if not names:
    raise RuntimeError("LC-PGRPO cached training requires at least one visible GPU")
if any(not any(family in name.upper() for family in ("H100","H200")) for name in names):
    raise RuntimeError(f"formal cached training requires Hopper H100/H200, got {names}")
if any(torch.cuda.get_device_capability(i)!=(9,0) for i in range(len(names))):
    raise RuntimeError("formal cached training requires sm_90")
if torch.version.cuda!="11.8" or not str(torch.__version__).startswith("2.0.1+cu118"):
    raise RuntimeError(f"environment drift: torch={torch.__version__}, runtime={torch.version.cuda}")
print(json.dumps({"status":"PASS","hardware_contract":"hopper_cached_at_least_one","devices":names,"torch":torch.__version__},sort_keys=True))
PY
}

visible_gpu_count() {
    "${ALGENGINE_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
}

verify_mechanism_gate() {
    "${ALGENGINE_PYTHON}" - "$1" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["schema_version"]==2
assert row["status"]=="PASS"
assert row["method"]=="lineage_consistent_mechanism_audit_v1"
assert row["decision"]=="AUTHORIZE_LCPGRPO_LOG_CV"
assert row["noise_seeds"]==[9,10,11]
assert row["prediction_metric_contract"]["gate"]["metric"]=="predicted_top1_is_in_heldout_top4_fraction"
assert row["scientific_contract"]["development_consumed_for_new_method_selection"] is False
assert row["scientific_contract"]["certification_consumed"] is False
PY
}

gate_decision() {
    "${ALGENGINE_PYTHON}" - "$1" <<'PY'
import json,sys
print(json.load(open(sys.argv[1])).get("decision",""))
PY
}

winner_fields() {
    "${ALGENGINE_PYTHON}" - "$1" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
winner=row.get("winner")
if not winner:
    raise RuntimeError("gate has no authorized winner")
print(winner["arm"])
print("" if winner.get("target_kl") is None else winner["target_kl"])
print(winner["epoch"])
files=winner.get("files",[])
if files:
    evaluation=json.load(open(files[0]["evaluation"]))
    print(evaluation["mechanism_gate"])
else:
    print("")
PY
}

mechanism_stage() {
    local run_id="$1"
    validate_id "${run_id}"
    local root="${EXPERIMENT_ROOT}/runs/${run_id}/mechanism"
    [[ ! -e "${root}" ]] || {
        echo "Refusing to reuse mechanism root: ${root}" >&2
        exit 1
    }
    mkdir -p "${root}"
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${AUDITOR}" \
        --cache "${FRESH_CACHE_ROOT}/train_seed9/cache.pt" \
        --cache "${FRESH_CACHE_ROOT}/train_seed10/cache.pt" \
        --cache "${FRESH_CACHE_ROOT}/train_seed11/cache.pt" \
        --pair-manifest "${TRAIN_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${TRAIN_PAIR_ROOT}/rare_data_audit.json" \
        --historical-direct-checkpoint "${HISTORICAL_DIRECT_CHECKPOINT}" \
        --reproduced-direct-checkpoint "${REPRODUCED_DIRECT_CHECKPOINT}" \
        --direct-fresh-evaluation "${DIRECT_FRESH_EVALUATION}" \
        --direct-development-evaluation "${DIRECT_DEVELOPMENT_EVALUATION}" \
        --output "${root}/mechanism_gate.json" \
        --batch-size 128 | tee "${root}/mechanism.log"
    echo "PASS LC-PGRPO mechanism stage: ${root}"
}

train_trial() {
    local gpu="$1"
    local fold="$2"
    local arm="$3"
    local target_kl="$4"
    local lineage_control="$5"
    local trial="$6"
    local mechanism_gate="$7"
    local target_args=()
    if [[ -n "${target_kl}" ]]; then
        target_args=(--target-kl "${target_kl}")
    fi
    mkdir -p "${trial}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --train-cache "${TRAIN_CACHE_ROOT}/train_seed0/cache.pt" \
        --train-cache "${TRAIN_CACHE_ROOT}/train_seed1/cache.pt" \
        --train-cache "${TRAIN_CACHE_ROOT}/train_seed2/cache.pt" \
        --pair-manifest "${TRAIN_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${TRAIN_PAIR_ROOT}/rare_data_audit.json" \
        --v3-checkpoint "${V3_CHECKPOINT}" \
        --v3-checkpoint-sha256 "${V3_SHA256}" \
        --mechanism-gate "${mechanism_gate}" \
        --output-dir "${trial}" \
        --arm "${arm}" \
        --heldout-fold "${fold}" \
        "${target_args[@]}" \
        --lineage-control "${lineage_control}" \
        --temperature 1 \
        --learning-rate 3e-5 \
        --kl-weight 1e-3 \
        --retention-headroom 0.005 \
        --reward-epsilon 1e-6 \
        --epochs 8 \
        --examples-per-epoch 6339 \
        --batch-size 64 \
        --checkpoint-epochs 1,2,4,8 \
        --require-full-coverage \
        --seed 0 \
        --device cuda
}

evaluate_cv_checkpoint() {
    local gpu="$1"
    local fold="$2"
    local epoch="$3"
    local trial="$4"
    local mechanism_gate="$5"
    local output_name="$6"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${EVALUATOR}" \
        --checkpoint "${trial}/epoch_${epoch}_scene_selector.pt" \
        --cache "${FRESH_CACHE_ROOT}/train_seed9/cache.pt" \
        --cache "${FRESH_CACHE_ROOT}/train_seed10/cache.pt" \
        --cache "${FRESH_CACHE_ROOT}/train_seed11/cache.pt" \
        --pair-manifest "${TRAIN_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${TRAIN_PAIR_ROOT}/rare_data_audit.json" \
        --mechanism-gate "${mechanism_gate}" \
        --split cv \
        --heldout-fold "${fold}" \
        --output "${trial}/${output_name}" \
        --bootstrap-repetitions 5000 \
        --device cuda
}

smoke_stage() {
    local mechanism_gate="$1"
    local run_id="$2"
    verify_mechanism_gate "${mechanism_gate}"
    validate_id "${run_id}"
    local root="${EXPERIMENT_ROOT}/smoke/${run_id}"
    [[ ! -e "${root}" ]] || {
        echo "Refusing to reuse smoke root: ${root}" >&2
        exit 1
    }
    mkdir -p "${root}"
    local trial="${root}/a3_lineage_proximal_d003_fold0"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${TRAINER}" \
        --train-cache "${TRAIN_CACHE_ROOT}/train_seed0/cache.pt" \
        --train-cache "${TRAIN_CACHE_ROOT}/train_seed1/cache.pt" \
        --train-cache "${TRAIN_CACHE_ROOT}/train_seed2/cache.pt" \
        --pair-manifest "${TRAIN_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${TRAIN_PAIR_ROOT}/rare_data_audit.json" \
        --v3-checkpoint "${V3_CHECKPOINT}" \
        --v3-checkpoint-sha256 "${V3_SHA256}" \
        --mechanism-gate "${mechanism_gate}" \
        --output-dir "${trial}" \
        --arm lineage_proximal \
        --heldout-fold 0 \
        --target-kl 0.03 \
        --lineage-control real \
        --temperature 1 \
        --learning-rate 3e-5 \
        --kl-weight 1e-3 \
        --retention-headroom 0.005 \
        --reward-epsilon 1e-6 \
        --epochs 1 \
        --examples-per-epoch 64 \
        --batch-size 64 \
        --checkpoint-epochs 1 \
        --seed 0 \
        --device cuda | tee "${root}/train.log"
    evaluate_cv_checkpoint \
        0 0 1 "${trial}" "${mechanism_gate}" smoke_evaluation.json \
        | tee "${root}/evaluation.log"
    echo "PASS LC-PGRPO one-step contract smoke: ${root}"
    echo "Smoke validates plumbing only and cannot select a method."
}

cv_stage() {
    local mechanism_gate="$1"
    local run_id="$2"
    verify_mechanism_gate "${mechanism_gate}"
    validate_id "${run_id}"
    local root="${EXPERIMENT_ROOT}/runs/${run_id}/cv"
    [[ ! -e "${root}" ]] || {
        echo "Refusing to reuse CV root: ${root}" >&2
        exit 1
    }
    mkdir -p "${root}"
    local specs=(
        "direct||a0_direct"
        "lineage_direct||a1_lineage_direct"
        "per_draw_proximal|0.01|a2_per_draw_proximal_d001"
        "per_draw_proximal|0.03|a2_per_draw_proximal_d003"
        "per_draw_proximal|0.10|a2_per_draw_proximal_d010"
        "lineage_proximal|0.01|a3_lineage_proximal_d001"
        "lineage_proximal|0.03|a3_lineage_proximal_d003"
        "lineage_proximal|0.10|a3_lineage_proximal_d010"
    )
    local gpu_count num_specs num_tasks lanes
    gpu_count="$(visible_gpu_count)"
    num_specs="${#specs[@]}"
    num_tasks=$((5 * num_specs))
    lanes="${gpu_count}"
    if (( lanes > 8 )); then lanes=8; fi
    if (( lanes > num_tasks )); then lanes="${num_tasks}"; fi

    run_configuration() {
        local gpu="$1"
        local fold="$2"
        local spec="$3"
        local arm target label trial epoch
        IFS='|' read -r arm target label <<< "${spec}"
        trial="${root}/fold_${fold}/${label}"
        train_trial "${gpu}" "${fold}" "${arm}" "${target}" real "${trial}" "${mechanism_gate}"
        for epoch in 1 2 4 8; do
            evaluate_cv_checkpoint \
                "${gpu}" "${fold}" "${epoch}" "${trial}" "${mechanism_gate}" \
                "epoch_${epoch}_evaluation.json"
        done
    }

    run_lane() {
        local gpu="$1"
        local task fold spec_index
        for ((task=gpu; task<num_tasks; task+=lanes)); do
            fold=$((task / num_specs))
            spec_index=$((task % num_specs))
            run_configuration "${gpu}" "${fold}" "${specs[spec_index]}"
        done
    }

    local pids=()
    local gpu
    for ((gpu=0; gpu<lanes; gpu++)); do
        (run_lane "${gpu}") > "${root}/lane_${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then failed=1; fi
    done
    if (( failed != 0 )); then
        echo "One or more LC-PGRPO CV lanes failed; inspect ${root}/lane_*.log" >&2
        exit 1
    fi
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${CV_SELECTOR}" \
        --cv-root "${root}" \
        --output "${root}/cv_gate.json" \
        --bootstrap-repetitions 10000 | tee "${root}/cv_gate.log"
    echo "PASS LC-PGRPO log-CV stage: ${root}"
}

control_stage() {
    local cv_gate="$1"
    local run_id="$2"
    local decision
    decision="$(gate_decision "${cv_gate}")"
    case "${decision}" in
        AUTHORIZE_A1_LINEAGE_DIRECT_NEGATIVE_CONTROL|AUTHORIZE_A3_LINEAGE_PROXIMAL_NEGATIVE_CONTROL) ;;
        *)
            echo "CV decision does not authorize lineage control: ${decision}" >&2
            exit 3
            ;;
    esac
    local fields=()
    mapfile -t fields < <(winner_fields "${cv_gate}")
    local arm="${fields[0]}"
    local target="${fields[1]}"
    local epoch="${fields[2]}"
    local mechanism_gate="${fields[3]}"
    verify_mechanism_gate "${mechanism_gate}"
    validate_id "${run_id}"
    local root="${EXPERIMENT_ROOT}/runs/${run_id}/control"
    [[ ! -e "${root}" ]] || {
        echo "Refusing to reuse control root: ${root}" >&2
        exit 1
    }
    mkdir -p "${root}"
    local gpu_count lanes
    gpu_count="$(visible_gpu_count)"
    lanes="${gpu_count}"
    if (( lanes > 5 )); then lanes=5; fi

    run_control_lane() {
        local gpu="$1"
        local fold trial
        for ((fold=gpu; fold<5; fold+=lanes)); do
            trial="${root}/fold_${fold}/shuffled_lineage"
            train_trial \
                "${gpu}" "${fold}" "${arm}" "${target}" independent_shuffle \
                "${trial}" "${mechanism_gate}"
            evaluate_cv_checkpoint \
                "${gpu}" "${fold}" "${epoch}" "${trial}" "${mechanism_gate}" \
                control_evaluation.json
        done
    }
    local pids=()
    local gpu
    for ((gpu=0; gpu<lanes; gpu++)); do
        (run_control_lane "${gpu}") > "${root}/lane_${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    local failed=0 pid
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then failed=1; fi
    done
    if (( failed != 0 )); then
        echo "One or more lineage-control lanes failed; inspect ${root}/lane_*.log" >&2
        exit 1
    fi
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${CONTROL_SELECTOR}" \
        --cv-gate "${cv_gate}" \
        --control-root "${root}" \
        --output "${root}/control_gate.json" \
        --bootstrap-repetitions 10000 | tee "${root}/control_gate.log"
    echo "PASS LC-PGRPO lineage-control stage: ${root}"
}

development_stage() {
    local control_gate="$1"
    local run_id="$2"
    [[ "$(gate_decision "${control_gate}")" == "AUTHORIZE_SINGLE_WINNER_DEVELOPMENT" ]] || {
        echo "Control gate did not authorize development" >&2
        exit 3
    }
    local fields=()
    mapfile -t fields < <(winner_fields "${control_gate}")
    local arm="${fields[0]}"
    local target="${fields[1]}"
    local epoch="${fields[2]}"
    # The control gate winner keeps the CV files, so winner_fields resolves the
    # original mechanism gate from the first real-lineage evaluation.
    local mechanism_gate="${fields[3]}"
    verify_mechanism_gate "${mechanism_gate}"
    validate_id "${run_id}"
    local root="${EXPERIMENT_ROOT}/runs/${run_id}/development"
    [[ ! -e "${root}" ]] || {
        echo "Refusing to reuse development root: ${root}" >&2
        exit 1
    }
    mkdir -p "${root}"
    local trial="${root}/winner"
    train_trial 0 -1 "${arm}" "${target}" real "${trial}" "${mechanism_gate}"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${EVALUATOR}" \
        --checkpoint "${trial}/epoch_${epoch}_scene_selector.pt" \
        --cache "${TRAIN_CACHE_ROOT}/development_seed3/cache.pt" \
        --cache "${TRAIN_CACHE_ROOT}/development_seed4/cache.pt" \
        --cache "${TRAIN_CACHE_ROOT}/development_seed5/cache.pt" \
        --pair-manifest "${DEVELOPMENT_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${DEVELOPMENT_PAIR_ROOT}/rare_data_audit.json" \
        --mechanism-gate "${mechanism_gate}" \
        --split development \
        --heldout-fold -1 \
        --output "${trial}/development_evaluation.json" \
        --bootstrap-repetitions 10000 \
        --device cuda | tee "${root}/development.log"
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${DEVELOPMENT_SELECTOR}" \
        --control-gate "${control_gate}" \
        --evaluation "${trial}/development_evaluation.json" \
        --output "${root}/development_gate.json" | tee "${root}/development_gate.log"
    echo "PASS LC-PGRPO single-winner development stage: ${root}"
}

certification_stage() {
    local development_gate="$1"
    local run_id="$2"
    [[ "$(gate_decision "${development_gate}")" == "AUTHORIZE_BLIND_CERTIFICATION" ]] || {
        echo "Development gate did not authorize certification" >&2
        exit 3
    }
    validate_id "${run_id}"
    local root="${EXPERIMENT_ROOT}/runs/${run_id}/certification"
    [[ ! -e "${root}" ]] || {
        echo "Refusing to reuse certification root: ${root}" >&2
        exit 1
    }
    mkdir -p "${root}"
    local certification_fields=()
    mapfile -t certification_fields < <("${ALGENGINE_PYTHON}" - "${development_gate}" <<'PY'
import json,sys
gate=json.load(open(sys.argv[1]))
evaluation=json.load(open(gate["evaluation"]))
print(evaluation["checkpoint"])
print(evaluation["mechanism_gate"])
PY
    )
    local checkpoint="${certification_fields[0]}"
    local mechanism_gate="${certification_fields[1]}"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${EVALUATOR}" \
        --checkpoint "${checkpoint}" \
        --cache "${TRAIN_CACHE_ROOT}/certification_seed6/cache.pt" \
        --cache "${TRAIN_CACHE_ROOT}/certification_seed7/cache.pt" \
        --cache "${TRAIN_CACHE_ROOT}/certification_seed8/cache.pt" \
        --pair-manifest "${CERTIFICATION_PAIR_ROOT}/pairs.jsonl" \
        --rare-data-audit "${CERTIFICATION_PAIR_ROOT}/rare_data_audit.json" \
        --mechanism-gate "${mechanism_gate}" \
        --split certification \
        --heldout-fold -1 \
        --output "${root}/certification_evaluation.json" \
        --bootstrap-repetitions 10000 \
        --device cuda | tee "${root}/certification.log"
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${ALGENGINE_PYTHON}" "${CERTIFICATION_SUMMARIZER}" \
        --development-gate "${development_gate}" \
        --evaluation "${root}/certification_evaluation.json" \
        --output "${root}/certification_summary.json" | tee "${root}/certification_summary.log"
    echo "PASS LC-PGRPO blind certification stage: ${root}"
}

if [[ "${MODE}" != "mechanism" ]]; then
    cached_hopper_preflight
fi
case "${MODE}" in
    mechanism)
        run_id="${1:-mechanism_$(date -u +%Y%m%dT%H%M%SZ)}"
        mechanism_stage "${run_id}"
        ;;
    smoke)
        [[ $# -ge 1 ]] || { echo "smoke needs MECHANISM_GATE" >&2; exit 2; }
        smoke_stage "$1" "${2:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
        ;;
    cv)
        [[ $# -ge 1 ]] || { echo "cv needs MECHANISM_GATE" >&2; exit 2; }
        cv_stage "$1" "${2:-cv_$(date -u +%Y%m%dT%H%M%SZ)}"
        ;;
    control)
        [[ $# -ge 1 ]] || { echo "control needs CV_GATE" >&2; exit 2; }
        control_stage "$1" "${2:-control_$(date -u +%Y%m%dT%H%M%SZ)}"
        ;;
    development)
        [[ $# -ge 1 ]] || { echo "development needs CONTROL_GATE" >&2; exit 2; }
        development_stage "$1" "${2:-development_$(date -u +%Y%m%dT%H%M%SZ)}"
        ;;
    certification)
        [[ $# -ge 1 ]] || { echo "certification needs DEVELOPMENT_GATE" >&2; exit 2; }
        certification_stage "$1" "${2:-certification_$(date -u +%Y%m%dT%H%M%SZ)}"
        ;;
    all)
        run_id="${1:-run_$(date -u +%Y%m%dT%H%M%SZ)}"
        validate_id "${run_id}"
        mechanism_stage "${run_id}"
        mechanism_gate="${EXPERIMENT_ROOT}/runs/${run_id}/mechanism/mechanism_gate.json"
        verify_mechanism_gate "${mechanism_gate}"
        cv_stage "${mechanism_gate}" "${run_id}"
        cv_gate="${EXPERIMENT_ROOT}/runs/${run_id}/cv/cv_gate.json"
        case "$(gate_decision "${cv_gate}")" in
            AUTHORIZE_A1_LINEAGE_DIRECT_NEGATIVE_CONTROL|AUTHORIZE_A3_LINEAGE_PROXIMAL_NEGATIVE_CONTROL)
                control_stage "${cv_gate}" "${run_id}"
                ;;
            *)
                echo "LC-PGRPO stopped after CV: $(gate_decision "${cv_gate}")"
                exit 0
                ;;
        esac
        control_gate="${EXPERIMENT_ROOT}/runs/${run_id}/control/control_gate.json"
        if [[ "$(gate_decision "${control_gate}")" != "AUTHORIZE_SINGLE_WINNER_DEVELOPMENT" ]]; then
            echo "LC-PGRPO stopped after lineage control: $(gate_decision "${control_gate}")"
            exit 0
        fi
        development_stage "${control_gate}" "${run_id}"
        development_gate="${EXPERIMENT_ROOT}/runs/${run_id}/development/development_gate.json"
        if [[ "$(gate_decision "${development_gate}")" != "AUTHORIZE_BLIND_CERTIFICATION" ]]; then
            echo "LC-PGRPO stopped after development: $(gate_decision "${development_gate}")"
            exit 0
        fi
        certification_stage "${development_gate}" "${run_id}"
        ;;
esac
