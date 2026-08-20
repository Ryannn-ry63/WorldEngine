#!/usr/bin/env bash
set -euo pipefail

SIMENGINE_PYTHON=${SIMENGINE_PYTHON:-/root/miniconda3/envs/simengine/bin/python}
ALGENGINE_PYTHON=${ALGENGINE_PYTHON:-/root/miniconda3/envs/algengine/bin/python}
for python_bin in "$SIMENGINE_PYTHON" "$ALGENGINE_PYTHON"; do
  [ -x "$python_bin" ] || { echo "ERROR: Python not executable: $python_bin" >&2; exit 1; }
done

CFG=$1
CKPT=$2
MODEL_NAME=$3
DATA_TYPE=$4
REACT_TYPE=$5
ASSET_NAME=${6:-$DATA_TYPE}
SCENARIO_PKL=${7:-}
SPLIT_COUNT=${8:-8}
if ! [[ "${SPLIT_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SPLIT_COUNT must be a positive integer" >&2
    exit 2
fi
if [[ -n "${SCENARIO_PKL}" && ! -f "${SCENARIO_PKL}" ]]; then
    echo "ERROR: Scenario pickle does not exist: ${SCENARIO_PKL}" >&2
    exit 1
fi

# Resume flag - set to true to skip already completed scenarios
ENABLE_RESUME=true

# Convert relative paths to absolute paths based on WORLDENGINE_ROOT
SIMENGINE_ROOT="$WORLDENGINE_ROOT/projects/SimEngine"
ALGENGINE_ROOT="$WORLDENGINE_ROOT/projects/AlgEngine"
export PYTHONPATH=$SIMENGINE_ROOT:$ALGENGINE_ROOT:${PYTHONPATH:-}

# SimEngine setting
ASSET_FOLDER_PATH="$WORLDENGINE_ROOT/data/sim_engine/assets/${ASSET_NAME}/assets"
DATAFILE_FOLDER_PATH="data/sim_engine/scenarios/original/${DATA_TYPE}"
DATA_OVERRIDES=("data_file_folder_path=${DATAFILE_FOLDER_PATH}" "data_pkl_file_name=all_scenarios.pkl")
if [[ -n "${SCENARIO_PKL}" ]]; then
    SCENARIO_PKL="$(cd "$(dirname "${SCENARIO_PKL}")" && pwd)/$(basename "${SCENARIO_PKL}")"
    DATA_OVERRIDES=("data_file_path=${SCENARIO_PKL}")
fi

# Test path (absolute, relative to WORLDENGINE_ROOT)
test_path="$WORLDENGINE_ROOT/experiments/closed_loop_exps/${MODEL_NAME}/${DATA_TYPE}_${REACT_TYPE}"

# Check if test_path already exists (skip check if resume is enabled)
if [ -d "$test_path" ] && [ "$ENABLE_RESUME" = false ]; then
    echo "ERROR: Test path already exists!"
    echo "Run the following command to remove it:"
    echo "rm -rf $test_path"
    echo "Or set ENABLE_RESUME=true to resume from where you left off."
    exit 1
fi

cleanup() {
  echo "Cleaning up processes..."
  trap - SIGINT SIGTERM EXIT

  kill 0 || true
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Main execution
echo "Starting distributed simulation with ${SPLIT_COUNT} splits..."
echo "Model: $MODEL_NAME, Type: $REACT_TYPE, Data: $DATA_TYPE, Asset: $ASSET_NAME"
echo "Scenario pickle: ${SCENARIO_PKL:-default for DATA_TYPE}"
echo "Resume mode: $ENABLE_RESUME"

cd $SIMENGINE_ROOT
"$SIMENGINE_PYTHON" worldengine/runner/run_simulation.py \
    debug_mode=True \
    debug_scene_name=null \
    "${DATA_OVERRIDES[@]}" \
    asset_folder_path=$ASSET_FOLDER_PATH \
    output_dir=$test_path/__WORKER_ID__/WE_output \
    job_name=${DATA_TYPE}_${REACT_TYPE}_${MODEL_NAME} \
    use_planner_actions=true \
    ego_policy=env_input_policy \
    ego_client=navformer_client \
    ego_controller=log_play_controller \
    ego_navigation=trajectory_navigation \
    $([ "$REACT_TYPE" = "R" ] && echo "agent_policy=idm_policy") \
    $([ "$REACT_TYPE" = "R" ] && echo "agent_navigation=idm_navigation") \
    planner_data_path=$test_path/__WORKER_ID__/plan_traj \
    planner_client_folder=$test_path/__WORKER_ID__/frames \
    with_metric_manager=true \
    with_dense_reward_manager=false \
    distributed_mode=SCENARIO_BASED \
    worker=ray_distributed \
    worker_id_prefix=split_ \
    enable_resume=$ENABLE_RESUME \
    completed_scenarios_dir=$test_path/__WORKER_ID__/completed_scenarios &

we_pid=$!
echo "WorldEngine started with PID: $we_pid with ray distributed mode!"

sleep 30

# Function to run single simulation
run_planner() {
    local split_id=$1
    local gpu_id=$1

    # Set GPU environment
    export CUDA_VISIBLE_DEVICES=${gpu_id}

    local split_suffix="split_${split_id}"
    local test_path_worker="$test_path/${split_suffix}"

    mkdir -p $test_path_worker/WE_output
    mkdir -p $test_path_worker/plan_traj
    mkdir -p $test_path_worker/frames
    mkdir -p $test_path_worker/merged_ann_files

    rm -rf $test_path_worker/merged_ann_files/*.pkl
    rm -rf $test_path_worker/frames/*.pkl
    rm -rf $test_path_worker/plan_traj/*.npy

    # Clean up previous simulation completed flag if exists
    rm -f $test_path_worker/WE_output/simulation_completed.flag

    # Start AlgEngine client
    cd $ALGENGINE_ROOT
    "$ALGENGINE_PYTHON" closed_loop/sim_test.py \
        $CFG \
        $CKPT \
        --log-dir $test_path_worker \
        --cfg-options sim.monitored_folder="$test_path_worker/frames" \
        sim.plan_save_path="$test_path_worker/plan_traj" \
        sim.merged_ann_save_dir="$test_path_worker/merged_ann_files" \
        sim.clean_temp_files=True \
        sim.clean_record_data=False \
        data_root="$test_path_worker/WE_output/openscene_format/" &

    local alg_pid=$!
    echo "AlgEngine started with PID: $alg_pid for split ${split_id}"

    wait $alg_pid
    local alg_exit_code=$?

    # Check exit codes
    if [ $alg_exit_code -eq 0 ]; then
        echo "Split ${split_id} completed successfully on GPU $gpu_id"
    else
        echo "Split ${split_id} failed - AlgEngine exit code: $alg_exit_code"
        return 1
    fi
}

# Run simulations in parallel
planner_pids=()
for ((i = 0; i < SPLIT_COUNT; i++)); do
    run_planner "$i" &
    planner_pids+=("$!")
done

# Wait for all simulations to complete
planner_failed=0
for planner_pid in "${planner_pids[@]}"; do
    if ! wait "$planner_pid"; then
        planner_failed=1
    fi
done
if ! wait "$we_pid"; then
    echo "WorldEngine process failed" >&2
    planner_failed=1
fi
if [[ "$planner_failed" -ne 0 ]]; then
    echo "One or more simulation processes failed" >&2
    exit 1
fi

echo "All simulation splits completed successfully."

trap - SIGINT SIGTERM EXIT

# Merge results
cd $SIMENGINE_ROOT
"$SIMENGINE_PYTHON" scripts/merge_simulation_results.py \
    --test_path "$test_path" \
    --react_type "$REACT_TYPE" \
    --num-splits "$SPLIT_COUNT"
