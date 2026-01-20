#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH}"

# Unzip and activate virtual environment
echo "Unzipping virtual environment..."
unzip -o -q venv.zip

echo "Activating virtual environment..."
source venv/bin/activate

echo "Setting up virtual environment paths explicitly..."
export PATH="${PWD}/venv/bin:${PATH}"
export VIRTUAL_ENV="${PWD}/venv"

echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Print job configuration
echo "=========================================="
echo "Running DOFM Full Conditioning Evaluation"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "Checkpoint: ${CHECKPOINT_FILENAME}"
echo "Config: ${CONFIG_FILENAME}"
echo "Graph Mode: ${GRAPH_MODE}"
echo "=========================================="

# Build extra args based on GRAPH_MODE
EXTRA_ARGS=""
if [ "${GRAPH_MODE}" = "all_unknown" ]; then
    EXTRA_ARGS="--all_unknown"
elif [ "${GRAPH_MODE}" = "t_to_y_only" ]; then
    EXTRA_ARGS="--t_to_y_only"
elif [ "${GRAPH_MODE}" = "x_to_t_only" ]; then
    EXTRA_ARGS="--x_to_t_only"
elif [ "${GRAPH_MODE}" = "x_to_y_only" ]; then
    EXTRA_ARGS="--x_to_y_only"
elif [ "${GRAPH_MODE}" = "full_graph" ]; then
    # No extra args needed - default is full graph knowledge
    EXTRA_ARGS=""
fi

# Run the evaluation with checkpoint and config paths
echo "Starting evaluation with extra args: ${EXTRA_ARGS}"
python run_baselines/dofm_full_conditioning.py \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --exp_name "${EXP_NAME}" \
    ${EXTRA_ARGS}

echo "Evaluation complete!"
echo "Results saved to: results/${EXP_NAME}/"
