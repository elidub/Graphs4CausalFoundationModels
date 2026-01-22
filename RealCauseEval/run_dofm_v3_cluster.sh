W#!/bin/bash
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
echo "Running DOFM v3 Baseline Evaluation"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "Graph Mode: ${GRAPH_MODE}"
echo "Target Encoding: ${TARGET_ENCODING}"
echo "=========================================="

# Build extra args based on settings
EXTRA_ARGS=""
if [ "${GRAPH_MODE}" = "full_graph" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --full_graph"
elif [ "${GRAPH_MODE}" = "all_unknown" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --all_unknown"
fi

if [ "${TARGET_ENCODING}" = "false" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --no_target_encoding"
fi

# Run the evaluation
echo "Starting evaluation with extra args: ${EXTRA_ARGS}"
python run_baselines/dofm_v3.py \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --exp_name "${EXP_NAME}" \
    ${EXTRA_ARGS}

echo "Evaluation complete!"
echo "Results saved to: results/${EXP_NAME}/"
