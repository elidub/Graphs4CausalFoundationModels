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
echo "Running DOFM Subsampled (1000) Evaluation"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Graph Mode: ${GRAPH_MODE}"
echo "Experiment: ${EXP_NAME}"
echo "=========================================="

# Run the evaluation
echo "Starting evaluation..."
python run_baselines/dofm_subsampled.py \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --exp_name "${EXP_NAME}" \
    --graph_mode "${GRAPH_MODE}"

echo "Evaluation complete!"
echo "Results saved to: results/${EXP_NAME}/"
