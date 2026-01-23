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
echo "Running X-Learner Baseline Evaluation"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Experiment: ${EXP_NAME}"
echo "Checkpoint: ${CHECKPOINT_FILENAME}"
echo "Config: ${CONFIG_FILENAME}"
echo "=========================================="

# Run the evaluation
echo "Starting evaluation..."
python run_baselines/predmodel_Xlearner.py \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --exp_name "${EXP_NAME}"

echo "Evaluation complete!"
echo "Results saved to: results/${EXP_NAME}/"
