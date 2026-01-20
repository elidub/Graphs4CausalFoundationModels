#!/bin/bash
set -e

export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH}"

echo "Unzipping virtual environment..."
unzip -o -q venv.zip

echo "Activating virtual environment..."
source venv/bin/activate

export PATH="${PWD}/venv/bin:${PATH}"
export VIRTUAL_ENV="${PWD}/venv"

echo "Python: $(which python) ($(python --version))"

echo "=========================================="
echo "Running DOFM PSID-Balanced Evaluation"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Graph Mode: ${GRAPH_MODE}"
echo "Experiment: ${EXP_NAME}"
echo "=========================================="

python run_baselines/dofm_psid_balanced.py \
    --dataset "${DATASET}" \
    --model "${MODEL_NAME}" \
    --exp_name "${EXP_NAME}" \
    --graph_mode "${GRAPH_MODE}"

echo "Evaluation complete!"
