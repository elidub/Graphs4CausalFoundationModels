#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PWD}:${PYTHONPATH}"

# Unzip and activate virtual environment
echo "Unzipping virtual environment..."
unzip -o -q venv.zip

echo "Activating virtual environment..."
source venv/bin/activate

export PATH="${PWD}/venv/bin:${PATH}"
export VIRTUAL_ENV="${PWD}/venv"

echo "Python location: $(which python)"
echo "Python version: $(python --version)"

echo "=========================================="
echo "Running DOFM Plant Evaluation (v2)"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Graph mode: ${GRAPH_MODE}"
echo "=========================================="

python run_dofm2.py \
    --dataset "${DATASET}" \
    --model "dofm_plant_v2_label" \
    --exp_name "dofm_plant_v2_label_${GRAPH_MODE}" \
    --graph_mode "${GRAPH_MODE}" \
    --checkpoint_path "final_model_with_bardist.pt" \
    --config_path "final_model_with_bardist_config.yaml"

echo "Evaluation complete!"
