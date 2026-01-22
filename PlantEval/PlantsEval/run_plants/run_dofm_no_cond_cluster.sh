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
echo "Running DOFM Plant Evaluation (Non-Conditioned)"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "=========================================="

# Non-conditioned model - no graph conditioning
python run_dofm_no_cond.py \
    --dataset "${DATASET}" \
    --model "dofm_plant_no_cond" \
    --exp_name "dofm_plant_no_cond" \
    --graph_mode "all_unknown" \
    --checkpoint_path "final_model_with_bardist.pt" \
    --config_path "final_model_with_bardist_config.yaml"

echo "Evaluation complete!"
