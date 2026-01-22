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
echo "Running DOFM Plant Evaluation"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Graph mode: ${GRAPH_MODE}"
echo "Use one-hot: ${USE_ONEHOT}"
echo "=========================================="

# Build command with optional --onehot flag
CMD="python run_dofm.py \
    --dataset \"${DATASET}\" \
    --model \"dofm_plant\" \
    --exp_name \"dofm_plant_${GRAPH_MODE}${ONEHOT_SUFFIX}\" \
    --graph_mode \"${GRAPH_MODE}\" \
    --checkpoint_path \"final_model_with_bardist.pt\" \
    --config_path \"final_model_with_bardist_config.yaml\""

if [ "${USE_ONEHOT}" = "true" ]; then
    CMD="${CMD} --onehot"
fi

eval ${CMD}

echo "Evaluation complete!"
