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
echo "Running DOFM Plant Evaluation (PlantEval2)"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Graph mode: ${GRAPH_MODE}"
echo "Checkpoint: ${CHECKPOINT_FILE:-final_model_with_bardist.pt}"
echo "Config: ${CONFIG_FILE:-final_model_with_bardist_config.yaml}"
echo "=========================================="

# Use environment variables for checkpoint paths, with defaults
CKPT="${CHECKPOINT_FILE:-final_model_with_bardist.pt}"
CFG="${CONFIG_FILE:-final_model_with_bardist_config.yaml}"

# Create exp_name based on checkpoint
if [[ "$CKPT" == "step_40000.pt" ]]; then
    EXP_NAME="planteval2_step40k_${GRAPH_MODE}"
else
    EXP_NAME="planteval2_${GRAPH_MODE}"
fi

# DATASET already includes .pkl extension
python run_dofm2.py \
    --dataset "${DATASET}" \
    --model "dofm_planteval2" \
    --exp_name "${EXP_NAME}" \
    --graph_mode "${GRAPH_MODE}" \
    --checkpoint_path "${CKPT}" \
    --config_path "${CFG}"

echo "Evaluation complete!"
