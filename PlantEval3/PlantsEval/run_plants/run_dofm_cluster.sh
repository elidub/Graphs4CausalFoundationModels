#!/bin/bash
set -e

# Setup environment - match PlantEval2 structure
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

# Debug: List transferred files to verify dataset is present
echo "=========================================="
echo "Files in execution directory:"
ls -lh
echo "=========================================="

echo "=========================================="
echo "Running DOFM Plant Evaluation (PlantEval3)"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Graph mode: ${GRAPH_MODE}"
echo "Use target encoding: ${USE_TARGET_ENCODING}"
echo "Hide by correlation: ${HIDE_BY_CORRELATION}"
echo "Top K edges: ${TOP_K_EDGES}"
echo "Checkpoint: ${CHECKPOINT_FILE:-final_model_with_bardist.pt}"
echo "Config: ${CONFIG_FILE:-final_model_with_bardist_config.yaml}"
echo "=========================================="

# Use environment variables for checkpoint paths, with defaults
CKPT="${CHECKPOINT_FILE:-final_model_with_bardist.pt}"
CFG="${CONFIG_FILE:-final_model_with_bardist_config.yaml}"

# Build command - run directly like PlantEval2
CMD="python run_dofm.py \
    --dataset \"${DATASET}\" \
    --model \"dofm\" \
    --exp_name \"${EXP_NAME}\" \
    --graph_mode \"${GRAPH_MODE}\" \
    --checkpoint_path \"${CKPT}\" \
    --config_path \"${CFG}\""

# Add optional flags
if [ "${USE_TARGET_ENCODING}" = "true" ]; then
    CMD="${CMD} --use_target_encoding"
fi

if [ "${HIDE_BY_CORRELATION}" = "true" ]; then
    CMD="${CMD} --hide_by_correlation --top_k_edges ${TOP_K_EDGES}"
fi

echo "Running command: ${CMD}"
eval ${CMD}

echo "Evaluation complete!"
