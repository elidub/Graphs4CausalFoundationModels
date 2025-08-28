#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Unzip and activate virtual environment
unzip -q venv.zip
source venv/bin/activate

# Run SimplePFN training with config specified here
CONFIG_PATH="../../early_test.yaml"
cd src/training
python simple_run.py --config "$CONFIG_PATH"
