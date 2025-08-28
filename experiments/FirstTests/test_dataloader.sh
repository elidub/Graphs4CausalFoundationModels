#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
#export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Unzip and activate virtual environment
unzip -q venv.zip
source venv/bin/activate

# Run dataloader test
if [ "$1" ]; then
    echo "Running with config file: $1"
    # Convert relative path to absolute path before changing directory
    CONFIG_PATH="$(pwd)/$1"
    cd src/training
    python test_dataloader.py --config "$CONFIG_PATH"
else
    echo "Running with default configs from BasicConfigs.py"
    cd src/training
    python test_dataloader.py
fi
