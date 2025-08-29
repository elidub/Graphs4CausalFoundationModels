#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Unzip and activate virtual environment
unzip -q venv.zip
source venv/bin/activate

# Debug: Show directory structure
echo "=== Current directory contents ==="
pwd
ls -la
echo "=== Contents after cd src/training ==="
cd src/training
pwd
ls -la
echo "=== Looking for config file ==="
ls -la ../../early_test.yaml || echo "Config file not found"

# Run SimplePFN training with config specified here
python3 simple_run.py --config "../../early_test.yaml"
