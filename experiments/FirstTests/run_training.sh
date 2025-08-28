#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Unzip and activate virtual environment
unzip -q venv.zip
source venv/bin/activate

# Run training with dataloader (no arguments needed - uses default configs)
cd src/training
python train_with_dataloader.py
