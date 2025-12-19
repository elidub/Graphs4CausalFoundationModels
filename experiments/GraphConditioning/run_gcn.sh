#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Unzip and activate virtual environment
echo "Unzipping virtual environment..."
unzip -o -q venv.zip

echo "Checking venv directory..."
ls -la venv/

echo "Checking activation script..."
ls -la venv/bin/activate

echo "Activating virtual environment..."
source venv/bin/activate

echo "Setting up virtual environment paths explicitly..."
export PATH="${PWD}/venv/bin:${PATH}"
export VIRTUAL_ENV="${PWD}/venv"

echo "Unzipping data_cache.zip if present..."
if [ -f "data_cache.zip" ]; then
	unzip -o -q data_cache.zip
	echo "data_cache unzipped. Contents:"
	ls -la data_cache | head -n 50 || true
    # Ensure Benchmark picks the correct path regardless of cwd
    export DATA_CACHE_DIR="${PWD}/data_cache"
	export JOB_ROOT_DIR="${PWD}"
    echo "Exported DATA_CACHE_DIR=${DATA_CACHE_DIR}"
	echo "Exported JOB_ROOT_DIR=${JOB_ROOT_DIR}"
else
	echo "No data_cache.zip found; proceeding without cached datasets"
fi

# Run SimplePFN training with config
cd src/training
python3 run.py --config "../../lingaus_ultimate_gcn.yaml"
