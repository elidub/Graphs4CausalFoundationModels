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

# Unzip LinGaus benchmark data
echo "Unzipping LinGaus benchmark data..."
if [ -f "lingaus_benchmark_data.zip" ]; then
	mkdir -p Benchmarks/LinGaus/data_cache
	unzip -o -q lingaus_benchmark_data.zip -d Benchmarks/LinGaus/data_cache
	echo "LinGaus benchmark data unzipped. Contents:"
	ls -la Benchmarks/LinGaus/data_cache || true
	export LINGAUS_BENCHMARK_DIR="${PWD}/Benchmarks/LinGaus"
	echo "Exported LINGAUS_BENCHMARK_DIR=${LINGAUS_BENCHMARK_DIR}"
else
	echo "No lingaus_benchmark_data.zip found; LinGaus benchmark will be disabled"
fi

# Copy benchmark module files to working directory
echo "Setting up LinGaus benchmark module..."
if [ -d "Benchmarks/LinGaus" ]; then
	# Ensure the benchmark module is in PYTHONPATH
	export PYTHONPATH="${PWD}/Benchmarks/LinGaus:${PYTHONPATH}"
	echo "Added Benchmarks/LinGaus to PYTHONPATH"
	echo "Contents of Benchmarks/LinGaus:"
	ls -la Benchmarks/LinGaus/ || true
	echo "Verifying benchmark data:"
	ls -la Benchmarks/LinGaus/data_cache/ | head -10 || true
else
	echo "WARNING: Benchmarks/LinGaus directory not found!"
	echo "Current directory: ${PWD}"
	echo "Directory contents:"
	ls -la
fi

# Run SimplePFN training with config
cd src/training
python3 run.py --config "../../configs_50node_ancestor/lingaus_ancestor_50node_gcn_and_softatt.yaml"

# Display debug log if it exists
echo "=== LinGaus Import Debug Log ==="
if [ -f "/tmp/lingaus_import_debug.log" ]; then
    cat /tmp/lingaus_import_debug.log
else
    echo "Debug log not found at /tmp/lingaus_import_debug.log"
fi
echo "================================"
