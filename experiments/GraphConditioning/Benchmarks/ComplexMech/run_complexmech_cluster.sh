#!/bin/bash
set -e

# Setup environment
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Unzip and activate virtual environment
echo "Unzipping virtual environment..."
unzip -o -q venv.zip

echo "Activating virtual environment..."
source venv/bin/activate

echo "Setting up virtual environment paths explicitly..."
export PATH="${PWD}/venv/bin:${PATH}"
export VIRTUAL_ENV="${PWD}/venv"

# Unzip ComplexMech benchmark data
echo "Unzipping ComplexMech benchmark data..."
if [ -f "Archive.zip" ]; then
	mkdir -p data_cache
	unzip -o -q Archive.zip -d data_cache
	echo "ComplexMech benchmark data unzipped. Contents:"
	ls -la data_cache | head -20 || true
else
	echo "ERROR: No Archive.zip found; cannot run benchmark"
	exit 1
fi

# Set benchmark directory
export COMPLEXMECH_BENCHMARK_DIR="${PWD}"
echo "Exported COMPLEXMECH_BENCHMARK_DIR=${COMPLEXMECH_BENCHMARK_DIR}"

# Run the benchmark
echo "Running ComplexMech benchmark..."
python3 run_complexmech.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --fidelity "${FIDELITY:-high}" \
    --output_dir "${OUTPUT_DIR}"

echo "Benchmark complete!"
