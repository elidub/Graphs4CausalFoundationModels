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

# Unzip ComplexMechNTest benchmark data
echo "Unzipping ComplexMechNTest benchmark data..."
if [ -f "data_cache_ComplexMechNTest.zip" ]; then
	unzip -o -q data_cache_ComplexMechNTest.zip
	# The zip creates data_cache/data_cache/ structure, fix it
	if [ -d "data_cache/data_cache" ]; then
		mv data_cache/data_cache/* data_cache/ 2>/dev/null || true
		rmdir data_cache/data_cache 2>/dev/null || true
	fi
	echo "ComplexMechNTest benchmark data unzipped. Contents:"
	ls -la data_cache | head -20 || true
else
	echo "ERROR: No data_cache_ComplexMechNTest.zip found; cannot run benchmark"
	exit 1
fi

# Set benchmark directory
export COMPLEXMECH_NTEST_BENCHMARK_DIR="${PWD}"
echo "Exported COMPLEXMECH_NTEST_BENCHMARK_DIR=${COMPLEXMECH_NTEST_BENCHMARK_DIR}"

# Run the benchmark
echo "Running ComplexMechNTest benchmark..."
python3 run_complexmech_ntest.py \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --fidelity "${FIDELITY:-high}" \
    --output_dir "${OUTPUT_DIR}"

echo "Benchmark complete!"
