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

# Unzip ComplexMech2 benchmark data
echo "Unzipping ComplexMech2 benchmark data..."
if [ -f "samples.zip" ]; then
	mkdir -p data_cache
	unzip -o -q samples.zip -d data_cache
	echo "ComplexMech2 benchmark data unzipped. Contents:"
	ls -la data_cache | head -20 || true
else
	echo "ERROR: No samples.zip found; cannot run benchmark"
	exit 1
fi

# Set benchmark directory
export COMPLEXMECH_BENCHMARK_DIR="${PWD}"
echo "Exported COMPLEXMECH_BENCHMARK_DIR=${COMPLEXMECH_BENCHMARK_DIR}"

# Run the benchmark
echo "Running ComplexMech2 benchmark..."
echo "Using config: ${CONFIG_FILENAME}"
echo "Using checkpoint: ${CHECKPOINT_FILENAME}"
echo "Model ID: ${MODEL_ID}"
echo "Max samples: ${MAX_SAMPLES:-10}"
echo "Output directory: ${OUTPUT_DIR}"

if [ -n "${OUTPUT_DIR}" ]; then
    python3 run_complex_mech_benchmark.py \
        --config_path "${PWD}/${CONFIG_FILENAME}" \
        --checkpoint_path "${PWD}/${CHECKPOINT_FILENAME}" \
        --max_samples "${MAX_SAMPLES:-10}" \
        --output_dir "${OUTPUT_DIR}"
else
    python3 run_complex_mech_benchmark.py \
        --config_path "${PWD}/${CONFIG_FILENAME}" \
        --checkpoint_path "${PWD}/${CHECKPOINT_FILENAME}" \
        --max_samples "${MAX_SAMPLES:-10}"
fi

echo "Benchmark complete!"
echo "Results should be in: ${OUTPUT_DIR}"
