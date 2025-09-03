#!/usr/bin/env bash
set -euo pipefail

# Very simple launcher for run_benchmark.py
# Usage: ./run_benchmark_simple.sh [MAX_TASKS] [OUTPUT_PATH]
# Defaults: MAX_TASKS=3, OUTPUT_PATH=experiments/FirstTests/benchmark_results.csv

MAX_TASKS=${1:-3}
OUTPUT_PATH=${2:-experiments/FirstTests/benchmark_results.csv}

# Export ALL_CAPS env vars consumed by src/benchmarking/run_benchmark.py
export TASKS=""
export MAX_TASKS="${MAX_TASKS}"
export DATA_DIR="data_cache"
export CONFIG="experiments/FirstTests/configs/early_test.yaml"
export CHECKPOINT="experiments/FirstTests/checkpoints/early_test1_32bs/step_100000.pt"
export DEVICE="cpu"
export OUTPUT="${OUTPUT_PATH}"
export NO_TARGET_ENCODING="0"
export QUIET="0"

echo "Running benchmark: max_tasks=${MAX_TASKS}, output=${OUTPUT_PATH}"

# Run in the job execute directory (Condor transfers inputs here). Use PWD.
cd "${PWD}"

# Ensure logs directory exists so Condor can transfer output files back
mkdir -p logs

# prefer any existing venv in job dir; if venv.zip exists, unzip it
if [ -f venv.zip ] && [ ! -d venv ]; then
  echo "Found venv.zip - extracting..."
  unzip -o -q venv.zip
fi

if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

# Set PYTHONPATH to the package copied into the job directory
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python3 src/benchmarking/run_benchmark.py

echo "Benchmark finished. Results saved to ${OUTPUT_PATH}"
