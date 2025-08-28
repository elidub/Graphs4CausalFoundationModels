#!/bin/bash
# Simple benchmark run script that takes config name as argument

# Extract environment
unzip venv.zip
source venv/bin/activate

# Set up Python path
cd CausalPriorFitting
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Run the benchmark with config
cd src/priordata_processing/Datasets/Benchmark_Dataloader
python3 -m Benchmark_Dataloader "$1"
