#!/bin/bash
# Retry script for ComplexMechNTest data generation
# This script restarts data generation whenever it crashes due to the 
# "none_dealloc: deallocating None" C extension bug

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON="/Users/arikreuter/miniforge/envs/causal_clean/bin/python"
GENERATE_SCRIPT="generate_all_variants_data.py"
LOG_FILE="data_generation.log"
NUM_SAMPLES=1000
MAX_RETRIES=100

echo "=============================================="
echo "ComplexMechNTest Data Generation with Auto-Retry"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Python: $PYTHON"
echo "Samples per config: $NUM_SAMPLES"
echo "Max retries: $MAX_RETRIES"
echo "=============================================="

retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    echo ""
    echo "[Attempt $((retry_count + 1))/$MAX_RETRIES] Starting data generation..."
    echo "Timestamp: $(date)"
    
    # Run the generation script (will skip already-generated files)
    $PYTHON $GENERATE_SCRIPT --num-samples $NUM_SAMPLES 2>&1 | tee -a $LOG_FILE
    
    exit_code=$?
    
    # Check how many files we have
    num_files=$(ls -1 data_cache/*.pkl 2>/dev/null | wc -l | tr -d ' ')
    echo ""
    echo "Exit code: $exit_code"
    echo "Files generated so far: $num_files / 96"
    
    # Check if we're done (96 configs total)
    if [ "$num_files" -ge 96 ]; then
        echo ""
        echo "=============================================="
        echo "SUCCESS! All 96 config variants generated!"
        echo "=============================================="
        exit 0
    fi
    
    # Always retry if we haven't generated all files yet
    retry_count=$((retry_count + 1))
    echo ""
    echo "Not all files generated yet. Waiting 2 seconds before retry..."
    sleep 2
done

echo ""
echo "=============================================="
echo "ERROR: Max retries ($MAX_RETRIES) exceeded!"
echo "Files generated: $num_files / 96"
echo "=============================================="
exit 1
