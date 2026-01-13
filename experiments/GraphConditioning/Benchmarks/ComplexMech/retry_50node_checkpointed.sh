#!/bin/bash

# Retry wrapper for 50 node checkpointed data generation
# Processes each configuration individually with automatic retry on crashes

# Change to the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SCRIPT_NAME="generate_50node_checkpointed.py"
MAX_ATTEMPTS=10
SUCCESS=false

echo "=========================================="
echo "50 Node ComplexMech Checkpointed Generation"
echo "=========================================="
echo "Working directory: $SCRIPT_DIR"
echo "Max attempts per config: $MAX_ATTEMPTS"
echo "Start time: $(date)"
echo "=========================================="

# First, show what needs to be generated
echo ""
echo "Checking current status..."
/Users/arikreuter/miniforge/envs/fastpy/bin/python $SCRIPT_NAME
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ All 50 node configurations already complete!"
    exit 0
fi

echo ""
echo "Some configurations need generation. Starting individual processing..."
echo ""

# Get list of all 50 node configs that need processing
CONFIGS_TO_PROCESS=(
    "base.yaml"
    "path_TY/ntest_500.yaml"
    "path_TY/ntest_700.yaml"
    "path_TY/ntest_800.yaml"
    "path_TY/ntest_900.yaml"
    "path_TY/ntest_950.yaml"
    "path_YT/ntest_500.yaml"
    "path_YT/ntest_700.yaml"
    "path_YT/ntest_800.yaml"
    "path_YT/ntest_900.yaml"
    "path_YT/ntest_950.yaml"
    "path_independent_TY/ntest_500.yaml"
    "path_independent_TY/ntest_700.yaml"
    "path_independent_TY/ntest_800.yaml"
    "path_independent_TY/ntest_900.yaml"
    "path_independent_TY/ntest_950.yaml"
)

TOTAL_CONFIGS=${#CONFIGS_TO_PROCESS[@]}
COMPLETED_CONFIGS=0
FAILED_CONFIGS=0

echo "Processing ${TOTAL_CONFIGS} configurations individually..."

# Process each config separately with retry
for i in "${!CONFIGS_TO_PROCESS[@]}"; do
    CONFIG="${CONFIGS_TO_PROCESS[$i]}"
    CONFIG_NUM=$((i + 1))
    
    echo ""
    echo "=========================================="
    echo "CONFIG $CONFIG_NUM/$TOTAL_CONFIGS: $CONFIG"
    echo "=========================================="
    
    CURRENT_ATTEMPT=1
    CONFIG_SUCCESS=false
    
    while [ $CURRENT_ATTEMPT -le $MAX_ATTEMPTS ] && [ "$CONFIG_SUCCESS" = false ]; do
        echo ""
        echo "--- Attempt $CURRENT_ATTEMPT/$MAX_ATTEMPTS for $CONFIG ---"
        echo "Starting at $(date)"
        
        # Run the checkpointed script for this specific config
        /Users/arikreuter/miniforge/envs/fastpy/bin/python $SCRIPT_NAME --config "$CONFIG"
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ Config $CONFIG completed successfully!"
            CONFIG_SUCCESS=true
            COMPLETED_CONFIGS=$((COMPLETED_CONFIGS + 1))
        else
            echo "❌ Config $CONFIG failed (exit code: $EXIT_CODE)"
            
            if [ $CURRENT_ATTEMPT -lt $MAX_ATTEMPTS ]; then
                echo "Waiting 5 seconds before retry..."
                sleep 5
            fi
            
            CURRENT_ATTEMPT=$((CURRENT_ATTEMPT + 1))
        fi
    done
    
    if [ "$CONFIG_SUCCESS" = false ]; then
        echo "❌ Config $CONFIG failed after $MAX_ATTEMPTS attempts"
        FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
    fi
    
    # Progress summary
    echo ""
    echo "Progress: $COMPLETED_CONFIGS/$TOTAL_CONFIGS completed, $FAILED_CONFIGS failed"
done

echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo "Total configurations: $TOTAL_CONFIGS"
echo "✓ Completed: $COMPLETED_CONFIGS"
echo "❌ Failed: $FAILED_CONFIGS"
echo "End time: $(date)"

if [ $FAILED_CONFIGS -eq 0 ]; then
    echo "✓ All 50 node configurations completed successfully!"
    exit 0
else
    echo "❌ Some configurations failed after maximum retries"
    exit 1
fi