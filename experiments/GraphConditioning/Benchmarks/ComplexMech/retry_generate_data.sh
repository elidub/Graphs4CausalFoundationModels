#!/bin/bash

# Hacky auto-retry wrapper for generate_all_variants_data.py
# Reruns the script up to 25 times when it crashes
# Runs exactly as VSCode would run it (no command line args, default behavior)

# Change to the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SCRIPT_NAME="generate_all_variants_data.py"
MAX_ATTEMPTS=25
CURRENT_ATTEMPT=1
SUCCESS=false

echo "=========================================="
echo "Auto-Retry Wrapper for $SCRIPT_NAME"
echo "=========================================="
echo "Working directory: $SCRIPT_DIR"
echo "Max attempts: $MAX_ATTEMPTS"
echo "Running with default settings (as VSCode would)"
echo "Start time: $(date)"
echo "=========================================="

while [ $CURRENT_ATTEMPT -le $MAX_ATTEMPTS ] && [ "$SUCCESS" = false ]; do
    echo ""
    echo "==================== ATTEMPT $CURRENT_ATTEMPT/$MAX_ATTEMPTS ===================="
    echo "Starting attempt $CURRENT_ATTEMPT at $(date)"
    
    # Run the Python script exactly as VSCode would (no args = defaults)
    /Users/arikreuter/miniforge/envs/fastpy/bin/python $SCRIPT_NAME
    EXIT_CODE=$?
    
    echo "Exit code: $EXIT_CODE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ SUCCESS! Script completed successfully on attempt $CURRENT_ATTEMPT"
        SUCCESS=true
    else
        echo "✗ FAILED! Script crashed with exit code $EXIT_CODE"
        
        # Check for specific error patterns
        if [ $EXIT_CODE -eq 134 ]; then
            echo "  Error type: SIGABRT (likely C extension refcount error)"
        elif [ $EXIT_CODE -eq 139 ]; then
            echo "  Error type: SIGSEGV (segmentation fault)"
        elif [ $EXIT_CODE -eq 6 ]; then
            echo "  Error type: SIGABRT (abort trap)"
        fi
        
        if [ $CURRENT_ATTEMPT -lt $MAX_ATTEMPTS ]; then
            echo "  Waiting 5 seconds before retry..."
            sleep 5
            CURRENT_ATTEMPT=$((CURRENT_ATTEMPT + 1))
        else
            echo "  Max attempts ($MAX_ATTEMPTS) reached. Giving up."
            break
        fi
    fi
done

echo ""
echo "=========================================="
echo "Final Summary:"
echo "=========================================="
if [ "$SUCCESS" = true ]; then
    echo "✓ SUCCESS after $CURRENT_ATTEMPT attempt(s)"
    echo "End time: $(date)"
    exit 0
else
    echo "✗ FAILED after $MAX_ATTEMPTS attempts"
    echo "End time: $(date)"
    echo "The script consistently crashes - there may be a fundamental issue."
    exit 1
fi