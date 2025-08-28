#!/bin/bash

# Simple script to run training with the test configuration

# HTCondor memory management - prevent virtual memory issues

# Function to monitor memory usage
monitor_memory() {
    echo "=== Memory Status at $(date) ==="
    echo "Process tree memory usage:"
    ps aux --forest | head -20
    echo ""
    echo "System memory:"
    free -h
    echo ""
    echo "Virtual memory limits:"
    ulimit -a | grep -E "(virtual|data|resident)"
    echo ""
    echo "Process memory maps (top 10 largest):"
    if [ -f /proc/$$/smaps ]; then
        awk '/^Size:/{size+=$2} /^Rss:/{rss+=$2} /^Pss:/{pss+=$2} END{print "Size:", size/1024, "MB, RSS:", rss/1024, "MB, PSS:", pss/1024, "MB"}' /proc/$$/smaps
    fi
    echo "========================="
    echo ""
}

# Log initial memory state
echo "Starting job with PID: $$"
monitor_memory

# Check if we're in HTCondor environment (files transferred to working directory)
if [ -d "src" ] && [ -d "configs" ]; then
    # HTCondor environment - files are in current directory
    echo "Running in HTCondor environment"
    
    # Set up virtual environment if venv.zip exists
    if [ -f "venv.zip" ]; then
        echo "Setting up virtual environment..."
        unzip -q venv.zip
        source venv/bin/activate
        echo "Virtual environment activated"
        monitor_memory
    fi
    
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    echo "About to start Python training..."
    monitor_memory
    
    # Run with memory monitoring in background
    (
        while true; do
            sleep 30
            monitor_memory
        done
    ) &
    MONITOR_PID=$!
    
    # Run the actual training with simple_run.py (no config argument needed)
    python3 src/training/simple_run.py
    PYTHON_EXIT_CODE=$?
    
    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null
    
    echo "Python training finished with exit code: $PYTHON_EXIT_CODE"
    monitor_memory
    
    exit $PYTHON_EXIT_CODE
else
    # Local environment - navigate to the CausalPriorFitting directory
    echo "Running in local environment"
    cd "$(dirname "$0")/../.."
    export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
    python3 src/training/simple_run.py
fi
