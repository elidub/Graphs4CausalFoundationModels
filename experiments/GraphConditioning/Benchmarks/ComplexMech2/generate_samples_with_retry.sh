#!/bin/bash

# Robust sample generation script with automatic retry on crash
# This script will keep retrying until all samples are generated or max attempts reached

PYTHON_BIN="/Users/arikreuter/miniforge/envs/fastpy/bin/python"
BENCHMARK_DIR="/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/GraphConditioning/Benchmarks/ComplexMech2"
TARGET_SAMPLES=50000
MAX_ATTEMPTS=100

echo "=========================================="
echo "ComplexMech2 Sample Generation with Retry"
echo "=========================================="
echo "Target samples: $TARGET_SAMPLES"
echo "Max attempts: $MAX_ATTEMPTS"
echo "Starting at: $(date)"
echo ""

cd "$BENCHMARK_DIR" || exit 1

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "----------------------------------------"
    echo "Attempt $attempt / $MAX_ATTEMPTS"
    echo "Time: $(date)"
    echo "----------------------------------------"
    
    # Count existing samples
    EXISTING_SAMPLES=$(find data_cache/samples -name "sample_*.pkl" 2>/dev/null | wc -l | tr -d ' ')
    echo "Existing samples: $EXISTING_SAMPLES / $TARGET_SAMPLES"
    
    # Check if we're done
    if [ "$EXISTING_SAMPLES" -ge "$TARGET_SAMPLES" ]; then
        echo ""
        echo "=========================================="
        echo "SUCCESS! All $TARGET_SAMPLES samples generated!"
        echo "Completed at: $(date)"
        echo "Total attempts: $attempt"
        echo "=========================================="
        exit 0
    fi
    
    # Calculate remaining samples
    REMAINING=$((TARGET_SAMPLES - EXISTING_SAMPLES))
    echo "Remaining: $REMAINING samples"
    echo ""
    
    # Run the generation script
    $PYTHON_BIN -c "
from ComplexMechBenchmark import ComplexMechBenchmark

print('Initializing benchmark...')
benchmark = ComplexMechBenchmark(verbose=True)

print('Loading config...')
config = benchmark.load_config()

print('Creating dataset...')
dataset = benchmark.create_dataset(seed=42)

print('Creating collator...')
collator = benchmark.create_collator()

print('Starting sample generation...')
samples = benchmark.generate_data(
    num_samples=$TARGET_SAMPLES,
    seed=42,
    save_to_cache=True,
    overwrite=False,
    keep_in_memory=False,
)

print('\nGeneration attempt complete!')
"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "Generation completed successfully!"
        # Check again if we have all samples
        EXISTING_SAMPLES=$(find data_cache/samples -name "sample_*.pkl" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$EXISTING_SAMPLES" -ge "$TARGET_SAMPLES" ]; then
            echo ""
            echo "=========================================="
            echo "SUCCESS! All $TARGET_SAMPLES samples generated!"
            echo "Completed at: $(date)"
            echo "Total attempts: $attempt"
            echo "=========================================="
            exit 0
        fi
    else
        echo ""
        echo "Process crashed with exit code $EXIT_CODE"
        echo "Saved samples before crash, will retry..."
    fi
    
    # Small delay before retry
    sleep 2
done

# If we get here, we hit max attempts
echo ""
echo "=========================================="
echo "WARNING: Reached maximum attempts ($MAX_ATTEMPTS)"
echo "Final sample count: $(find data_cache/samples -name 'sample_*.pkl' 2>/dev/null | wc -l | tr -d ' ') / $TARGET_SAMPLES"
echo "Ended at: $(date)"
echo "=========================================="
exit 1
