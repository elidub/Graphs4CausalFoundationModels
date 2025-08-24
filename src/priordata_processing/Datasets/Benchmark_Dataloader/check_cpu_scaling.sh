#!/bin/bash
# Simple analysis script to check CPU scaling results

echo "CPU Scaling Benchmark Results Analysis"
echo "======================================"

configs=("cpu_8" "cpu_16" "cpu_32" "cpu_64" "cpu_96" "cpu_128")

echo ""
echo "Job Status:"
echo "-----------"
condor_q $USER

echo ""
echo "Checking output files for completion:"
echo "------------------------------------"

for config in "${configs[@]}"; do
    out_file="Logs/${config}_job.out"
    err_file="Logs/${config}_job.err"
    
    if [ -f "$out_file" ]; then
        # Check if benchmark completed
        if grep -q "BENCHMARK COMPLETE" "$out_file"; then
            # Extract key metrics
            avg_time=$(grep "Average time per batch:" "$out_file" | sed 's/.*: //' | sed 's/ms//')
            batches_per_sec=$(grep "Batches per second:" "$out_file" | sed 's/.*: //')
            
            echo "✓ ${config}: ${avg_time}ms avg, ${batches_per_sec} batches/sec"
        else
            echo "⚠ ${config}: Job may still be running or failed"
        fi
    else
        echo "✗ ${config}: No output file found"
    fi
    
    # Check for errors
    if [ -f "$err_file" ] && [ -s "$err_file" ]; then
        echo "  ⚠ Warning: Error file not empty for ${config}"
    fi
done

echo ""
echo "Use 'cat Logs/{config}_job.out' to see full results for any configuration"
