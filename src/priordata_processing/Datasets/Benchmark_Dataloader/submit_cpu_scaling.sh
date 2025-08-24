#!/bin/bash
# Submit all CPU scaling benchmark jobs

echo "Submitting CPU scaling benchmark jobs..."

configs=("cpu_8" "cpu_16" "cpu_32" "cpu_64" "cpu_96" "cpu_128")

for config in "${configs[@]}"; do
    echo "Submitting ${config}.sub..."
    condor_submit "${config}.sub"
    if [ $? -eq 0 ]; then
        echo "✓ Successfully submitted ${config}"
    else
        echo "✗ Failed to submit ${config}"
    fi
    echo ""
done

echo "All CPU scaling jobs submitted!"
echo "Check status with: condor_q \$USER"
