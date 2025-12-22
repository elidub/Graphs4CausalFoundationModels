#!/bin/bash

# Script to submit all .sub files in this directory with condor_submit_bid

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

echo "Submitting all HTCondor jobs in $SCRIPT_DIR"
echo "============================================"
echo ""

# Submit each .sub file
for sub_file in *.sub; do
    if [ -f "$sub_file" ]; then
        echo "Submitting: $sub_file"
        condor_submit_bid 200 "$sub_file"
        echo ""
    fi
done

echo "============================================"
echo "All submissions complete!"
