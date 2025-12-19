#!/bin/bash
# Master script to submit all config_5node experiments
# Usage: ./submit_all.sh [bid_amount]

BID=${1:-2000}  # Default bid is 2000 if not specified

echo "=========================================="
echo "Submitting all configs_5node experiments"
echo "Bid amount: $BID"
echo "=========================================="
echo ""

# Array of submit files
SUBMIT_FILES=(
    "submit_gcn.sub"
    "submit_gcn_hard.sub"
    "submit_gcn_soft.sub"
    "submit_hard.sub"
    "submit_soft.sub"
)

# Array of config names for display
CONFIG_NAMES=(
    "GCN only"
    "GCN + Hard Attention"
    "GCN + Soft Attention"
    "Hard Attention only"
    "Soft Attention only"
)

# Submit each job
for i in "${!SUBMIT_FILES[@]}"; do
    submit_file="${SUBMIT_FILES[$i]}"
    config_name="${CONFIG_NAMES[$i]}"
    
    echo "[$((i+1))/${#SUBMIT_FILES[@]}] Submitting: $config_name"
    echo "    Submit file: $submit_file"
    
    if condor_submit_bid "$BID" "$submit_file"; then
        echo "    ✓ Successfully submitted"
    else
        echo "    ✗ Failed to submit"
    fi
    echo ""
done

echo "=========================================="
echo "Submission complete!"
echo "=========================================="
echo ""
echo "To check job status:"
echo "    condor_q"
echo ""
echo "To monitor logs:"
echo "    tail -f logs/*.out"
echo ""
