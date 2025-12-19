#!/bin/bash
# Submit both 20-node and 50-node GCN + Hard Attention experiments
# Usage: ./submit_gcn_hard_both.sh [bid_amount]

BID=${1:-2000}  # Default bid is 2000 if not specified

echo "=========================================="
echo "Submitting GCN + Hard Attention experiments"
echo "Bid amount: $BID"
echo "=========================================="
echo ""

# Array of submit files
SUBMIT_FILES=(
    "submit_gcn_hard_20node.sub"
    "submit_gcn_hard_50node.sub"
)

# Array of config names for display
CONFIG_NAMES=(
    "GCN + Hard Attention (20 nodes)"
    "GCN + Hard Attention (50 nodes)"
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
echo "    tail -f logs/gcn_hard_*node*.out"
echo ""
