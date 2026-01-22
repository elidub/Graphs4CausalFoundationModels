#!/bin/bash
# Generate all data using batched approach with subprocess isolation
# Compatible with macOS bash 3.2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use causal_clean environment
PYTHON="/Users/arikreuter/miniforge/envs/causal_clean/bin/python"

# Settings
NUM_SAMPLES=1000
BATCH_SIZE=50  # Small batches to avoid crash accumulation
SEED=42

# Directories
CONFIG_DIR="./configs"
OUTPUT_DIR="./data_cache"

echo "Starting batched generation..."
echo "Python: $PYTHON"
echo "Samples per config: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"

# Count total configs
TOTAL_CONFIGS=$(find "$CONFIG_DIR" -name "*.yaml" | wc -l | tr -d ' ')
echo "Total configs: $TOTAL_CONFIGS"

CURRENT=0
SKIPPED=0
SUCCESS=0
FAILED=0

# Process each config file
find "$CONFIG_DIR" -name "*.yaml" | sort | while read config_file; do
    CURRENT=$((CURRENT + 1))
    
    # Parse the config path to create the standard naming convention
    # e.g., configs/10node/base.yaml -> complexmech_10nodes_base_1000samples_seed42.pkl
    # e.g., configs/10node/path_TY/hide_0.0.yaml -> complexmech_10nodes_path_TY_hide0.0_1000samples_seed42.pkl
    
    # Get the directory name containing the yaml
    parent_dir=$(dirname "$config_file")
    parent_name=$(basename "$parent_dir")
    
    # Check if parent is a node directory or a path directory
    if [[ "$parent_name" =~ ^[0-9]+node$ ]]; then
        # This is a base file like configs/10node/base.yaml
        node_dir="$parent_name"
        config_base=$(basename "$config_file" .yaml)  # e.g., base
        output_name="complexmech_${node_dir}s_${config_base}_${NUM_SAMPLES}samples_seed${SEED}"
    else
        # This is a hide file like configs/10node/path_TY/hide_0.0.yaml
        grandparent_dir=$(basename "$(dirname "$parent_dir")")
        node_dir="$grandparent_dir"
        path_type="$parent_name"  # e.g., path_TY
        config_base=$(basename "$config_file" .yaml)  # e.g., hide_0.0
        # Convert hide_0.0 to hide0.0
        hide_value=$(echo "$config_base" | sed 's/hide_/hide/')
        output_name="complexmech_${node_dir}s_${path_type}_${hide_value}_${NUM_SAMPLES}samples_seed${SEED}"
    fi
    
    output_file="$OUTPUT_DIR/${output_name}.pkl"
    
    # Skip if already exists
    if [ -f "$output_file" ]; then
        echo "[$CURRENT/$TOTAL_CONFIGS] SKIP: $output_name (exists)"
        continue
    fi
    
    echo ""
    echo "[$CURRENT/$TOTAL_CONFIGS] Generating: $output_name"
    echo "  Config: $config_file"
    echo "  Output: $output_file"
    echo "============================================"
    
    # Run batched generator
    if $PYTHON generate_batched.py \
        --config "$config_file" \
        --output "$output_file" \
        --num-samples $NUM_SAMPLES \
        --batch-size $BATCH_SIZE; then
        echo "[$CURRENT/$TOTAL_CONFIGS] SUCCESS: $output_name"
    else
        echo "[$CURRENT/$TOTAL_CONFIGS] FAILED: $output_name"
    fi
    
    # Small pause between configs
    sleep 1
done

echo ""
echo "============================================"
echo "GENERATION COMPLETE"
echo "============================================"
