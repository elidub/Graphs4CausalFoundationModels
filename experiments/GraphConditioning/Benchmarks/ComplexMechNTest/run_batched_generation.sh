#!/bin/bash
# Run batched generation for all ComplexMechNTest configs
# macOS bash 3.2 compatible (no mapfile)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"
DATA_DIR="$SCRIPT_DIR/data_cache"
PYTHON="/Users/arikreuter/miniforge/envs/causal_clean/bin/python"
GENERATOR="$SCRIPT_DIR/generate_batched.py"

NUM_SAMPLES=1000
BATCH_SIZE=50
MAX_RETRIES=5
SEED=42

mkdir -p "$DATA_DIR"

# Function to get expected output name for a config
get_output_name() {
    local config_path="$1"
    local rel_path="${config_path#$CONFIG_DIR/}"
    
    # Parse the path: node_dir/base.yaml or node_dir/path_variant/ntest_N.yaml
    local node_dir=$(echo "$rel_path" | cut -d'/' -f1)
    local rest=$(echo "$rel_path" | cut -d'/' -f2-)
    
    if [[ "$rest" == "base.yaml" ]]; then
        # Base config: complexmech_ntest_{node_dir}s_base_1000samples_seed42.pkl
        echo "complexmech_ntest_${node_dir}s_base_${NUM_SAMPLES}samples_seed${SEED}"
    else
        # Variant config: node_dir/path_XXX/ntest_N.yaml
        local path_type=$(echo "$rel_path" | cut -d'/' -f2)
        local ntest_file=$(echo "$rel_path" | cut -d'/' -f3)
        local ntest_value=$(echo "$ntest_file" | sed 's/ntest_/ntest/' | sed 's/.yaml//')
        # Format: complexmech_{node_dir}s_{path_type}_{ntest_value}_1000samples_seed42.pkl
        echo "complexmech_${node_dir}s_${path_type}_${ntest_value}_${NUM_SAMPLES}samples_seed${SEED}"
    fi
}

# Count configs
total_configs=0
for config in $(find "$CONFIG_DIR" -name "*.yaml" -type f); do
    total_configs=$((total_configs + 1))
done

echo "========================================"
echo "ComplexMechNTest Batched Data Generation"
echo "========================================"
echo "Found $total_configs configs"
echo "Data directory: $DATA_DIR"
echo "Batch size: $BATCH_SIZE"
echo ""

# Process each config
processed=0
skipped=0

for config_path in $(find "$CONFIG_DIR" -name "*.yaml" -type f | sort); do
    processed=$((processed + 1))
    
    output_name=$(get_output_name "$config_path")
    output_file="$DATA_DIR/${output_name}.pkl"
    
    # Check if already exists
    if [[ -f "$output_file" ]]; then
        echo "[$processed/$total_configs] SKIP: $output_name (exists)"
        skipped=$((skipped + 1))
        continue
    fi
    
    echo ""
    echo "========================================"
    echo "[$processed/$total_configs] Generating: $output_name"
    echo "Config: $config_path"
    echo "========================================"
    
    "$PYTHON" "$GENERATOR" \
        --config "$config_path" \
        --output "$output_file" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --max-retries "$MAX_RETRIES"
    
    if [[ $? -eq 0 ]]; then
        echo "SUCCESS: $output_name"
    else
        echo "FAILED: $output_name"
    fi
done

echo ""
echo "========================================"
echo "Generation Complete"
echo "========================================"
echo "Total configs: $total_configs"
echo "Skipped (existing): $skipped"
echo "Generated: $((processed - skipped))"
