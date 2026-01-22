#!/bin/bash
# Process-isolated retry script for ComplexMechIDK data generation
# Runs each config in a SEPARATE Python process to avoid memory corruption

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON="/Users/arikreuter/miniforge/envs/causal_clean/bin/python"
SINGLE_CONFIG_SCRIPT="generate_single_config.py"
CACHE_DIR="data_cache"
NUM_SAMPLES=1000
BASE_SEED=42
MAX_RETRIES_PER_CONFIG=5

# Define all configs to generate
NODE_COUNTS=(2 5 10 20 35 50)
VARIANTS=("base" "path_TY" "path_YT" "path_independent_TY")
HIDE_FRACTIONS=("0.0" "0.25" "0.5" "0.75" "1.0")

mkdir -p "$CACHE_DIR"

echo "=============================================="
echo "ComplexMechIDK - Process-Isolated Generation"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Python: $PYTHON"
echo "Samples per config: $NUM_SAMPLES"
echo "=============================================="

total_configs=0
completed_configs=0
failed_configs=0

# Count total configs
for node_count in "${NODE_COUNTS[@]}"; do
    # Base config
    ((total_configs++))
    
    # Path variants with hide fractions
    for variant in "${VARIANTS[@]}"; do
        if [ "$variant" != "base" ]; then
            for hide in "${HIDE_FRACTIONS[@]}"; do
                ((total_configs++))
            done
        fi
    done
done

echo "Total configs to generate: $total_configs"
echo "=============================================="

config_num=0

for node_count in "${NODE_COUNTS[@]}"; do
    # Process base config
    ((config_num++))
    config_path="configs/${node_count}node/base.yaml"
    output_path="${CACHE_DIR}/complexmech_${node_count}nodes_base_${NUM_SAMPLES}samples_seed${BASE_SEED}.pkl"
    
    if [ -f "$output_path" ]; then
        echo "[$config_num/$total_configs] SKIP: ${node_count}node base (exists)"
        ((completed_configs++))
    else
        echo "[$config_num/$total_configs] Generating: ${node_count}node base"
        
        retry=0
        success=0
        while [ $retry -lt $MAX_RETRIES_PER_CONFIG ] && [ $success -eq 0 ]; do
                        # Run in subprocess (no timeout on macOS)
                        seed=$((BASE_SEED + config_num * 1000 + retry))
                        $PYTHON $SINGLE_CONFIG_SCRIPT \
                            --config "$config_path" \
                            --output "$output_path" \
                            --num-samples $NUM_SAMPLES \
                            --seed $seed            exit_code=$?
            
            if [ $exit_code -eq 0 ] && [ -f "$output_path" ]; then
                echo "  ✓ Success"
                ((completed_configs++))
                success=1
            else
                ((retry++))
                echo "  ✗ Failed (attempt $retry/$MAX_RETRIES_PER_CONFIG), retrying..."
                rm -f "$output_path" 2>/dev/null
                sleep 1
            fi
        done
        
        if [ $success -eq 0 ]; then
            echo "  ✗✗ FAILED after $MAX_RETRIES_PER_CONFIG attempts"
            ((failed_configs++))
        fi
    fi
    
    # Process path variants
    for variant in "${VARIANTS[@]}"; do
        if [ "$variant" != "base" ]; then
            for hide in "${HIDE_FRACTIONS[@]}"; do
                ((config_num++))
                config_path="configs/${node_count}node/${variant}/hide_${hide}.yaml"
                output_path="${CACHE_DIR}/complexmech_${node_count}nodes_${variant}_hide${hide}_${NUM_SAMPLES}samples_seed${BASE_SEED}.pkl"
                
                if [ -f "$output_path" ]; then
                    echo "[$config_num/$total_configs] SKIP: ${node_count}node ${variant} hide=${hide} (exists)"
                    ((completed_configs++))
                else
                    echo "[$config_num/$total_configs] Generating: ${node_count}node ${variant} hide=${hide}"
                    
                    retry=0
                    success=0
                    while [ $retry -lt $MAX_RETRIES_PER_CONFIG ] && [ $success -eq 0 ]; do
                        seed=$((BASE_SEED + config_num * 1000 + retry))
                        $PYTHON $SINGLE_CONFIG_SCRIPT \
                            --config "$config_path" \
                            --output "$output_path" \
                            --num-samples $NUM_SAMPLES \
                            --seed $seed
                        
                        exit_code=$?
                        
                        if [ $exit_code -eq 0 ] && [ -f "$output_path" ]; then
                            echo "  ✓ Success"
                            ((completed_configs++))
                            success=1
                        else
                            ((retry++))
                            echo "  ✗ Failed (attempt $retry/$MAX_RETRIES_PER_CONFIG), retrying..."
                            rm -f "$output_path" 2>/dev/null
                            sleep 1
                        fi
                    done
                    
                    if [ $success -eq 0 ]; then
                        echo "  ✗✗ FAILED after $MAX_RETRIES_PER_CONFIG attempts"
                        ((failed_configs++))
                    fi
                fi
            done
        fi
    done
done

echo ""
echo "=============================================="
echo "Generation Complete!"
echo "=============================================="
echo "Completed: $completed_configs / $total_configs"
echo "Failed: $failed_configs"
echo "=============================================="

if [ $failed_configs -gt 0 ]; then
    exit 1
else
    exit 0
fi
