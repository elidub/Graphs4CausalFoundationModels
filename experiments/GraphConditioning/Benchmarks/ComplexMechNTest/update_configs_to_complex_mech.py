#!/usr/bin/env python3
"""
Script to update all LinGausIDK configs to ComplexMechIDK configs.

Changes made:
1. Update experiment_name and description to reflect ComplexMech
2. Update scm_config to use complex mechanisms:
   - xgboost_prob: categorical distribution with 0.0-0.3 probabilities
   - mlp_nonlins: "tabicl" instead of "id"
   - mlp_num_hidden_layers: categorical distribution with 0-3 layers
   - mlp_activation_mode: equal probabilities for pre/post/mixed_in
   - mlp_use_batch_norm: 50/50 instead of always true
   - noise_mixture_proportions: [0.33, 0.33, 0.34] instead of [1.0, 0.0, 0.0]
   - use_exogenous_mechanisms: true instead of false
"""

import os
import yaml
from pathlib import Path


# Define the complex mechanism scm_config updates
COMPLEX_MECH_SCM_UPDATES = {
    # XGBoost probability: allow some XGBoost-based mechanisms
    "xgboost_prob": {
        "distribution": "categorical",
        "distribution_parameters": {
            "choices": [0.0, 0.1, 0.2, 0.3],
            "probabilities": [0.9, 0.1, 0.01, 0.001]
        }
    },
    
    # MLP nonlinearities: use tabicl instead of identity
    "mlp_nonlins": {
        "value": "tabicl"
    },
    
    # MLP hidden layers: allow 0-3 layers with distribution
    "mlp_num_hidden_layers": {
        "distribution": "categorical",
        "distribution_parameters": {
            "choices": [0, 1, 2, 3],
            "probabilities": [0.875, 0.1, 0.025, 0.01]
        }
    },
    
    # MLP activation mode: equal distribution
    "mlp_activation_mode": {
        "distribution": "categorical",
        "distribution_parameters": {
            "choices": ["pre", "post", "mixed_in"],
            "probabilities": [0.3, 0.3, 0.3]
        }
    },
    
    # MLP batch norm: 50/50 instead of always true
    "mlp_use_batch_norm": {
        "distribution": "categorical",
        "distribution_parameters": {
            "choices": [True, False],
            "probabilities": [0.5, 0.5]
        }
    },
    
    # Noise mixture: use all three noise types
    "noise_mixture_proportions": {
        "value": [0.33, 0.33, 0.34]
    },
    
    # Use exogenous mechanisms
    "use_exogenous_mechanisms": {
        "value": True
    }
}


def update_experiment_name_and_description(config: dict) -> dict:
    """Update experiment_name and description to reflect ComplexMech."""
    if "experiment_name" in config:
        config["experiment_name"] = config["experiment_name"].replace("lingaus", "complexmech")
        config["experiment_name"] = config["experiment_name"].replace("LinGaus", "ComplexMech")
    
    if "description" in config:
        config["description"] = config["description"].replace("Linear-Gaussian", "Complex Mechanism")
        config["description"] = config["description"].replace("linear-gaussian", "complex mechanism")
        config["description"] = config["description"].replace("LinGaus", "ComplexMech")
    
    return config


def update_scm_config(config: dict) -> dict:
    """Update scm_config to use complex mechanisms."""
    if "scm_config" not in config:
        return config
    
    scm_config = config["scm_config"]
    
    for key, value in COMPLEX_MECH_SCM_UPDATES.items():
        if key in scm_config:
            scm_config[key] = value
    
    return config


def process_yaml_file(filepath: Path) -> None:
    """Process a single YAML file and update it."""
    print(f"Processing: {filepath}")
    
    # Read the file
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        print(f"  Skipping empty file: {filepath}")
        return
    
    # Apply updates
    config = update_experiment_name_and_description(config)
    config = update_scm_config(config)
    
    # Write back
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"  Updated: {filepath}")


def main():
    # Get the configs directory
    script_dir = Path(__file__).parent
    configs_dir = script_dir / "configs"
    
    if not configs_dir.exists():
        print(f"Error: configs directory not found at {configs_dir}")
        return
    
    # Find all YAML files
    yaml_files = list(configs_dir.rglob("*.yaml"))
    print(f"Found {len(yaml_files)} YAML files to update\n")
    
    # Process each file
    for yaml_file in sorted(yaml_files):
        process_yaml_file(yaml_file)
    
    print(f"\n✅ Successfully updated {len(yaml_files)} config files!")
    print("\nNext steps:")
    print("1. Rename LingausBenchmarkIDK.py to ComplexMechBenchmarkIDK.py")
    print("2. Update class name inside the file")
    print("3. Update generate_benchmark_data.py and generate_all_variants_data.py")
    print("4. Clear data_cache and regenerate data")


if __name__ == "__main__":
    main()
