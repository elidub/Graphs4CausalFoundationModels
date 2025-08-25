#!/usr/bin/env python3
"""
Simple Training Example - CausalPriorFitting

Demonstrates basic usage of all configuration modules together.
"""

import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from training.SetupTraining import SetupTraining
from training.configs import small_config
from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config
from priordata_processing.ExampleConfigs.BasicConfigs import default_dataset_config, default_preprocessing_config
from models.ExampleConfigs.SimplePFN_Configs import small_simplepfn_config


def create_configs():
    """Create simple configurations for quick training."""
    
    # Minimal SCM config
    scm_config = {
        **default_sampling_config,
        "num_nodes": {"value": 4},
        "graph_edge_prob": {"value": 0.3},
    }
    
    # Small dataset config  
    dataset_config = {
        **default_dataset_config,
        "dataset_size": {"value": 10_000},
        "max_number_samples": {"value": 1000},  # Increase this to accommodate larger datasets
        "max_number_features": {"value": 10},
        "number_samples_per_dataset": {
            "distribution": "uniform", 
            "distribution_parameters": {"low": 20, "high": 100}  # Keep within reasonable bounds
        }
    }
    
    # Simple preprocessing
    preprocessing_config = {
        **default_preprocessing_config,
        "train_fraction": {"value": 0.7},
    }
    
    # Model config with required num_features 
    # Note: num_features = max_number_features - 1 (because one feature becomes target Y)
    expected_features = dataset_config["max_number_features"]["value"] - 1
    model_config = {
        **small_simplepfn_config,
        "num_features": expected_features,  # 10 - 1 = 9 features after target removal
        "d_model": 64,
        "depth": 2,
    }
    
    # Training config
    training_config = {
        **small_config,
        "max_steps": {"value": 100},
        "batch_size": {"value": 4},
        "device": {"value": "cpu"},
        "wandb_project": {"value": None},
        "num_workers": {"value": 0},  # Avoid multiprocessing issues on CPU
        "precision": {"value": "32"},  # Use 32-bit precision for CPU
        "early_stopping_patience": {"value": 0},  # Disable early stopping for simple example
    }
    
    data_config = {
        'scm_config': scm_config,
        'preprocessing_config': preprocessing_config,
        'dataset_config': dataset_config
    }
    
    return model_config, data_config, training_config


def main():
    """Run simple training example."""
    
    print("Starting training example...")
    
    # Create configurations
    model_config, data_config, training_config = create_configs()
    
    # Initialize training
    setup = SetupTraining(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config
    )
    
    print(f"Model: {type(setup.model).__name__}")
    print(f"Train dataset size: {len(setup.train_dataset)}")
    
    # Train
    setup.run_training()
    
    # Save model
    setup.save_model("trained_model.pth")
    
    print("Training completed.")


if __name__ == "__main__":
    main()
