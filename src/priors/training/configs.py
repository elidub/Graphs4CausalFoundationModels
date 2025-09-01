"""
Training configurations for SimplePFN experiments.

This module contains training hyperparameter configurations for SimplePFN models
following the rigorous pattern established by SCMHyperparameterSampler and 
priordata_processing configurations.

Each parameter can either have:
- A "value" key for fixed values
- A "distribution" key with "distribution_parameters" for sampling

Note: PFN training focuses on steps rather than epochs since all data is synthetic.
We typically train for one epoch but with many training steps.
"""

# Debug configuration for quick testing and development
debug_config = {
    # Model Configuration
    "model_config_name": {
        "value": "small"
    },
    
    # Training Hyperparameters (step-based)
    "learning_rate": {
        "value": 1e-3
    },
    "weight_decay": {
        "value": 1e-4
    },
    "max_steps": {
        "value": 100      # Focus on steps instead of epochs
    },
    "max_epochs": {
        "value": 1        # Usually 1 for synthetic data
    },
    "eta_min": {
        "value": 1e-6
    },
    
    # Data Loading
    "batch_size": {
        "value": 8
    },
    "num_workers": {
        "value": 0
    },
    
    # Early Stopping (in steps)
    "early_stopping_patience": {
        "value": 50
    },
    
    # Hardware Configuration
    "device": {
        "value": "cpu"
    },
    "precision": {
        "value": "32"
    },
    
    # Experiment Configuration
    "experiment_name": {
        "value": "debug_experiment"
    },
    "save_dir": {
        "value": "./experiments/debug"
    },
    "log_every_n_steps": {
        "value": 5
    },
    
    # Checkpointing (step-based)
    "checkpoint_every_n_steps": {
        "value": 25
    },
    
    # Weights & Biases
    "wandb_project": {
        "value": "simplepfn-debug"
    },
    "wandb_tags": {
        "value": ["debug", "small"]
    },
    "wandb_notes": {
        "value": "Quick debugging experiment with PyTorch Lightning"
    },
    
    # Gradient Settings
    "gradient_clip_val": {
        "value": 1.0
    },
}

# Small configuration for quick experiments
small_config = {
    # Model Configuration
    "model_config_name": {
        "value": "small"
    },
    
    # Training Hyperparameters (step-based)
    "learning_rate": {
        "value": 1e-3
    },
    "weight_decay": {
        "value": 1e-4
    },
    "max_steps": {
        "value": 5000     # Focus on steps
    },
    "max_epochs": {
        "value": 1        # Usually 1 for synthetic data
    },
    "eta_min": {
        "value": 1e-6
    },
    
    # Data Loading
    "batch_size": {
        "value": 16
    },
    "num_workers": {
        "value": 2
    },
    
    # Early Stopping (in steps)
    "early_stopping_patience": {
        "value": 500
    },
    
    # Hardware Configuration
    "device": {
        "value": "auto"
    },
    "precision": {
        "value": "16-mixed"
    },
    
    # Experiment Configuration
    "experiment_name": {
        "value": "small_experiment"
    },
    "save_dir": {
        "value": "./experiments/small"
    },
    "log_every_n_steps": {
        "value": 10
    },
    
    # Checkpointing (step-based)
    "checkpoint_every_n_steps": {
        "value": 250
    },
    
    # Weights & Biases
    "wandb_project": {
        "value": "simplepfn-experiments"
    },
    "wandb_tags": {
        "value": ["small", "quick"]
    },
    "wandb_notes": {
        "value": "Small-scale experiment for rapid iteration with PyTorch Lightning"
    },
    
    # Gradient Settings
    "gradient_clip_val": {
        "value": 1.0
    },
}

# Medium configuration for standard experiments
medium_config = {
    # Model Configuration
    "model_config_name": {
        "value": "medium"
    },
    
    # Training Hyperparameters (step-based)
    "learning_rate": {
        "value": 8e-4
    },
    "weight_decay": {
        "value": 1e-4
    },
    "max_steps": {
        "value": 10000    # Focus on steps
    },
    "max_epochs": {
        "value": 1        # Usually 1 for synthetic data
    },
    "eta_min": {
        "value": 1e-6
    },
    
    # Data Loading
    "batch_size": {
        "value": 32
    },
    "num_workers": {
        "value": 4
    },
    
    # Early Stopping (in steps)
    "early_stopping_patience": {
        "value": 1000
    },
    
    # Hardware Configuration
    "device": {
        "value": "auto"
    },
    "precision": {
        "value": "16-mixed"
    },
    
    # Experiment Configuration
    "experiment_name": {
        "value": "medium_experiment"
    },
    "save_dir": {
        "value": "./experiments/medium"
    },
    "log_every_n_steps": {
        "value": 20
    },
    
    # Checkpointing (step-based)
    "checkpoint_every_n_steps": {
        "value": 500
    },
    
    # Weights & Biases
    "wandb_project": {
        "value": "simplepfn-experiments"
    },
    "wandb_tags": {
        "value": ["medium", "standard"]
    },
    "wandb_notes": {
        "value": "Standard medium-scale experiment with PyTorch Lightning"
    },
    
    # Gradient Settings
    "gradient_clip_val": {
        "value": 1.0
    },
}

# Large configuration for comprehensive experiments
large_config = {
    # Model Configuration
    "model_config_name": {
        "value": "medium"  # Can be changed to "large" when available
    },
    
    # Training Hyperparameters (step-based)
    "learning_rate": {
        "value": 5e-4
    },
    "weight_decay": {
        "value": 2e-4
    },
    "max_steps": {
        "value": 20000    # Focus on steps
    },
    "max_epochs": {
        "value": 1        # Usually 1 for synthetic data
    },
    "eta_min": {
        "value": 1e-7
    },
    
    # Data Loading
    "batch_size": {
        "value": 64
    },
    "num_workers": {
        "value": 8
    },
    
    # Early Stopping (in steps)
    "early_stopping_patience": {
        "value": 2000
    },
    
    # Hardware Configuration
    "device": {
        "value": "auto"
    },
    "precision": {
        "value": "16-mixed"
    },
    
    # Experiment Configuration
    "experiment_name": {
        "value": "large_experiment"
    },
    "save_dir": {
        "value": "./experiments/large"
    },
    "log_every_n_steps": {
        "value": 50
    },
    
    # Checkpointing (step-based)
    "checkpoint_every_n_steps": {
        "value": 1000
    },
    
    # Weights & Biases
    "wandb_project": {
        "value": "simplepfn-experiments"
    },
    "wandb_tags": {
        "value": ["large", "comprehensive"]
    },
    "wandb_notes": {
        "value": "Large-scale comprehensive experiment with PyTorch Lightning"
    },
    
    # Gradient Settings
    "gradient_clip_val": {
        "value": 1.0
    },
}

# Fast configuration for rapid prototyping
fast_config = {
    # Model Configuration
    "model_config_name": {
        "value": "small"
    },
    
    # Training Hyperparameters (step-based)
    "learning_rate": {
        "value": 2e-3
    },
    "weight_decay": {
        "value": 5e-5
    },
    "max_steps": {
        "value": 1000     # Quick training
    },
    "max_epochs": {
        "value": 1        # Usually 1 for synthetic data
    },
    "eta_min": {
        "value": 1e-5
    },
    
    # Data Loading
    "batch_size": {
        "value": 32
    },
    "num_workers": {
        "value": 4
    },
    
    # Early Stopping (in steps)
    "early_stopping_patience": {
        "value": 200
    },
    
    # Hardware Configuration
    "device": {
        "value": "auto"
    },
    "precision": {
        "value": "16-mixed"
    },
    
    # Experiment Configuration
    "experiment_name": {
        "value": "fast_experiment"
    },
    "save_dir": {
        "value": "./experiments/fast"
    },
    "log_every_n_steps": {
        "value": 5
    },
    
    # Checkpointing (step-based)
    "checkpoint_every_n_steps": {
        "value": 100
    },
    
    # Weights & Biases
    "wandb_project": {
        "value": "simplepfn-prototyping"
    },
    "wandb_tags": {
        "value": ["fast", "prototype"]
    },
    "wandb_notes": {
        "value": "Fast prototyping experiment with PyTorch Lightning"
    },
    
    # Gradient Settings
    "gradient_clip_val": {
        "value": 1.0
    },
}

# Precision configuration for final model training
precision_config = {
    # Model Configuration
    "model_config_name": {
        "value": "medium"
    },
    
    # Training Hyperparameters (step-based)
    "learning_rate": {
        "value": 3e-4
    },
    "weight_decay": {
        "value": 1e-4
    },
    "max_steps": {
        "value": 30000    # Long training for precision
    },
    "max_epochs": {
        "value": 1        # Usually 1 for synthetic data
    },
    "eta_min": {
        "value": 1e-8
    },
    
    # Data Loading
    "batch_size": {
        "value": 16
    },
    "num_workers": {
        "value": 4
    },
    
    # Early Stopping (in steps)
    "early_stopping_patience": {
        "value": 3000
    },
    
    # Hardware Configuration
    "device": {
        "value": "auto"
    },
    "precision": {
        "value": "32"     # Higher precision
    },
    
    # Experiment Configuration
    "experiment_name": {
        "value": "precision_experiment"
    },
    "save_dir": {
        "value": "./experiments/precision"
    },
    "log_every_n_steps": {
        "value": 20
    },
    
    # Checkpointing (step-based)
    "checkpoint_every_n_steps": {
        "value": 1500
    },
    
    # Weights & Biases
    "wandb_project": {
        "value": "simplepfn-final"
    },
    "wandb_tags": {
        "value": ["precision", "final"]
    },
    "wandb_notes": {
        "value": "High-precision final model training with PyTorch Lightning"
    },
    
    # Gradient Settings
    "gradient_clip_val": {
        "value": 1.0
    },
}

# Dictionary for easy access to all configurations
TRAINING_CONFIGS = {
    "debug": debug_config,
    "small": small_config,
    "medium": medium_config,
    "large": large_config,
    "fast": fast_config,
    "precision": precision_config,
}


def get_training_config(name: str):
    """
    Get a training configuration by name.
    
    Args:
        name: Configuration name
        
    Returns:
        Configuration dictionary (copy to prevent modification)
        
    Raises:
        KeyError: If configuration name is not found
    """
    if name not in TRAINING_CONFIGS:
        available = list(TRAINING_CONFIGS.keys())
        raise KeyError(f"Unknown config '{name}'. Available: {available}")
    
    return TRAINING_CONFIGS[name].copy()


def extract_config_values(config: dict) -> dict:
    """
    Extract values from config following the value/distribution pattern.
    
    Args:
        config: Configuration dictionary with value/distribution structure
        
    Returns:
        Dictionary with extracted values for direct use
    """
    extracted = {}
    for key, param_config in config.items():
        if "value" in param_config:
            extracted[key] = param_config["value"]
        elif "distribution" in param_config:
            # For now, we don't support sampling in training configs
            # This would require implementing a TrainingHyperparameterSampler similar to SCMHyperparameterSampler
            raise ValueError(f"Parameter {key} uses distribution sampling, which is not yet implemented for training configs")
        else:
            raise ValueError(f"Parameter {key} must specify 'value' or 'distribution'")
    
    return extracted


def list_configs():
    """List all available training configurations."""
    return list(TRAINING_CONFIGS.keys())


def print_config_summary():
    """Print a summary of all training configurations."""
    print("=== SimplePFN Training Configurations ===\n")
    
    for name, config in TRAINING_CONFIGS.items():
        print(f"--- {name.upper()} CONFIG ---")
        print(f"  Model: {config['model_config_name']['value']}")
        print(f"  Learning Rate: {config['learning_rate']['value']}")
        print(f"  Batch Size: {config['batch_size']['value']}")
        print(f"  Max Steps: {config['max_steps']['value']}")
        print(f"  Max Epochs: {config['max_epochs']['value']}")
        print(f"  Early Stopping: {config['early_stopping_patience']['value']} steps")
        print(f"  Precision: {config['precision']['value']}")
        print(f"  Save Dir: {config['save_dir']['value']}")
        print()


if __name__ == "__main__":
    print_config_summary()
