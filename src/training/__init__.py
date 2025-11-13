"""
SimplePFN Training Package.

This package provides a two-level training API for SimplePFN models:

1. Trainer: Low-level training class with explicit parameters
2. SetupTraining: High-level API that takes configuration dictionaries

Main Components:
- Trainer: Core training implementation with explicit parameters
- SetupTraining: Configuration-based training setup and execution
- configs: Training configuration management (follows priordata_processing pattern)
- utils: Utility functions for data loading and experiment management

Example Usage:

## Low-level API (Trainer):
```python
from training import Trainer
from models.SimplePFN import SimplePFNRegressor

# Create model and datasets explicitly
model = SimplePFNRegressor(**model_config)
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=1e-3,
    max_steps=10000,       # Step-based training
    max_epochs=1,          # Usually 1 for synthetic data
    batch_size=32
)
trained_model = trainer.fit()
```

## High-level API (SetupTraining):
```python
from training import SetupTraining
from training.configs import debug_config

# Setup with configurations (using new value/distribution pattern)
setup = SetupTraining(
    model_config='medium',
    data_config=data_config,
    training_config=debug_config  # Uses rigorous config structure
)
```
trained_model = setup.run_training()
```
"""

from .trainer import SimplePFNTrainer
# from .SetupTraining import SetupTraining, quick_train
from .configs import (
    get_training_config, 
    list_configs, 
    print_config_summary,
    extract_config_values,
    TRAINING_CONFIGS
)
# from .utils import (
#     SimplePFNDataset,
#     create_dataloaders,
#     count_parameters,
#     set_seed,
#     plot_training_history,
#     save_experiment_summary,
#     load_experiment_summary,
#     find_best_checkpoint,
#     compare_models,
#     print_model_comparison,
#     EarlyStopping
# )
# from .load_utils import (
#     load_model,
#     find_checkpoint_in_run,
#     find_run_directory
# )

__version__ = "2.0.0"
__author__ = "SimplePFN Team"

# Main exports
__all__ = [
    # Core training classes
    "SimplePFNTrainer",  # Low-level trainer with explicit parameters (renamed from Trainer)
    # "SetupTraining",     # High-level config-based setup
    # "quick_train",       # Convenience function for quick training
    
    # Configuration management
    "get_training_config",
    "list_configs", 
    "print_config_summary",
    "extract_config_values",
    "TRAINING_CONFIGS",
    
    # # Utilities
    # "SimplePFNDataset",
    # "create_dataloaders",
    # "count_parameters",
    # "set_seed",
    # "plot_training_history",
    # "save_experiment_summary",
    # "load_experiment_summary",
    # "find_best_checkpoint",
    
    # # Model loading utilities
    # "load_model",
    # "find_checkpoint_in_run",
    # "find_run_directory",
    # "compare_models",
    # "print_model_comparison",
    # "EarlyStopping",
]
