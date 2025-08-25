# SimplePFN Training Package

PyTorch Lightning-based training for SimplePFN models with synthetic causal data.

## Quick Start

```python
from training import SetupTraining
from training.configs import small_config

# Basic training setup
setup = SetupTraining(
    model_config='small',  # or custom config dict
    data_config={
        'scm_config': scm_config,
        'preprocessing_config': preprocessing_config, 
        'dataset_config': dataset_config
    },
    training_config=small_config
)
trained_model = setup.run_training()
```

## Configuration Structure

All configs use the rigorous pattern:
- Fixed values: `{"value": X}`
- Distributions: `{"distribution": "uniform", "distribution_parameters": {"low": 1, "high": 10}}`

## Available Configs

- `debug_config`: 10 steps, CPU, fast testing
- `small_config`: 1K steps, basic training
- `medium_config`: 10K steps, production ready
- `large_config`: 100K steps, full experiments

## Example Configs

See `configs.py` for complete examples or run:
```python
from training.configs import *
print_all_configs()  # View all available configurations
```

## Key Components

- **SetupTraining**: High-level training API
- **Trainer**: PyTorch Lightning wrapper
- **configs.py**: Pre-defined training configurations

## Notes

- Training focuses on steps rather than epochs (ideal for synthetic data)
- Model expects `num_features = max_number_features - 1` (one feature becomes target)
- Use `early_stopping_patience: 0` to disable early stopping
