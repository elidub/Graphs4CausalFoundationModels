# SimplePFN Training Package

A comprehensive, two-tier training API for SimplePFN models with **PyTorch Lightning**, Weights & Biases logging, and step-based training for synthetic data.

## 🏗️ Architecture Overview

This package provides a **two-tier training architecture** using **PyTorch Lightning**:

1. **Trainer** (Low-level): PyTorch Lightning-based implementation with explicit parameter control
2. **SetupTraining** (High-level): Configuration-based API for production and experiments

### � Key Features for PFN Training

- **Step-based training**: Focus on training steps rather than epochs (ideal for synthetic data)
- **PyTorch Lightning**: Robust training with automatic mixed precision, gradient clipping, and checkpointing
- **Synthetic data optimized**: Typically train for 1 epoch with many steps of generated data
- **Lightning callbacks**: Built-in early stopping, model checkpointing, and W&B logging

## �🚀 Quick Start

### High-Level API (Recommended)

```python
from training import SetupTraining, quick_train
from training.configs import debug_config, small_config

# Option 1: Using rigorous configuration structure (new pattern)
setup = SetupTraining(
    model_config='medium',  # or direct config dict
    data_config={
        'scm_config': scm_config,
        'preprocessing_config': preprocessing_config,
        'dataset_config': dataset_config
    },
    training_config=debug_config  # Uses rigorous value/distribution structure
)
trained_model = setup.run_training()

# Option 2: Custom config with rigorous pattern
custom_config = {
    'learning_rate': {'value': 1e-3},
    'max_steps': {'value': 10000},    # Focus on steps, not epochs
    'max_epochs': {'value': 1},       # Usually 1 for synthetic data
    'batch_size': {'value': 32},
    'precision': {'value': '16-mixed'}
}
setup = SetupTraining(
    model_config='small',
    data_config=data_config,
    training_config=custom_config
)

# Option 3: Quick training using predefined configs
trained_model = quick_train(
    model_config='small',
    data_config=data_config,
    training_config=small_config  # Predefined config with rigorous structure
)
```

### Low-Level API (For Research)

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
    max_epochs=1,          # Typically 1 for synthetic data
    batch_size=32,
    device='cuda',
    precision='16-mixed',  # Lightning format
    use_wandb=True
)
trained_model = trainer.fit()
```

## 📁 Package Structure

```
training/
├── Trainer.py          # Low-level training implementation
├── SetupTraining.py    # High-level configuration-based API
├── configs.py          # Training configurations
├── utils.py           # Utility functions
├── __init__.py        # Package exports
└── README.md          # This file
```

## 🔧 Core Components

### Trainer (Low-Level API)

The core training implementation with explicit parameter control:

```python
from training import Trainer

```

**Key Features:**
- **PyTorch Lightning backbone**: Robust training with automatic device placement
- **Step-based focus**: Perfect for synthetic data generation workflows  
- **Direct parameter control**: All hyperparameters explicitly specified
- **Lightning callbacks**: Built-in checkpointing, early stopping, and logging
- **Mixed precision**: Automatic support via Lightning's precision parameter
- **Built-in evaluation and prediction methods**

**Methods:**
- `fit()`: Train the model using PyTorch Lightning trainer
- `evaluate(dataset)`: Evaluate model on a dataset
- `predict(dataset)`: Generate predictions
- `save_model(path)`: Save trained model
- `load_model(path)`: Load a saved model

### SetupTraining (High-Level API)

Configuration-based training setup that integrates with the entire SimplePFN pipeline:

```python
from training import SetupTraining

# Initialize with configs
setup = SetupTraining(
    model_config='medium',  # or config dict
    data_config=data_config,
    training_config={
        'max_steps': 10000,    # Step-based training
        'max_epochs': 1,       # Usually 1 for synthetic data
        'learning_rate': 1e-3,
        'batch_size': 32,
        'precision': '16-mixed'  # Lightning format
    }
)

# Run complete training pipeline
trained_model = setup.run_training()

# Evaluate on test data
metrics = setup.evaluate_model(test_config)
```

**Key Features:**
- **Automatic dataset creation** from `priordata_processing` pipeline
- **Model creation** from `models` configurations
- **Step-based training** optimized for synthetic data
- **PyTorch Lightning integration** with minimal configuration
- **Complete training setup** with minimal code

**Methods:**
- `run_training()`: Complete Lightning-based training pipeline
- `setup_trainer()`: Setup low-level trainer from configs
- `evaluate_model()`: Evaluate with automatic dataset creation
- `predict()`: Make predictions with automatic preprocessing
- `save_experiment()`: Save complete experiment state

## 🔧 Configuration Management

### Rigorous Configuration Structure

The training configs now follow the same **rigorous pattern** as `priordata_processing` and `SCMHyperparameterSampler`:

```python
from training.configs import debug_config, extract_config_values

# All parameters use value/distribution structure
debug_config = {
    'learning_rate': {'value': 1e-3},           # Fixed value
    'batch_size': {'value': 8},                 # Fixed value
    'max_steps': {'value': 100},                # Fixed value
    # In the future, could support:
    # 'learning_rate': {                        # Sampled value
    #     'distribution': 'uniform',
    #     'distribution_parameters': {'low': 1e-4, 'high': 1e-2}
    # }
}

# Extract values for use
training_params = extract_config_values(debug_config)
# training_params = {'learning_rate': 1e-3, 'batch_size': 8, ...}
```

### Training Configurations

```python
from training import get_training_config, list_configs

# List available configurations
list_configs()

# Load a specific configuration
config = get_training_config('medium')

# Print configuration summary
from training import print_config_summary
print_config_summary()
```

Available configurations (all step-based):
- `'small'`: Quick training with 5,000 steps
- `'medium'`: Balanced training with 10,000 steps  
- `'large'`: Extensive training with 20,000 steps
- `'debug'`: Fast training with 100 steps for debugging
- `'fast'`: Rapid prototyping with 1,000 steps
- `'precision'`: High-precision training with 30,000 steps

## 📊 Step-Based Training for PFNs

### Why Steps Instead of Epochs?

For PFN training with synthetic data:
- **All data is synthetic**: Generated on-the-fly during training
- **No fixed dataset size**: Can generate unlimited samples
- **Step-focused metrics**: Training progress measured by optimization steps
- **Typical workflow**: 1 epoch with many training steps

### Working with priordata_processing

The high-level API automatically integrates with the data generation pipeline:

```python
# Data configuration for step-based training
data_config = {
    'scm_config': {
        'n_nodes': 5,
        'edge_prob': 0.3,
        'function_family': 'mlp'
    },
    'preprocessing_config': {
        'normalize': True,
        'add_noise': False
    },
    'dataset_config': {
        'n_samples': 50000,  # Large synthetic dataset
        'test_size': 0.2,
        'val_size': 0.1
    }
}

# Training focused on steps
training_config = {
    'max_steps': 10000,     # Main training parameter
    'max_epochs': 1,        # Usually 1
    'batch_size': 32,
    'learning_rate': 1e-3,
    'precision': '16-mixed'  # Lightning format
}

setup = SetupTraining(
    model_config='medium',
    data_config=data_config,
    training_config=training_config
)
```

### Working with existing datasets

For the low-level API, you can use existing datasets:

```python
from training.utils import SimplePFNDataset

# Create datasets from your data
train_dataset = SimplePFNDataset(train_samples, train_targets)
val_dataset = SimplePFNDataset(val_samples, val_targets)

# Use with Trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    # ... other parameters
)
```

## 🧪 Experiment Management

### Weights & Biases Integration

```python
# Training with W&B logging (automatic detection)
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    use_wandb=True,
    wandb_project='SimplePFN-experiments',
    wandb_name='my-experiment'
)

# High-level API with W&B
setup = SetupTraining(
    model_config='medium',
    data_config=data_config,
    training_config={
        'use_wandb': True,
        'wandb_project': 'my-project'
    }
)
```

### Model Checkpointing

```python
# Automatic checkpointing during training
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    save_dir='./checkpoints',
    save_every_n_epochs=10
)

# Load best checkpoint
best_model = trainer.load_model('./checkpoints/best_model.pt')
```

## 🔧 Advanced Usage

### Custom Loss Functions

```python
import torch.nn as nn

# Using custom loss with Trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    criterion=nn.L1Loss(),  # Custom loss function
    # ... other parameters
)
```

### Learning Rate Scheduling

```python
# Built-in cosine annealing (default)
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=1e-3,
    use_scheduler=True,  # Cosine annealing
    # ... other parameters
)
```

### Mixed Precision Training

```python
# Enable automatic mixed precision for faster training
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    use_amp=True,  # Automatic Mixed Precision
    # ... other parameters
)
```

## 🛠️ Utility Functions

```python
from training.utils import (
    count_parameters,
    set_seed,
    plot_training_history,
    save_experiment_summary,
    EarlyStopping
)

# Count model parameters
print(f"Model has {count_parameters(model):,} parameters")

# Set random seed for reproducibility
set_seed(42)

# Plot training curves
plot_training_history(trainer.history)

# Early stopping
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
```

## 📋 Complete Examples

### Research Workflow (Low-Level API)

```python
from training import Trainer
from models.SimplePFN import SimplePFNRegressor
from training.utils import SimplePFNDataset, set_seed

# Set seed for reproducibility
set_seed(42)

# Create model
model = SimplePFNRegressor(
    input_dim=128,
    hidden_dim=256,
    num_layers=6
)

# Create datasets
train_dataset = SimplePFNDataset(train_samples, train_targets)
val_dataset = SimplePFNDataset(val_samples, val_targets)

# Setup trainer with explicit control
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=1e-3,
    max_epochs=100,
    batch_size=32,
    device='cuda',
    use_amp=True,
    use_wandb=True,
    wandb_project='research-experiments',
    save_dir='./models/',
    early_stopping_patience=15
)

# Train model
best_model = trainer.fit()

# Evaluate
test_dataset = SimplePFNDataset(test_samples, test_targets)
test_metrics = trainer.evaluate(test_dataset)
print(f"Test RMSE: {test_metrics['rmse']:.4f}")
```

### Production Workflow (High-Level API)

```python
from training import SetupTraining, quick_train

# Option 1: Full configuration
data_config = {
    'scm_config': {...},
    'preprocessing_config': {...},
    'dataset_config': {...}
}

training_config = {
    'learning_rate': 1e-3,
    'max_epochs': 100,
    'batch_size': 64,
    'use_wandb': True,
    'wandb_project': 'production-models'
}

setup = SetupTraining(
    model_config='large',
    data_config=data_config,
    training_config=training_config
)

# Run complete pipeline
trained_model = setup.run_training()

# Option 2: Quick training for prototyping
model = quick_train(
    model_config='small',
    data_config=data_config,
    max_epochs=20
)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or disable `use_amp`
2. **Slow training**: Enable `use_amp=True` and ensure `device='cuda'`
3. **Weights & Biases errors**: Set `use_wandb=False` or install with `pip install wandb`
4. **Import errors**: Ensure all dependencies are installed and paths are correct

### Performance Tips

- Use `use_amp=True` for faster training on modern GPUs
- Increase `batch_size` until you hit memory limits
- Use `num_workers > 0` in data loading for multi-core systems
- Enable W&B logging only when needed to reduce overhead

## 📚 API Reference

### Trainer Class

```python
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        batch_size: int = 32,
        device: str = 'auto',
        use_amp: bool = True,
        use_wandb: bool = False,
        wandb_project: str = None,
        wandb_name: str = None,
        save_dir: str = None,
        save_every_n_epochs: int = None,
        early_stopping_patience: int = None,
        criterion: torch.nn.Module = None,
        optimizer_class: torch.optim.Optimizer = None,
        use_scheduler: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    )
    
    def fit(self) -> torch.nn.Module
    def evaluate(self, dataset: Dataset) -> Dict[str, float]
    def predict(self, dataset: Dataset) -> np.ndarray
    def save_model(self, path: str) -> None
    def load_model(self, path: str) -> torch.nn.Module
```

### SetupTraining Class

```python
class SetupTraining:
    def __init__(
        self,
        model_config: Union[str, Dict],
        data_config: Dict,
        training_config: Dict = None
    )
    
    def run_training(self) -> torch.nn.Module
    def setup_trainer(self) -> Trainer
    def evaluate_model(self, test_config: Dict = None) -> Dict[str, float]
    def predict(self, predict_config: Dict) -> np.ndarray
    def save_experiment(self, path: str) -> None
    
    @staticmethod
    def from_config_files(
        model_config_path: str,
        data_config_path: str,
        training_config_path: str = None
    ) -> 'SetupTraining'

def quick_train(
    model_config: Union[str, Dict],
    data_config: Dict,
    max_epochs: int = 50,
    **kwargs
) -> torch.nn.Module
```

---

## 📄 License

This package is part of the SimplePFN project. See the main repository for license information.

## 🤝 Contributing

Please see the main repository's contributing guidelines for information on how to contribute to this package.
- 🔥 Pure PyTorch implementation (no Lightning dependency)
- 📊 Automatic Weights & Biases logging
- 💾 Smart checkpointing and model saving
- ⏰ Early stopping with patience
- 📈 Cosine annealing learning rate scheduling
- 🎯 Gradient clipping and precision control
- 📱 Comprehensive experiment tracking

### Training Configurations

Pre-defined configurations for different experiment types:

```python
from training import get_training_config, list_configs

# See available configurations
configs = list_configs()
print(configs)  # ['debug', 'small', 'medium', 'large', 'fast', 'precision']

# Load a specific configuration
config = get_training_config('medium')

# Customize configuration
config['learning_rate'] = 5e-4
config['max_epochs'] = 150
```

**Available Configurations:**

| Config | Model | Epochs | Batch Size | LR | Use Case |
|--------|-------|--------|------------|-----|----------|
| `debug` | small | 10 | 8 | 1e-3 | Quick testing & debugging |
| `small` | small | 50 | 16 | 1e-3 | Fast experiments |
| `medium` | medium | 100 | 32 | 8e-4 | Standard training |
| `large` | medium | 200 | 64 | 5e-4 | Comprehensive experiments |
| `fast` | small | 20 | 32 | 2e-3 | Rapid prototyping |
| `precision` | medium | 300 | 16 | 3e-4 | Final model training |

### Utility Functions

Helper functions for data management and experiment analysis:

```python
from training import (
    create_dataloaders, 
    set_seed, 
    count_parameters,
    plot_training_history,
    compare_models
)

# Set random seed for reproducibility
set_seed(42)

# Count model parameters
param_count = count_parameters(model)
print(f"Total parameters: {param_count['total']:,}")

# Create data loaders
dataloaders = create_dataloaders(
    train_samples, train_targets,
    val_samples, val_targets,
    batch_size=32
)

# Plot training curves
plot_training_history(trainer.training_history, 'training_plot.png')

# Compare multiple experiments
comparison = compare_models(['./experiments/exp1', './experiments/exp2'])
print_model_comparison(comparison)
```

## ⚙️ Configuration Reference

### Model Configuration
```python
config = {
    "model_config_name": "medium",  # SimplePFN model size: small, medium, large
}
```

### Training Hyperparameters
```python
config = {
    "learning_rate": 8e-4,          # Initial learning rate
    "weight_decay": 1e-4,           # L2 regularization
    "max_epochs": 100,              # Maximum training epochs
    "eta_min": 5e-7,                # Minimum LR for cosine annealing
}
```

### Data Loading
```python
config = {
    "batch_size": 32,               # Training batch size
    "num_workers": 4,               # Number of data loading processes
}
```

### Early Stopping
```python
config = {
    "early_stopping_patience": 15,  # Epochs to wait before stopping
}
```

### Hardware Settings
```python
config = {
    "accelerator": "auto",          # "cpu", "gpu", or "auto"
    "devices": "auto",              # Device selection
    "precision": "16-mixed",        # "16-mixed", "32", or "64"
}
```

### Experiment Management
```python
config = {
    "experiment_name": "my_experiment",     # Experiment identifier
    "save_dir": "./experiments/my_exp",     # Save directory
    "log_every_n_steps": 25,               # Logging frequency
}
```

### Checkpointing
```python
config = {
    "checkpoint_every_n_epochs": 10,       # Checkpoint frequency
    "checkpoint_top_k": 5,                 # Number of checkpoints to keep
}
```

### Weights & Biases
```python
config = {
    "wandb_project": "simplepfn-experiments",  # W&B project name
    "wandb_tags": ["medium", "experiment"],    # Experiment tags
    "wandb_notes": "Description of experiment", # Experiment notes
}
```

### Advanced Settings
```python
config = {
    "gradient_clip_val": 1.0,              # Gradient clipping threshold
}
```

## 📊 Weights & Biases Integration

The trainer automatically logs comprehensive metrics to W&B:

**Automatic Logging:**
- Training and validation loss curves
- Learning rate schedule
- Model gradients and parameters
- Hardware utilization
- Experiment configuration

**Custom Logging:**
```python
import wandb

# Additional custom logging
wandb.log({
    "custom_metric": value,
    "epoch": epoch
})
```

**W&B Configuration:**
- Set project name with `wandb_project`
- Add experiment tags with `wandb_tags`
- Include notes with `wandb_notes`
- Automatic model watching and gradient logging

## 🔄 Advanced Usage

### Custom Training Loop

```python
from training import SimpleTrainer

class CustomTrainer(SimpleTrainer):
    def _train_epoch(self, dataloader):
        # Custom training logic
        return super()._train_epoch(dataloader)
    
    def _validate_epoch(self, dataloader):
        # Custom validation logic
        return super()._validate_epoch(dataloader)

# Use custom trainer
trainer = CustomTrainer(config)
```

### Resume Training

```python
# Resume from checkpoint
trainer = SimpleTrainer(config)
model = trainer.fit(
    train_loader, 
    val_loader,
    resume_from_checkpoint='./experiments/my_exp/checkpoint_epoch_50.pt'
)
```

### Model Analysis

```python
from training import count_parameters, find_best_checkpoint

# Analyze model complexity
param_info = count_parameters(model)
print(f"Trainable parameters: {param_info['trainable']:,}")

# Find best checkpoint
best_checkpoint = find_best_checkpoint('./experiments/my_exp')
print(f"Best checkpoint: {best_checkpoint}")
```

### Experiment Comparison

```python
from training import compare_models, print_model_comparison

# Compare multiple experiments
experiments = [
    './experiments/small_lr_001',
    './experiments/medium_lr_0008',
    './experiments/large_lr_0005'
]

comparison = compare_models(experiments)
print_model_comparison(comparison)
```

## 📈 Monitoring and Debugging

### Training History

```python
# Access training history
history = trainer.training_history
print(f"Best validation loss: {min(history['val_loss'])}")

# Plot training curves
from training import plot_training_history
plot_training_history(history, save_path='training_curves.png')
```

### Model Inspection

```python
# Model parameter count
from training import count_parameters
params = count_parameters(trainer.model)
print(f"Model has {params['total']:,} parameters")

# Check device placement
print(f"Model device: {next(trainer.model.parameters()).device}")
```

### Debugging Configuration

```python
# Use debug config for quick iterations
debug_config = get_training_config('debug')
debug_config.update({
    'max_epochs': 3,
    'log_every_n_steps': 1,
    'early_stopping_patience': 2
})

trainer = SimpleTrainer(debug_config)
```

## 🛠️ Best Practices

### 1. Reproducibility
```python
from training import set_seed

# Always set seed for reproducible results
set_seed(42)
```

### 2. Configuration Management
```python
# Start with a base config and modify
config = get_training_config('medium')
config.update({
    'learning_rate': 5e-4,
    'experiment_name': 'medium_lr_5e4',
    'wandb_tags': ['medium', 'lr_experiment']
})
```

### 3. Experiment Organization
```python
# Use descriptive experiment names and directories
config['experiment_name'] = 'SimplePFN_medium_lr5e4_bs32'
config['save_dir'] = f"./experiments/{config['experiment_name']}"
```

### 4. Data Preparation
```python
# Use utility functions for consistent data loading
dataloaders = create_dataloaders(
    train_samples=train_data,
    train_targets=train_targets,
    val_samples=val_data,
    val_targets=val_targets,
    batch_size=config['batch_size'],
    num_workers=config['num_workers']
)
```

### 5. Model Evaluation
```python
# Always evaluate on test set
test_metrics = trainer.evaluate(test_loader)

# Save experiment results
from training import save_experiment_summary
save_experiment_summary(config, test_metrics, config['save_dir'])
```

## 🚀 Migration from Lightning

If you're migrating from the old PyTorch Lightning implementation:

### Old Lightning Code:
```python
from training import SimplePFNTrainer  # Lightning-based

trainer = SimplePFNTrainer(
    model_config=model_config,
    learning_rate=1e-3,
    max_epochs=100
)
model = trainer.fit(train_dl, val_dl)
```

### New SimpleTrainer Code:
```python
from training import SimpleTrainer, get_training_config

config = get_training_config('medium')
config.update({
    'learning_rate': 1e-3,
    'max_epochs': 100
})

trainer = SimpleTrainer(config)
model = trainer.fit(train_dl, val_dl)
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# Reduce batch size or use gradient accumulation
config['batch_size'] = 16  # Reduce from 32
```

**W&B Login Issues:**
```bash
wandb login
# Or set WANDB_MODE=offline for offline logging
```

**Model Not Learning:**
```python
# Check learning rate and use debug config
debug_config = get_training_config('debug')
debug_config['learning_rate'] = 1e-4  # Try different LR
```

**Slow Training:**
```python
# Increase num_workers and use mixed precision
config['num_workers'] = 8
config['precision'] = '16-mixed'
```

## 📚 Examples

Complete training examples:

### Basic Training Script
```python
from training import SimpleTrainer, get_training_config, create_dataloaders, set_seed

# Set seed for reproducibility
set_seed(42)

# Load configuration
config = get_training_config('medium')
config.update({
    'experiment_name': 'my_first_experiment',
    'wandb_project': 'my-simplepfn-project'
})

# Create data loaders (assuming you have your data)
dataloaders = create_dataloaders(
    train_samples=train_samples,
    train_targets=train_targets,
    val_samples=val_samples,
    val_targets=val_targets,
    test_samples=test_samples,
    test_targets=test_targets,
    batch_size=config['batch_size'],
    num_workers=config['num_workers']
)

# Create and train
trainer = SimpleTrainer(config)
best_model = trainer.fit(dataloaders['train'], dataloaders['val'])

# Evaluate
test_metrics = trainer.evaluate(dataloaders['test'])
print(f"Final test metrics: {test_metrics}")

# Save model
trainer.save_model(f"{config['save_dir']}/final_model.pt")
```

### Hyperparameter Search
```python
from training import SimpleTrainer, get_training_config

learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
results = {}

for lr in learning_rates:
    config = get_training_config('small')
    config.update({
        'learning_rate': lr,
        'experiment_name': f'lr_search_lr_{lr}',
        'max_epochs': 50
    })
    
    trainer = SimpleTrainer(config)
    model = trainer.fit(train_loader, val_loader)
    
    test_metrics = trainer.evaluate(test_loader)
    results[lr] = test_metrics['mse']
    
    print(f"LR {lr}: Test MSE = {test_metrics['mse']:.6f}")

# Find best learning rate
best_lr = min(results, key=results.get)
print(f"Best learning rate: {best_lr} (MSE: {results[best_lr]:.6f})")
```

## 📝 License

This training package is part of the CausalPriorFitting project.
