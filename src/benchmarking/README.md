# Benchmarking Module

This module contains tools for benchmarking models against standard datasets and tasks.

## OpenML Benchmark Loader

The `OpenMLBenchmarkLoader` class provides functionality for loading and preprocessing datasets from OpenML benchmarks, with a focus on the Tabular benchmark numerical regression (ID: 336).

### Features

- Download and cache OpenML benchmark datasets
- Standardize numerical features
- Handle missing values
- Split data into train and test sets
- Convert data to SimplePFN-compatible format
- Generate batches for model training

### Usage

#### Basic Usage

```python
from CausalPriorFitting.src.benchmarking.load_openml_benchmark import OpenMLBenchmarkLoader

# Initialize with the Tabular benchmark numerical regression (ID: 336)
loader = OpenMLBenchmarkLoader(benchmark_id=336)

# List available datasets
datasets = loader.list_datasets()
print(f"Found {len(datasets)} datasets")

# Load a specific dataset by task ID
task_id = datasets[0]['task_id']
data = loader.load_dataset(task_id)

# Access preprocessed data
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
```

#### Integration with SimplePFN

```python
# Create batches in SimplePFN format
X_context, y_context, X_target, y_target = loader.create_batch_for_pfn(
    task_id, 
    batch_size=16,    # Number of tasks in batch
    n_context=20,     # Context points per task
    n_target=10,      # Target points per task
    device='cuda'     # Device for tensors
)

# Create a dataloader for training
dataloader, n_features = prepare_dataloader(
    loader, task_id, batch_size=16, 
    n_context=20, n_target=10
)

# Use with SimplePFNTrainer
trainer = SimplePFNTrainer(
    model=pfn_model,
    dataloader=dataloader,
    # ... other parameters
)
```

## Example Scripts

The `examples` directory contains demonstration scripts:

- `openml_benchmark_example.py`: Basic usage of the benchmark loader with sklearn models
- `simplePFN_on_openml.py`: End-to-end example training SimplePFN on OpenML benchmarks

## OpenML Benchmarks of Interest

1. **Tabular benchmark numerical regression (ID: 336)**
   - 19 regression tasks
   - Focus on numerical features
   - Various dataset sizes

2. **AutoML benchmark regression (ID: 269)**
   - 30 regression tasks
   - More diverse datasets

## Dependencies

- openml
- numpy
- pandas
- scikit-learn
- torch (for integration with SimplePFN)
