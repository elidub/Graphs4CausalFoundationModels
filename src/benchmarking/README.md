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

The benchmark can be run with the run_benchmark.py script which is currently still under construction.