# ComplexMech Benchmark

This directory contains a benchmark for complex mechanism interventional models, similar to the LinGausIDK benchmark but adapted for the ComplexMech configuration structure.

## Structure

```
ComplexMech/
├── ComplexMechBenchmark.py     # Main benchmark class
├── test_benchmark.py           # Test script for local usage
└── configs/                    # Configuration files
    ├── 2node/
    │   ├── base.yaml
    │   ├── path_TY/
    │   │   ├── ntest_500.yaml
    │   │   ├── ntest_700.yaml
    │   │   ├── ntest_800.yaml
    │   │   ├── ntest_900.yaml
    │   │   └── ntest_950.yaml
    │   ├── path_YT/
    │   │   └── ... (same structure)
    │   └── path_independent_TY/
    │       └── ... (same structure)
    ├── 5node/
    │   └── ... (same structure as 2node)
    ├── 20node/
    │   └── ... (same structure as 2node)
    └── 50node/
        └── ... (same structure as 2node)
```

## Features

The ComplexMechBenchmark class provides:

1. **Configuration Management**: Load configs for different node counts (2, 5, 20, 50) and variants (base, path_TY, path_YT, path_independent_TY) with different sample sizes (500, 700, 800, 900, 950)

2. **Data Sampling**: Generate synthetic datasets using complex SCM mechanisms including:
   - XGBoost mechanisms with varying probability
   - MLP mechanisms with tabicl nonlinearities  
   - Mixed noise types (Normal, Laplace, StudentT)
   - Varying architectures and parameters

3. **Model Evaluation**: Support for both graph-conditioned and standard interventional models with metrics (MSE, R², NLL)

4. **Batch Processing**: Efficient evaluation across multiple configurations

## Usage

### Basic Usage
```python
from ComplexMechBenchmark import ComplexMechBenchmark

# Initialize benchmark
benchmark = ComplexMechBenchmark(
    benchmark_dir='/path/to/ComplexMech',
    max_samples=100,
    verbose=True
)

# Load a configuration
config = benchmark.load_config(node_count=5, variant="base")
config_path = benchmark.load_config(node_count=5, variant="path_TY", sample_size=500)

# Sample and save data
saved_path = benchmark.sample_and_save_data(
    node_count=5,
    num_samples=100,
    variant="base"
)

# Load and evaluate with a model
data, metadata = benchmark.load_data(filename)
benchmark.load_model(config_path, checkpoint_path)
results = benchmark.evaluate_dataset(data)
```

### Full Benchmark
```python
# Run complete benchmark across all configurations
results = benchmark.run_full_benchmark(
    node_counts=[2, 5, 20, 50],
    variants=["base", "path_TY", "path_YT", "path_independent_TY"],
    sample_sizes=[500, 700, 800, 900, 950],
    num_samples=1000
)
```

### Quick Test
```python
# Quick benchmark with limited configs
results = benchmark.quick_benchmark(
    config_path='/path/to/model_config.yaml',
    checkpoint_path='/path/to/model_checkpoint.pt',
    max_samples=50
)
```

## Configuration Structure

The benchmark supports:
- **Node counts**: 2, 5, 20, 50 nodes
- **Variants**: 
  - `base`: Base configuration
  - `path_TY`: Treatment→Outcome path constraint
  - `path_YT`: Outcome→Treatment path constraint  
  - `path_independent_TY`: Independent Treatment/Outcome constraint
- **Sample sizes**: 500, 700, 800, 900, 950 (for path variants)

## Differences from LinGausIDK

1. **Mechanisms**: Uses complex mechanisms (XGBoost, MLPs) instead of linear-Gaussian
2. **Configuration Structure**: Path variants use sample size files (`ntest_XXX.yaml`) instead of hide fraction files
3. **Node Counts**: Supports [2, 5, 20, 50] instead of [2, 5, 10, 20, 35, 50]
4. **Parameters**: Different parameter ranges and distributions for complex mechanisms

## Data Generation

Use the included data generation script to create datasets for all configurations:

```bash
# Generate data for all configs (default: 1000 samples each)
python generate_all_variants_data.py

# Quick test with fewer samples
python generate_all_variants_data.py --num-samples 100 --node-counts 2 5

# Generate only specific variants
python generate_all_variants_data.py --variants base path_TY --sample-sizes 500 700

# Overwrite existing data
python generate_all_variants_data.py --overwrite
```

The script generates data files in `data_cache/` with the naming pattern:
- Base configs: `complexmech_{N}nodes_base_{num_samples}samples_seed{seed}.pkl`
- Path configs: `complexmech_{N}nodes_{variant}_ntest{size}_{num_samples}samples_seed{seed}.pkl`

## Testing

Test the benchmark functionality:

```bash
# Test configuration loading
python test_benchmark.py

# Test data generation and loading  
python test_data_loading.py
```