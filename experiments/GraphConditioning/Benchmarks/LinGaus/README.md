# LinGaus Benchmark

This directory contains a benchmark for linear-Gaussian interventional models with graph conditioning. The benchmark tests causal inference models on synthetic linear-Gaussian structural causal models (SCMs).

## Structure

```
LinGaus/
├── LingausBenchmark.py             # Main benchmark class
├── generate_all_variants_data.py   # Generate data for all configurations
├── generate_benchmark_data.py      # Benchmark data generation utilities
├── __init__.py
├── configs/                        # Configuration files (flat structure)
│   ├── 2node.yaml
│   ├── 2node_path_TY.yaml
│   ├── 2node_path_YT.yaml
│   ├── 2node_path_independent_TY.yaml
│   ├── 5node.yaml
│   ├── 5node_path_TY.yaml
│   ├── ...
│   ├── 50node.yaml
│   └── 50node_path_independent_TY.yaml
├── benchmark_res/                  # Benchmark results
└── data_cache/                     # Generated benchmark data
```

## Features

The LinGausBenchmark class provides:

1. **Linear-Gaussian SCMs**: Synthetic causal models with:
   - Linear mechanisms between variables
   - Gaussian noise distributions
   - Varying graph sizes (2, 5, 10, 20, 35, 50 nodes)

2. **Configuration Variants**:
   - `base`: Standard configuration
   - `path_TY`: Ensures Treatment → Outcome path exists
   - `path_YT`: Ensures Outcome → Treatment path exists
   - `path_independent_TY`: Ensures no connection between Treatment and Outcome

3. **Graph Conditioning**: Support for graph-conditioned interventional models with full adjacency matrix knowledge

4. **Model Evaluation**: Metrics include MSE, R², and NLL

## Usage

### Basic Usage

```python
from LingausBenchmark import LinGausBenchmark

# Initialize benchmark
benchmark = LinGausBenchmark(
    benchmark_dir='/path/to/LinGaus',
    verbose=True
)

# Load a configuration
config = benchmark.load_config(node_count=5, variant="base")
config = benchmark.load_config(node_count=10, variant="path_TY")

# Sample and save data
saved_path = benchmark.sample_and_save_data(
    node_count=5,
    num_samples=1000,
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
    node_counts=[2, 5, 10, 20, 35, 50],
    variants=["base", "path_TY", "path_YT", "path_independent_TY"],
    num_samples=1000
)
```

## Data Generation

Use the included data generation scripts:

```bash
# Generate data for all configs (default: 1000 samples each)
python generate_all_variants_data.py

# Generate with specific parameters
python generate_all_variants_data.py --num-samples 1000 --node-counts 2 5 10

# Generate only specific variants
python generate_all_variants_data.py --variants base path_TY
```

## Configuration Structure

### Node Counts
- 2, 5, 10, 20, 35, 50 nodes

### Variants
- `base`: Standard linear-Gaussian SCM
- `path_TY`: Treatment → Outcome path constraint
- `path_YT`: Outcome → Treatment path constraint
- `path_independent_TY`: Independent Treatment/Outcome constraint

## Data File Naming

Generated data files follow this naming pattern:
- Base configs: `lingaus_{N}nodes_base_{num_samples}samples_seed{seed}.pkl`
- Path configs: `lingaus_{N}nodes_{variant}_{num_samples}samples_seed{seed}.pkl`

## Comparison with Other Benchmarks

| Feature | LinGaus | ComplexMechIDK | ComplexMech (NTest) |
|---------|---------|----------------|---------------------|
| Mechanisms | Linear | MLP/XGBoost | MLP/XGBoost |
| Noise | Gaussian | Mixed | Mixed |
| Edge states | {0, 1} | {-1, 0, 1} | {0, 1} |
| Node counts | [2,5,10,20,35,50] | [2,5,10,20,35,50] | [2,5,20,50] |
| Primary use | Full graph knowledge | Partial graph knowledge | Sample size experiments |
