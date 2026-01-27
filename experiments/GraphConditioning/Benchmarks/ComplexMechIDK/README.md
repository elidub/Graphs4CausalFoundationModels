# ComplexMechIDK Benchmark# ComplexMechIDK Benchmark - Partial Graph Knowledge



This directory contains a benchmark for complex mechanism interventional models with partial graph knowledge (IDK = "I Don't Know" edges). The benchmark uses three-state adjacency matrices where edges can be known present (1), known absent (-1), or unknown (0).This benchmark extends the ComplexMech benchmark to handle **partial graph knowledge** scenarios, where the causal structure is only partially known. This simulates realistic situations where some edges are confirmed, some are known to be absent, and others are uncertain.



## Structure## Overview



```The ComplexMechIDK benchmark generates **Complex Mechanism** Structural Causal Models (SCMs) with adjacency matrices in a **three-state format** `{-1, 0, 1}`:

ComplexMechIDK/- **-1.0**: Confirmed NO edge (known absence)

├── ComplexMechBenchmarkIDK.py      # Main benchmark class- **1.0**: Confirmed edge exists (known presence)

├── generate_all_variants_data.py   # Generate data for all configurations- **0.0**: UNKNOWN edge (uncertain/hidden)

├── generate_benchmark_data.py      # Benchmark data generation utilities

├── generate_single_config.py       # Generate data for single config### Complex Mechanism Features

├── run_complexmech_idk.py          # Run benchmark evaluation

├── configs/                        # Configuration filesUnlike the Linear-Gaussian benchmark, this benchmark uses:

│   ├── 2node/- **MLP mechanisms** with TabICL-style nonlinearities (`mlp_nonlins: tabicl`)

│   │   ├── base.yaml- **Hidden layers**: 0-3 layers with distribution favoring simpler (87.5% linear, 10% 1-layer, etc.)

│   │   ├── path_TY/- **Optional XGBoost mechanisms**: With probabilities 0.0-0.3

│   │   │   ├── hide_0.0.yaml- **Mixed noise distributions**: 33% Normal, 33% Laplace, 34% Student-T

│   │   │   ├── hide_0.25.yaml- **Exogenous mechanisms**: Enabled (`use_exogenous_mechanisms: true`)

│   │   │   ├── hide_0.5.yaml- **Batch normalization**: 50% probability

│   │   │   ├── hide_0.75.yaml

│   │   │   └── hide_1.0.yamlThis is implemented using:

│   │   ├── path_YT/- `use_partial_graph_format: true` - Enables three-state matrix format

│   │   │   └── ... (same structure)- `hide_fraction_matrix: Uniform(0.0, 1.0)` - Randomly hides a fraction of edges per dataset

│   │   └── path_independent_TY/

│   │       └── ... (same structure)## Directory Structure

│   ├── 5node/

│   ├── 10node/```

│   ├── 20node/ComplexMechIDK/

│   ├── 35node/├── configs/                     # Organized config files

│   └── 50node/│   ├── 2node/                   # 2-node SCM configs

│       └── ... (same structure as 2node)│   │   ├── base.yaml            # Uniform hide_fraction: U(0.0, 1.0)

└── data_cache/                     # Generated benchmark data│   │   ├── path_TY/             # T→Y path constraint

```│   │   │   ├── hide_0.0.yaml    # Fully known graph

│   │   │   ├── hide_0.25.yaml   # 25% edges hidden

## Features│   │   │   ├── hide_0.5.yaml    # 50% edges hidden

│   │   │   ├── hide_0.75.yaml   # 75% edges hidden

The ComplexMechBenchmarkIDK class provides:│   │   │   └── hide_1.0.yaml    # Fully unknown graph

│   │   ├── path_YT/             # Y→T path constraint

1. **Partial Graph Knowledge**: Three-state adjacency matrices {-1, 0, 1} where:│   │   │   ├── hide_0.0.yaml

   - `1`: Edge is known to be present│   │   │   ├── hide_0.25.yaml

   - `-1`: Edge is known to be absent│   │   │   ├── hide_0.5.yaml

   - `0`: Edge status is unknown (IDK)│   │   │   ├── hide_0.75.yaml

│   │   │   └── hide_1.0.yaml

2. **Configuration Management**: Load configs for different node counts (2, 5, 10, 20, 35, 50) and variants with varying hide fractions (0.0, 0.25, 0.5, 0.75, 1.0)│   │   └── path_independent_TY/ # T⊥Y constraint

│   │       ├── hide_0.0.yaml

3. **Complex Mechanisms**: Non-linear SCMs using:│   │       ├── hide_0.25.yaml

   - MLP mechanisms with TabICL-style nonlinearities│   │       ├── hide_0.5.yaml

   - Optional XGBoost mechanisms│   │       ├── hide_0.75.yaml

   - Mixed noise types (Normal, Laplace, StudentT)│   │       └── hide_1.0.yaml

│   ├── 5node/                   # 5-node SCM configs (same structure)

4. **Path Variants**:│   ├── 10node/                  # 10-node SCM configs (same structure)

   - `base`: Uniform hide fraction sampling from U(0, 1)│   ├── 20node/                  # 20-node SCM configs (same structure)

   - `path_TY`: Treatment → Outcome path constraint│   ├── 35node/                  # 35-node SCM configs (same structure)

   - `path_YT`: Outcome → Treatment path constraint│   └── 50node/                  # 50-node SCM configs (same structure)

   - `path_independent_TY`: Independent Treatment/Outcome constraint├── data_cache/                  # Cached benchmark datasets

├── benchmark_res/               # Benchmark results by model

5. **Model Evaluation**: Support for graph-conditioned interventional models with metrics (MSE, R², NLL)├── ComplexMechBenchmarkIDK.py          # Benchmark runner class

├── generate_configs.py          # Base config generation script

## Usage├── generate_path_variant_configs.py  # Path variant config generation

├── generate_all_variants_data.py     # Dataset generation script

### Basic Usage└── README.md                    # This file

```

```python

from ComplexMechBenchmarkIDK import ComplexMechBenchmarkIDK**Total configs**: 96 files

- 6 base configs (one per node count) with uniform distribution

# Initialize benchmark- 90 path variant configs (6 node counts × 3 variants × 5 hide fractions)

benchmark = ComplexMechBenchmarkIDK(

    benchmark_dir='/path/to/ComplexMechIDK',## Config Variants

    verbose=True

)### Base Variant (Uniform Distribution)



# Load a configurationEach node count has a `base.yaml` config with **uniform distribution** over hide fractions:

config = benchmark.load_config(node_count=5, variant="base")

config = benchmark.load_config(node_count=10, variant="path_TY", hide_fraction=0.5)```yaml

hide_fraction_matrix:

# Sample and save data  distribution: "uniform"

saved_path = benchmark.sample_and_save_data(  distribution_parameters:

    node_count=5,    low: 0.0   # Fully known graph

    num_samples=1000,    high: 1.0  # Completely unknown graph

    variant="base"```

)

**Use case**: Training models that need to handle varying levels of uncertainty

# Load and evaluate with a model

data, metadata = benchmark.load_data(filename)### Path Variants (Fixed Hide Fractions)

benchmark.load_model(config_path, checkpoint_path)

results = benchmark.evaluate_dataset(data)Each node count has 3 path constraint variants, each with **5 fixed hide fraction values** (0.0, 0.25, 0.5, 0.75, 1.0):

```

1. **path_TY/**: Treatment → Outcome path ensured

### Full Benchmark   - `ensure_treatment_outcome_path: true`

   - Guarantees at least one directed path from treatment to outcome

```python   - Files: `hide_0.0.yaml`, `hide_0.25.yaml`, `hide_0.5.yaml`, `hide_0.75.yaml`, `hide_1.0.yaml`

# Run complete benchmark across all configurations

results = benchmark.run_full_benchmark(2. **path_YT/**: Outcome → Treatment path ensured

    node_counts=[2, 5, 10, 20, 35, 50],   - `ensure_outcome_treatment_path: true`

    variants=["base", "path_TY", "path_YT", "path_independent_TY"],   - Guarantees at least one directed path from outcome to treatment (confounding)

    hide_fractions=[0.0, 0.25, 0.5, 0.75, 1.0],   - Files: `hide_0.0.yaml`, `hide_0.25.yaml`, `hide_0.5.yaml`, `hide_0.75.yaml`, `hide_1.0.yaml`

    num_samples=1000

)3. **path_independent_TY/**: No connection between Treatment and Outcome

```   - `ensure_no_connection_treatment_outcome: true`

   - Ensures treatment and outcome are d-separated

## Data Generation   - Files: `hide_0.0.yaml`, `hide_0.25.yaml`, `hide_0.5.yaml`, `hide_0.75.yaml`, `hide_1.0.yaml`



Use the included data generation scripts:**Hide fraction meanings**:

- `hide_0.0`: Fully known graph (baseline, no uncertainty)

```bash- `hide_0.25`: 25% of edges have unknown status

# Generate data for all configs (default: 1000 samples each)- `hide_0.5`: 50% of edges have unknown status (moderate uncertainty)

python generate_all_variants_data.py- `hide_0.75`: 75% of edges have unknown status (high uncertainty)

- `hide_1.0`: Fully unknown graph (maximum uncertainty)

# Generate with specific parameters

python generate_all_variants_data.py --num-samples 1000 --node-counts 2 5 10**Use cases**: 

- Evaluate model performance across different levels of graph knowledge

# Generate only specific variants and hide fractions- Study how uncertainty affects causal inference in specific path scenarios

python generate_all_variants_data.py --variants base path_TY --hide-fractions 0.0 0.5 1.0- Benchmark model robustness to missing graph information



# Generate for a single configuration## Key Configuration Parameters

python generate_single_config.py --node-count 5 --variant path_TY --hide-fraction 0.5

```### Dataset Config (Partial Graph Support)

```yaml

## Configuration Structuredataset_config:

  # Enable adjacency matrix output

### Node Counts  return_adjacency_matrix:

- 2, 5, 10, 20, 35, 50 nodes    value: true

  

### Variants  # CRITICAL: Enable partial graph format

- `base`: Base configuration with uniform hide fraction  use_partial_graph_format:

- `path_TY`: Treatment → Outcome path structure    value: true

- `path_YT`: Outcome → Treatment path structure  

- `path_independent_TY`: Independent Treatment/Outcome structure  # Randomly hide edges to simulate partial knowledge

  hide_fraction_matrix:

### Hide Fractions    distribution: "uniform"

- 0.0 (all edges known)    distribution_parameters:

- 0.25      low: 0.0   # Fully known graph

- 0.5      high: 1.0  # Completely unknown graph

- 0.75```

- 1.0 (no edges known)

### Model Config (For Graph-Conditioned Models)

## Data File Naming```yaml

model_config:

Generated data files follow this naming pattern:  use_graph_conditioning:

- Base configs: `complexmech_{N}nodes_base_{num_samples}samples_seed{seed}.pkl`    value: true  # Enable when using graph-conditioned models

- Path configs: `complexmech_{N}nodes_{variant}_hide{fraction}_{num_samples}samples_seed{seed}.pkl`  

  graph_conditioning_mode:

## Differences from ComplexMech (NTest)    value: "partial_gcn_and_soft_attention"

    # Recommended modes for partial graphs:

| Feature | ComplexMechIDK | ComplexMech (NTest) |    # - "partial_soft_attention": Soft attention bias only

|---------|----------------|---------------------|    # - "partial_gcn_and_soft_attention": Full conditioning (GCN+AdaLN+soft attention)

| Edge states | {-1, 0, 1} (with IDK) | {0, 1} (binary) |  

| Node counts | [2, 5, 10, 20, 35, 50] | [2, 5, 20, 50] |  # GCN parameters for partial graphs

| Path variant files | `hide_{fraction}.yaml` | `ntest_{size}.yaml` |  gcn_use_transpose:

| Primary use | Partial graph knowledge | Sample size experiments |    value: false  # Edge direction convention

  
  gcn_alpha_init:
    value: 0.1  # Initial weight for unknown edges (learned during training)
```

## Generating Configs

To regenerate all config files:

```bash
cd /path/to/ComplexMechIDK
python generate_configs.py
```

This creates 24 config files (6 node counts × 4 variants) in the `configs/` directory.

## Generating Benchmark Datasets

To generate datasets for all configs:

```bash
cd /path/to/ComplexMechIDK
python generate_all_variants_data.py
```

This will:
1. Sample 1000 datasets per config (using seed 42)
2. Store data in `data_cache/` as `.pkl` files
3. Each dataset contains adjacency matrices in three-state format

## Running Benchmarks

### Quick Benchmark (Single Model)

```python
from ComplexMechBenchmarkIDK import ComplexMechBenchmark

# Initialize benchmark
benchmark = ComplexMechBenchmark(
    benchmark_dir="/path/to/ComplexMechIDK",
    verbose=True,
    use_ancestor_matrix=False,  # Use adjacency (not ancestor) matrices
)

# Quick benchmark on all variants
results = benchmark.quick_benchmark(
    config_path="/path/to/model/config.yaml",
    checkpoint_path="/path/to/model/checkpoint.pt",
    max_samples=100,  # Limit to 100 samples per config
)
```

### Full Benchmark (All Nodes and Variants)

```python
# Load model
benchmark.load_model(
    config_path="/path/to/model/config.yaml",
    checkpoint_path="/path/to/model/checkpoint.pt",
)

# Run on all configs
results = benchmark.run_full_benchmark(
    node_counts=[2, 5, 10, 20, 35, 50],
    variants=["base", "path_TY", "path_YT", "path_independent_TY"],
    n_bootstrap=1000,
    num_samples=1000,
    base_seed=42,
)
```

### Fidelity Levels

The benchmark supports different evaluation fidelities:

```python
# Minimal: 3 samples per dataset
results = benchmark.run(
    fidelity="minimal",
    checkpoint_path="/path/to/checkpoint.pt",
)

# Low: 100 samples per dataset
results = benchmark.run(
    fidelity="low",
    checkpoint_path="/path/to/checkpoint.pt",
)

# High: 1000 samples per dataset (full evaluation)
results = benchmark.run(
    fidelity="high",
    checkpoint_path="/path/to/checkpoint.pt",
)
```

## Differences from Original ComplexMech

1. **Three-State Matrices**: Adjacency matrices now use {-1, 0, 1} instead of {0, 1}
2. **Partial Knowledge**: `hide_fraction_matrix` parameter randomly hides edges
3. **Model Requirements**: Models must support partial graph format (e.g., PartialGraphConditionedInterventionalPFN)
4. **Folder Structure**: Configs organized by node count for better navigation

## Model Compatibility

**Compatible Models:**
- PartialGraphConditionedInterventionalPFN (with partial graph conditioning modes)
- Standard InterventionalPFN (ignores graph information)

**Configuration Required:**
- Set `use_partial_graph_format: true` in dataset_config
- Set appropriate `graph_conditioning_mode` (e.g., "partial_gcn_and_soft_attention")
- Configure GCN parameters: `gcn_use_transpose`, `gcn_alpha_init`

## Example Results Structure

After running a benchmark, results are saved in:
```
benchmark_res/
└── model_name/
    ├── aggregated_2nodes_base.json
    ├── aggregated_2nodes_path_TY.json
    ├── ...
    ├── aggregated_50nodes_path_independent_TY.json
    ├── individual_*.json  (per-sample results)
    ├── summary_all_nodes.json  (aggregated summary)
    └── model_config.yaml  (copy of model config)
```

Each aggregated result contains:
- Mean and median metrics (MSE, R², NLL)
- Bootstrap confidence intervals (95%)
- Sample size and metadata

## Notes

- The `hide_fraction_matrix` distribution can be adjusted per config
- Benchmark datasets are cached in `data_cache/` for reproducibility
- All datasets use seed 42 by default for consistency
- The benchmark automatically handles batched inference if supported by the model
