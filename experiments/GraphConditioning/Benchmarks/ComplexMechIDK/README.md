# ComplexMechIDK Benchmark - Partial Graph Knowledge

This benchmark extends the ComplexMech benchmark to handle **partial graph knowledge** scenarios, where the causal structure is only partially known. This simulates realistic situations where some edges are confirmed, some are known to be absent, and others are uncertain.

## Overview

The ComplexMechIDK benchmark generates **Complex Mechanism** Structural Causal Models (SCMs) with adjacency matrices in a **three-state format** `{-1, 0, 1}`:
- **-1.0**: Confirmed NO edge (known absence)
- **1.0**: Confirmed edge exists (known presence)
- **0.0**: UNKNOWN edge (uncertain/hidden)

### Complex Mechanism Features

Unlike the Linear-Gaussian benchmark, this benchmark uses:
- **MLP mechanisms** with TabICL-style nonlinearities (`mlp_nonlins: tabicl`)
- **Hidden layers**: 0-3 layers with distribution favoring simpler (87.5% linear, 10% 1-layer, etc.)
- **Optional XGBoost mechanisms**: With probabilities 0.0-0.3
- **Mixed noise distributions**: 33% Normal, 33% Laplace, 34% Student-T
- **Exogenous mechanisms**: Enabled (`use_exogenous_mechanisms: true`)
- **Batch normalization**: 50% probability

This is implemented using:
- `use_partial_graph_format: true` - Enables three-state matrix format
- `hide_fraction_matrix: Uniform(0.0, 1.0)` - Randomly hides a fraction of edges per dataset

## Directory Structure

```
ComplexMechIDK/
├── configs/                     # Organized config files
│   ├── 2node/                   # 2-node SCM configs
│   │   ├── base.yaml            # Uniform hide_fraction: U(0.0, 1.0)
│   │   ├── path_TY/             # T→Y path constraint
│   │   │   ├── hide_0.0.yaml    # Fully known graph
│   │   │   ├── hide_0.25.yaml   # 25% edges hidden
│   │   │   ├── hide_0.5.yaml    # 50% edges hidden
│   │   │   ├── hide_0.75.yaml   # 75% edges hidden
│   │   │   └── hide_1.0.yaml    # Fully unknown graph
│   │   ├── path_YT/             # Y→T path constraint
│   │   │   ├── hide_0.0.yaml
│   │   │   ├── hide_0.25.yaml
│   │   │   ├── hide_0.5.yaml
│   │   │   ├── hide_0.75.yaml
│   │   │   └── hide_1.0.yaml
│   │   └── path_independent_TY/ # T⊥Y constraint
│   │       ├── hide_0.0.yaml
│   │       ├── hide_0.25.yaml
│   │       ├── hide_0.5.yaml
│   │       ├── hide_0.75.yaml
│   │       └── hide_1.0.yaml
│   ├── 5node/                   # 5-node SCM configs (same structure)
│   ├── 10node/                  # 10-node SCM configs (same structure)
│   ├── 20node/                  # 20-node SCM configs (same structure)
│   ├── 35node/                  # 35-node SCM configs (same structure)
│   └── 50node/                  # 50-node SCM configs (same structure)
├── data_cache/                  # Cached benchmark datasets
├── benchmark_res/               # Benchmark results by model
├── ComplexMechBenchmarkIDK.py          # Benchmark runner class
├── generate_configs.py          # Base config generation script
├── generate_path_variant_configs.py  # Path variant config generation
├── generate_all_variants_data.py     # Dataset generation script
└── README.md                    # This file
```

**Total configs**: 96 files
- 6 base configs (one per node count) with uniform distribution
- 90 path variant configs (6 node counts × 3 variants × 5 hide fractions)

## Config Variants

### Base Variant (Uniform Distribution)

Each node count has a `base.yaml` config with **uniform distribution** over hide fractions:

```yaml
hide_fraction_matrix:
  distribution: "uniform"
  distribution_parameters:
    low: 0.0   # Fully known graph
    high: 1.0  # Completely unknown graph
```

**Use case**: Training models that need to handle varying levels of uncertainty

### Path Variants (Fixed Hide Fractions)

Each node count has 3 path constraint variants, each with **5 fixed hide fraction values** (0.0, 0.25, 0.5, 0.75, 1.0):

1. **path_TY/**: Treatment → Outcome path ensured
   - `ensure_treatment_outcome_path: true`
   - Guarantees at least one directed path from treatment to outcome
   - Files: `hide_0.0.yaml`, `hide_0.25.yaml`, `hide_0.5.yaml`, `hide_0.75.yaml`, `hide_1.0.yaml`

2. **path_YT/**: Outcome → Treatment path ensured
   - `ensure_outcome_treatment_path: true`
   - Guarantees at least one directed path from outcome to treatment (confounding)
   - Files: `hide_0.0.yaml`, `hide_0.25.yaml`, `hide_0.5.yaml`, `hide_0.75.yaml`, `hide_1.0.yaml`

3. **path_independent_TY/**: No connection between Treatment and Outcome
   - `ensure_no_connection_treatment_outcome: true`
   - Ensures treatment and outcome are d-separated
   - Files: `hide_0.0.yaml`, `hide_0.25.yaml`, `hide_0.5.yaml`, `hide_0.75.yaml`, `hide_1.0.yaml`

**Hide fraction meanings**:
- `hide_0.0`: Fully known graph (baseline, no uncertainty)
- `hide_0.25`: 25% of edges have unknown status
- `hide_0.5`: 50% of edges have unknown status (moderate uncertainty)
- `hide_0.75`: 75% of edges have unknown status (high uncertainty)
- `hide_1.0`: Fully unknown graph (maximum uncertainty)

**Use cases**: 
- Evaluate model performance across different levels of graph knowledge
- Study how uncertainty affects causal inference in specific path scenarios
- Benchmark model robustness to missing graph information

## Key Configuration Parameters

### Dataset Config (Partial Graph Support)
```yaml
dataset_config:
  # Enable adjacency matrix output
  return_adjacency_matrix:
    value: true
  
  # CRITICAL: Enable partial graph format
  use_partial_graph_format:
    value: true
  
  # Randomly hide edges to simulate partial knowledge
  hide_fraction_matrix:
    distribution: "uniform"
    distribution_parameters:
      low: 0.0   # Fully known graph
      high: 1.0  # Completely unknown graph
```

### Model Config (For Graph-Conditioned Models)
```yaml
model_config:
  use_graph_conditioning:
    value: true  # Enable when using graph-conditioned models
  
  graph_conditioning_mode:
    value: "partial_gcn_and_soft_attention"
    # Recommended modes for partial graphs:
    # - "partial_soft_attention": Soft attention bias only
    # - "partial_gcn_and_soft_attention": Full conditioning (GCN+AdaLN+soft attention)
  
  # GCN parameters for partial graphs
  gcn_use_transpose:
    value: false  # Edge direction convention
  
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
