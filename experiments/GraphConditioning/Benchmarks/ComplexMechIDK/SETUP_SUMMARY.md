# ComplexMechIDK Configuration Update Summary

## What Was Done

Successfully reorganized and updated the ComplexMechIDK benchmark to support partial graph knowledge with a clean, scalable folder structure.

## Changes Made

### 1. Created New Folder Structure
```
ComplexMechIDK/
├── configs/
│   ├── 2node/
│   ├── 5node/
│   ├── 10node/
│   ├── 20node/
│   ├── 35node/
│   └── 50node/
│       ├── base.yaml
│       ├── path_TY.yaml
│       ├── path_YT.yaml
│       └── path_independent_TY.yaml
```

### 2. Generated 24 Config Files
- **6 node counts**: 2, 5, 10, 20, 35, 50
- **4 variants per node count**: base, path_TY, path_YT, path_independent_TY
- **Total**: 24 configuration files

### 3. Added Partial Graph Support

All new configs include:

```yaml
dataset_config:
  # Enable three-state format {-1, 0, 1}
  use_partial_graph_format:
    value: true
  
  # Randomly hide edges to simulate partial knowledge
  hide_fraction_matrix:
    distribution: "uniform"
    distribution_parameters:
      low: 0.0   # Fully known graph
      high: 1.0  # Completely unknown graph
```

### 4. Updated Model Configuration

Added GCN parameters for partial graph conditioning:

```yaml
model_config:
  graph_conditioning_mode:
    value: "partial_gcn_and_soft_attention"
  
  gcn_use_transpose:
    value: false
  
  gcn_alpha_init:
    value: 0.1  # Learnable weight for unknown edges
```

### 5. Archived Old Files

Moved all old flat YAML files to `old_configs/` folder for reference.

## Three-State Format Explained

The partial graph format uses three states:
- **-1.0**: Confirmed NO edge (known absence)
- **1.0**: Confirmed edge exists (known presence)
- **0.0**: UNKNOWN edge (uncertain/hidden by `hide_fraction_matrix`)

This simulates realistic scenarios where:
- Some edges are known from domain knowledge or strong evidence
- Some edges are known to be absent (e.g., temporal impossibility)
- Some edges are uncertain due to insufficient data or ambiguity

## Path Variants

Each node count has 4 path constraint variants:

| Variant | Description | Use Case |
|---------|-------------|----------|
| `base` | No constraints | General causal inference |
| `path_TY` | Treatment → Outcome path | Direct causal effect scenarios |
| `path_YT` | Outcome → Treatment path | Confounding scenarios |
| `path_independent_TY` | No T-Y connection | Null/negative control scenarios |

## Benefits of New Structure

1. **Scalability**: Easy to add new node counts or variants
2. **Organization**: Clear hierarchy (node count → variant)
3. **Maintainability**: Single script regenerates all configs
4. **Consistency**: All configs use same parameters and structure
5. **Flexibility**: `hide_fraction_matrix` distribution can be customized per config

## How to Customize

### Change hide_fraction_matrix Distribution

Edit `generate_configs.py` and modify the distribution in the `create_config()` function:

```python
# Example: Only hide 30-70% of edges
"hide_fraction_matrix": {
    "distribution": "uniform",
    "distribution_parameters": {"low": 0.3, "high": 0.7}
}
```

Then regenerate:
```bash
python generate_configs.py
```

### Add New Node Count

Edit `generate_configs.py` and add to `NODE_COUNTS` list:
```python
NODE_COUNTS = [2, 5, 10, 15, 20, 35, 50, 100]  # Added 15 and 100
```

Then regenerate:
```bash
python generate_configs.py
```

### Add New Variant

Edit `generate_configs.py` and add to `VARIANTS` dict:
```python
VARIANTS = {
    # ... existing variants ...
    "custom_variant": {
        "ensure_treatment_outcome_path": False,
        "ensure_outcome_treatment_path": False,
        "ensure_no_connection_treatment_outcome": False,
        # Add custom preprocessing flags here
    }
}
```

## Next Steps

1. **Generate Datasets**: Run `generate_all_variants_data.py` to create benchmark datasets
2. **Test Configuration**: Sample a few datasets to verify three-state format
3. **Train Models**: Use configs with PartialGraphConditionedInterventionalPFN
4. **Run Benchmarks**: Use `ComplexMechBenchmarkIDK.py` to evaluate models

## Files Created/Modified

### Created:
- `configs/` directory with subdirectories for each node count
- 24 YAML config files (6 node counts × 4 variants)
- `generate_configs.py` - Automated config generation script
- `README.md` - Comprehensive documentation
- `SETUP_SUMMARY.md` - This file

### Modified:
- None (all new files)

### Archived:
- `old_configs/` - Contains original flat YAML files

## Verification Checklist

- [x] Folder structure created
- [x] All 24 configs generated
- [x] `use_partial_graph_format: true` in all configs
- [x] `hide_fraction_matrix` distribution configured
- [x] GCN parameters added to model configs
- [x] Path constraints correctly set per variant
- [x] Old files archived
- [x] Documentation created (README.md)
- [x] Generation script created (generate_configs.py)

## Contact

For questions or issues with the ComplexMechIDK benchmark configuration, refer to:
- `README.md` - Full documentation
- `generate_configs.py` - Config generation logic
- `ComplexMechBenchmarkIDK.py` - Benchmark runner implementation
