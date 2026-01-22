# ComplexMechIDK Final Configuration Structure

## Overview

The ComplexMechIDK benchmark now has **96 configuration files** organized in a hierarchical structure:
- **6 base configs** (one per node count) with uniform hide_fraction distribution
- **90 path variant configs** (6 node counts × 3 variants × 5 hide fractions)

## Complete Structure

```
configs/
├── 2node/
│   ├── base.yaml                    # Uniform(0.0, 1.0) hide_fraction
│   ├── path_TY/
│   │   ├── hide_0.0.yaml           # Fully known graph
│   │   ├── hide_0.25.yaml          # 25% edges unknown
│   │   ├── hide_0.5.yaml           # 50% edges unknown
│   │   ├── hide_0.75.yaml          # 75% edges unknown
│   │   └── hide_1.0.yaml           # Fully unknown graph
│   ├── path_YT/
│   │   ├── hide_0.0.yaml
│   │   ├── hide_0.25.yaml
│   │   ├── hide_0.5.yaml
│   │   ├── hide_0.75.yaml
│   │   └── hide_1.0.yaml
│   └── path_independent_TY/
│       ├── hide_0.0.yaml
│       ├── hide_0.25.yaml
│       ├── hide_0.5.yaml
│       ├── hide_0.75.yaml
│       └── hide_1.0.yaml
├── 5node/          [same structure as 2node]
├── 10node/         [same structure as 2node]
├── 20node/         [same structure as 2node]
├── 35node/         [same structure as 2node]
└── 50node/         [same structure as 2node]
```

## Configuration Types

### 1. Base Configs (6 files)

**Location**: `{N}node/base.yaml`

**Characteristics**:
- `hide_fraction_matrix`: Uniform distribution U(0.0, 1.0)
- No path constraints
- General-purpose training and evaluation

**Example** (`5node/base.yaml`):
```yaml
dataset_config:
  use_partial_graph_format: true
  hide_fraction_matrix:
    distribution: "uniform"
    distribution_parameters:
      low: 0.0
      high: 1.0

preprocessing_config:
  ensure_treatment_outcome_path: false
  ensure_outcome_treatment_path: false
  ensure_no_connection_treatment_outcome: false
```

### 2. Path Variant Configs (90 files)

**Location**: `{N}node/{variant}/hide_{fraction}.yaml`

**Variants**:
1. **path_TY**: Ensures Treatment → Outcome path
2. **path_YT**: Ensures Outcome → Treatment path (confounding)
3. **path_independent_TY**: Ensures Treatment ⊥ Outcome (d-separated)

**Hide Fractions**:
- `0.0`: Fully known graph (no edges hidden)
- `0.25`: 25% of edges hidden (low uncertainty)
- `0.5`: 50% of edges hidden (moderate uncertainty)
- `0.75`: 75% of edges hidden (high uncertainty)
- `1.0`: Fully unknown graph (all edges hidden)

**Example** (`5node/path_TY/hide_0.5.yaml`):
```yaml
dataset_config:
  use_partial_graph_format: true
  hide_fraction_matrix:
    value: 0.5  # FIXED VALUE

preprocessing_config:
  ensure_treatment_outcome_path: true
  ensure_outcome_treatment_path: false
  ensure_no_connection_treatment_outcome: false
```

## Use Cases

### Training Scenarios

1. **Robust Model Training** (base configs):
   - Train on varying levels of uncertainty
   - Model learns to handle any hide fraction from 0% to 100%
   - Use: `{N}node/base.yaml`

2. **Specific Uncertainty Level** (path variant configs):
   - Train on fixed hide fraction (e.g., always 50% unknown)
   - Model specializes for that uncertainty level
   - Use: `{N}node/{variant}/hide_0.5.yaml`

### Evaluation Scenarios

1. **Uncertainty Robustness Study**:
   - Train on base config (varying uncertainty)
   - Evaluate on all fixed hide fractions (0.0, 0.25, 0.5, 0.75, 1.0)
   - Analyze: Does performance degrade gracefully with increasing uncertainty?

2. **Path-Specific Performance**:
   - Train on one path variant (e.g., path_TY)
   - Evaluate on different hide fractions within same variant
   - Analyze: How does graph knowledge affect causal effect estimation?

3. **Cross-Variant Generalization**:
   - Train on path_TY configs
   - Evaluate on path_YT or path_independent_TY configs
   - Analyze: Does model generalize across different causal structures?

4. **Ablation Study**:
   - Compare models trained with:
     * Full graph knowledge (hide_0.0)
     * Partial graph knowledge (hide_0.5)
     * No graph knowledge (hide_1.0)
   - Analyze: Value of partial graph information

## File Naming Convention

### Base Configs
- Format: `{N}node/base.yaml`
- Example: `5node/base.yaml`

### Path Variant Configs
- Format: `{N}node/{variant}/hide_{fraction}.yaml`
- Examples:
  - `5node/path_TY/hide_0.0.yaml`
  - `10node/path_YT/hide_0.5.yaml`
  - `20node/path_independent_TY/hide_1.0.yaml`

### Experiment Names (auto-generated in configs)
- Base: `complexmech_idk_{N}node_base`
- Path variants: `complexmech_idk_{N}node_{variant}_hide{fraction}`

Examples:
- `complexmech_idk_5node_base`
- `complexmech_idk_5node_path_TY_hide0.5`
- `complexmech_idk_20node_path_independent_TY_hide1.0`

## Generation Scripts

### Generate All Configs

```bash
# Generate base configs (6 files)
python generate_configs.py

# Generate path variant configs (90 files)
python generate_path_variant_configs.py
```

### Regenerate Specific Node Count

Modify either script to filter by node count:
```python
NODE_COUNTS = [5]  # Only generate 5-node configs
```

## Statistics

| Category | Count | Description |
|----------|-------|-------------|
| Node counts | 6 | 2, 5, 10, 20, 35, 50 |
| Base configs | 6 | One per node count |
| Path variants | 3 | path_TY, path_YT, path_independent_TY |
| Hide fractions | 5 | 0.0, 0.25, 0.5, 0.75, 1.0 |
| Path variant configs | 90 | 6 × 3 × 5 |
| **Total configs** | **96** | 6 + 90 |

## Model Compatibility

All configs are compatible with:
- **PartialGraphConditionedInterventionalPFN** (recommended)
  - Set `use_graph_conditioning: true`
  - Set `graph_conditioning_mode: "partial_gcn_and_soft_attention"`
  
- **InterventionalPFN** (baseline, ignores graph)
  - Graph information is available in dataset but not used
  - Good for ablation studies

## Next Steps

1. ✅ Configs generated and validated
2. 📝 Generate datasets using `generate_all_variants_data.py`
3. 🧪 Test with sample configs to verify three-state format
4. 🚀 Train models with different configurations
5. 📊 Run comprehensive benchmarks
