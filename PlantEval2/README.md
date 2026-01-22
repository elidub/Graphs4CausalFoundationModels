# PlantEval2 - Clean Plant Benchmark Setup

## Overview
This directory contains a clean version of the plant benchmark with properly formatted datasets that include causal graph information.

## Data Structure

### Datasets
Located in `plant_data/`:
- **Original files** (require dill):
  - `CID_10_ints_10_reals_normal (1).pkl`
  - `CID_10_ints_10_reals_four_std_uniform (2).pkl`

- **Enhanced files with graph info**:
  - `CID_10_ints_10_reals_normal (1)_with_graph.pkl` (dill format)
  - `CID_10_ints_10_reals_normal (1)_with_graph_std.pkl` (standard pickle)
  - `CID_10_ints_10_reals_four_std_uniform (2)_with_graph.pkl` (dill format)
  - `CID_10_ints_10_reals_four_std_uniform (2)_with_graph_std.pkl` (standard pickle)

### Dataset Properties
- **Realizations**: 10 per dataset
- **Training samples**: 256
- **Test samples**: 2560
- **Features**: 6

### Variables

**Features (X_train)**:
1. X (position)
2. Y (position)
3. Bed_Position
4. Red_Light_Intensity_umols
5. Blue_Light_Intensity_umols
6. Far_Red_Light_Intensity_umols

**Treatment (t)**: Light_Intensity_umols  
**Outcome (y)**: Biomass_g

**Total columns**: 8 (6 features + 1 treatment + 1 outcome)

### Matrix Variable Order
```
Index 0: X
Index 1: Y
Index 2: Bed_Position
Index 3: Red_Light_Intensity_umols
Index 4: Blue_Light_Intensity_umols
Index 5: Far_Red_Light_Intensity_umols
Index 6: Light_Intensity_umols (TREATMENT)
Index 7: Biomass_g (OUTCOME)
```

## Causal Graph

The causal graph encodes the following domain knowledge:

### Edge Types

1. **Position → Light Components** (9 edges)
   - X, Y, Bed_Position → Red_Light_Intensity_umols
   - X, Y, Bed_Position → Blue_Light_Intensity_umols
   - X, Y, Bed_Position → Far_Red_Light_Intensity_umols

2. **Light Components → Total Light** (3 edges)
   - Red_Light_Intensity_umols → Light_Intensity_umols
   - Blue_Light_Intensity_umols → Light_Intensity_umols
   - Far_Red_Light_Intensity_umols → Light_Intensity_umols

3. **Position → Total Light** (3 edges)
   - X, Y, Bed_Position → Light_Intensity_umols

4. **Everything → Outcome** (7 edges)
   - X, Y, Bed_Position → Biomass_g
   - Red_Light_Intensity_umols → Biomass_g
   - Blue_Light_Intensity_umols → Biomass_g
   - Far_Red_Light_Intensity_umols → Biomass_g
   - Light_Intensity_umols → Biomass_g

**Total edges**: 22

### Matrix Format

Both `adjacency_matrix` and `ancestor_matrix` are included as attributes:
- Shape: (8, 8)
- Values: 
  - `1` = edge/ancestor relationship exists
  - `-1` = no edge/ancestor relationship
- Format: `matrix[i,j] = 1` means variable i → variable j

Note: For this acyclic graph, the adjacency and ancestor matrices are identical (22 edges each).

## Scripts

### Data Generation
- **`generate_datasets_with_graph.py`**: Creates datasets with adjacency and ancestor matrices
  - Reads original .pkl files
  - Adds graph information based on the edge list
  - Outputs both dill and standard pickle formats
  - Run: `python generate_datasets_with_graph.py`

### Data Inspection
- **`inspect_data.py`**: Detailed inspection of dataset structure and contents
  - Shows shapes, dtypes, ranges
  - Displays obs_df structure
  - Run: `python inspect_data.py`

- **`verify_datasets.py`**: Verifies datasets with graph information
  - Checks adjacency/ancestor matrices
  - Tests pipeline compatibility
  - Run: `python verify_datasets.py`

- **`data_format_summary.py`**: Prints detailed adjacency matrix and edge information
  - Shows full adjacency matrix
  - Lists all edges
  - Counts by variable
  - Run: `python data_format_summary.py`

### Testing
- **`run_plants/test_graph_access.py`**: Quick test of adjacency matrix accessibility
  - Verifies attributes are present
  - Displays adjacency matrix visualization
  - Run from run_plants: `python test_graph_access.py`

- **`run_plants/dummy.py`**: Example baseline model
  - Uses DummyRegressor
  - Compatible with the new dataset format
  - Run: `python dummy.py --dataset "CID_10_ints_10_reals_normal (1)_with_graph" --model dummy --exp_name test`

## Usage Example

```python
import dill
from pathlib import Path

# Load dataset
data_path = Path('plant_data') / 'CID_10_ints_10_reals_normal (1)_with_graph.pkl'
with open(data_path, 'rb') as f:
    benchmark = dill.load(f)

# Access first realization
real = benchmark.realizations[0]

# Data arrays
X_train = real.X_train  # (256, 6)
t_train = real.t_train  # (256,)
y_train = real.y_train  # (256,)

# Graph information
adjacency = real.adjacency_matrix  # (8, 8)
ancestor = real.ancestor_matrix    # (8, 8)

# Create input for models
import numpy as np
t_X_train = np.concatenate([t_train.reshape(-1, 1), X_train], axis=1)  # (256, 7)
```

## Key Differences from PlantEval

1. **Clean directory structure** - Fresh start without accumulated experiment results
2. **Graph information included** - Adjacency and ancestor matrices as dataset attributes
3. **Proper variable ordering** - Matches the actual data structure (features → treatment → outcome)
4. **Updated edge list** - Reflects the reduced feature set (6 features instead of 8)
5. **Both pickle formats** - Dill for compatibility, standard pickle for cluster use

## Notes

- The graph encodes that treatment (Light_Intensity_umols) is both caused by its components AND by position
- All variables causally affect the outcome (Biomass_g)
- The treatment variable mediates the effect of light components on the outcome
- Position variables (X, Y, Bed_Position) have both direct and indirect effects on outcome
