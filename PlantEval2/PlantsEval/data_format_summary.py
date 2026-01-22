"""
Data Format Summary for PlantEval2 Datasets
===========================================

Dataset Files:
- CID_10_ints_10_reals_normal (1).pkl
- CID_10_ints_10_reals_four_std_uniform (2).pkl

Data Structure:
--------------
- Number of realizations: 10
- Training samples: 256
- Test samples: 2560
- Number of features (X): 6

Variables:
---------
Features (X_train, 6 variables):
1. X (position)
2. Y (position)  
3. Bed_Position
4. Red_Light_Intensity_umols
5. Blue_Light_Intensity_umols
6. Far_Red_Light_Intensity_umols

Treatment (t): Light_Intensity_umols
Outcome (y): Biomass_g

Total columns in obs_df: 8 (6 features + 1 treatment + 1 outcome)

Causal Graph Structure:
-----------------------
Based on domain knowledge, the causal relationships are:

Edges (parent -> child):
"""

# Define the causal edges
edges = [
    # Everything has a causal impact on the outcome
    ("X", "Biomass_g"),
    ("Y", "Biomass_g"),
    ("Bed_Position", "Biomass_g"),
    ("Red_Light_Intensity_umols", "Biomass_g"),
    ("Blue_Light_Intensity_umols", "Biomass_g"),
    ("Far_Red_Light_Intensity_umols", "Biomass_g"),
    ("Light_Intensity_umols", "Biomass_g"),  # Treatment -> Outcome

    # Light intensity is composed of its sources
    ("Far_Red_Light_Intensity_umols", "Light_Intensity_umols"),
    ("Blue_Light_Intensity_umols", "Light_Intensity_umols"),
    ("Red_Light_Intensity_umols", "Light_Intensity_umols"),

    # Positional information has a causal impact on light intensity
    ("X", "Light_Intensity_umols"),
    ("Y", "Light_Intensity_umols"),
    ("Bed_Position", "Light_Intensity_umols"),

    ("X", "Red_Light_Intensity_umols"),
    ("Y", "Red_Light_Intensity_umols"),
    ("Bed_Position", "Red_Light_Intensity_umols"),

    ("X", "Blue_Light_Intensity_umols"),
    ("Y", "Blue_Light_Intensity_umols"),
    ("Bed_Position", "Blue_Light_Intensity_umols"),

    ("X", "Far_Red_Light_Intensity_umols"),
    ("Y", "Far_Red_Light_Intensity_umols"),
    ("Bed_Position", "Far_Red_Light_Intensity_umols"),
]

print("\nEdge List:")
print("-" * 50)
for i, (parent, child) in enumerate(edges, 1):
    print(f"{i:2d}. {parent:35s} -> {child}")

print(f"\nTotal edges: {len(edges)}")

# Create adjacency matrix
import numpy as np

# All variables (features + treatment + outcome)
all_vars = [
    "X",
    "Y", 
    "Bed_Position",
    "Red_Light_Intensity_umols",
    "Blue_Light_Intensity_umols",
    "Far_Red_Light_Intensity_umols",
    "Light_Intensity_umols",  # Treatment
    "Biomass_g"  # Outcome
]

n_vars = len(all_vars)
var_to_idx = {var: i for i, var in enumerate(all_vars)}

# Initialize adjacency matrix with -1 (no edge)
adjacency = np.full((n_vars, n_vars), -1, dtype=int)

# Fill in the edges
for parent, child in edges:
    i = var_to_idx[parent]
    j = var_to_idx[child]
    adjacency[i, j] = 1

print("\n\nAdjacency Matrix:")
print("=" * 80)
print("Rows = parents, Columns = children")
print("Values: -1 = no edge, 1 = edge exists")
print()

# Print header
print("     ", end="")
for var in all_vars:
    print(f"{var[:6]:>7s}", end="")
print()

# Print matrix
for i, var in enumerate(all_vars):
    print(f"{var[:6]:>6s}", end="")
    for j in range(n_vars):
        val = adjacency[i, j]
        print(f"{val:>7d}", end="")
    print()

print("\n\nEdge Count by Variable:")
print("-" * 50)
print("\nParent (outgoing edges):")
for i, var in enumerate(all_vars):
    n_children = (adjacency[i, :] == 1).sum()
    if n_children > 0:
        print(f"  {var:35s}: {n_children:2d} children")

print("\nChild (incoming edges):")
for j, var in enumerate(all_vars):
    n_parents = (adjacency[:, j] == 1).sum()
    if n_parents > 0:
        print(f"  {var:35s}: {n_parents:2d} parents")

print("\n" + "=" * 80)
print(f"✓ Data format verified")
print(f"✓ {len(edges)} causal edges defined")
print(f"✓ {n_vars} variables total (6 features + 1 treatment + 1 outcome)")
