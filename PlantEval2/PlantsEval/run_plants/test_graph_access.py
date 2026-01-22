#!/usr/bin/env python3
"""
Quick test to verify adjacency matrix is accessible in the pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dill
from pathlib import Path

# Load a dataset
data_path = Path('../plant_data') / 'CID_10_ints_10_reals_normal (1)_with_graph.pkl'

print("Loading dataset...")
with open(data_path, 'rb') as f:
    benchmark = dill.load(f)

print(f"Number of realizations: {len(benchmark.realizations)}")
print()

# Check first realization
real = benchmark.realizations[0]
print("First realization attributes:")
print(f"  - X_train: {real.X_train.shape}")
print(f"  - t_train: {real.t_train.shape}")
print(f"  - y_train: {real.y_train.shape}")
print()

# Check for graph matrices
if hasattr(real, 'adjacency_matrix'):
    print("✓ Adjacency matrix found!")
    print(f"  Shape: {real.adjacency_matrix.shape}")
    print(f"  Edges: {(real.adjacency_matrix == 1).sum()}")
    print()
    print("  Adjacency matrix (first 8x8):")
    var_names = ["X", "Y", "Bed", "Red", "Blue", "FarRed", "Light", "Biomass"]
    print("     ", end="")
    for name in var_names:
        print(f"{name:>7s}", end="")
    print()
    for i in range(8):
        print(f"{var_names[i]:>5s}", end="")
        for j in range(8):
            val = real.adjacency_matrix[i, j]
            if val == 1:
                print(f"{'→':>7s}", end="")
            else:
                print(f"{'·':>7s}", end="")
        print()
    print()
else:
    print("✗ No adjacency matrix attribute!")
    print()

if hasattr(real, 'ancestor_matrix'):
    print("✓ Ancestor matrix found!")
    print(f"  Shape: {real.ancestor_matrix.shape}")
    print(f"  Ancestor relationships: {(real.ancestor_matrix == 1).sum()}")
else:
    print("✗ No ancestor matrix attribute!")

print()
print("=" * 80)
print("Dataset is ready to use!")
print()
print("Variable order in matrices:")
print("  0: X (feature)")
print("  1: Y (feature)")
print("  2: Bed_Position (feature)")
print("  3: Red_Light_Intensity_umols (feature)")
print("  4: Blue_Light_Intensity_umols (feature)")
print("  5: Far_Red_Light_Intensity_umols (feature)")
print("  6: Light_Intensity_umols (TREATMENT)")
print("  7: Biomass_g (OUTCOME)")
