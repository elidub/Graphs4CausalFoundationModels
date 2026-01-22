#!/usr/bin/env python3
"""
Script to inspect the structure and content of plant benchmark datasets.
"""

import pickle as pkl
import numpy as np
from pathlib import Path


def inspect_dataset(filepath):
    """Inspect a single dataset file."""
    print("=" * 80)
    print(f"Inspecting: {filepath.name}")
    print("=" * 80)
    print()
    
    # Try standard pickle first, then dill if available
    try:
        with open(filepath, 'rb') as f:
            benchmark = pkl.load(f)
    except Exception as e:
        print(f"Error loading with pickle: {e}")
        try:
            import dill
            with open(filepath, 'rb') as f:
                benchmark = dill.load(f)
            print("(Loaded successfully with dill)")
        except Exception as e2:
            print(f"Error loading with dill: {e2}")
            return
    
    print(f"Type: {type(benchmark)}")
    print(f"Has 'realizations' attribute: {hasattr(benchmark, 'realizations')}")
    print()
    
    if not hasattr(benchmark, 'realizations'):
        print("No 'realizations' attribute found. Top-level attributes:")
        print([attr for attr in dir(benchmark) if not attr.startswith('_')])
        return
    
    realizations = benchmark.realizations
    print(f"Number of realizations: {len(realizations)}")
    print()
    
    # Inspect first realization
    print("-" * 80)
    print("First Realization Details:")
    print("-" * 80)
    real = realizations[0]
    print(f"Type: {type(real)}")
    print()
    
    # List all attributes
    attrs = [attr for attr in dir(real) if not attr.startswith('_')]
    print(f"Attributes: {attrs}")
    print()
    
    # Check data arrays
    if hasattr(real, 'X_train'):
        print("Data Shapes:")
        print(f"  X_train: {real.X_train.shape}")
        print(f"  t_train: {real.t_train.shape}")
        print(f"  y_train: {real.y_train.shape}")
        print(f"  X_test:  {real.X_test.shape}")
        print(f"  t_test:  {real.t_test.shape}")
        if hasattr(real, 'true_cid'):
            print(f"  true_cid: {real.true_cid.shape}")
        print()
        
        print("Data Types:")
        print(f"  X_train dtype: {real.X_train.dtype}")
        print(f"  t_train dtype: {real.t_train.dtype}")
        print(f"  y_train dtype: {real.y_train.dtype}")
        print()
        
        print("Data Ranges:")
        print(f"  X_train: [{real.X_train.min():.2f}, {real.X_train.max():.2f}]")
        print(f"  t_train: [{real.t_train.min():.2f}, {real.t_train.max():.2f}]")
        print(f"  y_train: [{real.y_train.min():.2f}, {real.y_train.max():.2f}]")
        print()
        
        print(f"Number of features: {real.X_train.shape[1]}")
        print(f"Training samples: {real.X_train.shape[0]}")
        print(f"Test samples: {real.X_test.shape[0]}")
        print(f"Unique treatment values: {np.unique(real.t_train)}")
        print()
    
    # Check for feature information
    if hasattr(real, 'feature_names'):
        print("Feature Names:")
        print(f"  {real.feature_names}")
        print()
    
    if hasattr(real, 'obs_df'):
        print("obs_df Information:")
        print(f"  Shape: {real.obs_df.shape}")
        print(f"  Columns: {list(real.obs_df.columns)}")
        print()
        print("  Column dtypes:")
        for col in real.obs_df.columns:
            print(f"    {col}: {real.obs_df[col].dtype}")
        print()
        print("  First 5 rows:")
        print(real.obs_df.head())
        print()
        print("  Summary statistics:")
        print(real.obs_df.describe())
        print()
    
    # Check for adjacency/graph information
    if hasattr(real, 'adjacency_matrix'):
        print("Adjacency Matrix:")
        print(f"  Shape: {real.adjacency_matrix.shape}")
        print(f"  Number of edges: {(real.adjacency_matrix == 1).sum()}")
        print(f"  Unique values: {np.unique(real.adjacency_matrix)}")
        print()
    
    if hasattr(real, 'causal_graph'):
        print(f"Has causal_graph: {real.causal_graph}")
        print()
    
    # Check other realizations for consistency
    if len(realizations) > 1:
        print("-" * 80)
        print("Checking consistency across realizations:")
        print("-" * 80)
        all_consistent = True
        for i, r in enumerate(realizations[1:], start=2):
            if hasattr(r, 'X_train'):
                if r.X_train.shape != real.X_train.shape:
                    print(f"  Realization {i}: X_train shape differs: {r.X_train.shape}")
                    all_consistent = False
                if r.X_test.shape != real.X_test.shape:
                    print(f"  Realization {i}: X_test shape differs: {r.X_test.shape}")
                    all_consistent = False
        
        if all_consistent:
            print("  ✓ All realizations have consistent shapes")
        print()


def main():
    """Main function to inspect all datasets in plant_data directory."""
    data_dir = Path('plant_data')
    
    if not data_dir.exists():
        print(f"Error: {data_dir} directory not found!")
        return
    
    pkl_files = list(data_dir.glob('*.pkl'))
    
    if not pkl_files:
        print(f"No .pkl files found in {data_dir}")
        return
    
    print(f"Found {len(pkl_files)} dataset(s) in {data_dir}")
    print()
    
    for pkl_file in sorted(pkl_files):
        inspect_dataset(pkl_file)
        print()


if __name__ == "__main__":
    main()
