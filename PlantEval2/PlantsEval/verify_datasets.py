#!/usr/bin/env python3
"""
Verify that the generated datasets with graph information are correct.
"""

import dill
import pickle as pkl
import numpy as np
from pathlib import Path


def verify_dataset(filepath, use_dill=True):
    """Verify a single dataset file."""
    print("=" * 80)
    print(f"Verifying: {filepath.name}")
    print("=" * 80)
    print()
    
    # Load dataset
    if use_dill:
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        realizations = data.realizations
    else:
        with open(filepath, 'rb') as f:
            data = pkl.load(f)
        realizations = data['realizations']
    
    print(f"Number of realizations: {len(realizations)}")
    print()
    
    # Check first realization
    if use_dill:
        real = realizations[0]
    else:
        # For dict format, create a simple wrapper
        class SimpleReal:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        real = SimpleReal(realizations[0])
    
    # Verify data arrays
    print("Data shapes:")
    print(f"  X_train: {real.X_train.shape}")
    print(f"  t_train: {real.t_train.shape}")
    print(f"  y_train: {real.y_train.shape}")
    print(f"  X_test: {real.X_test.shape}")
    print(f"  t_test: {real.t_test.shape}")
    print(f"  true_cid: {real.true_cid.shape}")
    print()
    
    # Verify graph matrices
    if hasattr(real, 'adjacency_matrix') and real.adjacency_matrix is not None:
        print("✓ Adjacency matrix present")
        print(f"  Shape: {real.adjacency_matrix.shape}")
        print(f"  Edges (value=1): {(real.adjacency_matrix == 1).sum()}")
        print(f"  No edges (value=-1): {(real.adjacency_matrix == -1).sum()}")
        print()
        
        # Print a few entries
        print("  Sample edges (showing first 5):")
        var_names = [
            "X", "Y", "Bed_Position", "Red_Light_Intensity_umols",
            "Blue_Light_Intensity_umols", "Far_Red_Light_Intensity_umols",
            "Light_Intensity_umols", "Biomass_g"
        ]
        edge_count = 0
        for i in range(real.adjacency_matrix.shape[0]):
            for j in range(real.adjacency_matrix.shape[1]):
                if real.adjacency_matrix[i, j] == 1:
                    print(f"    {var_names[i]:35s} -> {var_names[j]}")
                    edge_count += 1
                    if edge_count >= 5:
                        break
            if edge_count >= 5:
                break
        print(f"    ... ({(real.adjacency_matrix == 1).sum() - 5} more edges)")
        print()
    else:
        print("✗ No adjacency matrix found!")
        print()
    
    if hasattr(real, 'ancestor_matrix') and real.ancestor_matrix is not None:
        print("✓ Ancestor matrix present")
        print(f"  Shape: {real.ancestor_matrix.shape}")
        print(f"  Ancestor relationships: {(real.ancestor_matrix == 1).sum()}")
        print()
    else:
        print("✗ No ancestor matrix found!")
        print()
    
    # Test compatibility with dummy.py-style usage
    print("Testing compatibility with pipeline functions:")
    try:
        t_X_train = np.concatenate([real.t_train.reshape(-1, 1), real.X_train], axis=1)
        print(f"  ✓ Can create t_X_train: shape {t_X_train.shape}")
        
        t_X_test = np.concatenate([real.t_test.reshape(-1, 1), real.X_test], axis=1)
        print(f"  ✓ Can create t_X_test: shape {t_X_test.shape}")
        print()
    except Exception as e:
        print(f"  ✗ Error creating concatenated arrays: {e}")
        print()
    
    print("✓ Dataset verification complete!")
    print()


def main():
    """Verify all generated datasets."""
    data_dir = Path('plant_data')
    
    # Find datasets with graph info
    graph_files = list(data_dir.glob('*_with_graph.pkl'))
    graph_std_files = list(data_dir.glob('*_with_graph_std.pkl'))
    
    if not graph_files and not graph_std_files:
        print("No datasets with graph information found!")
        print("Run generate_datasets_with_graph.py first.")
        return
    
    print(f"Found {len(graph_files)} dill datasets and {len(graph_std_files)} std pickle datasets")
    print()
    
    # Verify dill format
    for filepath in sorted(graph_files):
        verify_dataset(filepath, use_dill=True)
    
    # Verify standard pickle format
    for filepath in sorted(graph_std_files):
        verify_dataset(filepath, use_dill=False)
    
    print("=" * 80)
    print("All verifications complete!")


if __name__ == "__main__":
    main()
