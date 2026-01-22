#!/usr/bin/env python3
"""
Script to generate CID datasets with proper adjacency matrices for PlantEval2.

This script:
1. Loads the original plant datasets
2. Creates the adjacency matrix based on the causal graph
3. Converts to ancestor matrix
4. Saves datasets with adjacency/ancestor matrices as attributes
"""

import dill
import pickle as pkl
import numpy as np
from pathlib import Path
from typing import List, Tuple


def create_adjacency_matrix(all_vars: List[str], edges: List[Tuple[str, str]]) -> np.ndarray:
    """
    Create adjacency matrix from edge list.
    
    Args:
        all_vars: List of all variable names in order
        edges: List of (parent, child) tuples
        
    Returns:
        Adjacency matrix where adjacency[i,j] = 1 if i->j edge exists, -1 otherwise
    """
    n_vars = len(all_vars)
    var_to_idx = {var: i for i, var in enumerate(all_vars)}
    
    # Initialize with -1 (no edge)
    adjacency = np.full((n_vars, n_vars), -1, dtype=int)
    
    # Fill in edges
    for parent, child in edges:
        if parent not in var_to_idx or child not in var_to_idx:
            print(f"Warning: Variable {parent} or {child} not found in variable list")
            continue
        i = var_to_idx[parent]
        j = var_to_idx[child]
        adjacency[i, j] = 1
    
    return adjacency


def adjacency_to_ancestor(adjacency: np.ndarray) -> np.ndarray:
    """
    Convert adjacency matrix to ancestor matrix.
    
    Ancestor matrix contains transitive closure of adjacency matrix.
    ancestor[i,j] = 1 if there exists a directed path from i to j.
    
    Args:
        adjacency: Adjacency matrix (1 = edge, -1 = no edge)
        
    Returns:
        Ancestor matrix (1 = ancestor relationship, -1 = no relationship)
    """
    n = adjacency.shape[0]
    
    # Convert to binary adjacency (1 if edge exists, 0 otherwise)
    adj_binary = (adjacency == 1).astype(int)
    
    # Compute transitive closure using matrix multiplication
    # Keep adding paths of increasing length until convergence
    ancestor_binary = adj_binary.copy()
    power = adj_binary.copy()
    
    for _ in range(n - 1):
        power = power @ adj_binary
        ancestor_binary = np.clip(ancestor_binary + power, 0, 1)
    
    # Convert back to {-1, 1} format
    ancestor = np.where(ancestor_binary == 1, 1, -1)
    
    return ancestor


class CID_Dataset:
    """Enhanced CID_Dataset with adjacency and ancestor matrices."""
    
    def __init__(self, original_dataset, adjacency_matrix=None, ancestor_matrix=None):
        # Copy attributes from original dataset
        self.X_train = original_dataset.X_train
        self.t_train = original_dataset.t_train
        self.y_train = original_dataset.y_train
        self.X_test = original_dataset.X_test
        self.t_test = original_dataset.t_test
        self.true_cid = original_dataset.true_cid
        
        # Copy optional attributes if they exist
        if hasattr(original_dataset, 'obs_df'):
            self.obs_df = original_dataset.obs_df
        if hasattr(original_dataset, 'int_df'):
            self.int_df = original_dataset.int_df
        if hasattr(original_dataset, 'intervention_type'):
            self.intervention_type = original_dataset.intervention_type
            
        # Add graph information
        self.adjacency_matrix = adjacency_matrix
        self.ancestor_matrix = ancestor_matrix


class CID_Benchmark:
    """Enhanced CID_Benchmark with graph information."""
    
    def __init__(self, realizations):
        self.realizations = realizations


def process_dataset(input_path: Path, output_path: Path, use_standard_pickle: bool = False):
    """
    Process a dataset file to add adjacency and ancestor matrices.
    
    Args:
        input_path: Path to input .pkl file
        output_path: Path to output .pkl file
        use_standard_pickle: If True, save with standard pickle (no dill)
    """
    print(f"Processing: {input_path.name}")
    print("-" * 80)
    
    # Load original dataset
    with open(input_path, 'rb') as f:
        benchmark = dill.load(f)
    
    print(f"Loaded {len(benchmark.realizations)} realizations")
    
    # Define the causal graph
    # All variables in the order they appear in the data
    # X_train has 6 features, t is treatment, y is outcome
    all_vars = [
        "X",  # Feature 0
        "Y",  # Feature 1
        "Bed_Position",  # Feature 2
        "Red_Light_Intensity_umols",  # Feature 3
        "Blue_Light_Intensity_umols",  # Feature 4
        "Far_Red_Light_Intensity_umols",  # Feature 5
        "Light_Intensity_umols",  # Treatment (index 6 in concatenated [t, X])
        "Biomass_g"  # Outcome (index 7)
    ]
    
    edges = [
        # Everything has a causal impact on the outcome
        ("X", "Biomass_g"),
        ("Y", "Biomass_g"),
        ("Bed_Position", "Biomass_g"),
        ("Red_Light_Intensity_umols", "Biomass_g"),
        ("Blue_Light_Intensity_umols", "Biomass_g"),
        ("Far_Red_Light_Intensity_umols", "Biomass_g"),
        ("Light_Intensity_umols", "Biomass_g"),

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
    
    print(f"Creating adjacency matrix with {len(edges)} edges")
    adjacency = create_adjacency_matrix(all_vars, edges)
    
    print("Converting to ancestor matrix...")
    ancestor = adjacency_to_ancestor(adjacency)
    
    # Count edges
    n_adj_edges = (adjacency == 1).sum()
    n_anc_edges = (ancestor == 1).sum()
    print(f"Adjacency edges: {n_adj_edges}")
    print(f"Ancestor edges: {n_anc_edges}")
    print()
    
    # Print matrix info
    print("Variable order in matrices:")
    for i, var in enumerate(all_vars):
        print(f"  {i}: {var}")
    print()
    
    # Create enhanced realizations
    enhanced_realizations = []
    for i, real in enumerate(benchmark.realizations):
        enhanced_real = CID_Dataset(
            original_dataset=real,
            adjacency_matrix=adjacency.copy(),
            ancestor_matrix=ancestor.copy()
        )
        enhanced_realizations.append(enhanced_real)
    
    # Create enhanced benchmark
    enhanced_benchmark = CID_Benchmark(enhanced_realizations)
    
    # Save
    print(f"Saving to: {output_path}")
    if use_standard_pickle:
        # Convert to dict format for standard pickle
        data = {
            'realizations': [
                {
                    'X_train': r.X_train,
                    't_train': r.t_train,
                    'y_train': r.y_train,
                    'X_test': r.X_test,
                    't_test': r.t_test,
                    'true_cid': r.true_cid,
                    'adjacency_matrix': r.adjacency_matrix,
                    'ancestor_matrix': r.ancestor_matrix,
                }
                for r in enhanced_realizations
            ]
        }
        with open(output_path, 'wb') as f:
            pkl.dump(data, f)
        print("Saved with standard pickle (dict format)")
    else:
        with open(output_path, 'wb') as f:
            dill.dump(enhanced_benchmark, f)
        print("Saved with dill")
    
    print("✓ Done")
    print()
    
    return adjacency, ancestor


def main():
    """Process all datasets in plant_data directory."""
    data_dir = Path('plant_data')
    
    if not data_dir.exists():
        print(f"Error: {data_dir} not found")
        return
    
    # Find all input files
    input_files = list(data_dir.glob('*.pkl'))
    
    if not input_files:
        print(f"No .pkl files found in {data_dir}")
        return
    
    print(f"Found {len(input_files)} dataset(s)")
    print("=" * 80)
    print()
    
    for input_path in sorted(input_files):
        # Skip files that already have graph info in name
        if 'with_graph' in input_path.name:
            print(f"Skipping (already processed): {input_path.name}")
            continue
        
        # Create output filename
        output_name = input_path.stem + '_with_graph.pkl'
        output_path = data_dir / output_name
        
        # Also create standard pickle version
        output_std_name = input_path.stem + '_with_graph_std.pkl'
        output_std_path = data_dir / output_std_name
        
        # Process
        adj, anc = process_dataset(input_path, output_path, use_standard_pickle=False)
        process_dataset(input_path, output_std_path, use_standard_pickle=True)
        
        print()
    
    print("=" * 80)
    print("All datasets processed!")
    print()
    print("Output files:")
    print("  *_with_graph.pkl - Dill format with CID_Dataset objects")
    print("  *_with_graph_std.pkl - Standard pickle with dict format")


if __name__ == "__main__":
    main()
