"""
Detailed diagnostic test to understand the ancestor matrix mismatch issue.
"""

import sys
import os
import yaml
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset

# Import ancestor matrix computation with fallback logic
try:
    from utils.graph_utils import adjacency_to_ancestor_matrix
except ImportError:
    try:
        from src.utils.graph_utils import adjacency_to_ancestor_matrix
    except ImportError:
        from pathlib import Path
        utils_path = Path(__file__).resolve().parent / "src" / "utils"
        if str(utils_path) not in sys.path:
            sys.path.insert(0, str(utils_path))
        from graph_utils import adjacency_to_ancestor_matrix


def compute_ancestor_matrix_reference(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """Compute ancestor matrix using Warshall's algorithm."""
    n = adjacency_matrix.shape[0]
    ancestor = adjacency_matrix.clone().float()
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                ancestor[i, j] = torch.maximum(
                    ancestor[i, j],
                    torch.minimum(ancestor[i, k], ancestor[k, j])
                )
    
    return ancestor


def diagnose_single_sample(config_path: str, sample_idx: int = 0):
    """
    Diagnose a single sample in detail.
    """
    print(f"\n{'#'*80}")
    print(f"# Detailed Diagnostic for Sample {sample_idx}")
    print(f"{'#'*80}\n")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    scm_config = config['scm_config']
    dataset_config = config['dataset_config']
    preprocessing_config = config['preprocessing_config']
    
    # Configure to return ancestor matrix
    dataset_config_test = dataset_config.copy()
    dataset_config_test['return_ancestor_matrix'] = {'value': True}
    dataset_config_test['return_adjacency_matrix'] = {'value': False}
    
    print("Creating dataset...")
    dataset = InterventionalDataset(
        scm_config=scm_config,
        preprocessing_config=preprocessing_config,
        dataset_config=dataset_config_test,
        seed=42,
        return_scm=True
    )
    
    print(f"Fetching sample {sample_idx}...")
    result = dataset[sample_idx]
    
    X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, returned_matrix, scm, processor, intervention_node = result
    
    print(f"\n{'='*80}")
    print("SAMPLE INFORMATION")
    print(f"{'='*80}")
    print(f"Intervention node: {intervention_node}")
    print(f"Target node: {processor.selected_target_feature}")
    print(f"Kept features ({len(processor.kept_feature_indices)}): {processor.kept_feature_indices}")
    
    # Get node ordering
    target_node = processor.selected_target_feature
    kept_features = processor.kept_feature_indices
    ordered_nodes = [intervention_node, target_node] + kept_features
    
    print(f"\nOrdered nodes ({len(ordered_nodes)}): {ordered_nodes}")
    
    # Get SCM adjacency and compute reference ancestor
    scm_adjacency = scm.get_adjacency_matrix(node_order=ordered_nodes)
    reference_ancestor = compute_ancestor_matrix_reference(scm_adjacency)
    
    # Also get the ancestor matrix from the utility function
    utility_ancestor = adjacency_to_ancestor_matrix(scm_adjacency)
    
    print(f"\n{'='*80}")
    print("MATRIX SHAPES")
    print(f"{'='*80}")
    print(f"SCM adjacency shape: {scm_adjacency.shape}")
    print(f"Reference ancestor shape: {reference_ancestor.shape}")
    print(f"Utility ancestor shape: {utility_ancestor.shape}")
    print(f"Returned matrix shape: {returned_matrix.shape}")
    
    # Check if the utility function matches our reference
    real_n = len(ordered_nodes)
    util_matches_ref = torch.allclose(
        utility_ancestor[:real_n, :real_n],
        reference_ancestor[:real_n, :real_n]
    )
    print(f"\nUtility function matches reference: {util_matches_ref}")
    
    # Handle three-state format
    if dataset.use_partial_graph_format:
        print(f"\nDataset uses partial graph format (three-state: {{-1, 0, 1}})")
        returned_binary = (returned_matrix + 1.0) / 2.0
        known_mask = returned_matrix != 0.0
        
        num_known = known_mask[:real_n, :real_n].sum().item()
        total_entries = real_n * real_n
        print(f"Known entries: {num_known}/{total_entries} ({100*num_known/total_entries:.1f}%)")
        
        hide_frac = dataset.dataset_samplers.get('hide_fraction_matrix')
        if hide_frac:
            print(f"Hide fraction sampler: {hide_frac}")
    else:
        print(f"\nDataset uses binary format ({{0, 1}})")
        returned_binary = returned_matrix
        known_mask = torch.ones_like(returned_matrix, dtype=torch.bool)
    
    # Compare matrices
    ret_mat_real = returned_binary[:real_n, :real_n]
    ref_anc_real = reference_ancestor[:real_n, :real_n]
    known_mask_real = known_mask[:real_n, :real_n]
    
    print(f"\n{'='*80}")
    print("MISMATCH ANALYSIS")
    print(f"{'='*80}")
    
    # Find mismatches
    mismatches = []
    for i in range(real_n):
        for j in range(real_n):
            if not known_mask_real[i, j]:
                continue
            
            ref_val = ref_anc_real[i, j].item()
            ret_val = ret_mat_real[i, j].item()
            
            if abs(ref_val - ret_val) > 0.01:
                src_node = ordered_nodes[i]
                dst_node = ordered_nodes[j]
                mismatches.append((i, j, src_node, dst_node, ref_val, ret_val))
    
    if mismatches:
        print(f"\nFound {len(mismatches)} mismatches:")
        for i, j, src, dst, ref, ret in mismatches[:20]:
            print(f"  [{i:2d},{j:2d}] {str(src):>5s}→{str(dst):>5s}: ref={ref:.2f}, ret={ret:.2f}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
        
        # Analyze pattern of mismatches
        print(f"\n{'='*80}")
        print("MISMATCH PATTERNS")
        print(f"{'='*80}")
        
        # Check if mismatches are primarily in specific rows/columns
        mismatch_rows = {}
        mismatch_cols = {}
        for i, j, src, dst, ref, ret in mismatches:
            mismatch_rows[i] = mismatch_rows.get(i, 0) + 1
            mismatch_cols[j] = mismatch_cols.get(j, 0) + 1
        
        print("\nMismatches by source node (rows):")
        for i in sorted(mismatch_rows.keys(), key=lambda x: mismatch_rows[x], reverse=True)[:10]:
            print(f"  Row {i:2d} ({str(ordered_nodes[i]):>5s}): {mismatch_rows[i]} mismatches")
        
        print("\nMismatches by target node (columns):")
        for j in sorted(mismatch_cols.keys(), key=lambda x: mismatch_cols[x], reverse=True)[:10]:
            print(f"  Col {j:2d} ({str(ordered_nodes[j]):>5s}): {mismatch_cols[j]} mismatches")
        
        # Check specific case: treatment node (column 0)
        treatment_mismatches = [m for m in mismatches if m[1] == 0]
        if treatment_mismatches:
            print(f"\nMismatches to treatment node ({ordered_nodes[0]}): {len(treatment_mismatches)}")
            for i, j, src, dst, ref, ret in treatment_mismatches[:10]:
                # Check if there's a path in the adjacency matrix
                has_direct_edge = scm_adjacency[i, j].item() > 0.5
                print(f"  {str(src):>5s}→{str(dst):>5s}: ref={ref:.2f}, ret={ret:.2f}, direct_edge={has_direct_edge}")
    else:
        print("\n✓ No mismatches found! Matrices match perfectly.")
    
    print(f"\n{'='*80}")
    print("ADJACENCY MATRIX SAMPLE (first 10x10)")
    print(f"{'='*80}")
    print(scm_adjacency[:min(10, real_n), :min(10, real_n)])
    
    print(f"\n{'='*80}")
    print("REFERENCE ANCESTOR MATRIX SAMPLE (first 10x10)")
    print(f"{'='*80}")
    print(reference_ancestor[:min(10, real_n), :min(10, real_n)])
    
    print(f"\n{'='*80}")
    print("RETURNED MATRIX SAMPLE (first 10x10)")
    print(f"{'='*80}")
    if dataset.use_partial_graph_format:
        print("(Three-state format: -1=no edge, 0=unknown, 1=edge)")
    print(returned_matrix[:min(10, real_n), :min(10, real_n)])


if __name__ == "__main__":
    config_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/lingaus_ancestor_50node_idk_gcn_and_softatt_16760512.0_manipulated/final_config.yaml"
    
    # Diagnose sample 0 (which failed in the test)
    diagnose_single_sample(config_path, sample_idx=0)
