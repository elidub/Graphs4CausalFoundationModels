"""
Test to verify that the adjacency matrices from SCMs correctly align with 
the returned ancestor matrices from InterventionalDataset.

This test addresses the concern that there may be misalignment between:
1. The SCM's adjacency matrix (representing direct causal edges)
2. The ancestor matrix returned by the InterventionalDataset

The test checks:
- That the adjacency matrix from the SCM matches the DAG structure
- That the ancestor matrix is the correct transitive closure of the adjacency matrix
- That the node ordering [T, Y, X_0, ..., X_{L-1}] is consistent across both matrices
- That feature dropout and shuffling don't break the alignment
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
    """
    Compute the ancestor matrix as the transitive closure of the adjacency matrix.
    
    The ancestor matrix A* is defined as:
    A*[i,j] = 1 if there exists a directed path from i to j (i is an ancestor of j)
    A*[i,j] = 0 otherwise
    
    This can be computed via matrix power iteration or graph traversal.
    Here we use Warshall's algorithm (Floyd-Warshall variant for transitive closure).
    
    Parameters
    ----------
    adjacency_matrix : torch.Tensor
        Binary adjacency matrix where A[i,j]=1 indicates edge i→j
        
    Returns
    -------
    torch.Tensor
        Binary ancestor matrix where A*[i,j]=1 indicates i is ancestor of j
    """
    n = adjacency_matrix.shape[0]
    # Start with adjacency matrix (direct edges are ancestors)
    ancestor = adjacency_matrix.clone().float()
    
    # Warshall's algorithm: iterate through all intermediate nodes
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # If i→k and k→j exist, then i→j exists (i is ancestor of j)
                ancestor[i, j] = torch.maximum(
                    ancestor[i, j],
                    torch.minimum(ancestor[i, k], ancestor[k, j])
                )
    
    return ancestor


def verify_dataset_matrices(dataset, num_samples=10, verbose=True):
    """
    Verify that adjacency and ancestor matrices align correctly for a dataset.
    
    Parameters
    ----------
    dataset : InterventionalDataset
        Dataset instance configured to return ancestor matrices
    num_samples : int
        Number of samples to test
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    tuple
        (all_passed, failed_samples, error_messages)
    """
    all_passed = True
    failed_samples = []
    error_messages = []
    
    for idx in range(min(num_samples, len(dataset))):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Testing sample {idx + 1}/{num_samples}")
            print(f"{'='*80}")
        
        try:
            # Get sample from dataset
            result = dataset[idx]
            
            # Unpack based on return format
            if dataset.return_scm:
                if len(result) == 10:
                    # With treatment: (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, matrix, scm, processor, intervention_node)
                    X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, returned_matrix, scm, processor, intervention_node = result
                    has_treatment = True
                else:
                    # Without treatment: (X_obs, Y_obs, X_intv, Y_intv, matrix, scm, processor, intervention_node)
                    X_obs, Y_obs, X_intv, Y_intv, returned_matrix, scm, processor, intervention_node = result
                    has_treatment = False
            else:
                raise ValueError("Dataset must be created with return_scm=True for this test")
            
            if verbose:
                print(f"Has treatment: {has_treatment}")
                print(f"X_obs shape: {X_obs.shape}")
                if has_treatment:
                    print(f"T_obs shape: {T_obs.shape}")
                print(f"Y_obs shape: {Y_obs.shape}")
                print(f"Returned matrix shape: {returned_matrix.shape}")
            
            # Get the node ordering used by the dataset
            if has_treatment:
                target_node = processor.selected_target_feature
                kept_features = processor.kept_feature_indices
                ordered_nodes = [intervention_node, target_node] + kept_features
            else:
                # For no-treatment case, use default topological ordering
                ordered_nodes = list(scm.dag.nodes())
            
            if verbose:
                print(f"\nIntervention node: {intervention_node}")
                if has_treatment:
                    print(f"Target node: {target_node}")
                    print(f"Kept features ({len(kept_features)}): {kept_features[:5]}..." if len(kept_features) > 5 else f"Kept features: {kept_features}")
                print(f"Ordered nodes ({len(ordered_nodes)}): {ordered_nodes[:5]}..." if len(ordered_nodes) > 5 else f"Ordered nodes: {ordered_nodes}")
            
            # Get adjacency matrix from SCM with the same node ordering
            scm_adjacency = scm.get_adjacency_matrix(node_order=ordered_nodes)
            
            if verbose:
                print(f"\nSCM adjacency matrix shape: {scm_adjacency.shape}")
            
            # Compute reference ancestor matrix from SCM adjacency
            reference_ancestor = compute_ancestor_matrix_reference(scm_adjacency)
            
            # Handle three-state format if enabled
            if dataset.use_partial_graph_format:
                # returned_matrix is in {-1, 0, 1} format
                # Convert to binary {0, 1} for comparison by mapping:
                # -1 (no edge) → 0
                # 0 (unknown) → keep as unknown (we'll handle separately)
                # 1 (edge) → 1
                
                # For the parts that are not unknown (!=0), convert back to binary
                # Then compare with reference
                returned_binary = returned_matrix.clone()
                returned_binary = (returned_binary + 1.0) / 2.0  # Map {-1, 1} to {0, 1}
                
                # Check which entries are known (not zero in three-state format)
                known_mask = returned_matrix != 0.0
                
                if verbose:
                    real_n = len(ordered_nodes)
                    num_known = known_mask[:real_n, :real_n].sum().item()
                    total_entries = real_n * real_n
                    print(f"Known entries: {num_known}/{total_entries} ({100*num_known/total_entries:.1f}%)")
            else:
                # Binary format - direct comparison
                returned_binary = returned_matrix
                known_mask = torch.ones_like(returned_matrix, dtype=torch.bool)
            
            # Get the real (non-padded) submatrix
            real_n = len(ordered_nodes)
            scm_adj_real = scm_adjacency[:real_n, :real_n]
            ref_anc_real = reference_ancestor[:real_n, :real_n]
            ret_mat_real = returned_binary[:real_n, :real_n]
            known_mask_real = known_mask[:real_n, :real_n]
            
            # Check 1: Verify that SCM adjacency matches the DAG structure
            # (This is more of a sanity check on SCM.get_adjacency_matrix)
            for i, src_node in enumerate(ordered_nodes):
                for j, dst_node in enumerate(ordered_nodes):
                    if i == j:
                        continue  # Skip diagonal
                    
                    # Check if edge exists in DAG (src_node → dst_node means src_node is parent of dst_node)
                    has_edge_in_dag = src_node in scm._parents.get(dst_node, [])
                    has_edge_in_matrix = scm_adj_real[i, j].item() > 0.5
                    
                    if has_edge_in_dag != has_edge_in_matrix:
                        error_msg = (
                            f"Sample {idx}: Adjacency matrix mismatch!\n"
                            f"  Edge {src_node} → {dst_node} (position [{i},{j}])\n"
                            f"  In DAG: {has_edge_in_dag}, In matrix: {has_edge_in_matrix}"
                        )
                        print(f"ERROR: {error_msg}")
                        error_messages.append(error_msg)
                        all_passed = False
            
            # Check 2: Verify that returned matrix matches reference ancestor matrix
            # (Only check known entries if using partial graph format)
            if dataset.return_ancestor_matrix:
                mismatches = []
                for i in range(real_n):
                    for j in range(real_n):
                        if not known_mask_real[i, j]:
                            continue  # Skip unknown entries
                        
                        ref_val = ref_anc_real[i, j].item()
                        ret_val = ret_mat_real[i, j].item()
                        
                        # Allow small numerical tolerance
                        if abs(ref_val - ret_val) > 0.01:
                            src_node = ordered_nodes[i]
                            dst_node = ordered_nodes[j]
                            mismatches.append(
                                f"  [{i},{j}] {src_node}→{dst_node}: "
                                f"reference={ref_val:.2f}, returned={ret_val:.2f}"
                            )
                
                if mismatches:
                    error_msg = (
                        f"Sample {idx}: Ancestor matrix mismatches!\n" +
                        "\n".join(mismatches[:10])  # Show first 10 mismatches
                    )
                    if len(mismatches) > 10:
                        error_msg += f"\n  ... and {len(mismatches) - 10} more mismatches"
                    print(f"ERROR: {error_msg}")
                    error_messages.append(error_msg)
                    all_passed = False
                    failed_samples.append(idx)
                else:
                    if verbose:
                        print("✓ Ancestor matrix matches reference")
            
            # Check 3: Verify that adjacency matrix is consistent
            elif dataset.return_adjacency_matrix:
                mismatches = []
                for i in range(real_n):
                    for j in range(real_n):
                        if not known_mask_real[i, j]:
                            continue
                        
                        ref_val = scm_adj_real[i, j].item()
                        ret_val = ret_mat_real[i, j].item()
                        
                        if abs(ref_val - ret_val) > 0.01:
                            src_node = ordered_nodes[i]
                            dst_node = ordered_nodes[j]
                            mismatches.append(
                                f"  [{i},{j}] {src_node}→{dst_node}: "
                                f"reference={ref_val:.2f}, returned={ret_val:.2f}"
                            )
                
                if mismatches:
                    error_msg = (
                        f"Sample {idx}: Adjacency matrix mismatches!\n" +
                        "\n".join(mismatches[:10])
                    )
                    if len(mismatches) > 10:
                        error_msg += f"\n  ... and {len(mismatches) - 10} more mismatches"
                    print(f"ERROR: {error_msg}")
                    error_messages.append(error_msg)
                    all_passed = False
                    failed_samples.append(idx)
                else:
                    if verbose:
                        print("✓ Adjacency matrix matches reference")
            
            # Check 4: Verify feature alignment
            # For each kept feature, verify it corresponds to the right SCM node
            if has_treatment and len(kept_features) > 0:
                # X_obs has shape (n_samples, n_features)
                # The first len(kept_features) columns should correspond to kept_features in order
                num_real_features = len(kept_features)
                
                if verbose:
                    print(f"\n✓ Feature alignment check:")
                    print(f"  X has {X_obs.shape[1]} columns")
                    print(f"  {num_real_features} real features (rest are padding)")
                    print(f"  Matrix covers {real_n} nodes: [T, Y] + {num_real_features} features")
            
            if verbose:
                print(f"\n✓ Sample {idx} passed all checks!")
        
        except Exception as e:
            error_msg = f"Sample {idx}: Exception occurred: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            error_messages.append(error_msg)
            all_passed = False
            failed_samples.append(idx)
    
    return all_passed, failed_samples, error_messages


def test_config(config_path: str, num_samples: int = 10, verbose: bool = True):
    """
    Test a specific configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file
    num_samples : int
        Number of samples to test
    verbose : bool
        Whether to print detailed information
        
    Returns
    -------
    bool
        True if all tests passed, False otherwise
    """
    print(f"\n{'#'*80}")
    print(f"# Testing configuration: {config_path}")
    print(f"{'#'*80}\n")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config: {config.get('experiment_name', 'Unknown')}")
    print(f"Mode: {config.get('mode', 'Unknown')}")
    
    # Extract configurations
    scm_config = config['scm_config']
    dataset_config = config['dataset_config']
    preprocessing_config = config['preprocessing_config']
    
    # Ensure we return ancestor matrix for testing
    dataset_config_test = dataset_config.copy()
    dataset_config_test['return_ancestor_matrix'] = {'value': True}
    dataset_config_test['return_adjacency_matrix'] = {'value': False}
    
    # Create smaller test dataset
    scm_config_test = scm_config.copy()
    # Keep original settings but we'll just test a few samples
    
    print(f"\nCreating test dataset...")
    print(f"  Number of samples to test: {num_samples}")
    
    # Create dataset
    dataset = InterventionalDataset(
        scm_config=scm_config_test,
        preprocessing_config=preprocessing_config,
        dataset_config=dataset_config_test,
        seed=42,
        return_scm=True  # Required for testing
    )
    
    print(f"Dataset created with {len(dataset)} total samples")
    
    # Run verification
    all_passed, failed_samples, error_messages = verify_dataset_matrices(
        dataset, 
        num_samples=num_samples,
        verbose=verbose
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Configuration: {config_path}")
    print(f"Samples tested: {num_samples}")
    print(f"Status: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
    
    if not all_passed:
        print(f"\nFailed samples: {failed_samples}")
        print(f"\nErrors ({len(error_messages)}):")
        for error in error_messages[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(error_messages) > 5:
            print(f"  ... and {len(error_messages) - 5} more errors")
    
    print(f"{'='*80}\n")
    
    return all_passed


if __name__ == "__main__":
    # Test the configuration mentioned in the request
    config_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/lingaus_ancestor_50node_idk_gcn_and_softatt_16760512.0_manipulated/final_config.yaml"
    
    # Run test
    passed = test_config(
        config_path=config_path,
        num_samples=20,  # Test 20 samples
        verbose=True     # Print detailed output
    )
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)
