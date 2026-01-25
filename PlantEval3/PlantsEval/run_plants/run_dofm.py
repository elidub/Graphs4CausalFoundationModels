#!/usr/bin/env python3
"""
Simplified DOFM for PlantEval3.

Uses GraphConditionedInterventionalPFNSklearn wrapper directly.
Constructs adjacency/ancestor matrix in this script.
"""

import argparse
import sys
import os
import numpy as np

# Import from same directory (relative import for cluster compatibility)
from eval import evaluate_pipeline
from src.models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn


def compute_ancestor_matrix(adjacency_matrix):
    """
    Compute the ancestor matrix (transitive closure) from an adjacency matrix.
    
    Uses Warshall's algorithm to compute transitive closure.
    Works with three-state format {-1, 0, 1}:
    - If A[i,j] = 1, then i is a direct parent of j
    - After transitive closure, Anc[i,j] = 1 means i is an ancestor of j (directed path exists)
    - Entries with -1 (no edge) and 0 (unknown) are preserved where possible
    
    Args:
        adjacency_matrix: Binary or three-state adjacency matrix (n x n)
    
    Returns:
        Ancestor matrix (n x n) in same format
    """
    n = adjacency_matrix.shape[0]
    ancestor = adjacency_matrix.copy()
    
    # Warshall's algorithm for transitive closure
    # For three-state matrices: only propagate known edges (value = 1)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # If there's a known path i -> k (ancestor[i,k] == 1) 
                # and a known path k -> j (ancestor[k,j] == 1)
                # then i is an ancestor of j
                if ancestor[i, k] == 1 and ancestor[k, j] == 1:
                    ancestor[i, j] = 1
    
    return ancestor

def construct_ancestor_matrix(n_features=6, graph_mode="full_graph"):
    """
    Construct ancestor matrix for plant data.
    
    Plant causal structure (6 features):
      0: X (position)
      1: Y (position) 
      2: Bed_Position
      3: Red_Light_Intensity
      4: Blue_Light_Intensity
      5: Far_Red_Light_Intensity
      6: Treatment (Light_Intensity) 
      7: Outcome (Biomass)
    
    Matrix format for model: [T(0), Y(1), X(2), Y_pos(3), Bed(4), Red(5), Blue(6), Far_Red(7)]
    
    Three-state format:
      -1.0 = definitely NO ancestral relationship (we're certain)
       0.0 = UNKNOWN ancestral relationship (we don't know)
       1.0 = ANCESTRAL relationship EXISTS (i is ancestor of j)
    
    Args:
        n_features: Number of features (6 for plant data, may need padding to 50)
        graph_mode: "full_graph", "all_unknown", "t_to_y_only"
    
    Returns:
        Ancestor matrix of shape (n_features+2, n_features+2) in three-state format
        Entry [i,j] = 1 means i is an ancestor of j (directed path from i to j exists)
    """
    size = n_features + 2  # +2 for treatment and outcome
    n_real_features = 6  # Plant data has 6 real features
    
    # Initialize with -1 (no ancestral relationship) for padded features
    # This tells the model: "these padded nodes don't exist, so no ancestral relationships"
    adj = -1.0 * np.ones((size, size), dtype=np.float32)
    
    if graph_mode == "all_unknown":
        # All ancestral relationships unknown (0) for the real features
        adj[:n_real_features+2, :n_real_features+2] = 0.0
        # Padded features stay at -1 (definitely no ancestral relationship)
        print("[Graph] All unknown (zeros for real features, -1 for padding)")
        return adj
    
    elif graph_mode == "t_to_y_only":
        # Start with all unknown for real features
        adj[:n_real_features+2, :n_real_features+2] = 0.0
        # Only T is ancestor of Y (direct edge)
        adj[0, 1] = 1.0  # T -> Y
        print("[Graph] Only T ancestor of Y known, rest unknown")
        return adj
    
    elif graph_mode == "full_graph":
        # Build adjacency matrix first, then compute transitive closure
        # Start with all UNKNOWN (0) for real features
        adj[:n_real_features+2, :n_real_features+2] = 0.0
        
        # Known causal structure for plant data (direct edges)
        # Positions (X, Y) -> Bed_Position
        adj[2, 4] = 1.0  # X -> Bed
        adj[3, 4] = 1.0  # Y_pos -> Bed
        
        # Positions -> Light intensities (experimental design)
        adj[2, 5] = 1.0  # X -> Red
        adj[2, 6] = 1.0  # X -> Blue  
        adj[2, 7] = 1.0  # X -> Far_Red
        adj[3, 5] = 1.0  # Y_pos -> Red
        adj[3, 6] = 1.0  # Y_pos -> Blue
        adj[3, 7] = 1.0  # Y_pos -> Far_Red
        
        # Bed -> Light intensities
        adj[4, 5] = 1.0  # Bed -> Red
        adj[4, 6] = 1.0  # Bed -> Blue
        adj[4, 7] = 1.0  # Bed -> Far_Red
        
        # Light intensities -> Treatment (aggregate)
        adj[5, 0] = 1.0  # Red -> T
        adj[6, 0] = 1.0  # Blue -> T
        adj[7, 0] = 1.0  # Far_Red -> T
        
        # Treatment -> Outcome
        adj[0, 1] = 1.0  # T -> Y (Biomass)
        
        # Also: confounders (positions, bed) -> Outcome
        adj[2, 1] = 1.0  # X -> Biomass
        adj[3, 1] = 1.0  # Y_pos -> Biomass
        adj[4, 1] = 1.0  # Bed -> Biomass
        
        # Compute transitive closure to get ancestor matrix
        ancestor = compute_ancestor_matrix(adj)
        
        # Padded features remain at -1 (no ancestral relationships)
        print(f"[Graph] Adjacency: {(adj == 1).sum()} direct edges")
        print(f"[Graph] Ancestor matrix: {(ancestor == 1).sum()} ancestral relationships")
        print(f"[Graph] Real subgraph: {n_real_features+2}x{n_real_features+2}, Padded to: {size}x{size}")
        return ancestor
    
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")


def hide_edges_by_correlation(ancestor_matrix, X_train, t_train, y_train, top_k=10):
    """
    Hide all ancestral relationships except the top-k strongest correlations.
    
    Args:
        ancestor_matrix: Ancestor matrix with known relationships
        X_train: Training features
        t_train: Training treatment
        y_train: Training outcome
        top_k: Number of strongest correlations to keep
    
    Returns:
        Modified ancestor matrix with most relationships hidden (set to 0 = unknown)
    """
    n_real_features = X_train.shape[1]
    
    # Combine all variables: [T, Y, X_0, ..., X_{n-1}]
    all_vars = np.concatenate([t_train, y_train, X_train], axis=1)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(all_vars.T)
    
    # Get absolute correlations (we care about strength, not direction)
    abs_corr = np.abs(corr_matrix)
    
    # For each edge in the ancestor matrix, find its correlation
    # ancestor_matrix[i,j] = 1 means i is ancestor of j
    edge_correlations = []
    for i in range(n_real_features + 2):
        for j in range(n_real_features + 2):
            if ancestor_matrix[i, j] == 1:  # Known ancestral relationship
                corr_val = abs_corr[i, j]
                edge_correlations.append((i, j, corr_val))
    
    # Sort by correlation strength (descending)
    edge_correlations.sort(key=lambda x: x[2], reverse=True)
    
    # Create modified matrix: start by hiding all known edges (set to unknown)
    modified_matrix = ancestor_matrix.copy()
    
    # Set all known edges to unknown (0)
    for i, j, _ in edge_correlations:
        modified_matrix[i, j] = 0.0
    
    # Keep only top-k strongest correlations as known (+1)
    kept_edges = edge_correlations[:top_k]
    for i, j, corr_val in kept_edges:
        modified_matrix[i, j] = 1.0
        print(f"  Keeping edge [{i},{j}] with correlation {corr_val:.3f}")
    
    print(f"[Correlation filtering] Kept {len(kept_edges)} strongest edges, "
          f"hidden {len(edge_correlations) - len(kept_edges)} edges")
    
    return modified_matrix


def dofm_pipeline(model, cid_dataset):
    """
    Simple DOFM pipeline using GraphConditionedInterventionalPFNSklearn.
    
    Preprocessing:
    - Standardize all features (mean=0, std=1)
    - Standardize treatment (mean=0, std=1)
    - Scale outcome to (-1, 1)
    
    Args:
        model: GraphConditionedInterventionalPFNSklearn instance
        cid_dataset: Dataset with X_train, t_train, y_train, X_test, t_test, true_cid
    
    Returns:
        Predictions for test set (CID values)
    """
    # Extract data
    X_train = np.array(cid_dataset.X_train, dtype=np.float32)
    t_train = np.array(cid_dataset.t_train, dtype=np.float32).reshape(-1, 1)
    y_train = np.array(cid_dataset.y_train, dtype=np.float32).reshape(-1, 1)
    
    X_test = np.array(cid_dataset.X_test, dtype=np.float32)
    t_test = np.array(cid_dataset.t_test, dtype=np.float32).reshape(-1, 1)
    
    print(f"[Data] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X_train.shape[1]}")
    
    # ============ PREPROCESSING ============
    eps = 1e-8
    
    # 0. Target encoding for categorical features (if enabled)
    use_target_encoding = getattr(dofm_pipeline, 'use_target_encoding', False)
    if use_target_encoding:
        # Feature index 2 (Bed_Position) is categorical
        categorical_indices = [2]
        
        for cat_idx in categorical_indices:
            # Get unique categories from training data
            unique_cats = np.unique(X_train[:, cat_idx])
            
            # Compute mean target value for each category
            target_means = {}
            global_mean = y_train.mean()
            
            for cat in unique_cats:
                mask = X_train[:, cat_idx] == cat
                if mask.sum() > 0:
                    target_means[cat] = y_train[mask].mean()
                else:
                    target_means[cat] = global_mean
            
            # Replace categorical values with target means
            for cat in unique_cats:
                train_mask = X_train[:, cat_idx] == cat
                X_train[train_mask, cat_idx] = target_means[cat]
                
                test_mask = X_test[:, cat_idx] == cat
                # For unseen categories in test, use global mean
                if cat in target_means:
                    X_test[test_mask, cat_idx] = target_means[cat]
                else:
                    X_test[test_mask, cat_idx] = global_mean
            
            print(f"[Preprocessing] Target encoded feature {cat_idx} (Bed_Position) with {len(unique_cats)} categories")
    
    # 1. Standardize features (mean=0, std=1)
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std = X_train.std(axis=0, keepdims=True) + eps
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    print(f"[Preprocessing] Standardized features")
    
    # 2. Standardize treatment (mean=0, std=1)
    t_mean = t_train.mean()
    t_std = t_train.std() + eps
    t_train = (t_train - t_mean) / t_std
    t_test = (t_test - t_mean) / t_std
    print(f"[Preprocessing] Standardized treatment (mean={t_mean:.3f}, std={t_std:.3f})")
    
    # 3. Scale outcome to (-1, 1)
    y_min = y_train.min()
    y_max = y_train.max()
    y_range = y_max - y_min + eps
    y_train = 2.0 * (y_train - y_min) / y_range - 1.0
    # Note: y_test scaling not needed as we don't use it for prediction
    print(f"[Preprocessing] Scaled outcome to (-1, 1) (range: {y_min:.3f} to {y_max:.3f})")
    
    # Get model's expected number of features
    model_n_features = model.model.num_features
    n_real_features = X_train.shape[1]
    
    print(f"[Model] Expected features: {model_n_features}, Actual features: {n_real_features}")
    
    # Pad features if needed
    if n_real_features < model_n_features:
        pad_width = model_n_features - n_real_features
        X_train = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        X_test = np.pad(X_test, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        print(f"[Padding] Added {pad_width} zero features")
    
    # Construct ancestor matrix (transitive closure of causal graph)
    graph_mode = getattr(dofm_pipeline, 'graph_mode', 'full_graph')
    ancestor_matrix = construct_ancestor_matrix(model_n_features, graph_mode)
    
    # Optionally filter edges by correlation (hide all but top-k)
    hide_by_correlation = getattr(dofm_pipeline, 'hide_by_correlation', False)
    if hide_by_correlation and graph_mode == "full_graph":
        top_k = getattr(dofm_pipeline, 'top_k_edges', 10)
        print(f"[Correlation filtering] Hiding all but top-{top_k} strongest correlations")
        # Use standardized data for correlation computation (BEFORE padding)
        X_train_unpadded = X_train[:, :n_real_features]
        ancestor_matrix = hide_edges_by_correlation(
            ancestor_matrix, X_train_unpadded, t_train, y_train, top_k=top_k
        )
    
    print(f"[Graph] Matrix shape: {ancestor_matrix.shape}")
    print(f"[Graph] Ancestral relationships: {(ancestor_matrix == 1).sum()} known (+1), "
          f"{(ancestor_matrix == 0).sum()} unknown (0), "
          f"{(ancestor_matrix == -1).sum()} none (-1)")
    
    # Make predictions using the sklearn wrapper
    cid_pred_scaled = model.predict(
        X_obs=X_train,
        T_obs=t_train,
        Y_obs=y_train,
        X_intv=X_test,
        T_intv=t_test,
        adjacency_matrix=ancestor_matrix,  # Pass ancestor matrix
        prediction_type="mean",  # Use mean for continuous outcomes
        batched=False
    )
    
    # ============ UNDO SCALING ============
    # Reverse the outcome scaling from (-1, 1) back to original range
    # Original transformation: y_scaled = 2.0 * (y - y_min) / y_range - 1.0
    # Reverse: y = (y_scaled + 1.0) * y_range / 2.0 + y_min
    cid_pred = (cid_pred_scaled + 1.0) * y_range / 2.0 + y_min
    print(f"[Postprocessing] Unscaled predictions back to original outcome range")
    
    return cid_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DOFM on PlantEval3.")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)    
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--graph_mode", type=str, default="full_graph",
                        choices=["full_graph", "all_unknown", "t_to_y_only"])
    parser.add_argument("--hide_by_correlation", action="store_true",
                        help="Hide all edges except top-k strongest correlations")
    parser.add_argument("--top_k_edges", type=int, default=10,
                        help="Number of strongest correlation edges to keep")
    parser.add_argument("--use_target_encoding", action="store_true",
                        help="Apply target encoding to categorical features (Bed_Position)")

    args = parser.parse_args()
    
    # Default checkpoint (update this to your model path)
    if args.checkpoint_path is None:
        checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist.pt"
    else:
        checkpoint_path = args.checkpoint_path
        
    if args.config_path is None:
        config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist_config.yaml"
    else:
        config_path = args.config_path
    
    print(f"--- Starting DOFM PlantEval3 Experiment ---")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")
    print(f"Graph mode: {args.graph_mode}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    
    # Load model using the sklearn wrapper
    model = GraphConditionedInterventionalPFNSklearn(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=None,  # Auto-detect GPU
        verbose=True
    )
    model.load()
    
    # Store graph_mode and correlation filtering settings in the pipeline function for access
    dofm_pipeline.graph_mode = args.graph_mode
    dofm_pipeline.hide_by_correlation = args.hide_by_correlation
    dofm_pipeline.top_k_edges = args.top_k_edges
    dofm_pipeline.use_target_encoding = args.use_target_encoding
    
    # Run evaluation
    evaluate_pipeline(
        exp_name=args.exp_name,
        model_pipeline=dofm_pipeline,
        model=model,
        args=args
    )
    
    print(f"--- Experiment Finished ---")
