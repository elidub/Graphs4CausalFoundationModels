"""
DOFM for Plant CID benchmark using PreprocessingGraphConditionedPFN wrapper.

Uses the same preprocessing approach as dofm_full_conditioning.py from RealCauseEval.
"""

import argparse
import sys
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting')
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval/PlantsEval')

from run_plants.eval import evaluate_pipeline
from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN


def encode_mixed_data(X_train, X_test, y_train=None, max_cardinality_for_target_encoding=20):
    """
    Encode mixed numerical/categorical data using target encoding.
    
    For categorical columns:
    - If y_train is provided AND cardinality <= max_cardinality_for_target_encoding:
      use target encoding (mean of Y for each category)
    - Otherwise: fall back to label encoding
    
    High-cardinality features (like Unique_Identifier) use label encoding
    to avoid data leakage.
    
    Returns encoded arrays and list of categorical column indices.
    """
    n_cols = X_train.shape[1]
    X_train_encoded = np.zeros(X_train.shape, dtype=np.float32)
    X_test_encoded = np.zeros(X_test.shape, dtype=np.float32)
    categorical_indices = []
    
    # Flatten y_train if provided
    y_flat = y_train.flatten() if y_train is not None else None
    
    for col_idx in range(n_cols):
        train_col = X_train[:, col_idx]
        test_col = X_test[:, col_idx]
        
        try:
            X_train_encoded[:, col_idx] = train_col.astype(np.float32)
            X_test_encoded[:, col_idx] = test_col.astype(np.float32)
        except (ValueError, TypeError):
            categorical_indices.append(col_idx)
            
            n_unique = len(np.unique(train_col))
            
            # Only use target encoding for low-cardinality features
            if y_flat is not None and n_unique <= max_cardinality_for_target_encoding:
                # Target encoding: replace category with mean(Y|category)
                # Build mapping from category -> mean(Y)
                category_means = {}
                global_mean = np.mean(y_flat)
                
                for cat in np.unique(train_col):
                    mask = train_col == cat
                    if np.sum(mask) > 0:
                        category_means[cat] = np.mean(y_flat[mask])
                    else:
                        category_means[cat] = global_mean
                
                # Encode train
                X_train_encoded[:, col_idx] = np.array([
                    category_means.get(cat, global_mean) for cat in train_col
                ], dtype=np.float32)
                
                # Encode test (use global mean for unseen categories)
                X_test_encoded[:, col_idx] = np.array([
                    category_means.get(cat, global_mean) for cat in test_col
                ], dtype=np.float32)
                
                print(f"  Col {col_idx}: Target encoded ({n_unique} categories)")
            else:
                # Label encoding for high-cardinality or when y_train not available
                le = LabelEncoder()
                all_vals = np.concatenate([train_col, test_col])
                le.fit(all_vals)
                X_train_encoded[:, col_idx] = le.transform(train_col).astype(np.float32)
                X_test_encoded[:, col_idx] = le.transform(test_col).astype(np.float32)
                print(f"  Col {col_idx}: Label encoded ({n_unique} categories - high cardinality)")
    
    return X_train_encoded, X_test_encoded, categorical_indices


def encode_with_onehot(X_train, X_test, onehot_columns):
    """
    One-hot encode specified columns and return encoded data with feature mapping.
    
    Args:
        X_train: Training features (n_train, n_features)
        X_test: Test features (n_test, n_features)
        onehot_columns: List of column indices to one-hot encode (by original FEATURE_NAMES index)
    
    Returns:
        X_train_encoded: Encoded training features
        X_test_encoded: Encoded test features  
        feature_mapping: List where feature_mapping[new_idx] = original_feature_name
    """
    n_cols = X_train.shape[1]
    train_cols = []
    test_cols = []
    feature_mapping = []  # Maps new column index -> original feature name
    
    for col_idx in range(n_cols):
        orig_name = FEATURE_NAMES[col_idx]
        train_col = X_train[:, col_idx]
        test_col = X_test[:, col_idx]
        
        if col_idx in onehot_columns:
            # One-hot encode this column
            # Get all unique categories from both train and test
            all_cats = np.unique(np.concatenate([train_col, test_col]))
            all_cats = sorted([str(c) for c in all_cats])  # Sort for consistency
            
            for cat in all_cats:
                train_cols.append((train_col == cat).astype(np.float32))
                test_cols.append((test_col == cat).astype(np.float32))
                feature_mapping.append(orig_name)  # All one-hot columns map to original feature
            
            print(f"  Col {col_idx} ({orig_name}): One-hot encoded -> {len(all_cats)} columns")
        else:
            # Keep as-is (numeric or already encoded)
            try:
                train_cols.append(train_col.astype(np.float32))
                test_cols.append(test_col.astype(np.float32))
            except (ValueError, TypeError):
                # Label encode if categorical
                le = LabelEncoder()
                all_vals = np.concatenate([train_col, test_col])
                le.fit(all_vals)
                train_cols.append(le.transform(train_col).astype(np.float32))
                test_cols.append(le.transform(test_col).astype(np.float32))
                print(f"  Col {col_idx} ({orig_name}): Label encoded")
            feature_mapping.append(orig_name)
    
    X_train_encoded = np.column_stack(train_cols)
    X_test_encoded = np.column_stack(test_cols)
    
    print(f"  Total features after one-hot: {X_train_encoded.shape[1]} (from {n_cols})")
    
    return X_train_encoded, X_test_encoded, feature_mapping


# Feature name to index mapping for the Plant CID dataset
# X_train columns: X, Y, Position_in_Bed, Bed_Position, Unique_Identifier, Microgreen,
#                  Red_Light_Intensity_umols, Blue_Light_Intensity_umols, Far_Red_Light_Intensity_umols
FEATURE_NAMES = [
    "X", "Y", "Position_in_Bed", "Bed_Position", "Unique_Identifier", "Microgreen",
    "Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", "Far_Red_Light_Intensity_umols"
]
FEATURE_TO_IDX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

# Define causal roles for each original feature
POSITIONAL_FEATURES = ["X", "Y", "Position_in_Bed", "Bed_Position", "Unique_Identifier"]
LIGHT_FEATURES = ["Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", "Far_Red_Light_Intensity_umols"]
MICROGREEN_FEATURE = "Microgreen"


def build_plant_adjacency_with_onehot(model_n_features, n_real_features, feature_mapping, graph_mode="full_graph"):
    """
    Build adjacency matrix for plant experiment with one-hot encoded features.
    
    Uses feature_mapping to understand which encoded columns correspond to which
    original features, and applies causal structure accordingly.
    
    Args:
        model_n_features: Number of features the model expects
        n_real_features: Actual number of encoded features
        feature_mapping: List where feature_mapping[i] = original feature name for column i
        graph_mode: "all_unknown", "full_graph", or "true_graph"
    """
    T_idx = 0  # Light_Intensity_umols (treatment)
    Y_idx = 1  # Biomass_g (outcome)
    feature_offset = 2
    
    def get_feature_indices(feature_name):
        """Get all adjacency matrix indices for a given original feature name."""
        indices = []
        for i, name in enumerate(feature_mapping):
            if name == feature_name:
                indices.append(feature_offset + i)
        return indices
    
    if graph_mode == "all_unknown":
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        print(f"Graph knowledge: ALL UNKNOWN (one-hot encoding)")
    
    elif graph_mode == "t_to_y_only":
        # Only tell the model that T causes Y, nothing else known
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        adjacency_matrix[T_idx, Y_idx] = 1.0
        print(f"Graph knowledge: T->Y ONLY (one-hot encoding)")
        
    elif graph_mode == "full_graph":
        # Full knowledge: start with all -1 (no edges), then set known edges to 1
        adjacency_matrix = -np.ones((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        # T -> Y (Light_Intensity -> Biomass)
        adjacency_matrix[T_idx, Y_idx] = 1.0
        
        # All features -> Y (everything affects outcome)
        for orig_name in FEATURE_NAMES:
            for idx in get_feature_indices(orig_name):
                adjacency_matrix[idx, Y_idx] = 1.0
        
        # Light sources -> T
        for light_name in LIGHT_FEATURES:
            for idx in get_feature_indices(light_name):
                adjacency_matrix[idx, T_idx] = 1.0
        
        # Positional features -> T
        for pos_name in POSITIONAL_FEATURES:
            for idx in get_feature_indices(pos_name):
                adjacency_matrix[idx, T_idx] = 1.0
        
        # Positional features -> Light sources
        for pos_name in POSITIONAL_FEATURES:
            for pos_idx in get_feature_indices(pos_name):
                for light_name in LIGHT_FEATURES:
                    for light_idx in get_feature_indices(light_name):
                        adjacency_matrix[pos_idx, light_idx] = 1.0
        
        print(f"Graph knowledge: TRUE CAUSAL GRAPH (full knowledge, one-hot)")
        print(f"  - Real features: {n_real_features} (mapped from {len(FEATURE_NAMES)} original)")
        
    elif graph_mode == "true_graph":
        # Partial knowledge
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        # T -> Y
        adjacency_matrix[T_idx, Y_idx] = 1.0
        
        # All features -> Y
        for orig_name in FEATURE_NAMES:
            for idx in get_feature_indices(orig_name):
                adjacency_matrix[idx, Y_idx] = 1.0
        
        # Light sources -> T
        for light_name in LIGHT_FEATURES:
            for idx in get_feature_indices(light_name):
                adjacency_matrix[idx, T_idx] = 1.0
        
        # Positional features -> T
        for pos_name in POSITIONAL_FEATURES:
            for idx in get_feature_indices(pos_name):
                adjacency_matrix[idx, T_idx] = 1.0
        
        # Positional features -> Light sources
        for pos_name in POSITIONAL_FEATURES:
            for pos_idx in get_feature_indices(pos_name):
                for light_name in LIGHT_FEATURES:
                    for light_idx in get_feature_indices(light_name):
                        adjacency_matrix[pos_idx, light_idx] = 1.0
        
        # Known non-edges:
        # Y doesn't cause anything
        adjacency_matrix[Y_idx, :] = -1.0
        
        # Microgreen doesn't cause positional info, light components, or T
        for micro_idx in get_feature_indices(MICROGREEN_FEATURE):
            for pos_name in POSITIONAL_FEATURES:
                for pos_idx in get_feature_indices(pos_name):
                    adjacency_matrix[micro_idx, pos_idx] = -1.0
            for light_name in LIGHT_FEATURES:
                for light_idx in get_feature_indices(light_name):
                    adjacency_matrix[micro_idx, light_idx] = -1.0
            adjacency_matrix[micro_idx, T_idx] = -1.0
        
        print(f"Graph knowledge: TRUE CAUSAL GRAPH (partial knowledge, one-hot)")
        print(f"  - Real features: {n_real_features} (mapped from {len(FEATURE_NAMES)} original)")
        
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")
    
    # PADDED features: Set all edges to -1
    for i in range(n_real_features, model_n_features):
        feat_idx_pad = feature_offset + i
        adjacency_matrix[feat_idx_pad, :] = -1.0
        adjacency_matrix[:, feat_idx_pad] = -1.0
        adjacency_matrix[feat_idx_pad, feat_idx_pad] = -1.0
    
    return adjacency_matrix


def build_plant_causal_graph_adjacency(model_n_features, n_real_features, graph_mode="full_graph"):
    """
    Build adjacency matrix based on the true causal graph of the plant experiment.
    
    The causal structure is:
    - Treatment (T) = Light_Intensity_umols
    - Outcome (Y) = Biomass_g
    - Features affect T and Y according to the known causal structure
    
    Adjacency matrix layout: [T, Y, X_0, X_1, ..., X_n]
    - T_idx = 0
    - Y_idx = 1
    - Feature i has index (2 + i)
    
    Matrix values:
    - 1.0 = edge exists (i is ancestor of j)
    - -1.0 = NO edge (i is NOT ancestor of j) 
    - 0.0 = unknown
    """
    T_idx = 0  # Light_Intensity_umols
    Y_idx = 1  # Biomass_g
    feature_offset = 2
    
    def feat_idx(name):
        """Get adjacency matrix index for a feature name."""
        return feature_offset + FEATURE_TO_IDX[name]
    
    if graph_mode == "all_unknown":
        # All unknown: everything is 0
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        print(f"Graph knowledge: ALL UNKNOWN")
    
    elif graph_mode == "t_to_y_only":
        # Only tell the model that T causes Y, nothing else known
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        adjacency_matrix[T_idx, Y_idx] = 1.0
        print(f"Graph knowledge: T->Y ONLY")
        
    elif graph_mode == "full_graph":
        # Full knowledge: start with all -1 (no edges), then set known edges to 1
        # This means we KNOW the full graph structure
        adjacency_matrix = -np.ones((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        # No self-loops (set diagonal to -1, already done)
        
        # Use the true causal graph from the plant experiment
        
        # T -> Y (Light_Intensity -> Biomass)
        adjacency_matrix[T_idx, Y_idx] = 1.0
        
        # Everything has a causal impact on the outcome (Biomass_g)
        for name in ["X", "Y", "Position_in_Bed", "Bed_Position", "Unique_Identifier",
                     "Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", 
                     "Far_Red_Light_Intensity_umols", "Microgreen"]:
            if name in FEATURE_TO_IDX:
                adjacency_matrix[feat_idx(name), Y_idx] = 1.0
        
        # Light intensity (T) is composed of its sources - so sources -> T
        for name in ["Far_Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", 
                     "Red_Light_Intensity_umols"]:
            adjacency_matrix[feat_idx(name), T_idx] = 1.0
        
        # Positional information has a causal impact on light intensity (T)
        for name in ["X", "Y", "Position_in_Bed", "Bed_Position", "Unique_Identifier"]:
            adjacency_matrix[feat_idx(name), T_idx] = 1.0
        
        # Positional info -> individual light components (these are in X, not T)
        # Red_Light_Intensity_umols (idx 6)
        # Blue_Light_Intensity_umols (idx 7)
        # Far_Red_Light_Intensity_umols (idx 8)
        positional_features = ["X", "Y", "Position_in_Bed", "Bed_Position", "Unique_Identifier"]
        light_features = ["Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", 
                          "Far_Red_Light_Intensity_umols"]
        
        for pos_name in positional_features:
            for light_name in light_features:
                adjacency_matrix[feat_idx(pos_name), feat_idx(light_name)] = 1.0
        
        print(f"Graph knowledge: TRUE CAUSAL GRAPH (full knowledge)")
        print(f"  - Known edges (1): T->Y, Position->T, LightSources->T, All->Y, Position->LightSources")
        print(f"  - Known non-edges (-1): everything else")
    
    elif graph_mode == "true_graph":
        # Partial knowledge: only specify edges we're confident about (leave others as unknown)
        # Start with all unknown (0), set only the edges we're confident about
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        # T -> Y (Light_Intensity -> Biomass) - we KNOW this
        adjacency_matrix[T_idx, Y_idx] = 1.0
        
        # Everything has a causal impact on the outcome (Biomass_g) - we're confident
        for name in ["X", "Y", "Position_in_Bed", "Bed_Position", "Unique_Identifier",
                     "Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", 
                     "Far_Red_Light_Intensity_umols", "Microgreen"]:
            if name in FEATURE_TO_IDX:
                adjacency_matrix[feat_idx(name), Y_idx] = 1.0
        
        # Light intensity (T) is composed of its sources - we KNOW this
        for name in ["Far_Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", 
                     "Red_Light_Intensity_umols"]:
            adjacency_matrix[feat_idx(name), T_idx] = 1.0
        
        # Positional information has a causal impact on light intensity (T)
        for name in ["X", "Y", "Position_in_Bed", "Bed_Position", "Unique_Identifier"]:
            adjacency_matrix[feat_idx(name), T_idx] = 1.0
        
        # Positional info -> individual light components
        positional_features = ["X", "Y", "Position_in_Bed", "Bed_Position", "Unique_Identifier"]
        light_features = ["Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", 
                          "Far_Red_Light_Intensity_umols"]
        
        for pos_name in positional_features:
            for light_name in light_features:
                adjacency_matrix[feat_idx(pos_name), feat_idx(light_name)] = 1.0
        
        # Some things we're confident DON'T exist:
        # - Y doesn't cause anything (outcome is terminal)
        adjacency_matrix[Y_idx, :] = -1.0
        
        # - Microgreen doesn't cause positional info or light components
        for name in positional_features + light_features:
            adjacency_matrix[feat_idx("Microgreen"), feat_idx(name)] = -1.0
        adjacency_matrix[feat_idx("Microgreen"), T_idx] = -1.0  # Microgreen doesn't cause T
        
        print(f"Graph knowledge: TRUE CAUSAL GRAPH (partial knowledge)")
        print(f"  - Known edges (1): T->Y, Position->T, LightSources->T, All->Y, Position->LightSources")
        print(f"  - Known non-edges (-1): Y->*, Microgreen->position/lights")
        print(f"  - Unknown (0): everything else")
        
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")
    
    # PADDED features: Set all edges to -1
    for i in range(n_real_features, model_n_features):
        feat_idx_pad = feature_offset + i
        adjacency_matrix[feat_idx_pad, :] = -1.0
        adjacency_matrix[:, feat_idx_pad] = -1.0
        adjacency_matrix[feat_idx_pad, feat_idx_pad] = -1.0
    
    return adjacency_matrix


def build_adjacency_matrix(model_n_features, n_real_features, graph_mode="full_graph"):
    """Build adjacency matrix based on graph knowledge mode (same as dofm_full_conditioning)."""
    adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    
    T_idx = 0
    Y_idx = 1
    feature_offset = 2
    
    if graph_mode == "all_unknown":
        print(f"Graph knowledge: ALL UNKNOWN")
    elif graph_mode == "full_graph":
        adjacency_matrix[T_idx, Y_idx] = 1.0
        for i in range(n_real_features):
            adjacency_matrix[feature_offset + i, T_idx] = 1.0
            adjacency_matrix[feature_offset + i, Y_idx] = 1.0
        print(f"Graph knowledge: T->Y=1, X->T=1, X->Y=1 (full graph)")
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")
    
    # PADDED features: Set all edges to -1
    for i in range(n_real_features, model_n_features):
        feat_idx = feature_offset + i
        adjacency_matrix[feat_idx, :] = -1.0
        adjacency_matrix[:, feat_idx] = -1.0
        adjacency_matrix[feat_idx, feat_idx] = -1.0
    
    return adjacency_matrix


def create_dofm_plant_pipeline(graph_mode="full_graph", use_onehot=False):
    """Factory function to create a pipeline with specific graph mode."""
    
    def dofm_plant_pipeline(model, cid_dataset):
        """
        Use PreprocessingGraphConditionedPFN to predict dose-response (CID).
        
        Unlike CATE where T is binary, here T is continuous.
        We predict Y for each (X_test, t_test) pair.
        """
        # Extract raw data
        X_train_raw = np.array(cid_dataset.X_train)
        X_test_raw = np.array(cid_dataset.X_test)
        
        t_train = np.array(cid_dataset.t_train, dtype=np.float32).reshape(-1, 1)
        y_train = np.array(cid_dataset.y_train, dtype=np.float32).reshape(-1, 1)
        t_test = np.array(cid_dataset.t_test, dtype=np.float32).reshape(-1, 1)
        
        if use_onehot:
            # One-hot encode Position_in_Bed (idx 2) and Microgreen (idx 5)
            onehot_columns = [
                FEATURE_TO_IDX["Position_in_Bed"],  # idx 2
                FEATURE_TO_IDX["Microgreen"],       # idx 5
            ]
            print(f"Using one-hot encoding for columns: {onehot_columns}")
            X_train, X_test, feature_mapping = encode_with_onehot(X_train_raw, X_test_raw, onehot_columns)
        else:
            # Use target encoding (with label encoding for high-cardinality)
            X_train, X_test, cat_indices = encode_mixed_data(X_train_raw, X_test_raw, y_train)
            if cat_indices:
                print(f"Target-encoded categorical columns at indices: {cat_indices}")
            feature_mapping = None  # Not using one-hot
        
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_features = X_train.shape[1]
        print(f"[Dataset] Train: {n_train}, Test: {n_test}, Features: {n_features}")
        
        # Get model's expected number of features
        model_n_features = model.model.num_features
        n_features_orig = n_features
        
        # Fit preprocessing on training data (same as dofm_full_conditioning)
        model.fit(X_train, t_train, y_train)
        
        # Build adjacency matrix - use true causal graph for plant data
        n_real_features = min(n_features_orig, model_n_features)
        
        if use_onehot and feature_mapping is not None:
            # Use one-hot aware adjacency builder
            adjacency_matrix = build_plant_adjacency_with_onehot(
                model_n_features, n_real_features, feature_mapping, graph_mode)
        else:
            # Use standard adjacency builder
            adjacency_matrix = build_plant_causal_graph_adjacency(
                model_n_features, n_real_features, graph_mode)
        
        # Predict Y for each test sample at its intervention value t_test
        # Unlike CATE, we don't need to predict at T=0 and T=1, just at t_test
        cid_pred = model.predict(
            X_obs=X_train,
            T_obs=t_train,
            Y_obs=y_train,
            X_intv=X_test,
            T_intv=t_test,
            adjacency_matrix=adjacency_matrix,
            prediction_type="mean",
            inverse_transform=True,
        )
        
        return cid_pred.flatten()
    
    return dofm_plant_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DOFM on plant CID benchmark.")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)    
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--graph_mode", type=str, default="full_graph",
                        choices=["all_unknown", "t_to_y_only", "full_graph", "true_graph"])
    parser.add_argument("--onehot", action="store_true",
                        help="Use one-hot encoding for Position_in_Bed and Microgreen")

    args = parser.parse_args()
    
    graph_mode = args.graph_mode
    use_onehot = args.onehot

    if args.checkpoint_path is None:
        # Non-conditioned model checkpoint
        checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist.pt"
    else:
        checkpoint_path = args.checkpoint_path
        
    if args.config_path is None:
        # Non-conditioned model config
        model_config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist_config.yaml"
    else:
        model_config_path = args.config_path
    
    print(f"--- Starting DOFM Plant Experiment ---")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")
    print(f"Graph mode: {graph_mode}")
    print(f"One-hot encoding: {use_onehot}")
    print(f"Loading model from: {checkpoint_path}")
    
    model = PreprocessingGraphConditionedPFN(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
        use_clustering=False,  # No clustering for plant data
    )
    model.load()

    dofm_pipeline = create_dofm_plant_pipeline(graph_mode, use_onehot=use_onehot)

    evaluate_pipeline(
        exp_name=args.exp_name,
        model_pipeline=dofm_pipeline,
        model=model,
        args=args)
    
    print(f"--- Experiment Finished ---")
