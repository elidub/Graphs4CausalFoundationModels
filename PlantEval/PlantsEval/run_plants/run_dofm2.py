"""
DOFM for Plant CID benchmark using PreprocessingGraphConditionedPFN wrapper.

Simplified version:
- Removes Unique_Identifier from features
- Uses label encoding for categorical features
- No one-hot encoding complexity
"""

import argparse
import sys
import os
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting')
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval/PlantsEval')

from run_plants.eval import evaluate_pipeline
from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN


# Original feature names (before dropping Unique_Identifier)
# X_train columns: X, Y, Position_in_Bed, Bed_Position, Unique_Identifier, Microgreen,
#                  Red_Light_Intensity_umols, Blue_Light_Intensity_umols, Far_Red_Light_Intensity_umols
ORIGINAL_FEATURE_NAMES = [
    "X", "Y", "Position_in_Bed", "Bed_Position", "Unique_Identifier", "Microgreen",
    "Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", "Far_Red_Light_Intensity_umols"
]

# Feature names after dropping Unique_Identifier (index 4)
# New indices: X=0, Y=1, Position_in_Bed=2, Bed_Position=3, Microgreen=4, Red=5, Blue=6, Far_Red=7
FEATURE_NAMES = [
    "X", "Y", "Position_in_Bed", "Bed_Position", "Microgreen",
    "Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", "Far_Red_Light_Intensity_umols"
]
FEATURE_TO_IDX = {name: idx for idx, name in enumerate(FEATURE_NAMES)}

# Causal structure groups
POSITIONAL_FEATURES = ["X", "Y", "Position_in_Bed", "Bed_Position"]
LIGHT_FEATURES = ["Red_Light_Intensity_umols", "Blue_Light_Intensity_umols", "Far_Red_Light_Intensity_umols"]
MICROGREEN_FEATURE = "Microgreen"

# Index of Unique_Identifier in original data (to drop)
UNIQUE_IDENTIFIER_IDX = 4


def encode_and_drop_features(X_train, X_test, y_train):
    """
    Drop Unique_Identifier and encode categorical features with label encoding.
    
    Args:
        X_train: Training features (n_train, 9)
        X_test: Test features (n_test, 9)
        y_train: Training outcomes (not used, kept for compatibility)
    
    Returns:
        X_train_encoded: Encoded training features (n_train, 8)
        X_test_encoded: Encoded test features (n_test, 8)
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Drop Unique_Identifier (column 4)
    X_train_dropped = np.delete(X_train, UNIQUE_IDENTIFIER_IDX, axis=1)
    X_test_dropped = np.delete(X_test, UNIQUE_IDENTIFIER_IDX, axis=1)
    
    n_cols = X_train_dropped.shape[1]
    X_train_encoded = np.zeros(X_train_dropped.shape, dtype=np.float32)
    X_test_encoded = np.zeros(X_test_dropped.shape, dtype=np.float32)
    
    for col_idx in range(n_cols):
        train_col = X_train_dropped[:, col_idx]
        test_col = X_test_dropped[:, col_idx]
        
        try:
            # Try to convert to float (numeric column)
            X_train_encoded[:, col_idx] = train_col.astype(np.float32)
            X_test_encoded[:, col_idx] = test_col.astype(np.float32)
        except (ValueError, TypeError):
            # Categorical column - use label encoding
            le = LabelEncoder()
            all_vals = np.concatenate([train_col, test_col])
            le.fit(all_vals)
            X_train_encoded[:, col_idx] = le.transform(train_col).astype(np.float32)
            X_test_encoded[:, col_idx] = le.transform(test_col).astype(np.float32)
            print(f"  Col {col_idx} ({FEATURE_NAMES[col_idx]}): Label encoded ({len(le.classes_)} categories)")
    
    print(f"  Dropped Unique_Identifier. Features: {len(ORIGINAL_FEATURE_NAMES)} -> {n_cols}")
    
    return X_train_encoded, X_test_encoded


def build_plant_causal_graph_adjacency(model_n_features, n_real_features, graph_mode="full_graph"):
    """
    Build adjacency matrix based on the true causal graph of the plant experiment.
    
    This builds an ADJACENCY matrix (direct edges only), which will be automatically
    converted to an ANCESTOR matrix (transitive closure) by PreprocessingGraphConditionedPFN
    if the model was trained with ancestor matrices.
    
    Features (after dropping Unique_Identifier):
    0: X, 1: Y, 2: Position_in_Bed, 3: Bed_Position, 4: Microgreen,
    5: Red_Light, 6: Blue_Light, 7: Far_Red_Light
    
    Adjacency matrix layout: [T, Y, X_0, X_1, ..., X_n]
    - T_idx = 0 (Light_Intensity_umols - treatment)
    - Y_idx = 1 (Biomass_g - outcome)
    - Feature i has index (2 + i)
    
    Matrix values:
    - 1.0 = edge exists (i directly causes j)
    - -1.0 = NO edge (i does NOT cause j) 
    - 0.0 = unknown
    """
    T_idx = 0  # Light_Intensity_umols (treatment)
    Y_idx = 1  # Biomass_g (outcome)
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
        adjacency_matrix = -np.ones((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        # T -> Y (Light_Intensity -> Biomass)
        adjacency_matrix[T_idx, Y_idx] = 1.0
        
        # All features -> Y (everything affects outcome)
        for name in FEATURE_NAMES:
            adjacency_matrix[feat_idx(name), Y_idx] = 1.0
        
        # Light sources -> T
        for name in LIGHT_FEATURES:
            adjacency_matrix[feat_idx(name), T_idx] = 1.0
        
        # Positional features -> T
        for name in POSITIONAL_FEATURES:
            adjacency_matrix[feat_idx(name), T_idx] = 1.0
        
        # Positional features -> Light sources
        for pos_name in POSITIONAL_FEATURES:
            for light_name in LIGHT_FEATURES:
                adjacency_matrix[feat_idx(pos_name), feat_idx(light_name)] = 1.0
        
        print(f"Graph knowledge: TRUE CAUSAL GRAPH (full knowledge)")
        print(f"  - Known edges (1): T->Y, Position->T, LightSources->T, All->Y, Position->LightSources")
        print(f"  - Known non-edges (-1): everything else")
    
    elif graph_mode == "true_graph":
        # Partial knowledge: specify known edges, leave others as unknown
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        # T -> Y
        adjacency_matrix[T_idx, Y_idx] = 1.0
        
        # All features -> Y
        for name in FEATURE_NAMES:
            adjacency_matrix[feat_idx(name), Y_idx] = 1.0
        
        # Light sources -> T
        for name in LIGHT_FEATURES:
            adjacency_matrix[feat_idx(name), T_idx] = 1.0
        
        # Positional features -> T
        for name in POSITIONAL_FEATURES:
            adjacency_matrix[feat_idx(name), T_idx] = 1.0
        
        # Positional features -> Light sources
        for pos_name in POSITIONAL_FEATURES:
            for light_name in LIGHT_FEATURES:
                adjacency_matrix[feat_idx(pos_name), feat_idx(light_name)] = 1.0
        
        # Known non-edges:
        # Y doesn't cause anything (outcome is terminal)
        adjacency_matrix[Y_idx, :] = -1.0
        
        # Microgreen doesn't cause positional info, light components, or T
        for name in POSITIONAL_FEATURES + LIGHT_FEATURES:
            adjacency_matrix[feat_idx(MICROGREEN_FEATURE), feat_idx(name)] = -1.0
        adjacency_matrix[feat_idx(MICROGREEN_FEATURE), T_idx] = -1.0
        
        print(f"Graph knowledge: TRUE CAUSAL GRAPH (partial knowledge)")
        print(f"  - Known edges (1): T->Y, Position->T, LightSources->T, All->Y, Position->LightSources")
        print(f"  - Known non-edges (-1): Y->*, Microgreen->position/lights/T")
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


def create_dofm_plant_pipeline(graph_mode="full_graph"):
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
        
        # Drop Unique_Identifier and encode categoricals with target encoding
        X_train, X_test = encode_and_drop_features(X_train_raw, X_test_raw, y_train)
        
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_features = X_train.shape[1]
        print(f"[Dataset] Train: {n_train}, Test: {n_test}, Features: {n_features}")
        
        # Get model's expected number of features
        model_n_features = model.model.num_features
        n_features_orig = n_features
        
        # Fit preprocessing on training data
        model.fit(X_train, t_train, y_train)
        
        # Build adjacency matrix (direct causal edges)
        # This will be automatically converted to ancestor matrix if needed
        n_real_features = min(n_features_orig, model_n_features)
        adjacency_matrix = build_plant_causal_graph_adjacency(
            model_n_features, n_real_features, graph_mode)
        
        # Predict Y for each test sample at its intervention value t_test
        # The model will automatically convert adjacency -> ancestor matrix if needed
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
    parser = argparse.ArgumentParser(description="Run DOFM on plant CID benchmark (simplified).")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)    
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--graph_mode", type=str, default="full_graph",
                        choices=["all_unknown", "t_to_y_only", "full_graph", "true_graph"])

    args = parser.parse_args()
    
    graph_mode = args.graph_mode

    if args.checkpoint_path is None:
        checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist.pt"
    else:
        checkpoint_path = args.checkpoint_path
        
    if args.config_path is None:
        model_config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist_config.yaml"
    else:
        model_config_path = args.config_path
    
    print(f"--- Starting DOFM Plant Experiment (v2 - simplified) ---")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")
    print(f"Graph mode: {graph_mode}")
    print(f"Loading model from: {checkpoint_path}")
    
    model = PreprocessingGraphConditionedPFN(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
        use_clustering=False,  # No clustering for plant data
    )
    model.load()

    dofm_pipeline = create_dofm_plant_pipeline(graph_mode)

    evaluate_pipeline(
        exp_name=args.exp_name,
        model_pipeline=dofm_pipeline,
        model=model,
        args=args)
    
    print(f"--- Experiment Finished ---")
