import argparse
import sys
import os
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Determine base paths - works locally and on cluster
SCRIPT_DIR = Path(__file__).resolve().parent
PLANTSEVAL_DIR = SCRIPT_DIR.parent
# Navigate up to CausalPriorFitting root (PlantEval/PlantsEval/run_plants -> ../../..)
PROJECT_ROOT = PLANTSEVAL_DIR.parent.parent

# Add paths for imports
sys.path.insert(0, str(PLANTSEVAL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "RealCauseEval"))

from run_plants.eval import evaluate_pipeline
from src.models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn

# Model checkpoint path (relative to PROJECT_ROOT)
MODEL_CHECKPOINT_DIR = PROJECT_ROOT / "experiments" / "FirstTests" / "checkpoints" / "final_earlytest_full_conditioning_16773252.0"

# Global config for preprocessing
MODEL_CONFIG = None

# Plant data column names in order (from the dataset)
# The treatment is Light_Intensity_umols, the outcome is Biomass_g
PLANT_FEATURE_NAMES = [
    "X",  # 0
    "Y",  # 1  
    "Position_in_Bed",  # 2
    "Bed_Position",  # 3
    "Unique_Identifier",  # 4
    "Microgreen",  # 5
    "Red_Light_Intensity_umols",  # 6
    "Blue_Light_Intensity_umols",  # 7
    "Far_Red_Light_Intensity_umols",  # 8
]

# Causal edges for the plant data
# Treatment (T) is Light_Intensity_umols
# Outcome (Y_out) is Biomass_g
PLANT_CAUSAL_EDGES = [
    # Everything has a causal impact on the outcome (Biomass_g)
    ("X", "Biomass_g"),
    ("Y", "Biomass_g"),
    ("Position_in_Bed", "Biomass_g"),
    ("Bed_Position", "Biomass_g"),
    ("Unique_Identifier", "Biomass_g"),
    ("Red_Light_Intensity_umols", "Biomass_g"),
    ("Blue_Light_Intensity_umols", "Biomass_g"),
    ("Far_Red_Light_Intensity_umols", "Biomass_g"),
    ("Light_Intensity_umols", "Biomass_g"),  # T -> Y
    ("Microgreen", "Biomass_g"),

    # Light intensity is composed of its sources
    ("Far_Red_Light_Intensity_umols", "Light_Intensity_umols"),
    ("Blue_Light_Intensity_umols", "Light_Intensity_umols"),
    ("Red_Light_Intensity_umols", "Light_Intensity_umols"),

    # Positional information has a causal impact on light intensity (T)
    ("X", "Light_Intensity_umols"),
    ("Y", "Light_Intensity_umols"),
    ("Position_in_Bed", "Light_Intensity_umols"),
    ("Bed_Position", "Light_Intensity_umols"),
    ("Unique_Identifier", "Light_Intensity_umols"),

    ("X", "Red_Light_Intensity_umols"),
    ("Y", "Red_Light_Intensity_umols"),
    ("Position_in_Bed", "Red_Light_Intensity_umols"),
    ("Bed_Position", "Red_Light_Intensity_umols"),
    ("Unique_Identifier", "Red_Light_Intensity_umols"),

    ("X", "Blue_Light_Intensity_umols"),
    ("Y", "Blue_Light_Intensity_umols"),
    ("Position_in_Bed", "Blue_Light_Intensity_umols"),
    ("Bed_Position", "Blue_Light_Intensity_umols"),
    ("Unique_Identifier", "Blue_Light_Intensity_umols"),

    ("X", "Far_Red_Light_Intensity_umols"),
    ("Y", "Far_Red_Light_Intensity_umols"),
    ("Position_in_Bed", "Far_Red_Light_Intensity_umols"),
    ("Bed_Position", "Far_Red_Light_Intensity_umols"),
    ("Unique_Identifier", "Far_Red_Light_Intensity_umols"),
]


def encode_mixed_data(X_train, X_test):
    """
    Encode mixed numerical/categorical data.
    Converts string columns to label-encoded integers.
    Returns encoded arrays and list of categorical column indices.
    """
    n_cols = X_train.shape[1]
    X_train_encoded = np.zeros(X_train.shape, dtype=np.float32)
    X_test_encoded = np.zeros(X_test.shape, dtype=np.float32)
    categorical_indices = []
    
    for col_idx in range(n_cols):
        train_col = X_train[:, col_idx]
        test_col = X_test[:, col_idx]
        
        # Check if column is string/object type
        try:
            # Try to convert to float - if it works, it's numerical
            X_train_encoded[:, col_idx] = train_col.astype(np.float32)
            X_test_encoded[:, col_idx] = test_col.astype(np.float32)
        except (ValueError, TypeError):
            # Column contains strings - need to label encode
            categorical_indices.append(col_idx)
            le = LabelEncoder()
            # Fit on combined train+test to ensure consistent encoding
            all_vals = np.concatenate([train_col, test_col])
            le.fit(all_vals)
            X_train_encoded[:, col_idx] = le.transform(train_col).astype(np.float32)
            X_test_encoded[:, col_idx] = le.transform(test_col).astype(np.float32)
    
    return X_train_encoded, X_test_encoded, categorical_indices


def load_preprocessing_config(config_path):
    """Load preprocessing configuration from model config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('preprocessing_config', {}), config.get('dataset_config', {})


def get_config_value(config_dict, key, default=None):
    """Extract value from config entry that may be plain or dict with 'value' key."""
    raw = config_dict.get(key, default)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def dofm_pipeline(model, cid_dataset):
    """
    Use the GraphConditionedInterventionalPFN model to predict dose-response (CID).
    
    Unlike CATE prediction where T is binary, here T is continuous.
    We predict Y for each (X_test, t_test) pair.
    
    Args:
        model: GraphConditionedInterventionalPFNSklearn instance
        cid_dataset: CID_Dataset with X_train, t_train, y_train, X_test, t_test, true_cid
        
    Returns:
        cid_pred: Array of predictions for each test sample
    """
    global MODEL_CONFIG
    
    # Extract raw data from cid_dataset
    X_train_raw = np.array(cid_dataset.X_train)
    X_test_raw = np.array(cid_dataset.X_test)
    
    # Encode mixed categorical/numerical data
    X_train, X_test, cat_indices = encode_mixed_data(X_train_raw, X_test_raw)
    if cat_indices:
        print(f"Encoded categorical columns at indices: {cat_indices}")
    
    t_train = np.array(cid_dataset.t_train, dtype=np.float32).reshape(-1, 1)
    y_train = np.array(cid_dataset.y_train, dtype=np.float32).reshape(-1, 1)
    t_test = np.array(cid_dataset.t_test, dtype=np.float32).reshape(-1, 1)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features_orig = X_train.shape[1]
    
    # Get model's expected number of features
    model_n_features = model.model.num_features
    
    # Load preprocessing config if not already loaded
    if MODEL_CONFIG is None:
        config_path = MODEL_CHECKPOINT_DIR / "final_model_with_bardist_config.yaml"
        preprocessing_config, dataset_config = load_preprocessing_config(str(config_path))
        MODEL_CONFIG = {
            'preprocessing': preprocessing_config,
            'dataset': dataset_config
        }
    
    preprocessing_config = MODEL_CONFIG['preprocessing']
    dataset_config = MODEL_CONFIG['dataset']
    
    # Truncate or pad features to match model size
    if n_features_orig > model_n_features:
        print(f"Truncating features from {n_features_orig} to {model_n_features}")
        X_train = X_train[:, :model_n_features]
        X_test = X_test[:, :model_n_features]
        n_features = model_n_features
    elif n_features_orig < model_n_features:
        # Pad with zeros
        X_train = np.hstack([X_train, np.zeros((n_train, model_n_features - n_features_orig), dtype=np.float32)])
        X_test = np.hstack([X_test, np.zeros((n_test, model_n_features - n_features_orig), dtype=np.float32)])
        n_features = model_n_features
    else:
        n_features = n_features_orig
    
    # Apply feature preprocessing as done during training
    outlier_quantile = get_config_value(preprocessing_config, 'outlier_quantile', 0.99)
    remove_outliers = get_config_value(preprocessing_config, 'remove_outliers', True)
    feature_standardize = get_config_value(preprocessing_config, 'feature_standardize', True)
    
    if remove_outliers:
        # Compute outlier bounds from training data only
        lower_quantile = 1.0 - outlier_quantile
        upper_quantile = outlier_quantile
        lower_bounds = np.quantile(X_train, lower_quantile, axis=0)
        upper_bounds = np.quantile(X_train, upper_quantile, axis=0)
        
        # Clip both train and test
        X_train = np.clip(X_train, lower_bounds, upper_bounds)
        X_test = np.clip(X_test, lower_bounds, upper_bounds)
    
    if feature_standardize:
        # Standardize features: (X - mean) / std
        X_mean = np.mean(X_train, axis=0, keepdims=True)
        X_std = np.std(X_train, axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)
        
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
    
    # Apply target scaling to [-1, 1]
    ymin = float(np.min(y_train))
    ymax = float(np.max(y_train))
    y_range = max(ymax - ymin, 1e-8)
    
    # Scale targets to [-1, 1]
    y_train_scaled = 2.0 * (y_train - ymin) / y_range - 1.0
    
    # Prepare data for model
    T_train = t_train.astype(np.float32)
    Y_train = y_train_scaled.flatten().astype(np.float32)
    T_test = t_test.astype(np.float32)
    
    # Get max samples from config
    max_n_train_samples = get_config_value(dataset_config, 'max_number_train_samples_per_dataset', n_train)
    max_n_test_samples = get_config_value(dataset_config, 'max_number_test_samples_per_dataset', n_test)
    
    # Number of real (non-padded) features
    n_real_features = min(n_features_orig, model_n_features)
    
    # Create partial ancestral matrix with known causal structure for PLANT DATA
    # ANCESTOR MATRIX semantics: T[i,j] = 1 means i IS an ancestor of j
    #                            T[i,j] = 0 means UNKNOWN
    #                            T[i,j] = -1 means i is NOT an ancestor of j
    # Shape: (model_n_features + 2, model_n_features + 2)
    # Positions: 0 to model_n_features-1: Features, model_n_features: T, model_n_features+1: Y
    T_idx = model_n_features  # Treatment: Light_Intensity_umols
    Y_idx = model_n_features + 1  # Outcome: Biomass_g
    
    # Build name-to-index mapping for plant features
    # Features in order: X(0), Y(1), Position_in_Bed(2), Bed_Position(3), Unique_Identifier(4),
    #                    Microgreen(5), Red_Light(6), Blue_Light(7), Far_Red_Light(8)
    name_to_idx = {name: i for i, name in enumerate(PLANT_FEATURE_NAMES)}
    name_to_idx["Light_Intensity_umols"] = T_idx  # Treatment
    name_to_idx["Biomass_g"] = Y_idx  # Outcome
    
    # Initialize ancestor matrix with zeros (unknown)
    ancestor_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    
    # Mark diagonal as -1 (no self-ancestry)
    for i in range(model_n_features + 2):
        ancestor_matrix[i, i] = -1.0
    
    # Mark padded features as no-edge (-1) in all directions
    for i in range(n_real_features, model_n_features):
        ancestor_matrix[i, :] = -1.0
        ancestor_matrix[:, i] = -1.0
    
    # Build adjacency matrix first (direct edges), then compute transitive closure
    adj_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    
    for source, target in PLANT_CAUSAL_EDGES:
        src_idx = name_to_idx.get(source)
        tgt_idx = name_to_idx.get(target)
        if src_idx is not None and tgt_idx is not None:
            if src_idx < model_n_features + 2 and tgt_idx < model_n_features + 2:
                adj_matrix[src_idx, tgt_idx] = 1.0
    
    # Compute transitive closure (ancestor matrix) using Floyd-Warshall
    # T[i,j] = 1 iff there's a path from i to j
    reachability = adj_matrix.copy()
    n_nodes = model_n_features + 2
    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if reachability[i, k] > 0 and reachability[k, j] > 0:
                    reachability[i, j] = 1.0
    
    # Set known ancestors (1) in the ancestor matrix
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and reachability[i, j] > 0:
                # Only update if not a padded feature
                if not (n_real_features <= i < model_n_features or n_real_features <= j < model_n_features):
                    ancestor_matrix[i, j] = 1.0
    
    # Also mark known NON-ancestors as -1 based on the graph structure:
    # If we know the complete causal structure, nodes that aren't ancestors should be -1
    # For now, mark outcome Y and treatment T as not being ancestors of covariates
    # (since Y and T are downstream of covariates in our causal model)
    for i in range(n_real_features):
        # Y is not an ancestor of any covariate
        ancestor_matrix[Y_idx, i] = -1.0
        # T is not an ancestor of most covariates (only the light components, but let's be conservative)
        # Actually T (Light_Intensity) is composed of Red/Blue/Far_Red, so T is not an ancestor of X features
        ancestor_matrix[T_idx, i] = -1.0
    
    # Covariates don't cause each other in this specific plant model (they're exogenous)
    # X, Y_coord, Position_in_Bed, Bed_Position, Unique_Identifier, Microgreen are root nodes
    root_features = [0, 1, 2, 3, 4, 5]  # X, Y_coord, Position_in_Bed, Bed_Position, Unique_Identifier, Microgreen
    for i in root_features:
        for j in root_features:
            if i != j:
                ancestor_matrix[i, j] = -1.0  # Root features don't cause each other
    
    n_known_ancestors = np.sum(ancestor_matrix == 1.0)
    n_known_non_ancestors = np.sum(ancestor_matrix == -1.0)
    print(f"Built plant ancestor matrix: {n_known_ancestors:.0f} known ancestors, {n_known_non_ancestors:.0f} known non-ancestors")
    
    # Truncate samples if needed
    if n_train > max_n_train_samples:
        indices = np.random.choice(n_train, max_n_train_samples, replace=False)
        X_train = X_train[indices]
        T_train = T_train[indices]
        Y_train = Y_train[indices]
        n_train = max_n_train_samples
    
    # Pad training samples if needed
    if n_train < max_n_train_samples:
        pad_train = max_n_train_samples - n_train
        X_train = np.vstack([X_train, np.zeros((pad_train, X_train.shape[1]), dtype=np.float32)])
        T_train = np.vstack([T_train, np.zeros((pad_train, 1), dtype=np.float32)])
        Y_train = np.concatenate([Y_train, np.zeros(pad_train, dtype=np.float32)])
    
    # Ensure T_train is 2D
    if T_train.ndim == 1:
        T_train = T_train.reshape(-1, 1)
    
    # Process test samples in batches if needed
    if n_test > max_n_test_samples:
        all_y_pred = []
        
        for batch_start in range(0, n_test, max_n_test_samples):
            batch_end = min(batch_start + max_n_test_samples, n_test)
            batch_size = batch_end - batch_start
            
            X_test_batch = X_test[batch_start:batch_end]
            T_test_batch = T_test[batch_start:batch_end]
            
            # Pad batch if needed
            if batch_size < max_n_test_samples:
                pad_size = max_n_test_samples - batch_size
                X_test_batch = np.vstack([X_test_batch, np.zeros((pad_size, X_test_batch.shape[1]), dtype=np.float32)])
                T_test_batch = np.vstack([T_test_batch, np.zeros((pad_size, 1), dtype=np.float32)])
            
            y_pred_batch = model.predict(
                X_obs=X_train,
                T_obs=T_train,
                Y_obs=Y_train,
                X_intv=X_test_batch,
                T_intv=T_test_batch,
                adjacency_matrix=ancestor_matrix,
                prediction_type="mean",
                batched=False
            )
            
            all_y_pred.append(y_pred_batch[:batch_size])
        
        y_pred_scaled = np.concatenate(all_y_pred)
    else:
        # Pad test samples if needed
        if n_test < max_n_test_samples:
            pad_test = max_n_test_samples - n_test
            X_test_padded = np.vstack([X_test, np.zeros((pad_test, X_test.shape[1]), dtype=np.float32)])
            T_test_padded = np.vstack([T_test, np.zeros((pad_test, 1), dtype=np.float32)])
        else:
            X_test_padded = X_test
            T_test_padded = T_test
        
        y_pred = model.predict(
            X_obs=X_train,
            T_obs=T_train,
            Y_obs=Y_train,
            X_intv=X_test_padded,
            T_intv=T_test_padded,
            adjacency_matrix=ancestor_matrix,
            prediction_type="mean",
            batched=False
        )
        
        y_pred_scaled = y_pred[:n_test]
    
    # Inverse transform: Y_original = (Y_scaled + 1) * range / 2 + ymin
    cid_pred = (y_pred_scaled + 1.0) * y_range / 2.0 + ymin
    
    return cid_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DOFM experiments on plant data.")

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., CID_10_ints_10_reals_normal)")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")    
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")    

    args = parser.parse_args()
    
    print(f"--- Starting DOFM Plant Experiment ---")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")

    # Full conditioning model checkpoint
    checkpoint_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist.pt"
    model_config_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_full_conditioning_16773252.0/final_model_with_bardist_config.yaml"
    
    print(f"Loading model from: {checkpoint_path}")
    graphintpfn = GraphConditionedInterventionalPFNSklearn(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    graphintpfn.load()
    model = graphintpfn

    evaluate_pipeline(
        exp_name=args.exp_name,
        model_pipeline=dofm_pipeline,
        model=model,
        args=args)
    
    print(f"--- Experiment Finished ---")
