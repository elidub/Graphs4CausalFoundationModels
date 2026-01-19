import argparse
import os
import sys
import numpy as np
import torch
import os
import yaml

# Add paths
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting')
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting/RealCauseEval')

from src.models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn
from run_baselines.eval import evaluate_pipeline

# Global config for preprocessing
MODEL_CONFIG = None

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

def dofm_pipeline(model, cate_dataset):
    """
    Use the GraphConditionedInterventionalPFN model to predict CATE.
    
    Args:
        model: GraphConditionedInterventionalPFNSklearn instance
        cate_dataset: CATE_Dataset with X_train, t_train, y_train, X_test, true_cate
        
    Returns:
        cate_pred: Array of CATE predictions for X_test
    """
    global MODEL_CONFIG
    
    # Extract data from cate_dataset
    X_train = cate_dataset.X_train
    t_train = cate_dataset.t_train.reshape(-1, 1) if cate_dataset.t_train.ndim == 1 else cate_dataset.t_train
    y_train = cate_dataset.y_train.reshape(-1, 1) if cate_dataset.y_train.ndim == 1 else cate_dataset.y_train
    X_test = cate_dataset.X_test
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features_orig = X_train.shape[1]
    
    # Get model's expected number of features
    model_n_features = model.model.num_features
    
    # Load preprocessing config if not already loaded
    if MODEL_CONFIG is None:
        preprocessing_config, dataset_config = load_preprocessing_config(
            "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16768542.0/final_config.yaml"
        )
        MODEL_CONFIG = {
            'preprocessing': preprocessing_config,
            'dataset': dataset_config
        }
    
    preprocessing_config = MODEL_CONFIG['preprocessing']
    dataset_config = MODEL_CONFIG['dataset']
    
    # Check test feature mask fraction
    test_mask_frac = get_config_value(preprocessing_config, 'test_feature_mask_fraction', 0.0)
    if test_mask_frac > 0:
        print(f"Note: test_feature_mask_fraction={test_mask_frac} - {int(test_mask_frac*100)}% of test features will be masked (zeroed out)")
    
    # Truncate or pad features to match model size (before preprocessing)
    if n_features_orig > model_n_features:
        print(f"Truncating features from {n_features_orig} to {model_n_features}")
        X_train = X_train[:, :model_n_features]
        X_test = X_test[:, :model_n_features]
        n_features = model_n_features
    elif n_features_orig < model_n_features:
        print(f"Padding features from {n_features_orig} to {model_n_features}")
        # Pad with zeros (same as T-learner)
        X_train = np.hstack([X_train, np.zeros((n_train, model_n_features - n_features_orig))])
        X_test = np.hstack([X_test, np.zeros((n_test, model_n_features - n_features_orig))])
        n_features = model_n_features
    else:
        n_features = n_features_orig
    
    # Apply feature preprocessing as done during training (SAME AS T-LEARNER)
    # Training used: feature_standardize=True, remove_outliers=True, outlier_quantile=0.99
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
        print(f"Applied outlier removal at quantile {outlier_quantile}")
    
    if feature_standardize:
        # Standardize features: (X - mean) / std
        # Compute stats from training data only
        X_mean = np.mean(X_train, axis=0, keepdims=True)
        X_std = np.std(X_train, axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)  # Avoid division by zero
        
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        print(f"Applied feature standardization")
    
    # Apply target scaling to [-1, 1] as done during model training (SAME AS T-LEARNER)
    # This is CRITICAL: the model was trained with scaled targets!
    ymin = float(np.min(y_train))
    ymax = float(np.max(y_train))
    y_range = max(ymax - ymin, 1e-8)
    
    print(f"Target scaling stats: ymin={ymin:.4f}, ymax={ymax:.4f}, range={y_range:.4f}")
    
    # Scale targets to [-1, 1]
    # Formula: Y_scaled = 2.0 * (Y - ymin) / range - 1.0
    y_train_scaled = 2.0 * (y_train - ymin) / y_range - 1.0
    
    print(f"Scaled target range: [{np.min(y_train_scaled):.4f}, {np.max(y_train_scaled):.4f}]")
    
    # Prepare data for model (no more BasicProcessing!)
    # Convert to expected shapes
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    T_train = t_train.astype(np.float32)
    Y_train = y_train_scaled.flatten().astype(np.float32)
    
    # Get max samples from config (what the model was trained with)
    # OVERRIDE: Use 500 max train samples for faster evaluation
    max_n_train_samples = min(500, get_config_value(dataset_config, 'max_number_train_samples_per_dataset', n_train))
    max_n_test_samples = get_config_value(dataset_config, 'max_number_test_samples_per_dataset', n_test)
    # Don't pad features beyond what the model was trained on
    max_n_features = model_n_features
    
    # Features should already be exactly model_n_features from truncation/padding above
    # No additional padding needed here since we already handled it
    assert X_train.shape[1] == model_n_features, f"X_train has {X_train.shape[1]} features, expected {model_n_features}"
    assert X_test.shape[1] == model_n_features, f"X_test has {X_test.shape[1]} features, expected {model_n_features}"
    
    # Truncate samples if needed (random subsample to fit model capacity)
    if n_train > max_n_train_samples:
        print(f"Truncating training samples from {n_train} to {max_n_train_samples}")
        indices = np.random.choice(n_train, max_n_train_samples, replace=False)
        X_train = X_train[indices]
        T_train = T_train[indices]
        Y_train = Y_train[indices]
        n_train = max_n_train_samples
    
    if n_test > max_n_test_samples:
        print(f"Test samples ({n_test}) exceed model capacity ({max_n_test_samples}). Processing in batches.")
        # We'll process test samples in batches
        process_test_in_batches = True
    else:
        process_test_in_batches = False
    
    # Pad samples if needed
    if n_train < max_n_train_samples:
        pad_train = max_n_train_samples - n_train
        X_train = np.vstack([X_train, np.zeros((pad_train, X_train.shape[1]), dtype=np.float32)])
        T_train = np.vstack([T_train, np.zeros((pad_train, 1), dtype=np.float32)])
        Y_train = np.concatenate([Y_train, np.zeros(pad_train, dtype=np.float32)])
    
    if n_test < max_n_test_samples:
        pad_test = max_n_test_samples - n_test
        X_test = np.vstack([X_test, np.zeros((pad_test, X_test.shape[1]), dtype=np.float32)])
    
    # Ensure T_train is 2D
    if T_train.ndim == 1:
        T_train = T_train.reshape(-1, 1)
    
    # Create partial ancestral matrix with known causal structure (partial graph format)
    # Shape: (model_n_features + 2, model_n_features + 2) for features + treatment + outcome
    # Partial graph format uses three states:
    #   -1: No edge/ancestor relationship (known non-edge)
    #   0: Unknown whether edge/ancestor exists
    #   1: Edge/ancestor relationship exists (known edge)
    
    # Position mapping:
    # - Positions 0 to model_n_features-1: Feature variables
    # - Position model_n_features: Treatment (T)
    # - Position model_n_features+1: Outcome (Y)
    
    # Number of real (non-padded) features
    n_real_features = min(n_features_orig, model_n_features)
    
    # Initialize all as unknown (0)
    adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    
    # FULLY UNKNOWN GRAPH - all edges are unknown (0)
    # No known edges at all - the model must infer everything from the data
    T_idx = model_n_features
    Y_idx = model_n_features + 1
    
    # All relationships remain unknown (0):
    # - T -> Y: unknown
    # - Features -> T: unknown
    # - Features -> Y: unknown
    # - Relationships between features: unknown
    
    # 4. PADDED features: Set all edges to -1 (no edge) since they don't exist
    # Padded features are at positions n_real_features to model_n_features-1
    for i in range(n_real_features, model_n_features):
        # Padded feature i has no edges TO anything
        adjacency_matrix[i, :] = -1.0
        # Nothing has edges TO padded feature i
        adjacency_matrix[:, i] = -1.0
        # Diagonal is also -1 (no self-loops)
        adjacency_matrix[i, i] = -1.0
    
    # For CATE prediction, we need to predict Y for both T=0 and T=1
    # Then CATE = E[Y|T=1,X] - E[Y|T=0,X]
    
    print(f"Predicting CATE with test samples: {n_test}, train-samples: {n_train}")
    
    if process_test_in_batches:
        # Process test samples in batches of max_n_test_samples
        all_y_pred_1 = []
        all_y_pred_0 = []
        
        n_original_test = n_test  # Store original count before any modifications
        
        for batch_start in range(0, n_original_test, max_n_test_samples):
            batch_end = min(batch_start + max_n_test_samples, n_original_test)
            batch_size = batch_end - batch_start
            
            # Get batch of test samples (from original unpadded X_test)
            X_test_batch = X_test[batch_start:batch_end]
            
            # Pad batch if needed
            if batch_size < max_n_test_samples:
                pad_size = max_n_test_samples - batch_size
                X_test_batch = np.vstack([X_test_batch, np.zeros((pad_size, X_test_batch.shape[1]), dtype=np.float32)])
            
            # Create T arrays for this batch
            T_intv_1 = np.ones((max_n_test_samples, 1), dtype=np.float32)
            T_intv_0 = np.zeros((max_n_test_samples, 1), dtype=np.float32)
            
            # Predict Y for T=1
            y_pred_1_batch = model.predict(
                X_obs=X_train,
                T_obs=T_train,
                Y_obs=Y_train,
                X_intv=X_test_batch,
                T_intv=T_intv_1,
                adjacency_matrix=adjacency_matrix,
                prediction_type="mean",
                batched=False
            )
            
            # Predict Y for T=0
            y_pred_0_batch = model.predict(
                X_obs=X_train,
                T_obs=T_train,
                Y_obs=Y_train,
                X_intv=X_test_batch,
                T_intv=T_intv_0,
                adjacency_matrix=adjacency_matrix,
                prediction_type="mean",
                batched=False
            )
            
            # Only keep non-padded predictions
            all_y_pred_1.append(y_pred_1_batch[:batch_size])
            all_y_pred_0.append(y_pred_0_batch[:batch_size])
        
        y_pred_1 = np.concatenate(all_y_pred_1)
        y_pred_0 = np.concatenate(all_y_pred_0)
        cate_pred_scaled = y_pred_1 - y_pred_0
    else:
        # Create T arrays with proper padding
        T_intv_1 = np.ones((max_n_test_samples, 1), dtype=np.float32)
        T_intv_0 = np.zeros((max_n_test_samples, 1), dtype=np.float32)
        
        # Predict Y for T=1
        y_pred_1 = model.predict(
            X_obs=X_train,
            T_obs=T_train,
            Y_obs=Y_train,
            X_intv=X_test,
            T_intv=T_intv_1,
            adjacency_matrix=adjacency_matrix,
            prediction_type="mean",
            batched=False
        )
        
        # Predict Y for T=0
        y_pred_0 = model.predict(
            X_obs=X_train,
            T_obs=T_train,
            Y_obs=Y_train,
            X_intv=X_test,
            T_intv=T_intv_0,
            adjacency_matrix=adjacency_matrix,
            prediction_type="mean",
            batched=False
        )
        
        # CATE = E[Y|T=1,X] - E[Y|T=0,X]
        # Only keep predictions for non-padded test samples
        cate_pred_scaled = (y_pred_1 - y_pred_0)[:n_test]
    
    # Inverse transform predictions from [-1, 1] back to original scale
    # Original transform: Y_scaled = 2.0 * (Y - ymin) / rng - 1.0
    # Inverse: Y_original = (Y_scaled + 1.0) * rng / 2.0 + ymin
    # For CATE (difference), we only need to scale by the range:
    # CATE_original = CATE_scaled * rng / 2.0
    cate_pred = cate_pred_scaled * y_range / 2.0
    
    
    return cate_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline experiments.")

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (IHDP, ACIC, CPS, PSID) or 'all' for all datasets")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")    
    parser.add_argument("--exp_name", type=str, required=True, help="Current time of experiment")    

    args = parser.parse_args()

    # Hardcoded model initialization
    checkpoint_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist.pt"
    model_config_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist_config.yaml"
    
    print(f"Loading model from: {checkpoint_path}")
    graphintpfn = GraphConditionedInterventionalPFNSklearn(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    graphintpfn.load()
    model = graphintpfn

    # Determine which datasets to run
    ALL_DATASETS = ["IHDP", "ACIC", "CPS", "PSID"]
    if args.dataset.lower() == "all":
        datasets_to_run = ALL_DATASETS
    else:
        datasets_to_run = [args.dataset]
    
    # Run evaluation for each dataset
    for dataset in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"--- Starting Experiment ---")
        print(f"Dataset: {dataset}")
        print(f"Model:   {args.model}")
        print(f"{'='*60}\n")
        
        # Create a copy of args with the current dataset
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = dataset
        
        try:
            evaluate_pipeline(
                exp_name=args.exp_name,
                model_pipeline=dofm_pipeline,
                model=model,
                args=dataset_args)
            print(f"\n--- {dataset} Experiment Finished ---")
        except Exception as e:
            print(f"\n--- {dataset} Experiment FAILED: {e} ---")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"--- All Experiments Finished ---")
    print(f"{'='*60}")    
