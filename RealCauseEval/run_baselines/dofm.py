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
from src.priordata_processing.Preprocessor import Preprocessor
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


def preprocess_real_world_data(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    preprocessing_config: dict,
    max_n_features: int,
) -> tuple:
    """
    Apply exactly the same preprocessing as during training using the Preprocessor class.
    
    Args:
        X_train: Training features [n_train, n_features]
        Y_train: Training targets [n_train] or [n_train, 1]
        X_test: Test features [n_test, n_features]
        preprocessing_config: Dict with preprocessing settings from training config
        max_n_features: Maximum number of features the model was trained on
        
    Returns:
        X_train_processed, Y_train_processed, X_test_processed, ymin, ymax
    """
    # Extract preprocessing settings from config
    feature_standardize = get_config_value(preprocessing_config, 'feature_standardize', True)
    feature_negative_one_one_scaling = get_config_value(preprocessing_config, 'feature_negative_one_one_scaling', False)
    target_negative_one_one_scaling = get_config_value(preprocessing_config, 'target_negative_one_one_scaling', True)
    remove_outliers = get_config_value(preprocessing_config, 'remove_outliers', True)
    outlier_quantile = get_config_value(preprocessing_config, 'outlier_quantile', 0.99)
    yeo_johnson = get_config_value(preprocessing_config, 'yeo_johnson', False)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features = X_train.shape[1]
    
    # Ensure Y is 1D
    Y_train = Y_train.flatten()
    
    # Convert to torch tensors with batch dimension [1, n_samples, n_features]
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(0)  # [1, n_train, n_features]
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0)    # [1, n_test, n_features]
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(0)  # [1, n_train]
    
    # Concatenate X for processing (train first, then test) - same structure as Preprocessor expects
    X_combined = torch.cat([X_train_t, X_test_t], dim=1)  # [1, n_train + n_test, n_features]
    # Create dummy Y for test (we don't use it, but Preprocessor expects it)
    Y_dummy_test = torch.zeros(1, n_test, dtype=torch.float32)
    Y_combined = torch.cat([Y_train_t, Y_dummy_test], dim=1)  # [1, n_train + n_test]
    
    # Create Preprocessor with matching settings
    preprocessor = Preprocessor(
        n_features=n_features,
        max_n_features=max_n_features,
        n_train_samples=n_train,
        max_n_train_samples=n_train,  # Don't pad samples yet
        n_test_samples=n_test,
        max_n_test_samples=n_test,    # Don't pad samples yet
        feature_standardize=feature_standardize,
        feature_negative_one_one_scaling=feature_negative_one_one_scaling,
        target_negative_one_one_scaling=target_negative_one_one_scaling,
        yeo_johnson=yeo_johnson,
        remove_outliers=remove_outliers,
        outlier_quantile=outlier_quantile,
        shuffle_samples=False,  # Don't shuffle - we already have our split
        shuffle_features=False,  # Don't shuffle features
    )
    
    # Process the data
    result = preprocessor.process(X_combined, Y_combined)
    
    if result is None:
        raise ValueError("Preprocessing returned None - empty dataset?")
    
    X_train_processed, X_test_processed, Y_train_processed, Y_test_processed = result
    
    # Remove batch dimension and convert back to numpy
    X_train_processed = X_train_processed.squeeze(0).numpy()  # [n_train, n_features]
    X_test_processed = X_test_processed.squeeze(0).numpy()    # [n_test, n_features]
    Y_train_processed = Y_train_processed.squeeze(0).numpy()  # [n_train]
    
    # Compute original Y stats for inverse transform (before scaling)
    ymin = float(np.min(Y_train))
    ymax = float(np.max(Y_train))
    
    print(f"Preprocessing applied: remove_outliers={remove_outliers} (q={outlier_quantile}), "
          f"yeo_johnson={yeo_johnson}, feature_standardize={feature_standardize}, "
          f"target_scaling={target_negative_one_one_scaling}")
    
    return X_train_processed, Y_train_processed, X_test_processed, ymin, ymax

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
    X_train = cate_dataset.X_train.copy()
    t_train = cate_dataset.t_train.reshape(-1, 1) if cate_dataset.t_train.ndim == 1 else cate_dataset.t_train.copy()
    y_train = cate_dataset.y_train.copy()
    X_test = cate_dataset.X_test.copy()
    
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
    
    # Truncate or pad features to match model size (before preprocessing)
    if n_features_orig > model_n_features:
        print(f"Truncating features from {n_features_orig} to {model_n_features}")
        X_train = X_train[:, :model_n_features]
        X_test = X_test[:, :model_n_features]
        n_features = model_n_features
    elif n_features_orig < model_n_features:
        print(f"Padding features from {n_features_orig} to {model_n_features}")
        # Pad with zeros
        X_train = np.hstack([X_train, np.zeros((n_train, model_n_features - n_features_orig))])
        X_test = np.hstack([X_test, np.zeros((n_test, model_n_features - n_features_orig))])
        n_features = model_n_features
    else:
        n_features = n_features_orig
    
    # Number of real (non-padded) features for graph construction
    n_real_features = min(n_features_orig, model_n_features)
    
    # Apply preprocessing using the Preprocessor class (exactly matches training)
    X_train, Y_train, X_test, ymin, ymax = preprocess_real_world_data(
        X_train=X_train,
        Y_train=y_train,
        X_test=X_test,
        preprocessing_config=preprocessing_config,
        max_n_features=model_n_features,
    )
    y_range = max(ymax - ymin, 1e-8)
    
    print(f"Target scaling stats: ymin={ymin:.4f}, ymax={ymax:.4f}, range={y_range:.4f}")
    print(f"Scaled target range: [{np.min(Y_train):.4f}, {np.max(Y_train):.4f}]")
    
    # Convert to expected types
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    T_train = t_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    
    # Get max samples from config (what the model was trained with)
    max_n_train_samples = get_config_value(dataset_config, 'max_number_train_samples_per_dataset', n_train)
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
    # Shape: (model_n_features + 2, model_n_features + 2) for treatment + outcome + features
    # Partial graph format uses three states:
    #   -1: No edge/ancestor relationship (known non-edge)
    #   0: Unknown whether edge/ancestor exists
    #   1: Edge/ancestor relationship exists (known edge)
    
    # Position mapping (matching InterventionalDataset convention):
    # - Position 0: Treatment (T)
    # - Position 1: Outcome (Y)
    # - Positions 2 to model_n_features+1: Feature variables
    
    # Initialize all as unknown (0)
    adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
    
    # Add known causal edges for REAL features only:
    T_idx = 0
    Y_idx = 1
    feature_offset = 2  # Features start at position 2
    
    # 1. T -> Y (treatment causes outcome)
    adjacency_matrix[T_idx, Y_idx] = 1.0
    
    # 2. Real features -> T (features cause treatment)
    for i in range(n_real_features):
        adjacency_matrix[feature_offset + i, T_idx] = 1.0
    
    # 3. Real features -> Y (features cause outcome)
    for i in range(n_real_features):
        adjacency_matrix[feature_offset + i, Y_idx] = 1.0
    
    # Note: Relationships between real features remain unknown (0)
    
    # 4. PADDED features: Set all edges to -1 (no edge) since they don't exist
    # Padded features are at positions feature_offset + n_real_features to feature_offset + model_n_features - 1
    for i in range(n_real_features, model_n_features):
        feat_idx = feature_offset + i
        # Padded feature has no edges TO anything
        adjacency_matrix[feat_idx, :] = -1.0
        # Nothing has edges TO padded feature
        adjacency_matrix[:, feat_idx] = -1.0
        # Diagonal is also -1 (no self-loops)
        adjacency_matrix[feat_idx, feat_idx] = -1.0
    
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
