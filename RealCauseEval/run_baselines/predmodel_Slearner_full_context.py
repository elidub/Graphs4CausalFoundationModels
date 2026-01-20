"""
S-Learner baseline with full context size (no sample truncation).

Uses SimplePFN as an S-learner to predict CATE with the entire training set.
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml

# Add paths
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting')
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting/RealCauseEval')

from src.models.SimplePFN_sklearn import SimplePFNSklearn
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

def slearner_full_context_pipeline(model, cate_dataset):
    """
    Use SimplePFN as an S-learner with FULL context size (no truncation).
    
    S-learner approach:
    1. Train a single model with treatment as an additional feature
    2. Predict outcomes for test set with T=1 and T=0
    3. CATE = E[Y|T=1,X] - E[Y|T=0,X]
    """
    global MODEL_CONFIG
    
    # Extract data from cate_dataset
    X_train = cate_dataset.X_train
    t_train_orig = cate_dataset.t_train.reshape(-1, 1) if cate_dataset.t_train.ndim == 1 else cate_dataset.t_train
    y_train = cate_dataset.y_train.reshape(-1) if cate_dataset.y_train.ndim > 1 else cate_dataset.y_train
    X_test = cate_dataset.X_test
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features_orig = X_train.shape[1]
    
    print(f"[Full Context S-Learner] Using ALL {n_train} training samples (no truncation)")
    
    # Target encoding for treatment: replace T with mean(Y|T)
    t_flat = t_train_orig.flatten()
    mean_y_t0 = y_train[t_flat == 0].mean()
    mean_y_t1 = y_train[t_flat == 1].mean()
    t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)
    
    # For intervention values, use the same target encoding
    t_intv_0_encoded = mean_y_t0
    t_intv_1_encoded = mean_y_t1
    
    print(f"[Target Encoding] T=0 -> {mean_y_t0:.4f}, T=1 -> {mean_y_t1:.4f}")
    
    # Get model's expected number of features
    model_n_features = model.model.num_features
    
    # Load preprocessing config if not already loaded
    if MODEL_CONFIG is None:
        preprocessing_config, dataset_config = load_preprocessing_config(
            "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/basic_16691166.0.yaml"
        )
        MODEL_CONFIG = {
            'preprocessing': preprocessing_config,
            'dataset': dataset_config
        }
    
    preprocessing_config = MODEL_CONFIG['preprocessing']
    dataset_config = MODEL_CONFIG['dataset']
    
    # NO SAMPLE TRUNCATION - use full dataset
    # Only get test batch size for batched predictions
    max_test_samples = get_config_value(dataset_config, 'n_test_samples_per_dataset', 500)
    if isinstance(max_test_samples, dict):
        max_test_samples = max_test_samples.get('distribution_parameters', {}).get('high', 500)
    test_batch_size = max_test_samples
    
    # Reserve one feature slot for treatment
    max_features_for_X = model_n_features - 1
    
    # Truncate or pad features to match available slots
    if n_features_orig > max_features_for_X:
        print(f"Truncating features from {n_features_orig} to {max_features_for_X} (reserving 1 for treatment)")
        X_train = X_train[:, :max_features_for_X]
        X_test = X_test[:, :max_features_for_X]
        n_features = max_features_for_X
    elif n_features_orig < max_features_for_X:
        print(f"Padding features from {n_features_orig} to {max_features_for_X} (reserving 1 for treatment)")
        X_train = np.hstack([X_train, np.zeros((n_train, max_features_for_X - n_features_orig))])
        X_test = np.hstack([X_test, np.zeros((n_test, max_features_for_X - n_features_orig))])
        n_features = max_features_for_X
    else:
        n_features = n_features_orig
    
    # Apply feature preprocessing
    outlier_quantile = get_config_value(preprocessing_config, 'outlier_quantile', 0.99)
    remove_outliers = get_config_value(preprocessing_config, 'remove_outliers', True)
    feature_standardize = get_config_value(preprocessing_config, 'feature_standardize', True)
    
    if remove_outliers:
        lower_quantile = 1.0 - outlier_quantile
        upper_quantile = outlier_quantile
        lower_bounds = np.quantile(X_train, lower_quantile, axis=0)
        upper_bounds = np.quantile(X_train, upper_quantile, axis=0)
        X_train = np.clip(X_train, lower_bounds, upper_bounds)
        X_test = np.clip(X_test, lower_bounds, upper_bounds)
        print(f"Applied outlier removal at quantile {outlier_quantile}")
    
    if feature_standardize:
        X_mean = np.mean(X_train, axis=0, keepdims=True)
        X_std = np.std(X_train, axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        print(f"Applied feature standardization")
    
    # Add treatment as a feature (S-learner approach)
    X_train_with_T = np.hstack([X_train, t_train])
    
    print(f"S-learner: Combined features shape (X + T): {X_train_with_T.shape}")
    
    # Apply target scaling to [-1, 1]
    ymin = float(np.min(y_train))
    ymax = float(np.max(y_train))
    y_range = max(ymax - ymin, 1e-8)
    
    print(f"Target scaling stats: ymin={ymin:.4f}, ymax={ymax:.4f}, range={y_range:.4f}")
    
    y_train_scaled = 2.0 * (y_train - ymin) / y_range - 1.0
    
    print(f"Scaled target range: [{np.min(y_train_scaled):.4f}, {np.max(y_train_scaled):.4f}]")
    
    # Create test data with T=1 and T=0 using target-encoded values
    X_test_with_T1 = np.hstack([X_test, np.full((n_test, 1), t_intv_1_encoded)])
    X_test_with_T0 = np.hstack([X_test, np.full((n_test, 1), t_intv_0_encoded)])
    
    def batched_predict(model, X_train_with_T, y_train_scaled, X_test_with_T, batch_size):
        """Predict in batches to handle large test sets."""
        n_samples = X_test_with_T.shape[0]
        if n_samples <= batch_size:
            return model.predict(
                X_train=X_train_with_T,
                y_train=y_train_scaled,
                X_test=X_test_with_T,
                prediction_type="mean",
                aggregate="mean"
            )
        
        predictions = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_pred = model.predict(
                X_train=X_train_with_T,
                y_train=y_train_scaled,
                X_test=X_test_with_T[start_idx:end_idx],
                prediction_type="mean",
                aggregate="mean"
            )
            predictions.append(batch_pred)
        return np.concatenate(predictions)
    
    # Predict Y for T=1 and T=0
    y_pred_1_scaled = batched_predict(
        model, X_train_with_T, y_train_scaled, X_test_with_T1, test_batch_size
    )
    y_pred_0_scaled = batched_predict(
        model, X_train_with_T, y_train_scaled, X_test_with_T0, test_batch_size
    )
    
    # CATE in scaled space
    cate_pred_scaled = y_pred_1_scaled - y_pred_0_scaled
    
    # Inverse transform: CATE_original = CATE_scaled * range / 2.0
    cate_pred = cate_pred_scaled * y_range / 2.0
    
    print(f"CATE predictions shape: {cate_pred.shape}")
    print(f"CATE predictions range: [{np.min(cate_pred):.4f}, {np.max(cate_pred):.4f}]")
    
    return cate_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S-learner with full context size.")

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")    
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")    

    args = parser.parse_args()

    print(f"--- Starting S-Learner Full Context Experiment ---")
    print(f"Dataset: {args.dataset}")
    print(f"Model:   {args.model}")
    
    checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/step_55000.pt"
    model_config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/basic_16691166.0.yaml"
    
    print(f"Loading SimplePFN model from: {checkpoint_path}")
    simple_pfn = SimplePFNSklearn(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
        n_estimators=1,
        max_n_train=None,  # No clustering
        max_n_test=None,
    )
    simple_pfn.load()
    model = simple_pfn

    evaluate_pipeline(
        exp_name=args.exp_name,
        model_pipeline=slearner_full_context_pipeline,
        model=model,
        args=args)
    
    print(f"--- S-Learner Full Context Experiment Finished ---")
