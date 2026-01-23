"""
X-Learner variant for PSID imbalance: use all treated samples and up to 500 controls.

Training: all T=1 samples + random 500 T=0 samples (if available).
Testing: use full test set.

For X-learner:
- Stage 1: Outcome models trained on all T=1 and up to 500 T=0 samples
- Stage 2: Imputed treatment effects computed
- Stage 3: CATE models trained on the same samples
- Stage 4: Combine using propensity score weighting

Reference: Künzel et al. (2019) "Metalearners for estimating heterogeneous treatment effects using machine learning"
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
DATASET_NAME = None


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


def xlearner_psid_balanced_pipeline(model, cate_dataset):
    """
    X-learner pipeline that for PSID uses all treated samples and up to 500 controls.
    For other datasets this behaves like the regular X-learner.
    
    X-learner stages:
    1. Stage 1: Estimate outcome models μ₀(X) and μ₁(X)
    2. Stage 2: Compute imputed treatment effects D₁ and D₀
    3. Stage 3: Estimate CATE models τ₀(X) and τ₁(X)
    4. Stage 4: Combine using propensity score weighting
    
    Args:
        model: SimplePFNSklearn instance
        cate_dataset: CATE_Dataset with X_train, t_train, y_train, X_test, true_cate
        
    Returns:
        cate_pred: Array of CATE predictions for X_test
    """
    global MODEL_CONFIG, DATASET_NAME
    
    # Extract data from cate_dataset
    X_train_full = cate_dataset.X_train
    t_train_full = cate_dataset.t_train.reshape(-1) if cate_dataset.t_train.ndim > 1 else cate_dataset.t_train
    y_train_full = cate_dataset.y_train.reshape(-1) if cate_dataset.y_train.ndim > 1 else cate_dataset.y_train
    X_test = cate_dataset.X_test
    
    n_train_full = X_train_full.shape[0]
    n_test = X_test.shape[0]
    n_features_orig = X_train_full.shape[1]
    
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
    
    # Truncate or pad features to match model input
    if n_features_orig > model_n_features:
        print(f"Truncating features from {n_features_orig} to {model_n_features}")
        X_train = X_train_full[:, :model_n_features]
        X_test = X_test[:, :model_n_features]
        n_features = model_n_features
    elif n_features_orig < model_n_features:
        print(f"Padding features from {n_features_orig} to {model_n_features}")
        X_train = np.hstack([X_train_full, np.zeros((n_train_full, model_n_features - n_features_orig))])
        X_test = np.hstack([X_test, np.zeros((n_test, model_n_features - n_features_orig))])
        n_features = model_n_features
    else:
        X_train = X_train_full
        n_features = n_features_orig
    
    # Apply feature preprocessing as done during training
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
    
    # Split training data into treated and control groups (after preprocessing!)
    treated_mask = (t_train_full == 1)
    control_mask = (t_train_full == 0)
    
    X_train_treated = X_train[treated_mask]
    y_train_treated = y_train_full[treated_mask]
    
    X_train_control = X_train[control_mask]
    y_train_control = y_train_full[control_mask]
    
    # If PSID, subsample controls: keep all treated and up to 500 controls
    if DATASET_NAME is not None and DATASET_NAME.upper() == 'PSID':
        n_control = X_train_control.shape[0]
        n_keep_control = min(500, n_control)
        
        if n_control > n_keep_control:
            np.random.seed(42)
            indices = np.random.choice(n_control, n_keep_control, replace=False)
            X_train_control = X_train_control[indices]
            y_train_control = y_train_control[indices]
            print(f"PSID sampling: kept all {X_train_treated.shape[0]} treated, sampled {n_keep_control} / {n_control} controls")
        else:
            print(f"PSID sampling: kept all {X_train_treated.shape[0]} treated and all {n_control} controls (<=500)")
    else:
        print(f"Using full training set: {X_train_treated.shape[0]} treated, {X_train_control.shape[0]} control")
    
    print(f"Training data split: {X_train_treated.shape[0]} treated, {X_train_control.shape[0]} control")
    
    # Apply target scaling to [-1, 1]
    # Use combined y_train for computing scaling parameters
    y_train_combined = np.concatenate([y_train_treated, y_train_control])
    ymin = float(np.min(y_train_combined))
    ymax = float(np.max(y_train_combined))
    y_range = max(ymax - ymin, 1e-8)
    
    print(f"Target scaling stats: ymin={ymin:.4f}, ymax={ymax:.4f}, range={y_range:.4f}")
    
    y_train_treated_scaled = 2.0 * (y_train_treated - ymin) / y_range - 1.0
    y_train_control_scaled = 2.0 * (y_train_control - ymin) / y_range - 1.0
    
    # ========================================================================
    # STAGE 1: Train outcome models μ₀(X) and μ₁(X)
    # ========================================================================
    print("\n=== Stage 1: Training outcome models ===")
    
    # Predict μ₁(X) for all units (using treated model)
    mu1_train_scaled = model.predict(
        X_train=X_train_treated,
        y_train=y_train_treated_scaled,
        X_test=X_train,
        prediction_type="mean",
        aggregate="mean"
    )
    
    mu1_test_scaled = model.predict(
        X_train=X_train_treated,
        y_train=y_train_treated_scaled,
        X_test=X_test,
        prediction_type="mean",
        aggregate="mean"
    )
    
    # Predict μ₀(X) for all units (using control model)
    mu0_train_scaled = model.predict(
        X_train=X_train_control,
        y_train=y_train_control_scaled,
        X_test=X_train,
        prediction_type="mean",
        aggregate="mean"
    )
    
    mu0_test_scaled = model.predict(
        X_train=X_train_control,
        y_train=y_train_control_scaled,
        X_test=X_test,
        prediction_type="mean",
        aggregate="mean"
    )
    
    # ========================================================================
    # STAGE 2: Compute imputed treatment effects
    # ========================================================================
    print("=== Stage 2: Computing imputed treatment effects ===")
    
    # Scale y_train for imputation
    y_train_scaled = 2.0 * (y_train_full - ymin) / y_range - 1.0
    
    # For treated units: D₁ = Y - μ₀(X) (observed - imputed control)
    D1_scaled = y_train_scaled[treated_mask] - mu0_train_scaled[treated_mask]
    
    # For control units: D₀ = μ₁(X) - Y (imputed treated - observed)
    D0_scaled = mu1_train_scaled[control_mask] - y_train_scaled[control_mask]
    
    print(f"Imputed effects - D₁: mean={np.mean(D1_scaled):.4f}, std={np.std(D1_scaled):.4f}")
    print(f"Imputed effects - D₀: mean={np.mean(D0_scaled):.4f}, std={np.std(D0_scaled):.4f}")
    
    # ========================================================================
    # STAGE 3: Train CATE models τ₀(X) and τ₁(X)
    # ========================================================================
    print("=== Stage 3: Training CATE models ===")
    
    # Train τ₁(X) on treated units using D₁
    tau1_test_scaled = model.predict(
        X_train=X_train_treated,
        y_train=D1_scaled,
        X_test=X_test,
        prediction_type="mean",
        aggregate="mean"
    )
    
    # Train τ₀(X) on control units using D₀
    tau0_test_scaled = model.predict(
        X_train=X_train_control,
        y_train=D0_scaled,
        X_test=X_test,
        prediction_type="mean",
        aggregate="mean"
    )
    
    # ========================================================================
    # STAGE 4: Estimate propensity scores and combine
    # ========================================================================
    print("=== Stage 4: Combining with propensity scores ===")
    
    # Simple propensity score estimation: sample proportion in the balanced set
    propensity = X_train_treated.shape[0] / (X_train_treated.shape[0] + X_train_control.shape[0])
    print(f"Propensity score (balanced sample proportion): {propensity:.4f}")
    
    # For X-learner, we weight by propensity:
    # τ(X) = g(X)·τ₀(X) + (1-g(X))·τ₁(X)
    # where g(X) = P(T=1|X)
    # 
    # Using constant propensity: τ(X) = g·τ₀(X) + (1-g)·τ₁(X)
    cate_pred_scaled = propensity * tau0_test_scaled + (1 - propensity) * tau1_test_scaled
    
    # Inverse transform to original scale
    # For CATE (difference), the offset cancels: CATE_original = CATE_scaled * range / 2.0
    cate_pred = cate_pred_scaled * y_range / 2.0
    
    print(f"\nCombined CATE predictions:")
    print(f"  Shape: {cate_pred.shape}")
    print(f"  Range: [{np.min(cate_pred):.4f}, {np.max(cate_pred):.4f}]")
    print(f"  Mean: {np.mean(cate_pred):.4f}, Std: {np.std(cate_pred):.4f}")
    
    return cate_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run X-learner PSID-balanced experiment")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()

    DATASET_NAME = args.dataset

    print(f"Starting X-learner PSID-balanced: dataset={DATASET_NAME}")

    checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/step_55000.pt"
    model_config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/basic_16691166.0.yaml"

    simple_pfn = SimplePFNSklearn(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
        n_estimators=1,
        max_n_train=None,
        max_n_test=None,
    )
    simple_pfn.load()
    model = simple_pfn

    evaluate_pipeline(
        exp_name=args.exp_name,
        model_pipeline=xlearner_psid_balanced_pipeline,
        model=model,
        args=args
    )

    print("X-learner PSID-balanced finished")
