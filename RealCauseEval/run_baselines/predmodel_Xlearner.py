"""
X-Learner for CATE estimation using SimplePFN.

X-learner approach:
1. Train two outcome models: μ₀(X) on control units, μ₁(X) on treated units
2. Compute imputed treatment effects:
   - For treated units: D₁ᵢ = Yᵢ - μ₀(Xᵢ)  (observed outcome - imputed control outcome)
   - For control units: D₀ᵢ = μ₁(Xᵢ) - Yᵢ  (imputed treated outcome - observed outcome)
3. Train two CATE models: τ₀(X) on control units, τ₁(X) on treated units
4. Combine using propensity scores: τ(X) = g(X)·τ₀(X) + (1-g(X))·τ₁(X)
   where g(X) is the propensity score P(T=1|X)

Reference: Künzel et al. (2019) "Metalearners for estimating heterogeneous treatment effects using machine learning"
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml

# Add paths
sys.path.insert(0, '<REPO_ROOT>')
sys.path.insert(0, '<REPO_ROOT>/RealCauseEval')

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


def xlearner_pipeline(model, cate_dataset):
    """
    Use SimplePFN as an X-learner to predict CATE.
    
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
    global MODEL_CONFIG
    
    # Extract data from cate_dataset
    X_train = cate_dataset.X_train
    t_train = cate_dataset.t_train.reshape(-1) if cate_dataset.t_train.ndim > 1 else cate_dataset.t_train
    y_train = cate_dataset.y_train.reshape(-1) if cate_dataset.y_train.ndim > 1 else cate_dataset.y_train
    X_test = cate_dataset.X_test
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features_orig = X_train.shape[1]
    
    # Get model's expected number of features
    model_n_features = model.model.num_features
    
    # Load preprocessing config if not already loaded
    if MODEL_CONFIG is None:
        preprocessing_config, dataset_config = load_preprocessing_config(
            "<REPO_ROOT>/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/basic_16691166.0.yaml"
        )
        MODEL_CONFIG = {
            'preprocessing': preprocessing_config,
            'dataset': dataset_config
        }
    
    preprocessing_config = MODEL_CONFIG['preprocessing']
    dataset_config = MODEL_CONFIG['dataset']
    
    # Truncate features if needed
    if n_features_orig > model_n_features:
        print(f"Truncating features from {n_features_orig} to {model_n_features}")
        X_train = X_train[:, :model_n_features]
        X_test = X_test[:, :model_n_features]
        n_features = model_n_features
    elif n_features_orig < model_n_features:
        print(f"Padding features from {n_features_orig} to {model_n_features}")
        X_train = np.hstack([X_train, np.zeros((n_train, model_n_features - n_features_orig))])
        X_test = np.hstack([X_test, np.zeros((n_test, model_n_features - n_features_orig))])
        n_features = model_n_features
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
    
    # Split training data into treated and control groups
    treated_mask = (t_train == 1)
    control_mask = (t_train == 0)
    
    X_train_treated = X_train[treated_mask]
    y_train_treated = y_train[treated_mask]
    
    X_train_control = X_train[control_mask]
    y_train_control = y_train[control_mask]
    
    print(f"Training data split: {X_train_treated.shape[0]} treated, {X_train_control.shape[0]} control")
    
    # Apply target scaling to [-1, 1]
    y_train_all = y_train.flatten()
    ymin = float(np.min(y_train_all))
    ymax = float(np.max(y_train_all))
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
    y_train_scaled = 2.0 * (y_train - ymin) / y_range - 1.0
    
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
    
    # Simple propensity score estimation: sample proportion
    # For more sophisticated estimation, could use logistic regression or another model
    propensity = np.mean(treated_mask)
    print(f"Propensity score (sample proportion): {propensity:.4f}")
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run X-learner baseline experiments.")

    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (IHDP, ACIC, CPS, PSID) or 'all' for all datasets")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")    
    parser.add_argument("--exp_name", type=str, required=True, help="Current time of experiment")    

    args = parser.parse_args()

    # Hardcoded SimplePFN model initialization
    checkpoint_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/step_55000.pt"
    model_config_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints/simple_pfn_16691166.0_tabpfn_benchmark/basic_16691166.0.yaml"
    
    print(f"Loading SimplePFN model from: {checkpoint_path}")
    simple_pfn = SimplePFNSklearn(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        verbose=True,
        n_estimators=1,  # Single model for X-learner
        max_n_train=None,  # No clustering
        max_n_test=None,  # No test batching
    )
    simple_pfn.load()
    model = simple_pfn

    # Determine which datasets to run
    ALL_DATASETS = ["IHDP", "ACIC", "CPS", "PSID"]
    if args.dataset.lower() == "all":
        datasets_to_run = ALL_DATASETS
    else:
        datasets_to_run = [args.dataset]
    
    # Run evaluation for each dataset
    for dataset in datasets_to_run:
        print(f"\n{'='*60}")
        print(f"--- Starting X-Learner Experiment ---")
        print(f"Dataset: {dataset}")
        print(f"Model:   {args.model}")
        print(f"{'='*60}\n")
        
        # Create a copy of args with the current dataset
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = dataset
        
        try:
            evaluate_pipeline(
                exp_name=args.exp_name,
                model_pipeline=xlearner_pipeline,
                model=model,
                args=dataset_args)
            print(f"\n--- {dataset} X-Learner Experiment Finished ---")
        except Exception as e:
            print(f"\n--- {dataset} X-Learner Experiment FAILED: {e} ---")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"--- All X-Learner Experiments Finished ---")
    print(f"{'='*60}")
