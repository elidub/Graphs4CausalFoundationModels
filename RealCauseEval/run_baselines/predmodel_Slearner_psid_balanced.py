"""
S-Learner variant for PSID imbalance: use all treated samples and up to 500 controls.

Training: all T=1 samples + random 500 T=0 samples (if available).
Testing: use full test set.

Target encoding is computed on the sampled training set (the data the model sees).
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


def slearner_psid_balanced_pipeline(model, cate_dataset):
    """
    S-learner pipeline that uses full test set and for training uses all treated
    examples and up to 500 randomly selected control examples (for PSID only).
    For other datasets this behaves like the full-context S-learner.
    """
    global MODEL_CONFIG, DATASET_NAME

    # Extract data from cate_dataset
    X_train_full = cate_dataset.X_train
    t_train_full = cate_dataset.t_train.reshape(-1, 1) if cate_dataset.t_train.ndim == 1 else cate_dataset.t_train
    y_train_full = cate_dataset.y_train.reshape(-1) if cate_dataset.y_train.ndim > 1 else cate_dataset.y_train
    X_test = cate_dataset.X_test

    n_train_full = X_train_full.shape[0]
    n_test = X_test.shape[0]
    n_features_orig = X_train_full.shape[1]

    # Default: use full training set
    X_train = X_train_full
    t_train_orig = t_train_full
    y_train = y_train_full

    # If PSID, subsample controls: keep all treated (T==1) and up to 500 controls (T==0)
    if DATASET_NAME is not None and DATASET_NAME.upper() == 'PSID':
        t_flat_full = t_train_full.flatten()
        treated_mask = (t_flat_full == 1)
        control_mask = (t_flat_full == 0)

        X_treated = X_train_full[treated_mask]
        t_treated = t_train_full[treated_mask]
        y_treated = y_train_full[treated_mask]

        X_control = X_train_full[control_mask]
        t_control = t_train_full[control_mask]
        y_control = y_train_full[control_mask]

        n_control = X_control.shape[0]
        n_keep_control = min(500, n_control)

        if n_control > n_keep_control:
            np.random.seed(42)
            indices = np.random.choice(n_control, n_keep_control, replace=False)
            X_control_sel = X_control[indices]
            t_control_sel = t_control[indices]
            y_control_sel = y_control[indices]
            print(f"PSID sampling: kept all {X_treated.shape[0]} treated, sampled {n_keep_control} / {n_control} controls")
        else:
            X_control_sel = X_control
            t_control_sel = t_control
            y_control_sel = y_control
            print(f"PSID sampling: kept all {X_treated.shape[0]} treated and all {n_control} controls (<=500)")

        # Concatenate treated + selected controls
        X_train = np.vstack([X_treated, X_control_sel])
        t_train_orig = np.vstack([t_treated, t_control_sel])
        y_train = np.concatenate([y_treated, y_control_sel])

        # Shuffle so ordering isn't grouped
        perm = np.random.RandomState(42).permutation(X_train.shape[0])
        X_train = X_train[perm]
        t_train_orig = t_train_orig[perm]
        y_train = y_train[perm]

        n_train = X_train.shape[0]
        print(f"After PSID sampling: n_train={n_train}")
    else:
        n_train = n_train_full
        print(f"Using full training set: n_train={n_train}")

    # Compute target encoding on the training set the model will see
    t_flat = t_train_orig.flatten()
    mean_y_t0 = y_train[t_flat == 0].mean() if np.any(t_flat == 0) else 0.0
    mean_y_t1 = y_train[t_flat == 1].mean() if np.any(t_flat == 1) else 0.0
    t_train = np.where(t_train_orig == 0, mean_y_t0, mean_y_t1).astype(np.float32)

    # Intervention encodings
    t_intv_0_encoded = mean_y_t0
    t_intv_1_encoded = mean_y_t1

    print(f"[Target Encoding on training set] T=0 -> {mean_y_t0:.4f}, T=1 -> {mean_y_t1:.4f}")

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

    # Get test batch size for predictions
    max_test_samples = get_config_value(dataset_config, 'n_test_samples_per_dataset', 500)
    if isinstance(max_test_samples, dict):
        max_test_samples = max_test_samples.get('distribution_parameters', {}).get('high', 500)

    test_batch_size = max_test_samples

    # Reserve one feature for treatment
    max_features_for_X = model_n_features - 1

    # Truncate or pad features to match model input
    if n_features_orig > max_features_for_X:
        X_train = X_train[:, :max_features_for_X]
        X_test = X_test[:, :max_features_for_X]
        n_features = max_features_for_X
    elif n_features_orig < max_features_for_X:
        X_train = np.hstack([X_train, np.zeros((X_train.shape[0], max_features_for_X - n_features_orig))])
        X_test = np.hstack([X_test, np.zeros((X_test.shape[0], max_features_for_X - n_features_orig))])
        n_features = max_features_for_X
    else:
        n_features = n_features_orig

    # Apply outlier clipping and standardization (from preprocessing config)
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

    if feature_standardize:
        X_mean = np.mean(X_train, axis=0, keepdims=True)
        X_std = np.std(X_train, axis=0, keepdims=True)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)
        X_train = (X_train - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std

    # Add treatment as a feature
    X_train_with_T = np.hstack([X_train, t_train])

    # Scale targets to [-1,1]
    ymin = float(np.min(y_train))
    ymax = float(np.max(y_train))
    y_range = max(ymax - ymin, 1e-8)
    y_train_scaled = 2.0 * (y_train - ymin) / y_range - 1.0

    # Prepare test inputs with encoded T
    X_test_with_T1 = np.hstack([X_test, np.full((n_test, 1), t_intv_1_encoded)])
    X_test_with_T0 = np.hstack([X_test, np.full((n_test, 1), t_intv_0_encoded)])

    # Batched predict
    def batched_predict(model, X_train_with_T, y_train_scaled, X_test_with_T, batch_size):
        n_samples = X_test_with_T.shape[0]
        if n_samples <= batch_size:
            return model.predict(
                X_train=X_train_with_T,
                y_train=y_train_scaled,
                X_test=X_test_with_T,
                prediction_type="mean",
                aggregate="mean"
            )
        preds = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            p = model.predict(
                X_train=X_train_with_T,
                y_train=y_train_scaled,
                X_test=X_test_with_T[start:end],
                prediction_type="mean",
                aggregate="mean"
            )
            preds.append(p)
        return np.concatenate(preds)

    y_pred_1_scaled = batched_predict(model, X_train_with_T, y_train_scaled, X_test_with_T1, test_batch_size)
    y_pred_0_scaled = batched_predict(model, X_train_with_T, y_train_scaled, X_test_with_T0, test_batch_size)

    cate_pred_scaled = y_pred_1_scaled - y_pred_0_scaled
    cate_pred = cate_pred_scaled * y_range / 2.0

    return cate_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run S-learner PSID-balanced experiment")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    args = parser.parse_args()

    DATASET_NAME = args.dataset

    print(f"Starting S-learner PSID-balanced: dataset={DATASET_NAME}")

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
        model_pipeline=slearner_psid_balanced_pipeline,
        model=model,
        args=args
    )

    print("S-learner PSID-balanced finished")
