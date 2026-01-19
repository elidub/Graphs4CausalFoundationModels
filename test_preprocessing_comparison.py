"""
Diagnostic test for comparing preprocessing between training data and real-world data.

This script generates training data using InterventionalDataset and compares
the preprocessing statistics with ACTUAL real-world data (IHDP, ACIC, etc.)
as used in dofm.py.

The goal is to verify that:
1. Both use the same preprocessing pipeline
2. The resulting data has similar statistical properties
3. The scaling and normalization produce comparable ranges
"""

import sys
import os
import numpy as np
import torch
import yaml
from pathlib import Path

# Add paths
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting')
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting/src')
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting/RealCauseEval')

from src.priordata_processing.Preprocessor import Preprocessor
from src.priordata_processing.BasicProcessing import BasicProcessing
from src.priordata_processing.Datasets.InterventionalDataset import InterventionalDataset

# Import real-world CATE datasets
from CausalPFN.benchmarks import IHDPDataset, ACIC2016Dataset
from CausalPFN.benchmarks import RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset

REAL_WORLD_DATASETS = {
    "IHDP": IHDPDataset,
    "ACIC": ACIC2016Dataset,
    "CPS": RealCauseLalondeCPSDataset,
    "PSID": RealCauseLalondePSIDDataset,
}


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config_value(config_dict: dict, key: str, default=None):
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
    Apply preprocessing to real-world data using the Preprocessor class.
    This is the same function as in dofm.py.
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
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(0)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(0)
    
    # Concatenate X for processing (train first, then test)
    X_combined = torch.cat([X_train_t, X_test_t], dim=1)
    Y_dummy_test = torch.zeros(1, n_test, dtype=torch.float32)
    Y_combined = torch.cat([Y_train_t, Y_dummy_test], dim=1)
    
    # Create Preprocessor with matching settings
    preprocessor = Preprocessor(
        n_features=n_features,
        max_n_features=max_n_features,
        n_train_samples=n_train,
        max_n_train_samples=n_train,
        n_test_samples=n_test,
        max_n_test_samples=n_test,
        feature_standardize=feature_standardize,
        feature_negative_one_one_scaling=feature_negative_one_one_scaling,
        target_negative_one_one_scaling=target_negative_one_one_scaling,
        yeo_johnson=yeo_johnson,
        remove_outliers=remove_outliers,
        outlier_quantile=outlier_quantile,
        shuffle_samples=False,
        shuffle_features=False,
    )
    
    result = preprocessor.process(X_combined, Y_combined)
    
    if result is None:
        raise ValueError("Preprocessing returned None")
    
    X_train_processed, X_test_processed, Y_train_processed, Y_test_processed = result
    
    X_train_processed = X_train_processed.squeeze(0).numpy()
    X_test_processed = X_test_processed.squeeze(0).numpy()
    Y_train_processed = Y_train_processed.squeeze(0).numpy()
    
    return X_train_processed, Y_train_processed, X_test_processed


def compute_statistics(data: np.ndarray, name: str) -> dict:
    """Compute comprehensive statistics for a dataset."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    stats = {
        'name': name,
        'shape': data.shape,
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'mean_per_feature': np.mean(data, axis=0) if data.ndim == 2 else data.mean(),
        'std_per_feature': np.std(data, axis=0) if data.ndim == 2 else data.std(),
        'min_per_feature': np.min(data, axis=0) if data.ndim == 2 else data.min(),
        'max_per_feature': np.max(data, axis=0) if data.ndim == 2 else data.max(),
        'nan_count': np.sum(np.isnan(data)),
        'inf_count': np.sum(np.isinf(data)),
        'zero_count': np.sum(data == 0),
        'quantiles': {
            '1%': np.percentile(data, 1),
            '5%': np.percentile(data, 5),
            '25%': np.percentile(data, 25),
            '50%': np.percentile(data, 50),
            '75%': np.percentile(data, 75),
            '95%': np.percentile(data, 95),
            '99%': np.percentile(data, 99),
        }
    }
    return stats


def print_statistics(stats: dict, verbose: bool = True):
    """Print statistics in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Statistics for: {stats['name']}")
    print(f"{'='*60}")
    print(f"  Shape: {stats['shape']}")
    print(f"  Global Mean: {stats['mean']:.6f}")
    print(f"  Global Std:  {stats['std']:.6f}")
    print(f"  Global Min:  {stats['min']:.6f}")
    print(f"  Global Max:  {stats['max']:.6f}")
    print(f"  Median:      {stats['median']:.6f}")
    print(f"  NaN count:   {stats['nan_count']}")
    print(f"  Inf count:   {stats['inf_count']}")
    print(f"  Zero count:  {stats['zero_count']}")
    print(f"\n  Quantiles:")
    for q, v in stats['quantiles'].items():
        print(f"    {q}: {v:.6f}")
    
    if verbose:
        print(f"\n  Per-feature statistics (first 5 features):")
        n_show = min(5, len(stats['mean_per_feature']))
        for i in range(n_show):
            print(f"    Feature {i}: mean={stats['mean_per_feature'][i]:.4f}, "
                  f"std={stats['std_per_feature'][i]:.4f}, "
                  f"range=[{stats['min_per_feature'][i]:.4f}, {stats['max_per_feature'][i]:.4f}]")


def compare_statistics(stats1: dict, stats2: dict) -> dict:
    """Compare two sets of statistics and compute differences."""
    comparison = {
        'name': f"{stats1['name']} vs {stats2['name']}",
        'mean_diff': abs(stats1['mean'] - stats2['mean']),
        'std_diff': abs(stats1['std'] - stats2['std']),
        'min_diff': abs(stats1['min'] - stats2['min']),
        'max_diff': abs(stats1['max'] - stats2['max']),
        'range1': stats1['max'] - stats1['min'],
        'range2': stats2['max'] - stats2['min'],
    }
    return comparison


def print_comparison(comparison: dict):
    """Print comparison results."""
    print(f"\n{'='*60}")
    print(f"Comparison: {comparison['name']}")
    print(f"{'='*60}")
    print(f"  Mean difference:  {comparison['mean_diff']:.6f}")
    print(f"  Std difference:   {comparison['std_diff']:.6f}")
    print(f"  Min difference:   {comparison['min_diff']:.6f}")
    print(f"  Max difference:   {comparison['max_diff']:.6f}")
    print(f"  Range 1:          {comparison['range1']:.6f}")
    print(f"  Range 2:          {comparison['range2']:.6f}")


def generate_training_sample(config_path: str, n_samples: int = 10) -> list:
    """
    Generate samples from InterventionalDataset using the training config.
    
    Returns:
        List of (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv) tuples
    """
    config = load_config(config_path)
    
    scm_config = config.get('scm_config', {})
    preprocessing_config = config.get('preprocessing_config', {})
    dataset_config = config.get('dataset_config', {})
    
    # Override dataset size for testing
    dataset_config['dataset_size'] = {'value': n_samples}
    
    # Create dataset
    dataset = InterventionalDataset(
        scm_config=scm_config,
        preprocessing_config=preprocessing_config,
        dataset_config=dataset_config,
        seed=42,
    )
    
    samples = []
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        samples.append(sample)
    
    return samples


def load_real_world_data(dataset_name: str, realization: int = 0) -> tuple:
    """
    Load actual real-world data from CATE benchmarks.
    
    Args:
        dataset_name: One of 'IHDP', 'ACIC', 'CPS', 'PSID'
        realization: Which realization/table to use
        
    Returns:
        X_train, T_train, Y_train, X_test, true_cate
    """
    dataset_class = REAL_WORLD_DATASETS[dataset_name]
    dataset = dataset_class()
    
    cate_dataset = dataset[realization][0]
    
    X_train = cate_dataset.X_train
    T_train = cate_dataset.t_train
    Y_train = cate_dataset.y_train
    X_test = cate_dataset.X_test
    true_cate = cate_dataset.true_cate
    
    return X_train, T_train, Y_train, X_test, true_cate


def run_diagnostic_test(config_path: str):
    """
    Run the full diagnostic test comparing training data vs real-world data preprocessing.
    """
    print("\n" + "="*80)
    print("PREPROCESSING DIAGNOSTIC TEST")
    print("="*80)
    
    # Load config
    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)
    
    preprocessing_config = config.get('preprocessing_config', {})
    dataset_config = config.get('dataset_config', {})
    
    # Print preprocessing settings
    print("\n" + "-"*60)
    print("PREPROCESSING CONFIGURATION:")
    print("-"*60)
    for key in ['feature_standardize', 'feature_negative_one_one_scaling', 
                'target_negative_one_one_scaling', 'remove_outliers', 
                'outlier_quantile', 'yeo_johnson', 'standardize', 'negative_one_one_scaling']:
        val = get_config_value(preprocessing_config, key, 'NOT SET')
        print(f"  {key}: {val}")
    
    max_n_features = get_config_value(dataset_config, 'max_number_features', 50)
    print(f"  max_number_features: {max_n_features}")
    
    # =========================================================================
    # PART 1: Generate training data samples
    # =========================================================================
    print("\n" + "="*80)
    print("PART 1: TRAINING DATA (InterventionalDataset)")
    print("="*80)
    
    try:
        samples = generate_training_sample(config_path, n_samples=5)
        print(f"\nGenerated {len(samples)} training samples")
        
        # Analyze first sample
        if len(samples) > 0:
            sample = samples[0]
            if len(sample) == 6:
                X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv = sample
            elif len(sample) == 7:
                X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, _ = sample
            elif len(sample) == 4:
                X_obs, Y_obs, X_intv, Y_intv = sample
                T_obs = None
                T_intv = None
            else:
                print(f"Unexpected sample length: {len(sample)}")
                X_obs = sample[0]
                Y_obs = sample[1] if len(sample) > 1 else None
                T_obs = None
                T_intv = None
                X_intv = sample[2] if len(sample) > 2 else None
                Y_intv = sample[3] if len(sample) > 3 else None
            
            # Convert to numpy for analysis
            if isinstance(X_obs, torch.Tensor):
                X_obs = X_obs.numpy()
            if isinstance(Y_obs, torch.Tensor):
                Y_obs = Y_obs.numpy()
            if T_obs is not None and isinstance(T_obs, torch.Tensor):
                T_obs = T_obs.numpy()
            
            # Print statistics
            stats_X_train = compute_statistics(X_obs, "Training X_obs (from InterventionalDataset)")
            stats_Y_train = compute_statistics(Y_obs, "Training Y_obs (from InterventionalDataset)")
            print_statistics(stats_X_train)
            print_statistics(stats_Y_train)
            
            if T_obs is not None:
                stats_T_train = compute_statistics(T_obs, "Training T_obs (from InterventionalDataset)")
                print_statistics(stats_T_train)
    
    except Exception as e:
        print(f"\nError generating training samples: {e}")
        import traceback
        traceback.print_exc()
        X_obs = None
        Y_obs = None
    
    # =========================================================================
    # PART 2: Load and preprocess ACTUAL real-world data (IHDP)
    # =========================================================================
    print("\n" + "="*80)
    print("PART 2: REAL-WORLD DATA PREPROCESSING (IHDP)")
    print("="*80)
    
    # Load actual real-world data from IHDP benchmark
    dataset_name = "IHDP"
    print(f"\nLoading {dataset_name} dataset...")
    X_train_raw, T_train_raw, Y_train_raw, X_test_raw, true_cate = load_real_world_data(dataset_name, realization=0)
    
    n_features_real = X_train_raw.shape[1]
    print(f"Dataset: {dataset_name}")
    print(f"X_train shape: {X_train_raw.shape}")
    print(f"X_test shape: {X_test_raw.shape}")
    print(f"Y_train shape: {Y_train_raw.shape}")
    
    print("\nRaw data before preprocessing:")
    stats_X_raw = compute_statistics(X_train_raw, f"Raw X_train ({dataset_name})")
    stats_Y_raw = compute_statistics(Y_train_raw, f"Raw Y_train ({dataset_name})")
    print_statistics(stats_X_raw)
    print_statistics(stats_Y_raw)
    
    # Apply preprocessing (same as dofm.py)
    print("\nApplying preprocessing (same as dofm.py)...")
    
    # Pad features to match model
    if n_features_real < max_n_features:
        X_train_padded = np.hstack([X_train_raw, np.zeros((X_train_raw.shape[0], max_n_features - n_features_real))])
        X_test_padded = np.hstack([X_test_raw, np.zeros((X_test_raw.shape[0], max_n_features - n_features_real))])
    else:
        X_train_padded = X_train_raw[:, :max_n_features]
        X_test_padded = X_test_raw[:, :max_n_features]
    
    X_train_proc, Y_train_proc, X_test_proc = preprocess_real_world_data(
        X_train=X_train_padded,
        Y_train=Y_train_raw,
        X_test=X_test_padded,
        preprocessing_config=preprocessing_config,
        max_n_features=max_n_features,
    )
    
    print("\nProcessed data after preprocessing:")
    stats_X_proc = compute_statistics(X_train_proc, f"Processed X_train ({dataset_name})")
    stats_Y_proc = compute_statistics(Y_train_proc, f"Processed Y_train ({dataset_name})")
    print_statistics(stats_X_proc)
    print_statistics(stats_Y_proc)
    
    # =========================================================================
    # PART 3: Compare training data vs real-world preprocessed data
    # =========================================================================
    print("\n" + "="*80)
    print("PART 3: COMPARISON")
    print("="*80)
    
    if X_obs is not None:
        comparison_X = compare_statistics(stats_X_train, stats_X_proc)
        comparison_Y = compare_statistics(stats_Y_train, stats_Y_proc)
        print_comparison(comparison_X)
        print_comparison(comparison_Y)
    
    # =========================================================================
    # PART 4: Key checks
    # =========================================================================
    print("\n" + "="*80)
    print("PART 4: KEY CHECKS")
    print("="*80)
    
    issues = []
    
    # Check Y range (should be [-1, 1] if target_negative_one_one_scaling=True)
    target_scaling = get_config_value(preprocessing_config, 'target_negative_one_one_scaling', True)
    if target_scaling:
        if Y_train_proc.min() < -1.1 or Y_train_proc.max() > 1.1:
            issues.append(f"Y range [{Y_train_proc.min():.4f}, {Y_train_proc.max():.4f}] exceeds [-1, 1]")
        else:
            print(f"✓ Target scaling: Y range [{Y_train_proc.min():.4f}, {Y_train_proc.max():.4f}] is within [-1, 1]")
    
    # Check X standardization (should have ~mean=0, ~std=1 for non-zero features if feature_standardize=True)
    feature_std = get_config_value(preprocessing_config, 'feature_standardize', True)
    if feature_std:
        X_nonzero = X_train_proc[:, :n_features_real]
        mean_check = np.abs(np.mean(X_nonzero)) < 0.5
        std_check = np.abs(np.std(X_nonzero) - 1.0) < 0.5
        if not mean_check:
            issues.append(f"X mean {np.mean(X_nonzero):.4f} is not close to 0")
        if not std_check:
            issues.append(f"X std {np.std(X_nonzero):.4f} is not close to 1")
        else:
            print(f"✓ Feature standardization: X mean={np.mean(X_nonzero):.4f}, std={np.std(X_nonzero):.4f}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(X_train_proc)) or np.any(np.isnan(Y_train_proc)):
        issues.append("NaN values detected in processed data")
    else:
        print("✓ No NaN values in processed data")
    
    if np.any(np.isinf(X_train_proc)) or np.any(np.isinf(Y_train_proc)):
        issues.append("Inf values detected in processed data")
    else:
        print("✓ No Inf values in processed data")
    
    # Check padded features are zero
    if max_n_features > n_features_real:
        padded_values = X_train_proc[:, n_features_real:]
        if not np.allclose(padded_values, 0):
            issues.append(f"Padded features are not zero: max abs value = {np.abs(padded_values).max():.6f}")
        else:
            print(f"✓ Padded features are zero")
    
    # Compare with training data if available
    if X_obs is not None:
        print("\n" + "-"*60)
        print("Comparison with training data:")
        print("-"*60)
        
        # Check if ranges are similar
        range_X_train = stats_X_train['max'] - stats_X_train['min']
        range_X_proc = stats_X_proc['max'] - stats_X_proc['min']
        range_diff = abs(range_X_train - range_X_proc) / max(range_X_train, 0.001)
        
        if range_diff > 0.5:
            issues.append(f"X range differs significantly: training={range_X_train:.4f}, real-world={range_X_proc:.4f}")
        else:
            print(f"✓ X ranges similar: training={range_X_train:.4f}, real-world={range_X_proc:.4f}")
        
        range_Y_train = stats_Y_train['max'] - stats_Y_train['min']
        range_Y_proc = stats_Y_proc['max'] - stats_Y_proc['min']
        
        if abs(range_Y_train - range_Y_proc) > 0.5:
            issues.append(f"Y range differs: training={range_Y_train:.4f}, real-world={range_Y_proc:.4f}")
        else:
            print(f"✓ Y ranges similar: training={range_Y_train:.4f}, real-world={range_Y_proc:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ All checks passed! Preprocessing appears consistent.")
    
    return issues


if __name__ == "__main__":
    # Path to the model config used for training
    config_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16768542.0/final_config.yaml"
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Trying alternative path...")
        # Try to find a config file
        experiments_dir = Path("/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments")
        config_files = list(experiments_dir.rglob("*config*.yaml"))
        if config_files:
            config_path = str(config_files[0])
            print(f"Found config: {config_path}")
        else:
            print("No config files found. Please specify a valid config path.")
            sys.exit(1)
    
    issues = run_diagnostic_test(config_path)
    sys.exit(1 if issues else 0)
