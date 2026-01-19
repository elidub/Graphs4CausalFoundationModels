"""
Comprehensive investigation of CATE benchmark datasets (IHDP, ACIC, CPS, PSID).

This script provides:
1. Detailed statistics for each dataset
2. Distribution analysis (histograms, box plots)
3. Pairwise scatterplots
4. Correlation analysis
5. Treatment/outcome relationships
6. Comparison of raw vs preprocessed data
7. Cross-dataset comparisons
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add paths
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting')
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting/src')
sys.path.insert(0, '/Users/arikreuter/Documents/PhD/CausalPriorFitting/RealCauseEval')

from src.priordata_processing.Preprocessor import Preprocessor

# Import real-world CATE datasets
from CausalPFN.benchmarks import IHDPDataset, ACIC2016Dataset
from CausalPFN.benchmarks import RealCauseLalondeCPSDataset, RealCauseLalondePSIDDataset

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory for figures
OUTPUT_DIR = Path("/Users/arikreuter/Documents/PhD/CausalPriorFitting/benchmark_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset registry
DATASETS = {
    "IHDP": IHDPDataset,
    "ACIC": ACIC2016Dataset,
    "CPS": RealCauseLalondeCPSDataset,
    "PSID": RealCauseLalondePSIDDataset,
}


def load_preprocessing_config(config_path: str) -> Tuple[dict, dict]:
    """Load preprocessing configuration from model config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('preprocessing_config', {}), config.get('dataset_config', {})


def get_config_value(config_dict: dict, key: str, default=None):
    """Extract value from config entry."""
    raw = config_dict.get(key, default)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def load_dataset(name: str, realization: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset by name."""
    dataset_class = DATASETS[name]
    dataset = dataset_class()
    cate_dataset = dataset[realization][0]
    
    return (
        cate_dataset.X_train,
        cate_dataset.t_train,
        cate_dataset.y_train,
        cate_dataset.X_test,
        cate_dataset.true_cate
    )


def compute_detailed_statistics(data: np.ndarray, name: str) -> Dict:
    """Compute comprehensive statistics."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    stats = {
        'name': name,
        'shape': data.shape,
        'n_samples': data.shape[0],
        'n_features': data.shape[1] if data.ndim > 1 else 1,
        'global': {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'skewness': float(pd.Series(data.flatten()).skew()),
            'kurtosis': float(pd.Series(data.flatten()).kurtosis()),
        },
        'nan_count': int(np.sum(np.isnan(data))),
        'inf_count': int(np.sum(np.isinf(data))),
        'zero_count': int(np.sum(data == 0)),
        'zero_fraction': float(np.sum(data == 0) / data.size),
        'unique_values': int(len(np.unique(data))),
        'is_binary': bool(set(np.unique(data)).issubset({0, 1, 0.0, 1.0})),
        'quantiles': {
            '1%': float(np.percentile(data, 1)),
            '5%': float(np.percentile(data, 5)),
            '10%': float(np.percentile(data, 10)),
            '25%': float(np.percentile(data, 25)),
            '50%': float(np.percentile(data, 50)),
            '75%': float(np.percentile(data, 75)),
            '90%': float(np.percentile(data, 90)),
            '95%': float(np.percentile(data, 95)),
            '99%': float(np.percentile(data, 99)),
        },
    }
    
    # Per-feature statistics (for multi-feature data)
    if data.shape[1] > 1:
        stats['per_feature'] = {
            'means': np.mean(data, axis=0).tolist(),
            'stds': np.std(data, axis=0).tolist(),
            'mins': np.min(data, axis=0).tolist(),
            'maxs': np.max(data, axis=0).tolist(),
            'zero_fractions': (np.sum(data == 0, axis=0) / data.shape[0]).tolist(),
            'unique_counts': [len(np.unique(data[:, i])) for i in range(data.shape[1])],
            'binary_features': [bool(set(np.unique(data[:, i])).issubset({0, 1, 0.0, 1.0})) 
                               for i in range(data.shape[1])],
        }
    
    return stats


def print_dataset_summary(name: str, X_train: np.ndarray, T_train: np.ndarray, 
                          Y_train: np.ndarray, X_test: np.ndarray, true_cate: np.ndarray):
    """Print a summary of dataset characteristics."""
    print(f"\n{'='*80}")
    print(f"DATASET: {name}")
    print(f"{'='*80}")
    
    print(f"\n--- Shapes ---")
    print(f"  X_train: {X_train.shape}")
    print(f"  T_train: {T_train.shape}")
    print(f"  Y_train: {Y_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  true_cate: {true_cate.shape}")
    
    # Feature analysis
    print(f"\n--- Feature Analysis ---")
    n_features = X_train.shape[1]
    binary_features = sum(1 for i in range(n_features) 
                         if set(np.unique(X_train[:, i])).issubset({0, 1, 0.0, 1.0}))
    continuous_features = n_features - binary_features
    print(f"  Total features: {n_features}")
    print(f"  Binary features: {binary_features}")
    print(f"  Continuous features: {continuous_features}")
    
    # Identify which features are binary
    binary_idx = [i for i in range(n_features) 
                  if set(np.unique(X_train[:, i])).issubset({0, 1, 0.0, 1.0})]
    continuous_idx = [i for i in range(n_features) if i not in binary_idx]
    print(f"  Binary feature indices: {binary_idx[:10]}{'...' if len(binary_idx) > 10 else ''}")
    print(f"  Continuous feature indices: {continuous_idx[:10]}{'...' if len(continuous_idx) > 10 else ''}")
    
    # Treatment analysis
    print(f"\n--- Treatment Analysis ---")
    print(f"  Unique T values: {np.unique(T_train)}")
    print(f"  T mean: {T_train.mean():.4f}")
    print(f"  T=1 fraction: {(T_train == 1).mean():.4f}")
    print(f"  T=0 fraction: {(T_train == 0).mean():.4f}")
    
    # Outcome analysis
    print(f"\n--- Outcome Analysis ---")
    print(f"  Y range: [{Y_train.min():.4f}, {Y_train.max():.4f}]")
    print(f"  Y mean: {Y_train.mean():.4f}")
    print(f"  Y std: {Y_train.std():.4f}")
    print(f"  Y median: {np.median(Y_train):.4f}")
    
    # CATE analysis
    print(f"\n--- CATE Analysis ---")
    print(f"  True CATE range: [{true_cate.min():.4f}, {true_cate.max():.4f}]")
    print(f"  True CATE mean (ATE): {true_cate.mean():.4f}")
    print(f"  True CATE std: {true_cate.std():.4f}")
    
    # X statistics
    print(f"\n--- X Statistics ---")
    print(f"  X mean: {X_train.mean():.4f}")
    print(f"  X std: {X_train.std():.4f}")
    print(f"  X range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  Zero fraction: {(X_train == 0).mean():.4f}")
    
    # Check for pre-standardization
    print(f"\n--- Pre-standardization Check ---")
    per_feature_means = np.mean(X_train, axis=0)
    per_feature_stds = np.std(X_train, axis=0)
    
    # Only check continuous features
    if continuous_idx:
        cont_means = per_feature_means[continuous_idx]
        cont_stds = per_feature_stds[continuous_idx]
        print(f"  Continuous features mean of means: {np.mean(cont_means):.4f}")
        print(f"  Continuous features mean of stds: {np.mean(cont_stds):.4f}")
        is_standardized = np.abs(np.mean(cont_means)) < 0.5 and np.abs(np.mean(cont_stds) - 1.0) < 0.5
        print(f"  Appears pre-standardized: {is_standardized}")
    
    return binary_idx, continuous_idx


def plot_feature_distributions(name: str, X_train: np.ndarray, 
                               binary_idx: List[int], continuous_idx: List[int]):
    """Plot distributions of features."""
    n_cont = min(len(continuous_idx), 6)
    n_bin = min(len(binary_idx), 6)
    
    if n_cont > 0:
        fig, axes = plt.subplots(2, n_cont, figsize=(4*n_cont, 8))
        if n_cont == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feat_idx in enumerate(continuous_idx[:n_cont]):
            # Histogram
            axes[0, i].hist(X_train[:, feat_idx], bins=50, edgecolor='black', alpha=0.7)
            axes[0, i].set_title(f'Feature {feat_idx} (Continuous)')
            axes[0, i].set_xlabel('Value')
            axes[0, i].set_ylabel('Count')
            
            # Box plot
            axes[1, i].boxplot(X_train[:, feat_idx])
            axes[1, i].set_title(f'Feature {feat_idx} Box Plot')
        
        plt.suptitle(f'{name}: Continuous Feature Distributions', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{name}_continuous_features.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    if n_bin > 0:
        fig, axes = plt.subplots(1, n_bin, figsize=(3*n_bin, 4))
        if n_bin == 1:
            axes = [axes]
        
        for i, feat_idx in enumerate(binary_idx[:n_bin]):
            unique, counts = np.unique(X_train[:, feat_idx], return_counts=True)
            axes[i].bar(unique, counts, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'Feature {feat_idx} (Binary)')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Count')
        
        plt.suptitle(f'{name}: Binary Feature Distributions', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'{name}_binary_features.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_treatment_outcome_relationship(name: str, T_train: np.ndarray, Y_train: np.ndarray):
    """Plot treatment-outcome relationships."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Y distribution by treatment
    Y_t0 = Y_train[T_train.flatten() == 0]
    Y_t1 = Y_train[T_train.flatten() == 1]
    
    axes[0].hist(Y_t0, bins=30, alpha=0.6, label=f'T=0 (n={len(Y_t0)})', edgecolor='black')
    axes[0].hist(Y_t1, bins=30, alpha=0.6, label=f'T=1 (n={len(Y_t1)})', edgecolor='black')
    axes[0].set_xlabel('Outcome Y')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Outcome Distribution by Treatment')
    axes[0].legend()
    
    # Box plot by treatment
    axes[1].boxplot([Y_t0.flatten(), Y_t1.flatten()], labels=['T=0', 'T=1'])
    axes[1].set_ylabel('Outcome Y')
    axes[1].set_title('Outcome by Treatment Group')
    
    # Violin plot
    data_for_violin = pd.DataFrame({
        'Treatment': ['T=0'] * len(Y_t0) + ['T=1'] * len(Y_t1),
        'Outcome': np.concatenate([Y_t0.flatten(), Y_t1.flatten()])
    })
    sns.violinplot(data=data_for_violin, x='Treatment', y='Outcome', ax=axes[2])
    axes[2].set_title('Outcome Violin Plot by Treatment')
    
    plt.suptitle(f'{name}: Treatment-Outcome Relationship', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{name}_treatment_outcome.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print stats
    print(f"\n--- Treatment-Outcome Stats for {name} ---")
    print(f"  Y|T=0: mean={Y_t0.mean():.4f}, std={Y_t0.std():.4f}")
    print(f"  Y|T=1: mean={Y_t1.mean():.4f}, std={Y_t1.std():.4f}")
    print(f"  Observed ATE (naive): {Y_t1.mean() - Y_t0.mean():.4f}")


def plot_pairwise_scatterplot(name: str, X_train: np.ndarray, T_train: np.ndarray, 
                               Y_train: np.ndarray, continuous_idx: List[int], max_features: int = 5):
    """Create pairwise scatterplot for continuous features + T + Y."""
    # Select features for pairplot
    n_features = min(len(continuous_idx), max_features)
    if n_features < 2:
        print(f"  Not enough continuous features for pairplot in {name}")
        return
    
    selected_features = continuous_idx[:n_features]
    
    # Create DataFrame
    df = pd.DataFrame()
    for i, feat_idx in enumerate(selected_features):
        df[f'X{feat_idx}'] = X_train[:, feat_idx]
    df['Treatment'] = T_train.flatten().astype(str)
    df['Y'] = Y_train.flatten()
    
    # Pairplot with treatment coloring
    g = sns.pairplot(df, hue='Treatment', diag_kind='kde', 
                     plot_kws={'alpha': 0.5, 's': 20},
                     height=2.5, aspect=1)
    g.fig.suptitle(f'{name}: Pairwise Scatterplot (Continuous Features)', y=1.02)
    plt.savefig(OUTPUT_DIR / f'{name}_pairplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap
    X_subset = X_train[:, selected_features]
    corr_data = np.column_stack([X_subset, T_train.flatten(), Y_train.flatten()])
    corr_cols = [f'X{i}' for i in selected_features] + ['T', 'Y']
    corr_df = pd.DataFrame(corr_data, columns=corr_cols)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', ax=ax, square=True)
    ax.set_title(f'{name}: Correlation Matrix')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{name}_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_cate_distribution(name: str, true_cate: np.ndarray):
    """Plot CATE distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    axes[0].hist(true_cate, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(true_cate.mean(), color='red', linestyle='--', 
                    label=f'Mean (ATE)={true_cate.mean():.3f}')
    axes[0].set_xlabel('CATE')
    axes[0].set_ylabel('Count')
    axes[0].set_title('True CATE Distribution')
    axes[0].legend()
    
    # KDE
    sns.kdeplot(true_cate.flatten(), ax=axes[1], fill=True)
    axes[1].axvline(true_cate.mean(), color='red', linestyle='--', 
                    label=f'Mean={true_cate.mean():.3f}')
    axes[1].set_xlabel('CATE')
    axes[1].set_title('True CATE Density')
    axes[1].legend()
    
    plt.suptitle(f'{name}: True CATE Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{name}_cate.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_preprocessing_comparison(name: str, X_train_raw: np.ndarray, Y_train_raw: np.ndarray,
                                   X_train_proc: np.ndarray, Y_train_proc: np.ndarray,
                                   continuous_idx: List[int]):
    """Compare raw vs preprocessed data."""
    n_features = min(len(continuous_idx), 4)
    if n_features == 0:
        return
    
    fig, axes = plt.subplots(3, n_features, figsize=(4*n_features, 12))
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    
    for i, feat_idx in enumerate(continuous_idx[:n_features]):
        # Raw distribution
        axes[0, i].hist(X_train_raw[:, feat_idx], bins=30, alpha=0.7, edgecolor='black')
        axes[0, i].set_title(f'Raw X{feat_idx}')
        axes[0, i].set_xlabel('Value')
        
        # Processed distribution
        axes[1, i].hist(X_train_proc[:, feat_idx], bins=30, alpha=0.7, edgecolor='black', color='orange')
        axes[1, i].set_title(f'Processed X{feat_idx}')
        axes[1, i].set_xlabel('Value')
        
        # Q-Q plot comparing raw vs processed
        raw_sorted = np.sort(X_train_raw[:, feat_idx])
        proc_sorted = np.sort(X_train_proc[:, feat_idx])
        axes[2, i].scatter(raw_sorted, proc_sorted, alpha=0.5, s=10)
        axes[2, i].plot([raw_sorted.min(), raw_sorted.max()], 
                       [raw_sorted.min(), raw_sorted.max()], 'r--')
        axes[2, i].set_xlabel('Raw Values')
        axes[2, i].set_ylabel('Processed Values')
        axes[2, i].set_title(f'X{feat_idx} Transformation')
    
    plt.suptitle(f'{name}: Raw vs Preprocessed Features', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{name}_preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Y comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(Y_train_raw, bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_title('Raw Y')
    axes[0].set_xlabel('Value')
    
    axes[1].hist(Y_train_proc, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_title('Processed Y')
    axes[1].set_xlabel('Value')
    
    raw_sorted = np.sort(Y_train_raw.flatten())
    proc_sorted = np.sort(Y_train_proc.flatten())
    axes[2].scatter(raw_sorted, proc_sorted, alpha=0.5, s=10)
    axes[2].set_xlabel('Raw Y')
    axes[2].set_ylabel('Processed Y')
    axes[2].set_title('Y Transformation')
    
    plt.suptitle(f'{name}: Raw vs Preprocessed Outcome', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'{name}_Y_preprocessing.png', dpi=150, bbox_inches='tight')
    plt.close()


def preprocess_data(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray,
                    preprocessing_config: dict, max_n_features: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply preprocessing to data."""
    feature_standardize = get_config_value(preprocessing_config, 'feature_standardize', True)
    feature_negative_one_one_scaling = get_config_value(preprocessing_config, 'feature_negative_one_one_scaling', False)
    target_negative_one_one_scaling = get_config_value(preprocessing_config, 'target_negative_one_one_scaling', True)
    remove_outliers = get_config_value(preprocessing_config, 'remove_outliers', True)
    outlier_quantile = get_config_value(preprocessing_config, 'outlier_quantile', 0.99)
    yeo_johnson = get_config_value(preprocessing_config, 'yeo_johnson', False)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features = X_train.shape[1]
    
    Y_train = Y_train.flatten()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(0)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(0)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(0)
    
    X_combined = torch.cat([X_train_t, X_test_t], dim=1)
    Y_dummy_test = torch.zeros(1, n_test, dtype=torch.float32)
    Y_combined = torch.cat([Y_train_t, Y_dummy_test], dim=1)
    
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
    
    X_train_proc, X_test_proc, Y_train_proc, _ = result
    
    return (X_train_proc.squeeze(0).numpy(), 
            Y_train_proc.squeeze(0).numpy(), 
            X_test_proc.squeeze(0).numpy())


def create_cross_dataset_comparison(all_stats: Dict):
    """Create comparison plots across all datasets."""
    datasets = list(all_stats.keys())
    
    # Prepare data for comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Sample sizes
    train_sizes = [all_stats[d]['n_train'] for d in datasets]
    test_sizes = [all_stats[d]['n_test'] for d in datasets]
    x = np.arange(len(datasets))
    width = 0.35
    axes[0, 0].bar(x - width/2, train_sizes, width, label='Train')
    axes[0, 0].bar(x + width/2, test_sizes, width, label='Test')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_title('Sample Sizes')
    axes[0, 0].legend()
    
    # 2. Feature counts
    n_features = [all_stats[d]['n_features'] for d in datasets]
    n_binary = [all_stats[d]['n_binary_features'] for d in datasets]
    n_cont = [all_stats[d]['n_continuous_features'] for d in datasets]
    axes[0, 1].bar(x - width/2, n_binary, width, label='Binary')
    axes[0, 1].bar(x + width/2, n_cont, width, label='Continuous')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(datasets)
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].set_title('Feature Types')
    axes[0, 1].legend()
    
    # 3. Treatment proportions
    t1_fracs = [all_stats[d]['t1_fraction'] for d in datasets]
    axes[0, 2].bar(datasets, t1_fracs)
    axes[0, 2].axhline(0.5, color='red', linestyle='--', label='Balanced (0.5)')
    axes[0, 2].set_ylabel('Fraction T=1')
    axes[0, 2].set_title('Treatment Proportions')
    axes[0, 2].legend()
    
    # 4. Outcome ranges
    y_mins = [all_stats[d]['y_min'] for d in datasets]
    y_maxs = [all_stats[d]['y_max'] for d in datasets]
    y_means = [all_stats[d]['y_mean'] for d in datasets]
    for i, d in enumerate(datasets):
        axes[1, 0].errorbar(i, y_means[i], yerr=[[y_means[i]-y_mins[i]], [y_maxs[i]-y_means[i]]], 
                           fmt='o', capsize=5, capthick=2)
    axes[1, 0].set_xticks(range(len(datasets)))
    axes[1, 0].set_xticklabels(datasets)
    axes[1, 0].set_ylabel('Outcome Y')
    axes[1, 0].set_title('Outcome Ranges')
    
    # 5. ATE comparison
    ates = [all_stats[d]['ate'] for d in datasets]
    axes[1, 1].bar(datasets, ates)
    axes[1, 1].set_ylabel('True ATE')
    axes[1, 1].set_title('Average Treatment Effects')
    
    # 6. CATE heterogeneity (std of CATE)
    cate_stds = [all_stats[d]['cate_std'] for d in datasets]
    axes[1, 2].bar(datasets, cate_stds)
    axes[1, 2].set_ylabel('CATE Std')
    axes[1, 2].set_title('Treatment Effect Heterogeneity')
    
    plt.suptitle('Cross-Dataset Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cross_dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main investigation function."""
    print("="*80)
    print("COMPREHENSIVE BENCHMARK DATASET INVESTIGATION")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load preprocessing config
    config_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16768542.0/final_config.yaml"
    
    if os.path.exists(config_path):
        preprocessing_config, dataset_config = load_preprocessing_config(config_path)
        max_n_features = get_config_value(dataset_config, 'max_number_features', 51)
        print(f"\nLoaded preprocessing config from: {config_path}")
        print(f"Max features: {max_n_features}")
    else:
        print(f"\nWarning: Config not found at {config_path}, using defaults")
        preprocessing_config = {
            'feature_standardize': True,
            'feature_negative_one_one_scaling': False,
            'target_negative_one_one_scaling': True,
            'remove_outliers': True,
            'outlier_quantile': 0.99,
            'yeo_johnson': False,
        }
        max_n_features = 51
    
    # Store stats for cross-dataset comparison
    all_stats = {}
    
    # Analyze each dataset
    for name in DATASETS.keys():
        print(f"\n\n{'#'*80}")
        print(f"# ANALYZING: {name}")
        print(f"{'#'*80}")
        
        try:
            # Load data
            X_train, T_train, Y_train, X_test, true_cate = load_dataset(name)
            
            # Print summary
            binary_idx, continuous_idx = print_dataset_summary(
                name, X_train, T_train, Y_train, X_test, true_cate
            )
            
            # Store stats for comparison
            all_stats[name] = {
                'n_train': X_train.shape[0],
                'n_test': X_test.shape[0],
                'n_features': X_train.shape[1],
                'n_binary_features': len(binary_idx),
                'n_continuous_features': len(continuous_idx),
                't1_fraction': (T_train == 1).mean(),
                'y_min': Y_train.min(),
                'y_max': Y_train.max(),
                'y_mean': Y_train.mean(),
                'ate': true_cate.mean(),
                'cate_std': true_cate.std(),
            }
            
            # Plot feature distributions
            print(f"\n  Plotting feature distributions...")
            plot_feature_distributions(name, X_train, binary_idx, continuous_idx)
            
            # Plot treatment-outcome relationship
            print(f"  Plotting treatment-outcome relationship...")
            plot_treatment_outcome_relationship(name, T_train, Y_train)
            
            # Plot pairwise scatterplot
            print(f"  Plotting pairwise scatterplot...")
            plot_pairwise_scatterplot(name, X_train, T_train, Y_train, continuous_idx)
            
            # Plot CATE distribution
            print(f"  Plotting CATE distribution...")
            plot_cate_distribution(name, true_cate)
            
            # Apply preprocessing and compare
            print(f"  Applying preprocessing and comparing...")
            n_features_orig = X_train.shape[1]
            if n_features_orig < max_n_features:
                X_train_padded = np.hstack([X_train, np.zeros((X_train.shape[0], max_n_features - n_features_orig))])
                X_test_padded = np.hstack([X_test, np.zeros((X_test.shape[0], max_n_features - n_features_orig))])
            else:
                X_train_padded = X_train[:, :max_n_features]
                X_test_padded = X_test[:, :max_n_features]
            
            X_train_proc, Y_train_proc, X_test_proc = preprocess_data(
                X_train_padded, Y_train, X_test_padded, preprocessing_config, max_n_features
            )
            
            # Plot preprocessing comparison (only for original features)
            plot_preprocessing_comparison(name, X_train, Y_train, 
                                         X_train_proc[:, :n_features_orig], Y_train_proc,
                                         continuous_idx)
            
            print(f"\n  Preprocessed X range: [{X_train_proc[:, :n_features_orig].min():.4f}, {X_train_proc[:, :n_features_orig].max():.4f}]")
            print(f"  Preprocessed Y range: [{Y_train_proc.min():.4f}, {Y_train_proc.max():.4f}]")
            
            print(f"\n  ✓ {name} analysis complete!")
            
        except Exception as e:
            print(f"\n  ✗ Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create cross-dataset comparison
    print(f"\n\n{'#'*80}")
    print("# CROSS-DATASET COMPARISON")
    print(f"{'#'*80}")
    
    create_cross_dataset_comparison(all_stats)
    
    # Print summary table
    print("\n\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    summary_df = pd.DataFrame(all_stats).T
    print(summary_df.to_string())
    
    # Save summary to CSV
    summary_df.to_csv(OUTPUT_DIR / 'dataset_summary.csv')
    print(f"\n\nSummary saved to: {OUTPUT_DIR / 'dataset_summary.csv'}")
    print(f"All figures saved to: {OUTPUT_DIR}")
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
