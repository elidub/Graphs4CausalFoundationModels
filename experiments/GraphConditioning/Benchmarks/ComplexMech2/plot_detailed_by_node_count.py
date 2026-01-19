#!/usr/bin/env python3
"""
Generate detailed plots for ComplexMech2 benchmark results organized by node count.

Similar to generate_plots.py for LinGaus IDK, this creates multi-panel plots
showing performance differences across different graph sizes (node counts).
"""

import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")

# Define colors for models
MODEL_COLORS = {
    'baseline': '#cc4778',  # Pink/red for baseline (hide_all)
    'model': '#21918c',     # Teal for comparison model
}


def bootstrap_median_ci(data, n_bootstrap=10000, confidence=0.95):
    """Calculate bootstrap confidence interval for the median.
    
    Args:
        data: array of values
        n_bootstrap: number of bootstrap samples
        confidence: confidence level (default 0.95 for 95% CI)
        
    Returns:
        tuple: (median, lower_bound, upper_bound) of confidence interval
    """
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    
    medians = []
    n = len(data)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        medians.append(np.median(sample))
    
    # Calculate percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    median = np.median(data)
    lower_bound = np.percentile(medians, lower_percentile)
    upper_bound = np.percentile(medians, upper_percentile)
    
    return (median, lower_bound, upper_bound)


def bootstrap_mean_ci(data, n_bootstrap=10000, confidence=0.95):
    """Calculate bootstrap confidence interval for the mean.
    
    Args:
        data: array of values
        n_bootstrap: number of bootstrap samples
        confidence: confidence level (default 0.95 for 95% CI)
        
    Returns:
        tuple: (mean, lower_bound, upper_bound) of confidence interval
    """
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    
    means = []
    n = len(data)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    
    # Calculate percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean = np.mean(data)
    lower_bound = np.percentile(means, lower_percentile)
    upper_bound = np.percentile(means, upper_percentile)
    
    return (mean, lower_bound, upper_bound)


def load_sample_sizes(data_cache_dir, cache_file=None):
    """Load training sample sizes for each dataset index."""
    samples_dir = Path(data_cache_dir) / "samples"
    
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    # Try to load from cache first
    if cache_file and Path(cache_file).exists():
        print(f"Loading sample sizes from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            sample_sizes = json.load(f)
        # Convert string keys back to integers
        sample_sizes = {int(k): v for k, v in sample_sizes.items()}
        print(f"Loaded sample sizes for {len(sample_sizes)} datasets from cache")
        return sample_sizes
    
    raise FileNotFoundError(f"Cache file not found. Please run plot_performance_vs_sample_size.py first to generate cache.")


def count_active_features(data_cache_dir, max_samples=None):
    """Count the number of active (non-zero) features in each dataset by examining the data.
    
    Returns:
        dict: Mapping from sample index to number of active features (excluding T and Y)
    """
    samples_dir = Path(data_cache_dir) / "samples"
    
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    print("Counting active features in each dataset...")
    active_features = {}
    
    pickle_files = sorted(samples_dir.glob("*.pkl"))
    if max_samples:
        pickle_files = pickle_files[:max_samples]
    
    for pkl_file in tqdm(pickle_files, desc="Analyzing samples", unit="file"):
        # Extract index from filename
        try:
            stem = pkl_file.stem
            if stem.startswith("sample_"):
                index = int(stem.replace("sample_", ""))
            else:
                index = int(stem)
        except ValueError:
            continue
        
        # Load pickle file
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
        
        # Get X_obs to check for active features
        X_obs = data.get('X_obs')
        if X_obs is None:
            continue
        
        # Handle batch dimension if present
        if X_obs.ndim == 3 and X_obs.shape[0] == 1:
            X_obs = X_obs.squeeze(0)
        
        # Count non-zero features (columns with any non-zero values)
        # Exclude the last column which is the treatment indicator T
        feature_cols = X_obs[:, :-1]  # All columns except T
        
        # A feature is active if it has non-zero variance or any non-zero value
        active_mask = np.any(np.abs(feature_cols) > 1e-10, axis=0)
        n_active = np.sum(active_mask)
        
        # Total nodes = active features + T + Y (so +2)
        active_features[index] = int(n_active)
    
    print(f"Counted active features for {len(active_features)} datasets")
    if active_features:
        print(f"Active feature range: {min(active_features.values())} to {max(active_features.values())}")
    
    return active_features


def load_model_results(results_dir):
    """Load model evaluation results."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Look for detailed_results.json
    json_files = list(results_dir.glob("**/detailed_results.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No detailed_results.json found in {results_dir}")
    
    # Use the most recent results
    json_file = sorted(json_files)[-1]
    print(f"Loading results from: {json_file}")
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Convert to dict indexed by dataset index
    results_by_index = {}
    for result in results:
        if 'sample_idx' in result:
            idx = result['sample_idx']
            results_by_index[idx] = {
                'mse': result.get('mse', np.nan),
                'r2': result.get('r2', np.nan),
                'nll': result.get('nll', np.nan),
                'n_features': result.get('n_features', np.nan),
                'n_obs': result.get('n_obs', np.nan),
                'n_intv': result.get('n_intv', np.nan),
            }
        elif 'index' in result:
            idx = result['index']
            results_by_index[idx] = {
                'mse': result.get('mse', np.nan),
                'r2': result.get('r2', np.nan),
                'nll': result.get('nll', np.nan),
                'n_features': result.get('n_features', np.nan),
                'n_obs': result.get('n_obs', np.nan),
                'n_intv': result.get('n_intv', np.nan),
            }
    
    print(f"Loaded results for {len(results_by_index)} datasets")
    
    return results_by_index


def create_dataframe_by_node_count(sample_sizes, baseline_results, model_results,
                                   active_features_dict,
                                   node_count_bins=None):
    """Create dataframe with performance differences organized by node count bins.
    
    Args:
        sample_sizes: dict mapping index to training sample size
        baseline_results: dict mapping index to baseline performance metrics
        model_results: dict mapping index to model performance metrics
        active_features_dict: dict mapping index to number of active features
        node_count_bins: list of (min, max, label) tuples defining node count bins
        
    Returns:
        pandas DataFrame with differences organized by node count bins
    """
    if node_count_bins is None:
        # Default bins
        node_count_bins = [
            (0, 10, "2-10 nodes"),
            (10, 20, "10-20 nodes"),
            (20, 35, "20-35 nodes"),
            (35, 60, "35-60 nodes"),
        ]
    
    data_rows = []
    
    for idx in sorted(sample_sizes.keys()):
        if idx in baseline_results and idx in model_results and idx in active_features_dict:
            sample_size = sample_sizes[idx]
            baseline = baseline_results[idx]
            model = model_results[idx]
            
            # Get actual number of nodes: active features + T + Y
            n_active_features = active_features_dict[idx]
            n_nodes = n_active_features + 2  # +T and +Y
            
            # Skip if any metric is NaN
            if (np.isnan(baseline['mse']) or np.isnan(model['mse']) or
                np.isnan(baseline['r2']) or np.isnan(model['r2']) or
                np.isnan(baseline['nll']) or np.isnan(model['nll'])):
                continue
            
            # Determine node count bin
            node_bin_label = None
            for min_nodes, max_nodes, label in node_count_bins:
                if min_nodes <= n_nodes < max_nodes:
                    node_bin_label = label
                    break
            
            if node_bin_label is None:
                continue
            
            row = {
                'sample_idx': idx,
                'sample_size': sample_size,
                'n_nodes': n_nodes,
                'n_active_features': n_active_features,
                'node_bin': node_bin_label,
                'mse_baseline': baseline['mse'],
                'mse_model': model['mse'],
                'mse_diff': model['mse'] - baseline['mse'],
                'r2_baseline': baseline['r2'],
                'r2_model': model['r2'],
                'r2_diff': model['r2'] - baseline['r2'],
                'nll_baseline': baseline['nll'],
                'nll_model': model['nll'],
                'nll_diff': model['nll'] - baseline['nll'],
            }
            
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    print(f"\nDataFrame created with {len(df)} samples")
    if len(df) > 0:
        print(f"Node bins: {sorted(df['node_bin'].unique())}")
        print(f"Sample size range: {df['sample_size'].min()} to {df['sample_size'].max()}")
        print(f"Node count range: {df['n_nodes'].min()} to {df['n_nodes'].max()}")
        print(f"Active features range: {df['n_active_features'].min()} to {df['n_active_features'].max()}")
    
    return df


def plot_difference_by_node_count_and_sample_size(df, metrics=['mse', 'r2', 'nll'],
                                                   sample_size_bins=None,
                                                   n_sample_bins=8,
                                                   output_path=None,
                                                   baseline_name="Baseline",
                                                   model_name="Model",
                                                   statistic='median'):
    """Create multi-panel box plot showing differences by node count and sample size bins.
    
    Similar to generate_plots.py style with node counts as columns and metrics as rows.
    Each box shows the distribution of differences for that combination.
    
    Args:
        n_sample_bins: Number of sample size bins to create (default: 8)
        statistic: 'median' or 'mean' - which statistic to compute and display
    """
    if sample_size_bins is None:
        # Create quantile-based bins for sample sizes
        quantiles = np.linspace(0, 1, n_sample_bins + 1)
        bin_edges = df['sample_size'].quantile(quantiles).values
        sample_size_bins = []
        for i in range(len(bin_edges) - 1):
            min_size = int(bin_edges[i])
            max_size = int(bin_edges[i + 1])
            sample_size_bins.append((min_size, max_size, f"{min_size}-{max_size}"))
    
    # Get unique node bins
    node_bins = sorted(df['node_bin'].unique())
    n_node_bins = len(node_bins)
    
    metric_titles = {
        'mse': 'MSE Difference',
        'r2': 'R² Difference',
        'nll': 'NLL Difference'
    }
    
    # Create figure: metrics as rows, node bins as columns
    fig, axes = plt.subplots(len(metrics), n_node_bins,
                            figsize=(5.5 * n_node_bins, 4.5 * len(metrics)))
    
    # Handle single column case
    if n_node_bins == 1:
        axes = axes.reshape(-1, 1)
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)
    
    for metric_idx, metric in enumerate(metrics):
        metric_col = f'{metric}_diff'
        
        for node_idx, node_bin in enumerate(node_bins):
            ax = axes[metric_idx, node_idx]
            
            # Filter data for this node bin
            node_data = df[df['node_bin'] == node_bin].copy()
            
            if len(node_data) == 0:
                ax.text(0.5, 0.5, 'No data',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14)
                if metric_idx == 0:
                    ax.set_title(node_bin, fontsize=18, fontweight='bold')
                continue
            
            # Bin by sample size
            box_data = []
            positions = []
            colors = []
            labels = []
            medians_list = []
            ci_lowers = []
            ci_uppers = []
            
            for bin_idx, (min_size, max_size, label) in enumerate(sample_size_bins):
                bin_mask = (node_data['sample_size'] >= min_size) & (node_data['sample_size'] <= max_size)
                bin_values = node_data.loc[bin_mask, metric_col].values
                
                if len(bin_values) > 0:
                    # Compute bootstrap CI based on statistic
                    if statistic == 'median':
                        stat_value, ci_lower, ci_upper = bootstrap_median_ci(bin_values, n_bootstrap=10000, confidence=0.95)
                    else:  # mean
                        stat_value, ci_lower, ci_upper = bootstrap_mean_ci(bin_values, n_bootstrap=10000, confidence=0.95)
                    
                    # Create synthetic distribution for boxplot
                    if not np.isnan(ci_lower) and not np.isnan(ci_upper) and ci_lower != ci_upper:
                        synthetic_data = [
                            ci_lower,
                            stat_value - (stat_value - ci_lower) * 0.5,
                            stat_value,
                            stat_value + (ci_upper - stat_value) * 0.5,
                            ci_upper
                        ]
                    else:
                        synthetic_data = [stat_value] * 5
                    
                    box_data.append(synthetic_data)
                    positions.append(bin_idx)
                    labels.append(label)
                    medians_list.append(stat_value)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)
                    
                    # Color based on whether model is better
                    if metric in ['mse', 'nll']:
                        # Lower is better
                        color = '#35b779' if stat_value < 0 else '#cc4778'
                    else:  # r2
                        # Higher is better
                        color = '#35b779' if stat_value > 0 else '#cc4778'
                    colors.append(color)
            
            if box_data and positions:
                # Create box plot
                bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                               patch_artist=True, showmeans=False,
                               medianprops=dict(color='red', linewidth=2),
                               boxprops=dict(linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
                
                # Apply colors
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Set x-axis
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, fontsize=12, ha='right')
                if metric_idx == len(metrics) - 1:
                    ax.set_xlabel('Sample Size Range', fontsize=14, fontweight='bold')
                
                # Add value labels above boxes
                for i, pos in enumerate(positions):
                    median_val = medians_list[i]
                    # Get the upper whisker position from boxplot
                    upper_whisker = bp['whiskers'][i*2 + 1].get_ydata()[1]
                    ax.text(pos, upper_whisker, f'{median_val:.4f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Add horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Set titles and labels
            if metric_idx == 0:
                ax.set_title(node_bin, fontsize=18, fontweight='bold')
            
            if node_idx == 0:
                ax.set_ylabel(metric_titles.get(metric, metric.upper()),
                             fontsize=16, fontweight='bold')
            
            # Set tick label sizes
            ax.tick_params(axis='y', labelsize=14)
            
            # Grid and spines
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Overall title
    stat_label = statistic.capitalize()
    fig.suptitle(f'Performance Differences ({stat_label}): {model_name} vs {baseline_name}\n(Organized by Node Count and Sample Size, Whiskers = 95% Bootstrap CI)',
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_difference_by_node_count_aggregated(df, metrics=['mse', 'r2', 'nll'],
                                             output_path=None,
                                             baseline_name="Baseline",
                                             model_name="Model",
                                             statistic='median'):
    """Create single-row box plot showing differences aggregated across all sample sizes.
    
    Shows one box per node count bin, aggregating all sample sizes together.
    
    Args:
        statistic: 'median' or 'mean' - which statistic to compute and display
    """
    # Get unique node bins
    node_bins = sorted(df['node_bin'].unique())
    
    metric_titles = {
        'mse': 'MSE Difference',
        'r2': 'R² Difference',
        'nll': 'NLL Difference'
    }
    
    # Create figure: 1 row, metrics as columns
    fig, axes = plt.subplots(1, len(metrics),
                            figsize=(5.5 * len(metrics), 5))
    
    # Handle single metric case
    if len(metrics) == 1:
        axes = [axes]
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        metric_col = f'{metric}_diff'
        
        box_data = []
        positions = []
        colors = []
        labels = []
        medians_list = []
        ci_lowers = []
        ci_uppers = []
        
        for node_idx, node_bin in enumerate(node_bins):
            # Filter data for this node bin (all sample sizes)
            node_data = df[df['node_bin'] == node_bin].copy()
            
            if len(node_data) > 0:
                bin_values = node_data[metric_col].values
                
                # Compute bootstrap CI based on statistic
                if statistic == 'median':
                    stat_value, ci_lower, ci_upper = bootstrap_median_ci(bin_values, n_bootstrap=10000, confidence=0.95)
                else:  # mean
                    stat_value, ci_lower, ci_upper = bootstrap_mean_ci(bin_values, n_bootstrap=10000, confidence=0.95)
                
                # Create synthetic distribution for boxplot
                if not np.isnan(ci_lower) and not np.isnan(ci_upper) and ci_lower != ci_upper:
                    synthetic_data = [
                        ci_lower,
                        stat_value - (stat_value - ci_lower) * 0.5,
                        stat_value,
                        stat_value + (ci_upper - stat_value) * 0.5,
                        ci_upper
                    ]
                else:
                    synthetic_data = [stat_value] * 5
                
                box_data.append(synthetic_data)
                positions.append(node_idx)
                labels.append(node_bin)
                medians_list.append(stat_value)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
                
                # Color based on whether model is better
                if metric in ['mse', 'nll']:
                    # Lower is better
                    color = '#35b779' if stat_value < 0 else '#cc4778'
                else:  # r2
                    # Higher is better
                    color = '#35b779' if stat_value > 0 else '#cc4778'
                colors.append(color)
        
        if box_data and positions:
            # Create box plot
            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                           patch_artist=True, showmeans=False,
                           medianprops=dict(color='red', linewidth=2),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            # Apply colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Set x-axis
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=0, fontsize=16)
            ax.set_xlabel('Node Count Range', fontsize=18, fontweight='bold')
            
            # Add value labels above boxes
            for i, pos in enumerate(positions):
                median_val = medians_list[i]
                # Get the upper whisker position from boxplot
                upper_whisker = bp['whiskers'][i*2 + 1].get_ydata()[1]
                ax.text(pos, upper_whisker, f'{median_val:.4f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Set title and labels
        ax.set_title(metric_titles.get(metric, metric.upper()),
                    fontsize=18, fontweight='bold')
        ax.set_ylabel('Difference', fontsize=18, fontweight='bold')
        
        # Set tick label sizes
        ax.tick_params(axis='y', labelsize=16)
        
        # Grid and spines
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Overall title
    stat_label = statistic.capitalize()
    fig.suptitle(f'Performance Differences by Node Count ({stat_label}): {model_name} vs {baseline_name}\n(Aggregated Across All Sample Sizes, Whiskers = 95% Bootstrap CI)',
                fontsize=18, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_difference_by_sample_size_aggregated(df, metrics=['mse', 'r2', 'nll'],
                                              sample_size_bins=None,
                                              n_sample_bins=8,
                                              output_path=None,
                                              baseline_name="Baseline",
                                              model_name="Model"):
    """Create single-row box plot showing differences by sample size bins, aggregated across all node counts.
    
    Shows one box per sample size bin, aggregating all node counts together.
    
    Args:
        n_sample_bins: Number of sample size bins to create (default: 8)
    """
    if sample_size_bins is None:
        # Create quantile-based bins for sample sizes
        quantiles = np.linspace(0, 1, n_sample_bins + 1)
        bin_edges = df['sample_size'].quantile(quantiles).values
        sample_size_bins = []
        for i in range(len(bin_edges) - 1):
            min_size = int(bin_edges[i])
            max_size = int(bin_edges[i + 1])
            sample_size_bins.append((min_size, max_size, f"{min_size}-{max_size}"))
    
    metric_titles = {
        'mse': 'MSE Difference',
        'r2': 'R² Difference',
        'nll': 'NLL Difference'
    }
    
    # Create figure: 1 row, metrics as columns
    fig, axes = plt.subplots(1, len(metrics),
                            figsize=(5.5 * len(metrics), 5))
    
    # Handle single metric case
    if len(metrics) == 1:
        axes = [axes]
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        metric_col = f'{metric}_diff'
        
        box_data = []
        positions = []
        colors = []
        labels = []
        medians_list = []
        ci_lowers = []
        ci_uppers = []
        
        for bin_idx, (min_size, max_size, label) in enumerate(sample_size_bins):
            # Filter data for this sample size bin (all node counts)
            bin_mask = (df['sample_size'] >= min_size) & (df['sample_size'] <= max_size)
            bin_data = df[bin_mask].copy()
            
            if len(bin_data) > 0:
                bin_values = bin_data[metric_col].values
                
                # Compute bootstrap CI for median
                median, ci_lower, ci_upper = bootstrap_median_ci(bin_values, n_bootstrap=10000, confidence=0.95)
                
                # Create synthetic distribution for boxplot
                if not np.isnan(ci_lower) and not np.isnan(ci_upper) and ci_lower != ci_upper:
                    synthetic_data = [
                        ci_lower,
                        median - (median - ci_lower) * 0.5,
                        median,
                        median + (ci_upper - median) * 0.5,
                        ci_upper
                    ]
                else:
                    synthetic_data = [median] * 5
                
                box_data.append(synthetic_data)
                positions.append(bin_idx)
                labels.append(label)
                medians_list.append(median)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
                
                # Color based on whether model is better
                if metric in ['mse', 'nll']:
                    # Lower is better
                    color = '#35b779' if median < 0 else '#cc4778'
                else:  # r2
                    # Higher is better
                    color = '#35b779' if median > 0 else '#cc4778'
                colors.append(color)
        
        if box_data and positions:
            # Create box plot
            bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                           patch_artist=True, showmeans=False,
                           medianprops=dict(color='red', linewidth=2),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            # Apply colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Set x-axis
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=0, fontsize=16)
            ax.set_xlabel('Sample Size Range', fontsize=18, fontweight='bold')
            
            # Add value labels above boxes
            for i, pos in enumerate(positions):
                median_val = medians_list[i]
                # Get the upper whisker position from boxplot
                upper_whisker = bp['whiskers'][i*2 + 1].get_ydata()[1]
                ax.text(pos, upper_whisker, f'{median_val:.4f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Set title and labels
        ax.set_title(metric_titles.get(metric, metric.upper()),
                    fontsize=18, fontweight='bold')
        ax.set_ylabel('Difference', fontsize=18, fontweight='bold')
        
        # Set tick label sizes
        ax.tick_params(axis='y', labelsize=16)
        
        # Grid and spines
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Overall title
    fig.suptitle(f'Performance Differences by Sample Size (Median): {model_name} vs {baseline_name}\n(Aggregated Across All Node Counts, Whiskers = 95% Bootstrap CI)',
                fontsize=18, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close(fig)


def print_summary_statistics(df, output_file=None, baseline_name="Baseline", model_name="Model"):
    """Print and save summary statistics by node count and sample size."""
    def write_line(line="", file_handle=None):
        print(line)
        if file_handle:
            file_handle.write(line + "\n")
    
    file_handle = None
    if output_file:
        file_handle = open(output_file, 'w')
    
    write_line("="*80, file_handle)
    write_line(f"SUMMARY STATISTICS: {model_name} vs {baseline_name}", file_handle)
    write_line("="*80, file_handle)
    
    # Overall statistics
    write_line("\nOVERALL (All Node Counts, All Sample Sizes):", file_handle)
    write_line("-"*80, file_handle)
    for metric in ['mse', 'r2', 'nll']:
        diff_col = f'{metric}_diff'
        if diff_col in df.columns:
            median_diff = df[diff_col].median()
            mean_diff = df[diff_col].mean()
            std_diff = df[diff_col].std()
            
            if metric in ['mse', 'nll']:
                better_count = (df[diff_col] < 0).sum()
            else:  # r2
                better_count = (df[diff_col] > 0).sum()
            
            total_count = len(df)
            better_pct = 100 * better_count / total_count if total_count > 0 else 0
            
            write_line(f"  {metric.upper():5s} - Median: {median_diff:8.5f}, Mean: {mean_diff:8.5f}, Std: {std_diff:8.5f}", file_handle)
            write_line(f"         Model better: {better_count}/{total_count} ({better_pct:.1f}%)", file_handle)
    
    # By node count
    write_line("\n" + "="*80, file_handle)
    write_line("STATISTICS BY NODE COUNT BIN:", file_handle)
    write_line("="*80, file_handle)
    
    node_bins = sorted(df['node_bin'].unique())
    for node_bin in node_bins:
        write_line(f"\n{node_bin}:", file_handle)
        write_line("-"*80, file_handle)
        
        node_data = df[df['node_bin'] == node_bin]
        write_line(f"  Sample count: {len(node_data)}", file_handle)
        write_line(f"  Sample size range: {node_data['sample_size'].min()} to {node_data['sample_size'].max()}", file_handle)
        write_line(f"  Actual node counts: {node_data['n_nodes'].min()} to {node_data['n_nodes'].max()}", file_handle)
        write_line("", file_handle)
        
        for metric in ['mse', 'r2', 'nll']:
            diff_col = f'{metric}_diff'
            if diff_col in node_data.columns:
                median_diff = node_data[diff_col].median()
                mean_diff = node_data[diff_col].mean()
                std_diff = node_data[diff_col].std()
                
                if metric in ['mse', 'nll']:
                    better_count = (node_data[diff_col] < 0).sum()
                else:  # r2
                    better_count = (node_data[diff_col] > 0).sum()
                
                total_count = len(node_data)
                better_pct = 100 * better_count / total_count if total_count > 0 else 0
                
                write_line(f"  {metric.upper():5s} - Median: {median_diff:8.5f}, Mean: {mean_diff:8.5f}, Std: {std_diff:8.5f}", file_handle)
                write_line(f"         Model better: {better_count}/{total_count} ({better_pct:.1f}%)", file_handle)
    
    write_line("\n" + "="*80, file_handle)
    
    if file_handle:
        file_handle.close()
        print(f"\nSummary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed plots by node count for ComplexMech2 benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_cache",
        type=str,
        required=True,
        help="Path to data_cache directory containing samples/"
    )
    parser.add_argument(
        "--baseline_results",
        type=str,
        required=True,
        help="Path to baseline model results directory"
    )
    parser.add_argument(
        "--model_results",
        type=str,
        required=True,
        help="Path to comparison model results directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: model_results/plots)"
    )
    parser.add_argument(
        "--baseline_name",
        type=str,
        default="Baseline",
        help="Name of baseline model for plot titles"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Model",
        help="Name of comparison model for plot titles"
    )
    parser.add_argument(
        "--n_sample_bins",
        type=int,
        default=8,
        help="Number of sample size bins to create (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_cache_dir = Path(args.data_cache)
    baseline_dir = Path(args.baseline_results)
    model_dir = Path(args.model_results)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_dir / "plots" / "by_node_count"
    
    # Add comparison identifier and timestamp
    baseline_id = baseline_dir.name
    model_id = model_dir.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"{model_id}_vs_{baseline_id}" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache file for sample sizes
    cache_file = data_cache_dir / "sample_sizes_cache.json"
    
    print("="*80)
    print("ComplexMech2: Detailed Analysis by Node Count")
    print("="*80)
    print(f"Data cache: {data_cache_dir}")
    print(f"Baseline: {baseline_dir}")
    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Load data
    print("Loading sample sizes...")
    sample_sizes = load_sample_sizes(data_cache_dir, cache_file=cache_file)
    
    print("\nCounting active features in each dataset...")
    active_features_dict = count_active_features(data_cache_dir)
    
    print("\nLoading baseline results...")
    baseline_results = load_model_results(baseline_dir)
    
    print("\nLoading model results...")
    model_results = load_model_results(model_dir)
    
    # Create dataframe
    print("\nCreating dataframe organized by node count...")
    df = create_dataframe_by_node_count(sample_sizes, baseline_results, model_results,
                                       active_features_dict)
    
    if len(df) == 0:
        print("ERROR: No valid data found!")
        return
    
    # Generate plots
    print("\n" + "="*80)
    print(f"Generating detailed plots by node count and sample size (Median, {args.n_sample_bins} bins)...")
    print("="*80)
    
    output_path = output_dir / "difference_by_node_count_and_sample_size.png"
    plot_difference_by_node_count_and_sample_size(
        df,
        metrics=['mse', 'r2', 'nll'],
        n_sample_bins=args.n_sample_bins,
        output_path=output_path,
        baseline_name=args.baseline_name,
        model_name=args.model_name,
        statistic='median'
    )
    
    print("\n" + "="*80)
    print("Generating aggregated plot by node count (Median)...")
    print("="*80)
    
    output_path = output_dir / "difference_by_node_count_aggregated.png"
    plot_difference_by_node_count_aggregated(
        df,
        metrics=['mse', 'r2', 'nll'],
        output_path=output_path,
        baseline_name=args.baseline_name,
        model_name=args.model_name,
        statistic='median'
    )
    
    print("\n" + "="*80)
    print(f"Generating aggregated plot by sample size (Median, {args.n_sample_bins} bins)...")
    print("="*80)
    
    output_path = output_dir / "difference_by_sample_size_aggregated.png"
    plot_difference_by_sample_size_aggregated(
        df,
        metrics=['mse', 'r2', 'nll'],
        n_sample_bins=args.n_sample_bins,
        output_path=output_path,
        baseline_name=args.baseline_name,
        model_name=args.model_name
    )
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("Generating summary statistics...")
    print("="*80)
    
    summary_file = output_dir / "summary_by_node_count.txt"
    print_summary_statistics(df, output_file=summary_file,
                           baseline_name=args.baseline_name,
                           model_name=args.model_name)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
