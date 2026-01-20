"""Plot performance vs training data sample size for ComplexMech2 benchmark."""

import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def bootstrap_median_ci(data, n_bootstrap=10000, confidence=0.95):
    """Calculate bootstrap confidence interval for the median.
    
    Args:
        data: array of values
        n_bootstrap: number of bootstrap samples
        confidence: confidence level (default 0.95 for 95% CI)
        
    Returns:
        tuple: (lower_bound, upper_bound) of confidence interval
    """
    if len(data) == 0:
        return (np.nan, np.nan)
    
    medians = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        medians.append(np.median(sample))
    
    # Calculate percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(medians, lower_percentile)
    upper_bound = np.percentile(medians, upper_percentile)
    
    return (lower_bound, upper_bound)


def load_sample_sizes(data_cache_dir, cache_file=None, max_samples=None):
    """Load training sample sizes for each dataset index.
    
    Args:
        data_cache_dir: Path to data_cache directory containing samples/
        cache_file: Optional path to cache file to speed up loading
        max_samples: Maximum number of samples to load (for debugging)
        
    Returns:
        dict: Mapping from index to training sample size
    """
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
        if sample_sizes:
            print(f"Sample size range: {min(sample_sizes.values())} to {max(sample_sizes.values())}")
        return sample_sizes
    
    sample_sizes = {}
    
    # Load each pickle file to get sample size
    pickle_files = sorted(samples_dir.glob("*.pkl"))
    
    # Limit number of files for debugging if requested
    if max_samples:
        pickle_files = pickle_files[:max_samples]
        print(f"Loading sample sizes from {len(pickle_files)} files (limited to {max_samples} for debugging)...")
    else:
        print(f"Loading sample sizes from {len(pickle_files)} files...")
    
    for pkl_file in tqdm(pickle_files, desc="Loading samples", unit="file"):
        # Extract index from filename (e.g., "0000.pkl" -> 0 or "sample_0000.pkl" -> 0)
        try:
            stem = pkl_file.stem
            # Handle both "0000" and "sample_0000" formats
            if stem.startswith("sample_"):
                index = int(stem.replace("sample_", ""))
            else:
                index = int(stem)
        except ValueError:
            print(f"Skipping file with non-numeric name: {pkl_file.name}")
            continue
        
        # Load pickle file
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Get training sample size - check multiple possible keys
        if 'X_train' in data:
            X = data['X_train']
            # Handle possible batch dimension
            if X.ndim == 3:
                n_train = X.shape[1]  # (batch, samples, features)
            else:
                n_train = len(X)
        elif 'train' in data and 'X' in data['train']:
            X = data['train']['X']
            if X.ndim == 3:
                n_train = X.shape[1]
            else:
                n_train = len(X)
        elif 'X_obs' in data:
            # For ComplexMech2, X_obs is the observational training data
            X = data['X_obs']
            # Handle possible batch dimension
            if X.ndim == 3:
                n_train = X.shape[1]  # (batch, samples, features)
            else:
                n_train = len(X)
            # Debug: print first few to verify
            if index < 3:
                print(f"  Sample {index}: X_obs shape = {X.shape}, n_train = {n_train}")
        else:
            print(f"Warning: Could not find training data in {pkl_file.name}")
            continue
        
        sample_sizes[index] = n_train
    
    print(f"Loaded sample sizes for {len(sample_sizes)} datasets")
    if sample_sizes:
        print(f"Sample size range: {min(sample_sizes.values())} to {max(sample_sizes.values())}")
    else:
        raise ValueError("No valid sample files found. Check that pickle files contain training data.")
    
    # Save to cache if specified
    if cache_file:
        print(f"Saving sample sizes to cache: {cache_file}")
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(sample_sizes, f)
    
    return sample_sizes


def load_model_results(results_dir):
    """Load model evaluation results.
    
    Args:
        results_dir: Path to model results directory
        
    Returns:
        dict: Mapping from index to performance metrics
    """
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
            }
        elif 'index' in result:
            idx = result['index']
            results_by_index[idx] = {
                'mse': result.get('mse', np.nan),
                'r2': result.get('r2', np.nan),
                'nll': result.get('nll', np.nan),
            }
    
    print(f"Loaded results for {len(results_by_index)} datasets")
    
    return results_by_index


def create_scatter_plot(sample_sizes, results, metric='mse', output_path=None, model_name="Model"):
    """Create scatter plot of performance vs sample size.
    
    Args:
        sample_sizes: dict mapping index to training sample size
        results: dict mapping index to performance metrics
        metric: metric to plot ('mse', 'r2', or 'nll')
        output_path: path to save figure (optional)
        model_name: name of model for title
    """
    # Collect data points
    sizes = []
    values = []
    
    for idx in sorted(sample_sizes.keys()):
        if idx in results:
            size = sample_sizes[idx]
            value = results[idx][metric]
            
            # Skip NaN values
            if not np.isnan(value):
                sizes.append(size)
                values.append(value)
            else:
                if len(sizes) < 5:  # Debug first few
                    print(f"  Skipping sample {idx}: {metric} is NaN")
        else:
            if len(sizes) < 5:  # Debug first few
                print(f"  Sample {idx} not in results")
    
    if len(sizes) == 0:
        print(f"No valid data points for metric: {metric}")
        print(f"  Total samples in sample_sizes: {len(sample_sizes)}")
        print(f"  Total samples in results: {len(results)}")
        print(f"  Common indices: {len(set(sample_sizes.keys()) & set(results.keys()))}")
        # Check a few values
        common_idxs = list(set(sample_sizes.keys()) & set(results.keys()))[:5]
        for idx in common_idxs:
            print(f"  Sample {idx}: size={sample_sizes[idx]}, {metric}={results[idx][metric]}")
        return
    
    print(f"Plotting {len(sizes)} data points for {metric}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(sizes, values, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add trend line (moving average)
    if len(sizes) > 10:
        # Sort by size
        sorted_indices = np.argsort(sizes)
        sorted_sizes = np.array(sizes)[sorted_indices]
        sorted_values = np.array(values)[sorted_indices]
        
        # Compute moving average
        window_size = max(10, len(sizes) // 20)
        moving_avg = np.convolve(sorted_values, np.ones(window_size)/window_size, mode='valid')
        # Adjust size array to match moving average length
        moving_sizes = sorted_sizes[window_size-1:][:len(moving_avg)]
        
        ax.plot(moving_sizes, moving_avg, 'r-', linewidth=2, alpha=0.8, label=f'Moving Average (window={window_size})')
        ax.legend()
    
    # Labels and title
    ax.set_xlabel('Training Sample Size', fontsize=14, fontweight='bold')
    
    metric_labels = {
        'mse': 'Mean Squared Error (MSE)',
        'r2': 'R² Score',
        'nll': 'Negative Log-Likelihood (NLL)'
    }
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=14, fontweight='bold')
    
    ax.set_title(f'{model_name}: {metric_labels.get(metric, metric.upper())} vs Training Sample Size',
                fontsize=16, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Stats text box
    stats_text = f"N = {len(sizes)}\n"
    stats_text += f"Median {metric.upper()}: {np.median(values):.4f}\n"
    stats_text += f"Mean {metric.upper()}: {np.mean(values):.4f}\n"
    stats_text += f"Std {metric.upper()}: {np.std(values):.4f}"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_binned_bar_plot(sample_sizes, results, metric='mse', output_path=None, model_name="Model", n_bins=5):
    """Create binned bar plot of performance vs sample size.
    
    Args:
        sample_sizes: dict mapping index to training sample size
        results: dict mapping index to performance metrics
        metric: metric to plot ('mse', 'r2', or 'nll')
        output_path: path to save figure (optional)
        model_name: name of model for title
        n_bins: number of bins to create based on sample size quantiles
    """
    # Collect data points
    sizes = []
    values = []
    
    for idx in sorted(sample_sizes.keys()):
        if idx in results:
            size = sample_sizes[idx]
            value = results[idx][metric]
            
            # Skip NaN values
            if not np.isnan(value):
                sizes.append(size)
                values.append(value)
    
    if len(sizes) == 0:
        print(f"No valid data points for metric: {metric}")
        return
    
    sizes = np.array(sizes)
    values = np.array(values)
    
    print(f"Creating binned bar plot with {len(sizes)} data points for {metric}")
    
    # Create bins based on quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(sizes, quantiles)
    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1
    
    # Assign data to bins
    bin_indices = np.digitize(sizes, bin_edges[1:-1])
    
    # Compute statistics for each bin
    bin_medians = []
    bin_q25 = []
    bin_q75 = []
    bin_counts = []
    bin_labels = []
    
    for i in range(actual_n_bins):
        mask = bin_indices == i
        bin_values = values[mask]
        bin_sizes = sizes[mask]
        
        if len(bin_values) > 0:
            bin_medians.append(np.median(bin_values))
            bin_q25.append(np.percentile(bin_values, 25))
            bin_q75.append(np.percentile(bin_values, 75))
            bin_counts.append(len(bin_values))
            
            # Create label with size range
            size_min = int(bin_sizes.min())
            size_max = int(bin_sizes.max())
            bin_labels.append(f"{size_min}-{size_max}\n(n={len(bin_values)})")
        else:
            bin_medians.append(0)
            bin_q25.append(0)
            bin_q75.append(0)
            bin_counts.append(0)
            bin_labels.append(f"Empty")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot with interquartile range as error bars
    x_pos = np.arange(len(bin_labels))
    # Error bars show IQR: from Q1 to Q3
    yerr_lower = [bin_medians[i] - bin_q25[i] for i in range(len(bin_medians))]
    yerr_upper = [bin_q75[i] - bin_medians[i] for i in range(len(bin_medians))]
    bars = ax.bar(x_pos, bin_medians, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # Color bars by value
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bin_medians)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Labels and title
    ax.set_xlabel('Training Sample Size Range', fontsize=14, fontweight='bold')
    
    metric_labels = {
        'mse': 'Mean Squared Error (MSE)',
        'r2': 'R² Score',
        'nll': 'Negative Log-Likelihood (NLL)'
    }
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=14, fontweight='bold')
    
    ax.set_title(f'{model_name}: {metric_labels.get(metric, metric.upper())} by Training Sample Size\n(Median per Bin, Error Bars = IQR)',
                fontsize=16, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, fontsize=10)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (median_val, count) in enumerate(zip(bin_medians, bin_counts)):
        if count > 0:
            ax.text(i, bin_q75[i], f'{median_val:.3f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Stats text box
    stats_text = f"Total samples: {len(sizes)}\n"
    stats_text += f"Overall median: {np.median(values):.4f}\n"
    stats_text += f"Overall mean: {np.mean(values):.4f}\n"
    stats_text += f"Overall std: {np.std(values):.4f}"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_boxplot(sample_sizes, results, metric='mse', output_path=None, model_name="Model", n_bins=5):
    """Create box plot with bootstrap CIs for median performance vs sample size.
    
    Similar to the style in plot_performance_difference.py.
    Boxes represent synthetic data from CI, whiskers show 95% bootstrap CI for median.
    
    Args:
        sample_sizes: dict mapping index to training sample size
        results: dict mapping index to performance metrics
        metric: metric to plot ('mse', 'r2', or 'nll')
        output_path: path to save figure (optional)
        model_name: name of model for title
        n_bins: number of bins to create based on sample size quantiles
    """
    # Collect data points
    sizes = []
    values = []
    
    for idx in sorted(sample_sizes.keys()):
        if idx in results:
            size = sample_sizes[idx]
            value = results[idx][metric]
            
            # Skip NaN values
            if not np.isnan(value):
                sizes.append(size)
                values.append(value)
    
    if len(sizes) == 0:
        print(f"No valid data points for metric: {metric}")
        return
    
    sizes = np.array(sizes)
    values = np.array(values)
    
    print(f"Creating box plot with bootstrap CIs using {len(sizes)} data points for {metric}")
    
    # Create bins based on quantiles
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(sizes, quantiles)
    # Ensure unique bin edges
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1
    
    # Assign data to bins
    bin_indices = np.digitize(sizes, bin_edges[1:-1])
    
    # Compute statistics for each bin
    bin_medians = []
    bin_ci_lower = []
    bin_ci_upper = []
    bin_counts = []
    bin_labels = []
    
    print(f"Computing bootstrap confidence intervals (10,000 samples per bin)...")
    for i in range(actual_n_bins):
        mask = bin_indices == i
        bin_values = values[mask]
        bin_sizes = sizes[mask]
        
        if len(bin_values) > 0:
            median = np.median(bin_values)
            bin_medians.append(median)
            
            # Compute 95% bootstrap CI for median
            ci_lower, ci_upper = bootstrap_median_ci(bin_values, n_bootstrap=10000, confidence=0.95)
            bin_ci_lower.append(ci_lower)
            bin_ci_upper.append(ci_upper)
            
            bin_counts.append(len(bin_values))
            
            # Create label with size range
            size_min = int(bin_sizes.min())
            size_max = int(bin_sizes.max())
            bin_labels.append(f"{size_min}-{size_max}\n(n={len(bin_values)})")
            
            print(f"  Bin {i+1}: median={median:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            bin_medians.append(0)
            bin_ci_lower.append(0)
            bin_ci_upper.append(0)
            bin_counts.append(0)
            bin_labels.append(f"Empty")
    
    # Create figure with style matching plot_performance_difference.py
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare box plot data
    box_data = []
    positions = []
    
    for i in range(actual_n_bins):
        if bin_counts[i] > 0:
            median = bin_medians[i]
            ci_lower = bin_ci_lower[i]
            ci_upper = bin_ci_upper[i]
            
            # Create synthetic distribution for boxplot using CI bounds
            if ci_lower != ci_upper:
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
            positions.append(i)
    
    if box_data and positions:
        # Create box plot
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=False,
                       medianprops=dict(color='red', linewidth=2),
                       boxprops=dict(linewidth=1.5),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Color boxes using viridis colormap
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set x-axis
        ax.set_xticks(positions)
        ax.set_xticklabels([bin_labels[p] for p in positions], rotation=0, fontsize=14)
        ax.set_xlabel('Training Sample Size Range', fontsize=16, fontweight='bold')
        
        # Add value labels above boxes
        for i, pos in enumerate(positions):
            median_val = bin_medians[pos]
            # Get the upper whisker position from boxplot
            upper_whisker = bp['whiskers'][i*2 + 1].get_ydata()[1]
            ax.text(pos, upper_whisker, f'{median_val:.4f}', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Labels and title
    metric_labels = {
        'mse': 'Mean Squared Error (MSE)',
        'r2': 'R² Score',
        'nll': 'Negative Log-Likelihood (NLL)'
    }
    
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), fontsize=16, fontweight='bold')
    ax.set_title(f'{model_name}: {metric_labels.get(metric, metric.upper())} by Training Sample Size\n(Whiskers = 95% Bootstrap CI for Median)',
                fontsize=16, fontweight='bold')
    
    # Set tick label sizes
    ax.tick_params(axis='y', labelsize=14)
    
    # Grid and spines
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Stats text box
    stats_text = f"Total samples: {len(sizes)}\n"
    stats_text += f"Overall median: {np.median(values):.4f}\n"
    stats_text += f"Overall mean: {np.mean(values):.4f}\n"
    stats_text += f"Overall std: {np.std(values):.4f}"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=11)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot performance vs training sample size for ComplexMech2 benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_cache",
        type=str,
        required=True,
        help="Path to data_cache directory containing samples/"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to model results directory"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mse", "r2", "nll"],
        choices=["mse", "r2", "nll"],
        help="Metrics to plot"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: results_dir)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ComplexMech2 Model",
        help="Model name for plot titles"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to load (for debugging)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_cache_dir = Path(args.data_cache)
    results_dir = Path(args.results_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "plots"
    
    # Add model identifier to output directory based on results_dir name
    model_id = results_dir.name
    output_dir = output_dir / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache file for sample sizes
    cache_file = data_cache_dir / "sample_sizes_cache.json"
    
    print("="*80)
    print("ComplexMech2: Performance vs Training Sample Size")
    print("="*80)
    print(f"Data cache: {data_cache_dir}")
    print(f"Results: {results_dir}")
    print(f"Output: {output_dir}")
    print(f"Metrics: {args.metrics}")
    print()
    
    # Load data
    print("Loading sample sizes...")
    sample_sizes = load_sample_sizes(data_cache_dir, cache_file=cache_file, max_samples=args.max_samples)
    
    print("\nLoading model results...")
    results = load_model_results(results_dir)
    
    # Find common indices
    common_indices = set(sample_sizes.keys()) & set(results.keys())
    print(f"\nFound {len(common_indices)} datasets with both sample sizes and results")
    
    if len(common_indices) == 0:
        print("ERROR: No common datasets found!")
        return
    
    # Create plots for each metric
    print("\nGenerating scatter plots...")
    for metric in args.metrics:
        print(f"\n--- Plotting {metric.upper()} (Scatter) ---")
        output_path = output_dir / f"performance_vs_sample_size_{metric}_scatter.png"
        create_scatter_plot(sample_sizes, results, metric=metric, 
                          output_path=output_path, model_name=args.model_name)
    
    print("\nGenerating binned bar plots...")
    for metric in args.metrics:
        print(f"\n--- Plotting {metric.upper()} (Binned) ---")
        output_path = output_dir / f"performance_vs_sample_size_{metric}_binned.png"
        create_binned_bar_plot(sample_sizes, results, metric=metric, 
                             output_path=output_path, model_name=args.model_name, n_bins=5)
    
    print("\nGenerating box plots with bootstrap CIs...")
    for metric in args.metrics:
        print(f"\n--- Plotting {metric.upper()} (Box Plot with Bootstrap CIs) ---")
        output_path = output_dir / f"performance_vs_sample_size_{metric}_boxplot.png"
        create_boxplot(sample_sizes, results, metric=metric, 
                      output_path=output_path, model_name=args.model_name, n_bins=5)
    
    print()
    print("="*80)
    print("Plotting complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
