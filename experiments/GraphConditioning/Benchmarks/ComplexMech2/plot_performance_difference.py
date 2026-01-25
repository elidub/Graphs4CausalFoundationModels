"""Plot performance difference between two models vs training data sample size."""

import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats

# Set style with very large default font
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 40


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


def load_sample_sizes(data_cache_dir, cache_file=None):
    """Load training sample sizes for each dataset index.
    
    Args:
        data_cache_dir: Path to data_cache directory containing samples/
        cache_file: Optional path to cache file to speed up loading
        
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
    
    raise FileNotFoundError(f"Cache file not found. Please run plot_performance_vs_sample_size.py first to generate cache.")


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


def create_difference_scatter_plot(sample_sizes, baseline_results, model_results, metric='mse', 
                                   output_path=None, baseline_name="Baseline", model_name="Model"):
    """Create scatter plot of performance difference vs sample size.
    
    Args:
        sample_sizes: dict mapping index to training sample size
        baseline_results: dict mapping index to baseline performance metrics
        model_results: dict mapping index to model performance metrics
        metric: metric to plot ('mse', 'r2', or 'nll')
        output_path: path to save figure (optional)
        baseline_name: name of baseline model
        model_name: name of comparison model
    """
    # Collect data points
    sizes = []
    differences = []
    
    for idx in sorted(sample_sizes.keys()):
        if idx in baseline_results and idx in model_results:
            size = sample_sizes[idx]
            baseline_value = baseline_results[idx][metric]
            model_value = model_results[idx][metric]
            
            # Skip NaN values
            if not np.isnan(baseline_value) and not np.isnan(model_value):
                sizes.append(size)
                # Difference = Model - Baseline
                # Negative means model is better (for MSE, NLL)
                # Positive means model is better (for R2)
                differences.append(model_value - baseline_value)
    
    if len(sizes) == 0:
        print(f"No valid data points for metric: {metric}")
        return
    
    print(f"Plotting {len(sizes)} data points for {metric} difference")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Determine colors based on sign of difference
    colors = ['green' if d < 0 else 'red' for d in differences]
    if metric == 'r2':
        # For R2, positive is better
        colors = ['green' if d > 0 else 'red' for d in differences]
    
    # Scatter plot
    ax.scatter(sizes, differences, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='No difference')
    
    # Add trend line (moving average)
    if len(sizes) > 10:
        # Sort by size
        sorted_indices = np.argsort(sizes)
        sorted_sizes = np.array(sizes)[sorted_indices]
        sorted_diffs = np.array(differences)[sorted_indices]
        
        # Compute moving average
        window_size = max(10, len(sizes) // 20)
        moving_avg = np.convolve(sorted_diffs, np.ones(window_size)/window_size, mode='valid')
        # Adjust size array to match moving average length
        moving_sizes = sorted_sizes[window_size-1:][:len(moving_avg)]
        
        ax.plot(moving_sizes, moving_avg, 'b-', linewidth=2, alpha=0.8, label=f'Moving Average (window={window_size})')
        ax.legend()
    
    # Labels and title - simplified
    ax.set_xlabel('Training Sample Size', fontsize=14, fontweight='bold')
    
    metric_labels = {
        'mse': 'MSE Difference',
        'r2': 'R² Difference',
        'nll': 'NLL Difference'
    }
    metric_titles = {
        'mse': 'Improvement over Baseline (MSE)',
        'r2': 'Improvement over Baseline (R²)',
        'nll': 'Improvement over Baseline (NLL)'
    }
    
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), 
                  fontsize=14, fontweight='bold')
    
    ax.set_title(metric_titles.get(metric, f'Improvement over Baseline ({metric.upper()})'),
                fontsize=16, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_difference_binned_bar_plot(sample_sizes, baseline_results, model_results, metric='mse', 
                                     output_path=None, baseline_name="Baseline", model_name="Model", n_bins=5):
    """Create binned bar plot of performance difference vs sample size.
    
    Args:
        sample_sizes: dict mapping index to training sample size
        baseline_results: dict mapping index to baseline performance metrics
        model_results: dict mapping index to model performance metrics
        metric: metric to plot ('mse', 'r2', or 'nll')
        output_path: path to save figure (optional)
        baseline_name: name of baseline model
        model_name: name of comparison model
        n_bins: number of bins to create based on sample size quantiles
    """
    # Collect data points
    sizes = []
    differences = []
    
    for idx in sorted(sample_sizes.keys()):
        if idx in baseline_results and idx in model_results:
            size = sample_sizes[idx]
            baseline_value = baseline_results[idx][metric]
            model_value = model_results[idx][metric]
            
            # Skip NaN values
            if not np.isnan(baseline_value) and not np.isnan(model_value):
                sizes.append(size)
                differences.append(model_value - baseline_value)
    
    if len(sizes) == 0:
        print(f"No valid data points for metric: {metric}")
        return
    
    sizes = np.array(sizes)
    differences = np.array(differences)
    
    print(f"Creating binned bar plot with {len(sizes)} data points for {metric} difference")
    
    # Create fixed bins: 1-100, 101-200, 201-300, 301-500
    bin_edges = [1, 101, 201, 301, 501]
    bin_labels_fixed = ["1-100", "101-200", "201-300", "301-500"]
    actual_n_bins = len(bin_edges) - 1
    
    # Assign data to bins
    bin_indices = np.digitize(sizes, bin_edges) - 1  # 0-indexed
    
    # Compute statistics for each bin
    bin_medians = []
    bin_q25 = []
    bin_q75 = []
    bin_counts = []
    bin_labels = []
    
    for i in range(actual_n_bins):
        mask = bin_indices == i
        bin_diffs = differences[mask]
        
        if len(bin_diffs) > 0:
            bin_medians.append(np.median(bin_diffs))
            bin_q25.append(np.percentile(bin_diffs, 25))
            bin_q75.append(np.percentile(bin_diffs, 75))
            bin_counts.append(len(bin_diffs))
            bin_labels.append(bin_labels_fixed[i])
        else:
            bin_medians.append(0)
            bin_q25.append(0)
            bin_q75.append(0)
            bin_counts.append(0)
            bin_labels.append(bin_labels_fixed[i])
    
    # Create figure - large size for paper
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create bar plot with interquartile range as error bars
    x_pos = np.arange(len(bin_labels))
    # Error bars show IQR: from Q1 to Q3
    yerr_lower = [bin_medians[i] - bin_q25[i] for i in range(len(bin_medians))]
    yerr_upper = [bin_q75[i] - bin_medians[i] for i in range(len(bin_medians))]
    bars = ax.bar(x_pos, bin_medians, yerr=[yerr_lower, yerr_upper], capsize=14, alpha=0.7, 
                   edgecolor='black', linewidth=4)
    
    # Color bars by sign (green if model is better, red if baseline is better)
    for i, (bar, median) in enumerate(zip(bars, bin_medians)):
        if metric in ['mse', 'nll']:
            # Lower is better - green means improvement
            color = '#35b779' if median < 0 else '#cc4778'
        else:  # r2
            # Higher is better - green means improvement
            color = '#35b779' if median > 0 else '#cc4778'
        bar.set_color(color)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=4, alpha=0.7)
    
    # Labels and title - DOUBLED font sizes for paper
    ax.set_xlabel('Training Sample Size', fontsize=56, fontweight='bold')
    
    metric_labels = {
        'mse': 'MSE Difference',
        'r2': 'R² Difference',
        'nll': 'NLL Difference'
    }
    metric_titles = {
        'mse': 'Improvement over Baseline (MSE)',
        'r2': 'Improvement over Baseline (R²)',
        'nll': 'Improvement over Baseline (NLL)'
    }
    
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), 
                  fontsize=56, fontweight='bold')
    
    ax.set_title(metric_titles.get(metric, f'Improvement over Baseline ({metric.upper()})'),
                fontsize=64, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels, fontsize=48)
    ax.tick_params(axis='y', labelsize=48)
    
    # Grid and spines
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_difference_boxplot(sample_sizes, baseline_results, model_results, metric='mse', 
                             output_path=None, baseline_name="Baseline", model_name="Model", n_bins=5):
    """Create box plot with bootstrap CIs for median differences vs sample size.
    
    Similar to the style in generate_plots.py, but adapted for difference plotting.
    Boxes represent synthetic data from CI, whiskers show 95% bootstrap CI for median.
    
    Args:
        sample_sizes: dict mapping index to training sample size
        baseline_results: dict mapping index to baseline performance metrics
        model_results: dict mapping index to model performance metrics
        metric: metric to plot ('mse', 'r2', or 'nll')
        output_path: path to save figure (optional)
        baseline_name: name of baseline model
        model_name: name of comparison model
        n_bins: number of bins to create based on sample size quantiles
    """
    # Collect data points
    sizes = []
    differences = []
    
    for idx in sorted(sample_sizes.keys()):
        if idx in baseline_results and idx in model_results:
            size = sample_sizes[idx]
            baseline_value = baseline_results[idx][metric]
            model_value = model_results[idx][metric]
            
            # Skip NaN values
            if not np.isnan(baseline_value) and not np.isnan(model_value):
                sizes.append(size)
                differences.append(model_value - baseline_value)
    
    if len(sizes) == 0:
        print(f"No valid data points for metric: {metric}")
        return
    
    sizes = np.array(sizes)
    differences = np.array(differences)
    
    print(f"Creating box plot with bootstrap CIs using {len(sizes)} data points for {metric} difference")
    
    # Create fixed bins: 1-100, 101-200, 201-300, 301-500
    bin_edges = [1, 101, 201, 301, 501]
    bin_labels_fixed = ["1-100", "101-200", "201-300", "301-500"]
    actual_n_bins = len(bin_edges) - 1
    
    # Assign data to bins
    bin_indices = np.digitize(sizes, bin_edges) - 1  # 0-indexed
    
    # Compute statistics for each bin
    bin_medians = []
    bin_ci_lower = []
    bin_ci_upper = []
    bin_counts = []
    bin_labels = []
    
    print(f"Computing bootstrap confidence intervals (10,000 samples per bin)...")
    for i in range(actual_n_bins):
        mask = bin_indices == i
        bin_diffs = differences[mask]
        
        if len(bin_diffs) > 0:
            median = np.median(bin_diffs)
            bin_medians.append(median)
            
            # Compute 95% bootstrap CI for median
            ci_lower, ci_upper = bootstrap_median_ci(bin_diffs, n_bootstrap=10000, confidence=0.95)
            bin_ci_lower.append(ci_lower)
            bin_ci_upper.append(ci_upper)
            
            bin_counts.append(len(bin_diffs))
            bin_labels.append(bin_labels_fixed[i])
            
            print(f"  Bin {i+1}: median={median:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
        else:
            bin_medians.append(0)
            bin_ci_lower.append(0)
            bin_ci_upper.append(0)
            bin_counts.append(0)
            bin_labels.append(bin_labels_fixed[i])
    
    # Create figure - size for paper
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare box plot data
    box_data = []
    positions = []
    colors = []
    
    for i in range(actual_n_bins):
        if bin_counts[i] > 0:
            median = bin_medians[i]
            ci_lower = bin_ci_lower[i]
            ci_upper = bin_ci_upper[i]
            
            # Create synthetic distribution for boxplot using CI bounds
            # This matches the approach in generate_plots.py
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
            
            # Color based on whether model is better
            if metric in ['mse', 'nll']:
                # Lower is better - green means improvement (negative difference)
                color = '#35b779' if median < 0 else '#cc4778'
            else:  # r2
                # Higher is better - green means improvement (positive difference)
                color = '#35b779' if median > 0 else '#cc4778'
            colors.append(color)
    
    if box_data and positions:
        # Create box plot with thicker lines
        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=False,
                       medianprops=dict(color='red', linewidth=2),
                       boxprops=dict(linewidth=2),
                       whiskerprops=dict(linewidth=2),
                       capprops=dict(linewidth=2))
        
        # Apply colors to boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set x-axis labels
        ax.set_xticks(positions)
        ax.set_xticklabels([bin_labels[p] for p in positions], fontsize=25)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    
    # Labels and title - 25pt font sizes
    ax.set_xlabel('Training Sample Size', fontsize=25, fontweight='bold')
    
    metric_labels = {
        'mse': 'MSE Difference',
        'r2': 'R² Difference',
        'nll': 'NLL Difference'
    }
    metric_titles = {
        'mse': 'Improvement over Baseline (MSE)',
        'r2': 'Improvement over Baseline (R²)',
        'nll': 'Improvement over Baseline (NLL)'
    }
    
    ax.set_ylabel(metric_labels.get(metric, metric.upper()), 
                  fontsize=25, fontweight='bold')
    
    # Title
    ax.set_title(metric_titles.get(metric, f'Improvement over Baseline ({metric.upper()})'),
                fontsize=25, fontweight='bold')
    
    # Set tick label sizes - 25pt
    ax.tick_params(axis='both', labelsize=25)

    # Disable scientific notation for y-axis
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style='plain', axis='y')
    ax.get_yaxis().get_offset_text().set_visible(False)
    
    # Grid and spines
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
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
        description="Plot performance difference between two models vs training sample size",
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
    
    args = parser.parse_args()
    
    # Setup paths
    data_cache_dir = Path(args.data_cache)
    baseline_dir = Path(args.baseline_results)
    model_dir = Path(args.model_results)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_dir / "plots" / "difference"
    
    # Add comparison identifier to output directory
    baseline_id = baseline_dir.name
    model_id = model_dir.name
    output_dir = output_dir / f"{model_id}_vs_{baseline_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache file for sample sizes
    cache_file = data_cache_dir / "sample_sizes_cache.json"
    
    print("="*80)
    print("ComplexMech2: Performance Difference Analysis")
    print("="*80)
    print(f"Data cache: {data_cache_dir}")
    print(f"Baseline: {baseline_dir}")
    print(f"Model: {model_dir}")
    print(f"Output: {output_dir}")
    print(f"Metrics: {args.metrics}")
    print()
    
    # Load data
    print("Loading sample sizes...")
    sample_sizes = load_sample_sizes(data_cache_dir, cache_file=cache_file)
    
    print("\nLoading baseline results...")
    baseline_results = load_model_results(baseline_dir)
    
    print("\nLoading model results...")
    model_results = load_model_results(model_dir)
    
    # Find common indices
    common_indices = set(sample_sizes.keys()) & set(baseline_results.keys()) & set(model_results.keys())
    print(f"\nFound {len(common_indices)} datasets with sample sizes and both model results")
    
    if len(common_indices) == 0:
        print("ERROR: No common datasets found!")
        return
    
    # Create plots for each metric
    print("\nGenerating scatter plots...")
    for metric in args.metrics:
        print(f"\n--- Plotting {metric.upper()} Difference (Scatter) ---")
        output_path = output_dir / f"performance_difference_{metric}_scatter.png"
        create_difference_scatter_plot(sample_sizes, baseline_results, model_results, metric=metric, 
                                      output_path=output_path, baseline_name=args.baseline_name, 
                                      model_name=args.model_name)
    
    print("\nGenerating binned bar plots...")
    for metric in args.metrics:
        print(f"\n--- Plotting {metric.upper()} Difference (Binned Bar) ---")
        output_path = output_dir / f"performance_difference_{metric}_binned.png"
        create_difference_binned_bar_plot(sample_sizes, baseline_results, model_results, metric=metric, 
                                         output_path=output_path, baseline_name=args.baseline_name, 
                                         model_name=args.model_name, n_bins=5)
    
    print("\nGenerating box plots with bootstrap CIs...")
    for metric in args.metrics:
        print(f"\n--- Plotting {metric.upper()} Difference (Box Plot with Bootstrap CIs) ---")
        output_path = output_dir / f"performance_difference_{metric}_boxplot.png"
        create_difference_boxplot(sample_sizes, baseline_results, model_results, metric=metric, 
                                 output_path=output_path, baseline_name=args.baseline_name, 
                                 model_name=args.model_name, n_bins=5)
    
    print()
    print("="*80)
    print("Plotting complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
