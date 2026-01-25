#!/usr/bin/env python3
"""
ComplexMech Difference Analysis Script

This script analyzes ComplexMech benchmark results by computing the difference in metrics between
models (78, 80) and the baseline model (hide_all), then generates box plots showing the 
improvement (or degradation) relative to hide_all baseline.

Usage:
    python generate_complexmech_difference_plots.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

# Configuration
checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints"
checkpoint_base = Path(checkpoint_path)

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parent / "plots" / f"difference_plots_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['font.size'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['figure.titlesize'] = 30

# Define model order and colors
MODEL_ORDER = ['80', '78']

MODEL_COLORS = {
    '80': '#21918c',  # Teal color (IDK model)
    '78': '#cc4778',  # Pink/red (no graph knowledge)
}

MODEL_DISPLAY_NAMES = {
    '80': 'Model 80 (IDK)',
    '78': 'Model 78 (No Graph)',
}


def extract_model_name(checkpoint_name):
    """Extract model name from checkpoint directory name."""
    if 'hide_all' in checkpoint_name:
        return 'hide_all'
    elif '16773478' in checkpoint_name:
        return '78'
    else:
        return '80'


def load_individual_results(checkpoint_base, checkpoint_keys):
    """Load individual sample results for bootstrap analysis."""
    results_dict = {}
    
    print("Loading individual results...")
    for ckpt_key in tqdm(checkpoint_keys, desc="Loading checkpoints"):
        ckpt_dir = checkpoint_base / ckpt_key
        results_dir = ckpt_dir / "complexmech_idk_high"
        
        if not results_dir.exists():
            print(f"Missing results dir: {results_dir}")
            continue
        
        checkpoint_results = {}
        
        # Look for individual result files
        node_counts = [2, 5, 10, 20, 35, 50]
        variant_suffixes = ['path_YT', 'path_TY', 'path_independent_TY']
        
        for node_count in node_counts:
            # Check for base variant
            base_file = results_dir / f"individual_{node_count}nodes.json"
            if base_file.exists():
                if 'base' not in checkpoint_results:
                    checkpoint_results['base'] = {}
                try:
                    with open(base_file, 'r') as f:
                        checkpoint_results['base'][node_count] = json.load(f)
                except Exception as e:
                    print(f"Error loading {base_file}: {e}")
            
            # Check for path variants with hide fractions
            for variant_suffix in variant_suffixes:
                for hide_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    variant_file = results_dir / f"individual_{node_count}nodes_{variant_suffix}_hide{hide_frac}.json"
                    if variant_file.exists():
                        variant_key = f"{variant_suffix}_hide{hide_frac}"
                        if variant_key not in checkpoint_results:
                            checkpoint_results[variant_key] = {}
                        try:
                            with open(variant_file, 'r') as f:
                                checkpoint_results[variant_key][node_count] = json.load(f)
                        except Exception as e:
                            print(f"Error loading {variant_file}: {e}")
        
        if checkpoint_results:
            results_dict[ckpt_key] = checkpoint_results
            print(f"  Loaded: {ckpt_key} ({len(checkpoint_results)} variants)")
    
    return results_dict


def bootstrap_confidence_interval(data, confidence_level=0.95, n_bootstrap=1000, use_median=False, remove_outliers=False):
    """Compute bootstrap confidence interval for the mean or median of data."""
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Remove outliers if requested (clip to 5th-95th percentile range)
    if remove_outliers:
        data_array = np.array(data)
        q05 = np.percentile(data_array, 5)
        q95 = np.percentile(data_array, 95)
        # Keep only values within the 5th-95th percentile range
        data = data_array[(data_array >= q05) & (data_array <= q95)]
        n = len(data)
        if n == 0:
            return np.nan, np.nan, np.nan
    
    # Original statistic (mean or median)
    if use_median:
        original_stat = np.median(data)
    else:
        original_stat = np.mean(data)
    
    # Bootstrap resampling - vectorized for speed
    np.random.seed(42)  # For reproducibility
    bootstrap_samples = np.random.choice(data, size=(n_bootstrap, n), replace=True)
    
    if use_median:
        bootstrap_stats = np.median(bootstrap_samples, axis=1)
    else:
        bootstrap_stats = np.mean(bootstrap_samples, axis=1)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return original_stat, ci_lower, ci_upper


def compute_metric_differences_with_bootstrap(results_dict, metric='nll', confidence_level=0.95, use_median=False, remove_outliers=False):
    """
    Compute metric differences with bootstrap CI for each variant and hide fraction separately.
    
    INDIVIDUAL-LEVEL DIFFERENCES:
    For each configuration (variant + hide fraction + node count):
    1. Get baseline (hide_all) metric values [val1, val2, ..., val1000]
    2. Get model (78, 80) metric values [val1, val2, ..., val1000]
    3. Compute individual differences: [baseline[i] - model[i] for i in range(1000)]
    4. Apply bootstrap to the array of individual differences
    5. Keep hide fraction separate (don't aggregate)
    """
    difference_data = []
    
    # First get all baseline data organized by variant
    baseline_data = {}  # {variant: {node_count: [metric_values]}}
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        if model_name == 'hide_all':
            for variant, variant_data in ckpt_results.items():
                if variant not in baseline_data:
                    baseline_data[variant] = {}
                for node_count, individual_results in variant_data.items():
                    baseline_data[variant][node_count] = [sample[metric] for sample in individual_results]
    
    print(f"\nComputing {metric.upper()} individual-level differences...")
    print(f"Found baseline data for {len(baseline_data)} variants")
    
    # Compute differences for each model relative to baseline
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        
        if model_name == 'hide_all':
            continue  # Skip baseline
        
        print(f"  Processing model: {model_name}")
        
        for variant, variant_data in ckpt_results.items():
            if variant not in baseline_data:
                continue
            
            for node_count, individual_results in variant_data.items():
                if node_count not in baseline_data[variant]:
                    continue
                
                # Get metric values
                model_metric_values = np.array([sample[metric] for sample in individual_results])
                baseline_metric_values = np.array(baseline_data[variant][node_count])
                
                if len(model_metric_values) == 0 or len(baseline_metric_values) == 0:
                    continue
                
                # Check if we have matching sample counts for individual-level differences
                min_samples = min(len(model_metric_values), len(baseline_metric_values))
                if min_samples == 0:
                    continue
                
                # Truncate to matching length if needed
                model_metric_values = model_metric_values[:min_samples]
                baseline_metric_values = baseline_metric_values[:min_samples]
                
                # COMPUTE INDIVIDUAL-LEVEL DIFFERENCES
                if metric == 'r2':
                    # Positive = improvement for R2 (model better than baseline)
                    individual_differences = model_metric_values - baseline_metric_values
                else:
                    # Positive = improvement for NLL/MSE (baseline better than model)
                    individual_differences = baseline_metric_values - model_metric_values
                
                # Apply bootstrap to the array of individual differences
                stat_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                    individual_differences, confidence_level=confidence_level, 
                    use_median=use_median, remove_outliers=remove_outliers
                )
                
                # Extract hide fraction from variant name
                hide_frac = None
                if 'hide' in variant:
                    try:
                        hide_str = variant.split('hide')[-1]
                        hide_frac = float(hide_str)
                    except:
                        pass
                
                # Store the difference statistics with hide fraction preserved
                difference_data.append({
                    'model': model_name,
                    'variant': variant,
                    'hide_fraction': hide_frac,
                    'node_count': int(node_count),
                    'difference_mean': stat_diff,
                    'difference_ci_lower': ci_lower,
                    'difference_ci_upper': ci_upper,
                    'n_samples': min_samples,
                })
    
    return pd.DataFrame(difference_data)


def compute_aggregated_differences(diff_df, confidence_level=0.95, use_median=False, remove_outliers=False):
    """
    Aggregate differences across path variants BUT KEEP HIDE FRACTIONS SEPARATE.
    
    For each model, node count, and hide fraction, combines differences from 
    path_TY, path_YT, and path_independent_TY variants.
    
    This maintains the hide fraction differentiation that was computed at the individual level.
    """
    print("\nAggregating differences across path variants (keeping hide fractions separate)...")
    
    # Filter to only path variants with hide fractions
    path_variants = diff_df[diff_df['variant'].str.contains('path_', na=False)]
    
    if len(path_variants) == 0:
        print("No path variants found for aggregation")
        return pd.DataFrame()
    
    aggregated_data = []
    
    # Group by model, node_count, AND hide_fraction
    grouped = path_variants.groupby(['model', 'node_count', 'hide_fraction'])
    
    for (model, node_count, hide_frac), group in tqdm(grouped, desc="Aggregating"):
        if len(group) == 0 or pd.isna(hide_frac):
            continue
        
        # Get all difference values for this model/node_count/hide_fraction combo
        # These are already individual-level differences, just from different causal structures
        all_diffs = []
        for _, row in group.iterrows():
            # We have mean and CI, but we want to treat each variant's contribution
            # For proper aggregation, we'd need the raw difference arrays
            # Since we only have summary stats, we'll average the means
            all_diffs.append(row['difference_mean'])
        
        if len(all_diffs) == 0:
            continue
        
        # Compute mean/median of the variant means and bootstrap CI
        stat_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
            all_diffs, confidence_level=confidence_level, 
            use_median=use_median, remove_outliers=remove_outliers
        )
        
        aggregated_data.append({
            'model': model,
            'node_count': int(node_count),
            'hide_fraction': hide_frac,
            'variant': 'aggregated_paths',
            'difference_mean': stat_diff,
            'difference_ci_lower': ci_lower,
            'difference_ci_upper': ci_upper,
            'n_variants': len(all_diffs),
        })
    
    return pd.DataFrame(aggregated_data)


def compute_fully_aggregated_differences(diff_df, confidence_level=0.95, use_median=False, remove_outliers=False):
    """
    Fully aggregate differences across path variants AND node counts, keeping only hide fractions.
    
    For each model and hide fraction, combines differences from all path variants and all node counts.
    
    This gives the overall trend across all complexities.
    """
    print("\nFully aggregating differences across path variants AND node counts...")
    
    # Filter to only path variants with hide fractions
    path_variants = diff_df[diff_df['variant'].str.contains('path_', na=False)]
    
    if len(path_variants) == 0:
        print("No path variants found for full aggregation")
        return pd.DataFrame()
    
    aggregated_data = []
    
    # Group by model and hide_fraction ONLY (aggregate over node counts)
    grouped = path_variants.groupby(['model', 'hide_fraction'])
    
    for (model, hide_frac), group in tqdm(grouped, desc="Fully aggregating"):
        if len(group) == 0 or pd.isna(hide_frac):
            continue
        
        # Get all difference values for this model/hide_fraction combo (from all node counts)
        all_diffs = []
        for _, row in group.iterrows():
            all_diffs.append(row['difference_mean'])
        
        if len(all_diffs) == 0:
            continue
        
        # Compute mean/median of all the differences and bootstrap CI
        stat_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
            all_diffs, confidence_level=confidence_level, 
            use_median=use_median, remove_outliers=remove_outliers
        )
        
        aggregated_data.append({
            'model': model,
            'hide_fraction': hide_frac,
            'variant': 'fully_aggregated',
            'difference_mean': stat_diff,
            'difference_ci_lower': ci_lower,
            'difference_ci_upper': ci_upper,
            'n_samples': len(all_diffs),
        })
    
    return pd.DataFrame(aggregated_data)


def plot_fully_aggregated_differences(df, metric_name, output_prefix=""):
    """
    Create a single plot showing fully aggregated metric differences.
    X-axis: hide fractions (0.0, 0.25, 0.5, 0.75, 1.0)
    Different colors: different models (78, 80)
    Aggregated over all node counts and path variants.
    
    Args:
        df: DataFrame with fully aggregated difference data
        metric_name: Name of the metric (for labeling)
        output_prefix: Prefix for output filename
    """
    # Get unique hide fractions and sort
    hide_fractions = sorted([h for h in df['hide_fraction'].unique() if not pd.isna(h)])
    
    if len(hide_fractions) == 0:
        print("No hide fractions found in data!")
        return
    
    # Create single large figure
    fig, ax = plt.subplots(1, 1, figsize=(25, 22.5))
    
    # Get unique models and sort by predefined order
    available_models = df['model'].unique()
    models = [m for m in MODEL_ORDER if m in available_models]
    
    if not models:
        ax.text(0.5, 0.5, f'No models', ha='center', va='center', transform=ax.transAxes)
        plt.close(fig)
        return
    
    # Plot each model's differences across hide fractions
    n_models = len(models)
    box_width = 0.8 / n_models  # Divide space among models
    
    for model_idx, model in enumerate(models):
        model_data = df[df['model'] == model]
        
        if len(model_data) == 0:
            continue
        
        box_data = []
        positions = []
        
        for hide_idx, hide_frac in enumerate(hide_fractions):
            hide_data = model_data[model_data['hide_fraction'] == hide_frac]
            
            if len(hide_data) == 0:
                continue
            
            # Extract difference statistics
            mean_diff = hide_data['difference_mean'].values[0]
            ci_lower = hide_data['difference_ci_lower'].values[0]
            ci_upper = hide_data['difference_ci_upper'].values[0]
            
            # Get median (use mean as proxy since we computed mean differences)
            median_val = mean_diff
            
            # Create synthetic distribution for box plot (matching reference style)
            synthetic_data = [
                ci_lower,                                           # Lower whisker
                ci_lower + (median_val - ci_lower) * 0.5,          # Lower box edge
                median_val,                                         # Median
                median_val + (ci_upper - median_val) * 0.5,        # Upper box edge
                ci_upper                                            # Upper whisker
            ]
            
            box_data.append(synthetic_data)
            # Position: center of hide_frac group + offset for this model
            position = hide_idx + (model_idx - (n_models - 1) / 2) * box_width
            positions.append(position)
        
        if box_data:
            # Create box plot for this model
            bp = ax.boxplot(box_data, positions=positions, widths=box_width * 0.9,
                           patch_artist=True, showmeans=False, showfliers=False,
                           medianprops=dict(color='red', linewidth=2),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))
            
            # Color boxes for this model
            model_color = MODEL_COLORS.get(model, '#808080')
            for patch in bp['boxes']:
                patch.set_facecolor(model_color)
                patch.set_alpha(0.7)
    
    # Add horizontal line at zero (no difference) - thicker baseline
    ax.axhline(y=0, color='black', linestyle='-', linewidth=3, alpha=0.9)
    
    # Set x-axis labels (hide fractions)
    ax.set_xticks(range(len(hide_fractions)))
    ax.set_xticklabels([f'{h:.2f}' for h in hide_fractions], fontsize=45)
    ax.set_xlabel('Hide Fraction', fontsize=45, fontweight='bold')
    
    # Set title and labels
    ax.set_ylabel('Improvement', fontsize=45, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=45)
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=MODEL_COLORS.get(m, '#808080'), 
                                    alpha=0.7, edgecolor='black', linewidth=1.5,
                                    label=MODEL_DISPLAY_NAMES.get(m, m)) 
                      for m in models]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(models), 
              fontsize=45, frameon=True, bbox_to_anchor=(0.5, 0.95))
    
    # Main title
    fig.suptitle(f'Improvement over hide_all Baseline ({metric_name.upper()})\nAggregated over all node counts', 
                fontsize=60, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save plot
    filename = f"{output_prefix}_{metric_name}_fully_aggregated.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filepath}")
    plt.close(fig)


def plot_metric_differences(df, metric_name, node_counts_to_compare=None, output_prefix=""):
    """
    Create box plots showing metric differences relative to hide_all baseline.
    X-axis: hide fractions (0.0, 0.25, 0.5, 0.75, 1.0)
    Each subplot: different node count
    Different colors: different models (78, 80)
    
    Args:
        df: DataFrame with difference data including hide_fraction column
        metric_name: Name of the metric (for labeling)
        node_counts_to_compare: List of node counts to include
        output_prefix: Prefix for output filename
    """
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    # Get unique hide fractions and sort
    hide_fractions = sorted([h for h in df['hide_fraction'].unique() if not pd.isna(h)])
    
    if len(hide_fractions) == 0:
        print("No hide fractions found in data!")
        return
    
    # Create figure with subplots for each node count
    n_nodes = len(node_counts_to_compare)
    fig, axes = plt.subplots(1, n_nodes, figsize=(15 * n_nodes, 22.5))
    
    if n_nodes == 1:
        axes = [axes]
    
    for node_idx, node_count in enumerate(node_counts_to_compare):
        ax = axes[node_idx]
        
        # Filter data for this node count
        node_data = df[df['node_count'] == node_count].copy()
        
        if len(node_data) == 0:
            ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{node_count} Nodes')
            continue
        
        # Get unique models and sort by predefined order
        available_models = node_data['model'].unique()
        models = [m for m in MODEL_ORDER if m in available_models]
        
        if not models:
            ax.text(0.5, 0.5, f'No models', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{node_count} Nodes')
            continue
        
        # Plot each model's differences across hide fractions
        n_models = len(models)
        box_width = 0.8 / n_models  # Divide space among models
        
        for model_idx, model in enumerate(models):
            model_data = node_data[node_data['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            box_data = []
            positions = []
            
            for hide_idx, hide_frac in enumerate(hide_fractions):
                hide_data = model_data[model_data['hide_fraction'] == hide_frac]
                
                if len(hide_data) == 0:
                    continue
                
                # Extract difference statistics
                mean_diff = hide_data['difference_mean'].values[0]
                ci_lower = hide_data['difference_ci_lower'].values[0]
                ci_upper = hide_data['difference_ci_upper'].values[0]
                
                # Get median (use mean as proxy since we computed mean differences)
                median_val = mean_diff
                
                # Create synthetic distribution for box plot (matching reference style)
                synthetic_data = [
                    ci_lower,                                           # Lower whisker
                    ci_lower + (median_val - ci_lower) * 0.5,          # Lower box edge
                    median_val,                                         # Median
                    median_val + (ci_upper - median_val) * 0.5,        # Upper box edge
                    ci_upper                                            # Upper whisker
                ]
                
                box_data.append(synthetic_data)
                # Position: center of hide_frac group + offset for this model
                position = hide_idx + (model_idx - (n_models - 1) / 2) * box_width
                positions.append(position)
            
            if box_data:
                # Create box plot for this model
                bp = ax.boxplot(box_data, positions=positions, widths=box_width * 0.9,
                               patch_artist=True, showmeans=False, showfliers=False,
                               medianprops=dict(color='red', linewidth=2),
                               boxprops=dict(linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
                
                # Color boxes for this model
                model_color = MODEL_COLORS.get(model, '#808080')
                for patch in bp['boxes']:
                    patch.set_facecolor(model_color)
                    patch.set_alpha(0.7)
        
        # Add horizontal line at zero (no difference) - thicker baseline
        ax.axhline(y=0, color='black', linestyle='-', linewidth=3, alpha=0.9)
        
        # Set x-axis labels (hide fractions)
        ax.set_xticks(range(len(hide_fractions)))
        ax.set_xticklabels([f'{h:.2f}' for h in hide_fractions], fontsize=45)
        ax.set_xlabel('Hide Fraction', fontsize=45, fontweight='bold')
        
        # Set title and labels
        ax.set_title(f'{node_count} Nodes', fontsize=45, fontweight='bold')
        ax.set_ylabel('Improvement', fontsize=45, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=45)
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=MODEL_COLORS.get(m, '#808080'), 
                                    alpha=0.7, edgecolor='black', linewidth=1.5,
                                    label=MODEL_DISPLAY_NAMES.get(m, m)) 
                      for m in models]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(models), 
              fontsize=45, frameon=True, bbox_to_anchor=(0.5, 0.95))
    
    # Main title
    fig.suptitle(f'Improvement over hide_all Baseline ({metric_name.upper()})', 
                fontsize=60, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save plot
    filename = f"{output_prefix}_{metric_name}_difference_by_hide.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {filepath}")
    plt.close(fig)


def main():
    """Main function to generate difference analysis."""
    print("="*80)
    print("ComplexMech Difference Analysis - Models vs hide_all Baseline")
    print("Computing INDIVIDUAL-LEVEL differences with hide fractions preserved")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Define checkpoint keys
    checkpoint_keys = [
        "final_earlytest_16773480.0",  # Model 80 (IDK)
        "final_earlytest_16773478.0",  # Model 78 (No graph)
        "final_earlytest_hide_all_16771679.0",  # Baseline (hide_all)
    ]
    
    # Load individual results
    results_dict = load_individual_results(checkpoint_base, checkpoint_keys)
    
    if not results_dict:
        print("No results found!")
        return
    
    print(f"\nLoaded {len(results_dict)} checkpoints")
    
    # Process each metric with mean, median, AND cleaned mean (outliers removed)
    for stat_type in ['mean', 'median', 'cleaned_mean']:
        use_median = (stat_type == 'median')
        remove_outliers = (stat_type == 'cleaned_mean')
        
        if stat_type == 'cleaned_mean':
            stat_name = "CLEANED MEAN (5th-95th percentile)"
        else:
            stat_name = "MEDIAN" if use_median else "MEAN"
        
        print("\n" + "="*80)
        print(f"Computing {stat_name}-based differences")
        print("="*80)
        
        for metric in ['nll', 'mse', 'r2']:
            print("\n" + "-"*80)
            print(f"Processing {metric.upper()} with {stat_name}")
            print("-"*80)
            
            # Compute individual-level differences (preserving hide fractions)
            diff_df = compute_metric_differences_with_bootstrap(
                results_dict, metric=metric, use_median=use_median, remove_outliers=remove_outliers
            )
            
            if diff_df.empty:
                print(f"No difference data computed for {metric}!")
                continue
            
            print(f"Computed individual-level differences: {len(diff_df)} configurations")
            print(f"  Unique hide fractions: {sorted(diff_df['hide_fraction'].dropna().unique())}")
            print(f"  Unique variants: {sorted(diff_df['variant'].unique())}")
            
            # Aggregate across path variants (but keep hide fractions separate)
            agg_df = compute_aggregated_differences(diff_df, use_median=use_median, remove_outliers=remove_outliers)
            
            if not agg_df.empty:
                print(f"Aggregated to: {len(agg_df)} model-node-hide combinations")
                print(f"  Hide fractions in aggregated data: {sorted(agg_df['hide_fraction'].unique())}")
                
                # Plot aggregated differences BY HIDE FRACTION (separate subplots per node count)
                # Use all available node counts
                available_node_counts = sorted(agg_df['node_count'].unique())
                prefix = f"complexmech_{stat_type}"
                plot_metric_differences(agg_df, metric, 
                                      node_counts_to_compare=available_node_counts,
                                      output_prefix=prefix)
            else:
                print(f"No aggregated data for {metric}")
            
            # Fully aggregate across path variants AND node counts
            fully_agg_df = compute_fully_aggregated_differences(diff_df, use_median=use_median, remove_outliers=remove_outliers)
            
            if not fully_agg_df.empty:
                print(f"Fully aggregated to: {len(fully_agg_df)} model-hide combinations")
                
                # Plot fully aggregated differences (single plot)
                prefix = f"complexmech_{stat_type}"
                plot_fully_aggregated_differences(fully_agg_df, metric, output_prefix=prefix)
            else:
                print(f"No fully aggregated data for {metric}")
    
    print("\n" + "="*80)
    print(f"Analysis complete! All plots saved to: {output_dir}")
    print("="*80)
    
    print("\n" + "="*80)
    print(f"Analysis complete! All plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
