#!/usr/bin/env python3
"""
Metric Difference Analysis Script for ComplexMech Benchmark

This script analyzes ComplexMech benchmark results by computing the difference in metrics
(NLL, MSE, R²) between models 78/80 and the hide_all baseline model, then generates 
box plots showing the improvement (or degradation) relative to baseline.

Usage:
    python generate_metric_difference_plots.py
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
checkpoint_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints"
checkpoint_base = Path(checkpoint_path)

# Create timestamped output directory with subfolders for each metric
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = Path(__file__).parent / "plots" / f"metric_differences_{timestamp}"
base_output_dir.mkdir(parents=True, exist_ok=True)

# Create subfolders for each metric
nll_output_dir = base_output_dir / "nll"
mse_output_dir = base_output_dir / "mse"
r2_output_dir = base_output_dir / "r2"

nll_output_dir.mkdir(exist_ok=True)
mse_output_dir.mkdir(exist_ok=True)
r2_output_dir.mkdir(exist_ok=True)

# Styling
sns.set_style("whitegrid")

# Define model order and colors
MODEL_ORDER = ['78', '80']

MODEL_COLORS = {
    '78': '#cc4778',  # Pink/red (no knowledge)
    '80': '#21918c',  # Teal color (partial knowledge)
}

MODEL_DISPLAY_NAMES = {
    '78': 'No Graph (78)',
    '80': 'Partial Graph (80)',
}


def extract_model_name(checkpoint_name):
    """Extract model name from checkpoint directory name."""
    if 'hide_all' in checkpoint_name:
        return 'hide_all'
    elif '16773478' in checkpoint_name:  # No graph model
        return '78'
    else:
        return '80'


def bootstrap_confidence_interval(data, confidence_level=0.95, n_bootstrap=1000):
    """Compute bootstrap confidence interval for the mean of data."""
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Original mean
    original_mean = np.mean(data)
    
    # Bootstrap resampling - vectorized for speed
    np.random.seed(42)  # For reproducibility
    bootstrap_samples = np.random.choice(data, size=(n_bootstrap, n), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return original_mean, ci_lower, ci_upper


def load_individual_results(checkpoint_base, checkpoint_dirs):
    """Load individual sample results for bootstrap analysis."""
    results_dict = {}
    
    print("Loading individual results...")
    for ckpt_dir in tqdm(checkpoint_dirs, desc="Loading checkpoints"):
        results_dir = Path(ckpt_dir)
        
        if not results_dir.exists():
            print(f"Missing results dir: {results_dir}")
            continue
        
        checkpoint_name = results_dir.parent.name
        checkpoint_results = {}
        
        # Look for individual result files
        node_counts = [2, 5, 10, 20, 35, 50]
        variant_suffixes = ['path_YT', 'path_TY', 'path_independent_TY']
        
        for node_count in node_counts:
            # Base variant
            base_file = results_dir / f"individual_{node_count}nodes.json"
            if base_file.exists():
                try:
                    with open(base_file, 'r') as f:
                        checkpoint_results[('base', node_count, None)] = json.load(f)
                except Exception as e:
                    print(f"Error loading {base_file}: {e}")
            
            # Path variants with hide fractions
            for variant_suffix in variant_suffixes:
                # Full graph knowledge (no hide)
                variant_file_no_hide = results_dir / f"individual_{node_count}nodes_{variant_suffix}.json"
                if variant_file_no_hide.exists():
                    try:
                        with open(variant_file_no_hide, 'r') as f:
                            checkpoint_results[(variant_suffix, node_count, None)] = json.load(f)
                    except Exception as e:
                        print(f"Error loading {variant_file_no_hide}: {e}")
                
                # With hide fractions
                for hide_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    variant_file = results_dir / f"individual_{node_count}nodes_{variant_suffix}_hide{hide_frac}.json"
                    if variant_file.exists():
                        try:
                            with open(variant_file, 'r') as f:
                                checkpoint_results[(variant_suffix, node_count, hide_frac)] = json.load(f)
                        except Exception as e:
                            print(f"Error loading {variant_file}: {e}")
        
        if checkpoint_results:
            results_dict[checkpoint_name] = checkpoint_results
            print(f"  Loaded: {checkpoint_name} ({len(checkpoint_results)} configs)")
    
    return results_dict


def compute_metric_differences_with_bootstrap(results_dict, metric='nll', confidence_level=0.95):
    """
    Compute metric differences with bootstrap confidence intervals.
    
    For each (variant, node_count, hide) combination:
    1. Get baseline (hide_all) metric values
    2. Get model (78/80) metric values
    3. Compute difference (baseline - model) for positive = improvement
       Exception: For R², compute (model - baseline) since higher is better
    4. Bootstrap confidence intervals
    """
    print(f"\nComputing {metric.upper()} differences...")
    difference_data = []
    
    # First, organize baseline data by (variant, node_count, hide)
    baseline_data = {}  # {(variant, node_count, hide): [metric_values]}
    model_data = {}     # {model: {(variant, node_count, hide): [metric_values]}}
    
    for checkpoint_name, checkpoint_results in results_dict.items():
        model_name = extract_model_name(checkpoint_name)
        
        for (variant, node_count, hide), individual_results in checkpoint_results.items():
            metric_values = [sample[metric] for sample in individual_results if metric in sample]
            
            if model_name == 'hide_all':
                baseline_data[(variant, node_count, hide)] = metric_values
            else:
                if model_name not in model_data:
                    model_data[model_name] = {}
                model_data[model_name][(variant, node_count, hide)] = metric_values
    
    print(f"  Baseline configs: {len(baseline_data)}")
    print(f"  Models: {list(model_data.keys())}")
    
    # Compute differences for each model relative to baseline
    total_comparisons = sum(len(configs) for configs in model_data.values())
    
    with tqdm(total=total_comparisons, desc=f"  Computing {metric} differences") as pbar:
        for model_name, model_configs in model_data.items():
            for (variant, node_count, hide), model_metric_values in model_configs.items():
                # Get corresponding baseline values
                if (variant, node_count, hide) in baseline_data:
                    baseline_metric_values = baseline_data[(variant, node_count, hide)]
                    
                    if len(model_metric_values) == 0 or len(baseline_metric_values) == 0:
                        pbar.update(1)
                        continue
                    
                    # Compute pairwise differences
                    # For MSE and NLL: baseline - model (positive = improvement)
                    # For R²: model - baseline (positive = improvement, since higher is better)
                    if metric == 'r2':
                        differences = [m - b for m, b in zip(model_metric_values, baseline_metric_values)]
                    else:
                        differences = [b - m for b, m in zip(baseline_metric_values, model_metric_values)]
                    
                    # Bootstrap confidence interval
                    mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                        differences, confidence_level=confidence_level
                    )
                    
                    difference_data.append({
                        'model': model_name,
                        'variant': variant,
                        'node_count': node_count,
                        'hide': hide,
                        'n_samples': len(differences),
                        f'{metric}_difference_mean': mean_diff,
                        f'{metric}_difference_ci_lower': ci_lower,
                        f'{metric}_difference_ci_upper': ci_upper,
                        f'{metric}_raw_differences': differences,  # Store for plotting
                    })
                
                pbar.update(1)
    
    return pd.DataFrame(difference_data)


def compute_aggregated_differences(results_dict, metric='nll', confidence_level=0.95):
    """
    Compute metric differences aggregated across path variants.
    
    Combines path_TY, path_YT, and path_independent_TY for each (model, node_count, hide).
    """
    print(f"\nComputing aggregated {metric.upper()} differences...")
    difference_data = []
    
    path_variants = ['path_TY', 'path_YT', 'path_independent_TY']
    
    # Organize baseline data
    baseline_data = {}  # {(node_count, hide): {variant: [values]}}
    model_data = {}     # {model: {(node_count, hide): {variant: [values]}}}
    
    for checkpoint_name, checkpoint_results in results_dict.items():
        model_name = extract_model_name(checkpoint_name)
        
        for (variant, node_count, hide), individual_results in checkpoint_results.items():
            if variant not in path_variants:
                continue
            
            metric_values = [sample[metric] for sample in individual_results if metric in sample]
            
            if model_name == 'hide_all':
                key = (node_count, hide)
                if key not in baseline_data:
                    baseline_data[key] = {}
                baseline_data[key][variant] = metric_values
            else:
                if model_name not in model_data:
                    model_data[model_name] = {}
                key = (node_count, hide)
                if key not in model_data[model_name]:
                    model_data[model_name][key] = {}
                model_data[model_name][key][variant] = metric_values
    
    # Compute aggregated differences
    total_comparisons = sum(len(configs) for configs in model_data.values())
    
    with tqdm(total=total_comparisons, desc=f"  Aggregating {metric} differences") as pbar:
        for model_name, model_configs in model_data.items():
            for (node_count, hide), variants_dict in model_configs.items():
                # Combine all variants for this (node_count, hide)
                all_differences = []
                
                for variant in path_variants:
                    if variant in variants_dict and (node_count, hide) in baseline_data:
                        if variant in baseline_data[(node_count, hide)]:
                            model_vals = variants_dict[variant]
                            baseline_vals = baseline_data[(node_count, hide)][variant]
                            
                            if len(model_vals) > 0 and len(baseline_vals) > 0:
                                # Compute differences
                                if metric == 'r2':
                                    diffs = [m - b for m, b in zip(model_vals, baseline_vals)]
                                else:
                                    diffs = [b - m for b, m in zip(baseline_vals, model_vals)]
                                all_differences.extend(diffs)
                
                if len(all_differences) > 0:
                    # Bootstrap confidence interval on combined differences
                    mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                        all_differences, confidence_level=confidence_level
                    )
                    
                    difference_data.append({
                        'model': model_name,
                        'variant': 'aggregated_paths',
                        'node_count': node_count,
                        'hide': hide,
                        'n_samples': len(all_differences),
                        f'{metric}_difference_mean': mean_diff,
                        f'{metric}_difference_ci_lower': ci_lower,
                        f'{metric}_difference_ci_upper': ci_upper,
                        f'{metric}_raw_differences': all_differences,
                    })
                
                pbar.update(1)
    
    return pd.DataFrame(difference_data)


def plot_metric_differences(df, metric='nll', node_counts_to_compare=None, 
                           title_suffix="", output_prefix="", output_dir=None):
    """Create box plots showing metric differences relative to hide_all baseline."""
    if output_dir is None:
        if metric == 'nll':
            output_dir = nll_output_dir
        elif metric == 'mse':
            output_dir = mse_output_dir
        else:
            output_dir = r2_output_dir
    
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    variants = sorted(df['variant'].unique())
    
    variant_display_names = {
        'base': 'Base Causal Structure',
        'path_YT': 'Y → T Path',
        'path_TY': 'T → Y Path',
        'path_independent_TY': 'T ⊥ Y (Independent)',
        'aggregated_paths': 'Aggregated Paths (TY + YT + Independent)',
    }
    
    metric_titles = {
        'nll': 'NLL',
        'mse': 'MSE',
        'r2': 'R²',
    }
    
    print(f"\nGenerating {metric.upper()} difference plots for {len(variants)} variants...")
    
    # Create separate figure for each variant
    for variant in tqdm(variants, desc=f"  Plotting {metric} variants"):
        variant_data = df[df['variant'] == variant].copy()
        
        if len(variant_data) == 0:
            print(f"No data for variant: {variant}")
            continue
        
        # Create figure
        fig, axes = plt.subplots(1, len(node_counts_to_compare), 
                                figsize=(5.5 * len(node_counts_to_compare), 6))
        
        # Handle single column case
        if len(node_counts_to_compare) == 1:
            axes = [axes]
        
        for node_idx, node_count in enumerate(node_counts_to_compare):
            ax = axes[node_idx]
            
            # Filter data for this node count
            node_data = variant_data[variant_data['node_count'] == node_count].copy()
            
            if len(node_data) == 0:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{node_count} Nodes')
                continue
            
            # Get unique hide values and models
            hide_values = sorted([h for h in node_data['hide'].unique() if h is not None])
            if not hide_values:
                hide_values = [None]
            
            models = [m for m in MODEL_ORDER if m in node_data['model'].unique()]
            
            if not models:
                ax.text(0.5, 0.5, f'No models', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{node_count} Nodes')
                continue
            
            box_data = []
            positions = []
            colors = []
            
            # Create boxes grouped by hide, with models side by side
            position_offset = 0
            n_models = len(models)
            group_width = n_models * 0.7
            
            for hide_idx, hide in enumerate(hide_values):
                for local_model_idx, model in enumerate(models):
                    if hide is None:
                        hide_data = node_data[(node_data['hide'].isna()) & (node_data['model'] == model)]
                    else:
                        hide_data = node_data[(node_data['hide'] == hide) & (node_data['model'] == model)]
                    
                    if len(hide_data) == 0:
                        continue
                    
                    # Get difference statistics
                    mean_diff = hide_data[f'{metric}_difference_mean'].values[0]
                    ci_lower = hide_data[f'{metric}_difference_ci_lower'].values[0]
                    ci_upper = hide_data[f'{metric}_difference_ci_upper'].values[0]
                    
                    # Create synthetic box plot data (like in generate_nll_difference_plots.py)
                    synthetic_data = [
                        ci_lower,
                        ci_lower + (mean_diff - ci_lower) * 0.5,
                        mean_diff,
                        mean_diff + (ci_upper - mean_diff) * 0.5,
                        ci_upper
                    ]
                    
                    pos = position_offset + local_model_idx * 0.7
                    positions.append(pos)
                    box_data.append(synthetic_data)
                    colors.append(MODEL_COLORS.get(model, '#808080'))
                
                position_offset += group_width + 0.5
            
            if box_data:
                # Create box plot
                bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                               patch_artist=True, showmeans=False, showfliers=False,
                               medianprops=dict(color='red', linewidth=2),
                               boxprops=dict(linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
                
                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add horizontal line at zero
                ax.axhline(y=0, color='black', linestyle='-', linewidth=3, alpha=0.9)
                
                # Set x-axis labels for hide values
                hide_labels = [f'{h:.2f}' if h is not None else 'Full' for h in hide_values]
                group_centers = []
                pos_offset = 0
                for _ in hide_values:
                    center = pos_offset + (n_models - 1) * 0.7 / 2
                    group_centers.append(center)
                    pos_offset += group_width + 0.5
                
                ax.set_xticks(group_centers)
                ax.set_xticklabels(hide_labels, rotation=0, fontsize=18)
                ax.set_xlabel('Hide Fraction', fontsize=18)
            
            # Set titles and labels
            ax.set_title(f'{node_count} Nodes', fontsize=18, fontweight='bold')
            ax.set_ylabel(f'{metric_titles[metric]} Improvement', fontsize=18, fontweight='bold')
            
            ax.tick_params(axis='y', labelsize=18)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Add legend for models
        if len(models) > 1:
            legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLORS.get(m, '#808080'), 
                                            alpha=0.7, label=MODEL_DISPLAY_NAMES.get(m, m)) 
                             for m in models]
            fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                      fontsize=18, ncol=len(models), frameon=True)
        
        variant_display = variant_display_names.get(variant, variant)
        if title_suffix:
            fig.suptitle(f'{metric_titles[metric]} Improvement: {variant_display}\n{title_suffix}', 
                        fontsize=20, fontweight='bold', y=1.02)
        else:
            fig.suptitle(f'{metric_titles[metric]} Improvement over hide_all Baseline: {variant_display}', 
                        fontsize=20, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        filename = f"{output_prefix}_{variant}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    print(f"  ✓ Saved {len(variants)} {metric.upper()} plots to {output_dir}")


def main():
    """Main function to generate all metric difference plots."""
    print("="*80)
    print("ComplexMech Metric Difference Analysis")
    print("Comparing models 78 and 80 vs hide_all baseline")
    print(f"Output directory: {base_output_dir}")
    print("="*80)
    
    # Load data from all three models
    checkpoint_dirs = [
        checkpoint_base / "final_earlytest_16773480.0" / "complexmech_idk_high",  # Model 80
        checkpoint_base / "final_earlytest_16773478.0" / "complexmech_idk_high",  # Model 78
        checkpoint_base / "final_earlytest_hide_all_16771679.0" / "complexmech_idk_high",  # hide_all (baseline)
    ]
    
    results_dict = load_individual_results(checkpoint_base, checkpoint_dirs)
    
    if not results_dict:
        print("No results found!")
        return
    
    print(f"\nLoaded data from {len(results_dict)} checkpoints")
    
    # Process each metric
    for metric in ['nll', 'mse', 'r2']:
        print("\n" + "="*80)
        print(f"{metric.upper()} DIFFERENCE ANALYSIS")
        print("="*80)
        
        # Compute differences for individual variants
        diff_df = compute_metric_differences_with_bootstrap(results_dict, metric=metric)
        
        if not diff_df.empty:
            print(f"  Computed {len(diff_df)} difference points")
            
            # Plot individual variants
            plot_metric_differences(diff_df, metric=metric,
                                  node_counts_to_compare=[2, 5, 10, 20, 35, 50],
                                  output_prefix=f"{metric}_difference")
        
        # Compute aggregated differences
        agg_df = compute_aggregated_differences(results_dict, metric=metric)
        
        if not agg_df.empty:
            print(f"  Computed {len(agg_df)} aggregated difference points")
            
            # Plot aggregated
            plot_metric_differences(agg_df, metric=metric,
                                  node_counts_to_compare=[2, 5, 10, 20, 35, 50],
                                  output_prefix=f"{metric}_difference_aggregated")
    
    print("\n" + "="*80)
    print(f"✓ All plots saved to: {base_output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
