#!/usr/bin/env python3
"""
NLL Difference Analysis Script

This script analyzes benchmark results by computing the difference in NLL between
all approaches and the baseline model, then generates box plots showing the 
improvement (or degradation) relative to baseline.

Usage:
    python generate_nll_difference_plots.py
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


# Configuration
checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints"
checkpoint_base = Path(checkpoint_path)

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parent / "plots" / f"nll_differences_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 10)

# Define model order and colors (excluding baseline since it will be the reference)
MODEL_ORDER = [
    'hardatt', 'ancestor_hardatt',
    'softatt', 'ancestor_softatt', 
    'gcn', 'ancestor_gcn',
    'gcn_and_hartatt', 'ancestor_gcn_and_hartatt',
    'gcn_and_softatt', 'gcn_and_softatt_fixed', 'ancestor_gcn_and_softatt'
]

MODEL_COLORS = {
    'hardatt': '#31688e',
    'ancestor_hardatt': '#5a9bc4',
    'softatt': '#35b779',
    'ancestor_softatt': '#6dcd9f',
    'gcn': '#fde724',
    'ancestor_gcn': '#fee761',
    'gcn_and_hartatt': '#90d743',
    'ancestor_gcn_and_hartatt': '#ade577',
    'gcn_and_softatt': '#21918c',
    'gcn_and_softatt_fixed': '#cc4778',
    'ancestor_gcn_and_softatt': '#46c0b6'
}


def load_individual_results(checkpoint_base, checkpoint_keys):
    """Load individual sample results for bootstrap analysis."""
    results_dict = {}
    
    for ckpt_key in checkpoint_keys:
        ckpt_dir = checkpoint_base / ckpt_key
        results_dir = ckpt_dir / "lingaus_final"
        
        if not results_dir.exists():
            print(f"Missing results dir: {results_dir}")
            continue
        
        checkpoint_results = {}
        
        # Look for individual result files
        for node_count in [2, 5, 10, 20, 35, 50]:
            # Check for base variant
            base_file = results_dir / f"individual_{node_count}nodes.json"
            if base_file.exists():
                if 'base' not in checkpoint_results:
                    checkpoint_results['base'] = {}
                try:
                    with open(base_file, 'r') as f:
                        data = json.load(f)
                        checkpoint_results['base'][str(node_count)] = data
                except Exception as e:
                    print(f"Error loading {base_file}: {e}")
            
            # Check for path variants
            for variant_suffix in ['path_YT', 'path_TY', 'path_independent_TY']:
                variant_file = results_dir / f"individual_{node_count}nodes_{variant_suffix}.json"
                if variant_file.exists():
                    if variant_suffix not in checkpoint_results:
                        checkpoint_results[variant_suffix] = {}
                    try:
                        with open(variant_file, 'r') as f:
                            data = json.load(f)
                            checkpoint_results[variant_suffix][str(node_count)] = data
                    except Exception as e:
                        print(f"Error loading {variant_file}: {e}")
        
        if checkpoint_results:
            results_dict[ckpt_key] = checkpoint_results
            print(f"Loaded individual results: {ckpt_key} ({len(checkpoint_results)} variants)")
    
    return results_dict


def bootstrap_confidence_interval(data, confidence_level=0.90, n_bootstrap=10000):
    """Compute bootstrap confidence interval for the mean of data."""
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Original mean
    original_mean = np.mean(data)
    
    # Bootstrap resampling
    bootstrap_means = []
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return original_mean, ci_lower, ci_upper


def compute_nll_differences_with_bootstrap(results_dict, confidence_level=0.90):
    """Compute NLL differences with bootstrap confidence intervals."""
    difference_data = []
    
    # First, organize data by variant and node count
    baseline_data = {}  # {variant: {node_count: [nll_values]}}
    model_data = {}     # {model: {variant: {node_count: [nll_values]}}}
    
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        
        for variant, variant_data in ckpt_results.items():
            for node_count, individual_results in variant_data.items():
                nll_values = [sample['nll'] for sample in individual_results]
                
                if model_name == 'baseline':
                    if variant not in baseline_data:
                        baseline_data[variant] = {}
                    baseline_data[variant][node_count] = nll_values
                else:
                    if model_name not in model_data:
                        model_data[model_name] = {}
                    if variant not in model_data[model_name]:
                        model_data[model_name][variant] = {}
                    model_data[model_name][variant][node_count] = nll_values
    
    # Compute differences for each model relative to baseline
    for model_name, model_variants in model_data.items():
        for variant, variant_node_data in model_variants.items():
            for node_count, model_nll_values in variant_node_data.items():
                # Get corresponding baseline values
                if variant in baseline_data and node_count in baseline_data[variant]:
                    baseline_nll_values = baseline_data[variant][node_count]
                    
                    # Ensure same number of samples
                    min_samples = min(len(model_nll_values), len(baseline_nll_values))
                    model_nll_subset = model_nll_values[:min_samples]
                    baseline_nll_subset = baseline_nll_values[:min_samples]
                    
                    # Compute differences (baseline - model, so positive = improvement)
                    nll_differences = np.array(baseline_nll_subset) - np.array(model_nll_subset)
                    
                    # Bootstrap confidence interval for the mean difference
                    mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                        nll_differences, confidence_level=confidence_level
                    )
                    
                    difference_data.append({
                        'model': model_name,
                        'variant': variant,
                        'node_count': int(node_count),
                        'nll_difference_mean': float(mean_diff),
                        'nll_difference_ci_lower': float(ci_lower),
                        'nll_difference_ci_upper': float(ci_upper),
                        'n_samples': len(nll_differences),
                        'baseline_nll_mean': float(np.mean(baseline_nll_subset)),
                        'model_nll_mean': float(np.mean(model_nll_subset))
                    })
    
    return pd.DataFrame(difference_data)


def extract_model_name(checkpoint_name):
    """Extract model name from checkpoint directory name."""
    parts = checkpoint_name.split('_')
    model_name_parts = []
    is_ancestor = False
    
    if "ancestor" in parts:
        is_ancestor = True
    
    start_collecting = False
    for i, part in enumerate(parts):
        if part == "benchmarked":
            start_collecting = True
            continue
        
        if "node" in part and part.replace("node", "").isdigit():
            start_collecting = True
            continue
        
        if start_collecting and not part.replace('.', '').replace('-', '').isdigit():
            model_name_parts.append(part)
    
    model_name = '_'.join(model_name_parts) if model_name_parts else 'unknown'
    
    if is_ancestor and model_name != 'unknown':
        model_name = f'ancestor_{model_name}'
    
    return model_name


def extract_model_name(checkpoint_name):
    """Extract model name from checkpoint directory name."""
    parts = checkpoint_name.split('_')
    model_name_parts = []
    is_ancestor = False
    
    # Check if this is an ancestor model
    if "ancestor" in parts:
        is_ancestor = True
    
    start_collecting = False
    for i, part in enumerate(parts):
        # Start collecting after "benchmarked" keyword
        if part == "benchmarked":
            start_collecting = True
            continue
        
        # For ancestor pattern: start collecting after "Xnode" (e.g., "5node")
        if "node" in part and part.replace("node", "").isdigit():
            start_collecting = True
            continue
        
        # Collect parts that are not PIDs (numbers with dots)
        if start_collecting and not part.replace('.', '').replace('-', '').isdigit():
            model_name_parts.append(part)
    
    model_name = '_'.join(model_name_parts) if model_name_parts else 'unknown'
    
    # Add "ancestor_" prefix if this is an ancestor model
    if is_ancestor and model_name != 'unknown':
        model_name = f'ancestor_{model_name}'
    
    return model_name


def plot_nll_differences(df, node_counts_to_compare=None, title_suffix="", output_prefix="nll_diff"):
    """Create box plots showing NLL differences relative to baseline using bootstrap CI bounds."""
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    variants = sorted(df['variant'].unique())
    
    variant_display_names = {
        'base': 'Base Causal Structure',
        'path_YT': 'Y → T Path',
        'path_TY': 'T → Y Path', 
        'path_independent_TY': 'T ⊥ Y (Independent)'
    }
    
    # Create separate figure for each variant
    for variant in variants:
        variant_data = df[df['variant'] == variant].copy()
        
        if len(variant_data) == 0:
            print(f"No data for variant: {variant}")
            continue
        
        fig, axes = plt.subplots(1, len(node_counts_to_compare), figsize=(20, 8))
        
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
            
            # Get unique models and sort by predefined order
            available_models = node_data['model'].unique()
            models = [m for m in MODEL_ORDER if m in available_models]
            
            if not models:
                ax.text(0.5, 0.5, f'No models', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{node_count} Nodes')
                continue
            
            box_data = []
            positions = []
            colors = []
            mean_values = []
            
            for model_idx, model in enumerate(models):
                model_data = node_data[node_data['model'] == model]
                
                if len(model_data) == 0:
                    continue
                
                # Extract difference statistics
                mean_diff = model_data['nll_difference_mean'].values[0]
                ci_lower = model_data['nll_difference_ci_lower'].values[0]
                ci_upper = model_data['nll_difference_ci_upper'].values[0]
                
                # Create synthetic distribution from bootstrap CI bounds (same as original plots)
                synthetic_data = [
                    ci_lower,
                    ci_lower + (mean_diff - ci_lower) * 0.5,  # Q1
                    mean_diff,  # Median
                    mean_diff + (ci_upper - mean_diff) * 0.5,  # Q3
                    ci_upper
                ]
                
                box_data.append(synthetic_data)
                positions.append(model_idx)
                colors.append(MODEL_COLORS.get(model, '#808080'))
                mean_values.append(mean_diff)
            
            if box_data:
                # Create box plot (same style as original plots)
                bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                               patch_artist=True, showmeans=False,
                               medianprops=dict(color='red', linewidth=2),
                               boxprops=dict(linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
                
                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add mean value labels
                for pos, mean_val in zip(positions, mean_values):
                    ax.text(pos, mean_val, f'{mean_val:.4f}', 
                           ha='center', va='bottom', fontsize=7, rotation=0,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor='none', alpha=0.7))
                
                # Add horizontal line at zero (no difference)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
            
            # Set x-axis labels
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
            
            ax.set_title(f'{node_count} Nodes', fontsize=12, fontweight='bold')
            ax.set_ylabel('NLL Difference\n(Baseline - Model)', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Main title
        variant_display = variant_display_names.get(variant, variant)
        fig.suptitle(f'NLL Improvement over Baseline: {variant_display}\n{title_suffix}\n(90% Bootstrap CI)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        
        # Save plot
        filename = f"{output_prefix}_{variant}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)


def print_nll_difference_statistics(df, title_suffix="", output_prefix="nll_diff"):
    """Print summary statistics for NLL differences and save to file."""
    output_file = output_dir / f"{output_prefix}_summary.txt"
    
    with open(output_file, 'w') as f:
        def write_line(line=""):
            print(line)
            f.write(line + "\n")
        
        write_line("="*80)
        write_line(f"NLL DIFFERENCE ANALYSIS - {title_suffix}")
        write_line("Positive values indicate improvement over baseline")
        write_line("Negative values indicate degradation relative to baseline")
        write_line("90% Bootstrap Confidence Intervals")
        write_line("="*80)
        
        variants = sorted(df['variant'].unique())
        for variant in variants:
            write_line(f"\n{'='*80}")
            write_line(f"VARIANT: {variant}")
            write_line(f"{'='*80}")
            variant_data = df[df['variant'] == variant]
            
            for model in sorted(variant_data['model'].unique()):
                model_data = variant_data[variant_data['model'] == model]
                write_line(f"\n{str(model).upper()}")
                write_line("-"*80)
                
                mean_diff = model_data['nll_difference_mean'].values[0]
                ci_lower = model_data['nll_difference_ci_lower'].values[0]
                ci_upper = model_data['nll_difference_ci_upper'].values[0]
                n_samples = model_data['n_samples'].values[0]
                baseline_mean = model_data['baseline_nll_mean'].values[0]
                model_mean = model_data['model_nll_mean'].values[0]
                
                write_line(f"Mean difference: {mean_diff:.6f}")
                write_line(f"90% Bootstrap CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                write_line(f"Sample size: {n_samples}")
                write_line(f"Baseline NLL: {baseline_mean:.6f}")
                write_line(f"Model NLL: {model_mean:.6f}")
                
                # Check if CI includes zero (statistical significance at 90% level)
                significant = not (ci_lower <= 0 <= ci_upper)
                
                if significant:
                    if mean_diff > 0:
                        interpretation = "SIGNIFICANT IMPROVEMENT (90% confidence)"
                    else:
                        interpretation = "SIGNIFICANT DEGRADATION (90% confidence)"
                else:
                    interpretation = "NO SIGNIFICANT DIFFERENCE (90% confidence)"
                
                write_line(f"Statistical significance: {interpretation}")
                write_line(f"  (CI {'excludes' if significant else 'includes'} zero)")
        
        write_line("\n" + "="*80)
        write_line("Statistical significance based on 90% bootstrap confidence intervals")
        write_line("Differences computed from individual sample pairs (bootstrap resampling)")
        write_line("="*80)
    
    print(f"Saved summary: {output_file}")


def main():
    """Main function to generate NLL difference analysis with bootstrap CIs."""
    print("="*80)
    print("Generating NLL Difference Analysis with 90% Bootstrap CIs")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Define checkpoint keys (50-node models)
    checkpoint_keys = [
        # Baseline
        "lingaus_50node_benchmarked_baseline_16715917.0",
        # Regular models
        "lingaus_50node_benchmarked_gcn_16715920.0",
        "lingaus_50node_benchmarked_gcn_and_hartatt_16715918.0",
        "lingaus_50node_benchmarked_gcn_and_softatt_16715919.0",
        "lingaus_50node_benchmarked_hardatt_16715921.0",
        "lingaus_50node_benchmarked_softatt_16715922.0",
        # Ancestor models
        "lingaus_ancestor_50node_baseline_16723333.0",
        "lingaus_ancestor_50node_gcn_16723348.0",
        "lingaus_ancestor_50node_gcn_and_hartatt_16723334.0",
        "lingaus_ancestor_50node_gcn_and_softatt_16723335.0",
        "lingaus_ancestor_50node_hardatt_16723337.0",
        "lingaus_ancestor_50node_softatt_16723338.0",
    ]
    
    print("Loading individual sample results for bootstrap analysis...")
    results_dict = load_individual_results(checkpoint_base, checkpoint_keys)
    
    if not results_dict:
        print("No results found!")
        return
    
    print("Computing NLL differences with bootstrap confidence intervals...")
    diff_df = compute_nll_differences_with_bootstrap(results_dict, confidence_level=0.90)
    
    if diff_df.empty:
        print("No difference data computed!")
        return
    
    print(f"Computed differences for {len(diff_df)} model-variant-node combinations")
    print(f"Models: {sorted(diff_df['model'].unique())}")
    print(f"Node counts: {sorted(diff_df['node_count'].unique())}")
    print(f"Variants: {sorted(diff_df['variant'].unique())}")
    
    # Generate plots
    plot_nll_differences(diff_df, 
                        node_counts_to_compare=[2, 5, 20, 50],
                        title_suffix="50-Node Training Models vs Baseline",
                        output_prefix="nll_difference")
    
    # Generate statistics
    print_nll_difference_statistics(diff_df,
                                  title_suffix="50-Node Training Models vs Baseline", 
                                  output_prefix="nll_difference")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! All plots saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()