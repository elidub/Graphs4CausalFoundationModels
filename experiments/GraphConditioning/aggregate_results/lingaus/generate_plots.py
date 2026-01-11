#!/usr/bin/env python3
"""
Benchmark Analysis Script

This script analyzes benchmark results by combining regular and ancestor models,
then generates box plots and summary statistics. Saves plots to timestamped folders.

Usage:
    python benchmark_analysis_script.py --checkpoint-path /path/to/checkpoints --title "My Analysis"
"""

import argparse
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
output_dir = Path(__file__).parent / "plots" / timestamp

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (30, 8)


# Define model order and colors
MODEL_ORDER = [
    'baseline', 
    'hardatt', 'ancestor_hardatt',
    'softatt', 'ancestor_softatt',
    'gcn', 'ancestor_gcn',
    'gcn_and_hartatt', 'ancestor_gcn_and_hartatt',
    'gcn_and_softatt', 'gcn_and_softatt_fixed', 'ancestor_gcn_and_softatt'
]

MODEL_COLORS = {
    'baseline': '#440154',
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


def load_benchmark_results(checkpoint_base, pattern, eval_step=None):
    """Load all benchmark JSON files matching the given pattern."""
    results_dict = {}
    
    for ckpt_dir in sorted(checkpoint_base.glob(pattern)):
        checkpoint_results = {}
        
        # Determine which directory to load from
        if eval_step is not None:
            results_dir = ckpt_dir / f"lingaus_eval_step{eval_step}"
        else:
            results_dir = ckpt_dir / "lingaus_final"
        
        if not results_dir.exists():
            print(f"Missing results dir: {results_dir}")
            continue
        
        # Check for new format with multiple variant files
        has_variant_files = False
        variant_files = {}
        
        for node_count in [2, 5, 10, 20, 35, 50]:
            # Check for base variant (no suffix)
            base_file = results_dir / f"aggregated_{node_count}nodes.json"
            if base_file.exists():
                has_variant_files = True
                if 'base' not in variant_files:
                    variant_files['base'] = []
                variant_files['base'].append((node_count, base_file))
            
            # Check for path variants
            for variant_suffix in ['path_YT', 'path_TY', 'path_independent_TY']:
                variant_file = results_dir / f"aggregated_{node_count}nodes_{variant_suffix}.json"
                if variant_file.exists():
                    has_variant_files = True
                    if variant_suffix not in variant_files:
                        variant_files[variant_suffix] = []
                    variant_files[variant_suffix].append((node_count, variant_file))
        
        if has_variant_files:
            # New format: load from individual variant files
            for variant, files in variant_files.items():
                variant_data = {}
                for node_count, filepath in files:
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            variant_data[str(node_count)] = data
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                checkpoint_results[variant] = variant_data
            
            if checkpoint_results:
                results_dict[ckpt_dir.name] = checkpoint_results
                print(f"Loaded: {ckpt_dir.name} ({len(checkpoint_results)} variants)")
        else:
            # Old format: single lingaus_benchmark_final.json file
            json_path = results_dir / "lingaus_benchmark_final.json"
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        # Wrap in 'base' variant for consistency
                        checkpoint_results['base'] = data
                        results_dict[ckpt_dir.name] = checkpoint_results
                        print(f"Loaded: {ckpt_dir.name} (legacy format)")
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
            else:
                print(f"Missing JSON in: {ckpt_dir.name}")
    
    print(f"\nTotal loaded: {len(results_dict)} results")
    return results_dict


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


def create_dataframe(results_dict):
    """Convert results dictionary to pandas DataFrame."""
    data_rows = []
    
    for checkpoint_name, checkpoint_data in results_dict.items():
        model_name = extract_model_name(checkpoint_name)
        
        # Iterate through variants
        for variant, variant_data in checkpoint_data.items():
            # Iterate through each node count
            for node_count, node_data in variant_data.items():
                row = {
                    'model': model_name,
                    'variant': variant,
                    'node_count': int(node_count),
                }
                
                # Extract metrics (mse, r2, nll, etc.)
                for metric_name, metric_values in node_data.items():
                    if metric_name != 'metadata' and isinstance(metric_values, dict):
                        for stat_name, stat_value in metric_values.items():
                            if isinstance(stat_value, (int, float)):
                                row[f"{metric_name}_{stat_name}"] = stat_value
                
                data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    # Drop rows where key metrics are NaN (incomplete runs)
    print(f"Original shape: {df.shape}")
    df = df.dropna(subset=['mse_mean', 'r2_mean', 'nll_mean'])
    print(f"After dropping NaN rows: {df.shape}")
    
    # Apply consistent model order
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)
    
    return df


def plot_box_comparison_separate_variants(df, node_counts_to_compare=None, title_suffix="", output_prefix=""):
    """Create separate box plot figures for each variant."""
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    variants = sorted(df['variant'].unique())
    metrics_to_plot = ['mse', 'r2', 'nll']
    metric_titles = ['MSE', 'R²', 'NLL']
    
    # Variant display names
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
        
        fig, axes = plt.subplots(len(metrics_to_plot), len(node_counts_to_compare), 
                                figsize=(20, 12))
        
        for metric_idx, (metric, metric_title) in enumerate(zip(metrics_to_plot, metric_titles)):
            for node_idx, node_count in enumerate(node_counts_to_compare):
                ax = axes[metric_idx, node_idx]
                
                # Filter data for this node count
                node_data = variant_data[variant_data['node_count'] == node_count].copy()
                
                if len(node_data) == 0:
                    ax.text(0.5, 0.5, f'No data', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{node_count} Nodes')
                    continue
                
                # Sort by model order
                node_data = node_data.sort_values('model')
                
                # Get unique models
                models = node_data['model'].unique()
                n_models = len(models)
                
                metric_col = f'{metric}_mean'
                if metric_col not in node_data.columns:
                    continue
                
                # Store baseline value for reference line
                baseline_val = None
                
                box_data = []
                positions = []
                colors = []
                mean_values = []
                
                for model_idx, model in enumerate(models):
                    model_data = node_data[node_data['model'] == model]
                    
                    if len(model_data) == 0:
                        continue
                    
                    positions.append(model_idx)
                    
                    # Get metric statistics
                    mean_val = model_data[metric_col].values[0]
                    mean_values.append(mean_val)
                    
                    # Store baseline value
                    if model == 'baseline':
                        baseline_val = mean_val
                    
                    # Create synthetic distribution from CI bounds
                    ci_lower_col = f'{metric}_mean_ci_lower'
                    ci_upper_col = f'{metric}_mean_ci_upper'
                    
                    if ci_lower_col in model_data.columns and ci_upper_col in model_data.columns:
                        ci_lower = model_data[ci_lower_col].values[0]
                        ci_upper = model_data[ci_upper_col].values[0]
                        
                        # Create 5-point distribution: [ci_lower, Q1, median(mean), Q3, ci_upper]
                        synthetic_data = [
                            ci_lower,
                            ci_lower + (mean_val - ci_lower) * 0.5,  # Q1
                            mean_val,  # Median
                            mean_val + (ci_upper - mean_val) * 0.5,  # Q3
                            ci_upper
                        ]
                    else:
                        # No CI available, use mean only
                        synthetic_data = [mean_val] * 5
                    
                    box_data.append(synthetic_data)
                    
                    # Get color for model
                    colors.append(MODEL_COLORS.get(str(model), '#808080'))
                
                if box_data and positions:
                    # Create box plot
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
                        ax.text(pos, mean_val, f'{mean_val:.3f}', 
                               ha='center', va='bottom', fontsize=7, rotation=0,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor='none', alpha=0.7))
                
                # Add horizontal line at baseline level
                if baseline_val is not None:
                    ax.axhline(y=baseline_val, color='gray', linestyle='--', 
                             linewidth=2, alpha=0.7, zorder=0)
                
                # Set x-axis labels
                ax.set_xticks(range(n_models))
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
                
                if node_idx == 0:
                    ax.set_ylabel(metric_title, fontsize=11, fontweight='bold')
                
                if metric_idx == 0:
                    ax.set_title(f'{node_count} Nodes', fontsize=11, fontweight='bold')
                
                ax.grid(axis='y', alpha=0.3)
        
        # Create main title with variant info
        variant_name = variant_display_names.get(variant, variant)
        main_title = f'Model Performance Box Plots: {variant_name} (from 95% CI)'
        if title_suffix:
            main_title = f'{main_title} - {title_suffix}'
        plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save plot
        filename = f"{output_prefix}_boxplot_{variant}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)


def print_summary_statistics(df, title_suffix="", output_prefix="", checkpoint_keys=None):
    """Print summary statistics table and save to file."""
    output_file = output_dir / f"{output_prefix}_summary.txt"
    
    with open(output_file, 'w') as f:
        def write_line(line=""):
            print(line)
            f.write(line + "\n")
        
        write_line("="*80)
        if title_suffix:
            write_line(f"SUMMARY STATISTICS BY MODEL - {title_suffix}")
        else:
            write_line("SUMMARY STATISTICS BY MODEL")
        write_line("="*80)
        
        # Write checkpoint paths if provided
        if checkpoint_keys:
            write_line("\nCHECKPOINT PATHS:")
            write_line("-"*80)
            for key in sorted(checkpoint_keys):
                checkpoint_path = checkpoint_base / key
                write_line(f"  {key}")
                write_line(f"    -> {checkpoint_path}")
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
                
                if 'mse_mean' in model_data.columns:
                    write_line(f"MSE   - Mean: {model_data['mse_mean'].mean():.6f} ± {model_data['mse_mean'].std():.6f}")
                if 'r2_mean' in model_data.columns:
                    write_line(f"R²    - Mean: {model_data['r2_mean'].mean():.6f} ± {model_data['r2_mean'].std():.6f}")
                if 'nll_mean' in model_data.columns:
                    write_line(f"NLL   - Mean: {model_data['nll_mean'].mean():.6f} ± {model_data['nll_mean'].std():.6f}")
        
        write_line("\n" + "="*80)
    
    print(f"Saved summary: {output_file}")


def analyze_models(checkpoint_base, regular_keys, ancestor_keys=None, 
                  node_counts_to_compare=None, title_suffix="", output_prefix=""):
    """
    Main analysis function that combines regular and ancestor models.
    
    Parameters
    ----------
    checkpoint_base : str or Path
        Base directory containing checkpoint folders
    regular_keys : list
        List of regular model checkpoint names to include
    ancestor_keys : list, optional
        List of ancestor model checkpoint names to include
    node_counts_to_compare : list, optional
        List of node counts to compare in plots
    title_suffix : str, optional
        Suffix to add to plot titles
    output_prefix : str, optional
        Prefix for output filenames
    """
    checkpoint_base = Path(checkpoint_base)
    
    if node_counts_to_compare is None:
        node_counts_to_compare = [2, 5]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load regular results
    pattern_regular = "*"
    results_dict_regular = load_benchmark_results(checkpoint_base, pattern_regular)
    
    # Filter to only include specified regular keys
    results_dict_regular_filtered = {k: v for k, v in results_dict_regular.items() if k in regular_keys}
    
    # Load ancestor results if provided
    results_dict_ancestor = {}
    if ancestor_keys:
        pattern_ancestor = "*"
        results_dict_ancestor_all = load_benchmark_results(checkpoint_base, pattern_ancestor)
        results_dict_ancestor = {k: v for k, v in results_dict_ancestor_all.items() if k in ancestor_keys}
    
    # Combine both dictionaries
    results_dict_combined = {**results_dict_regular_filtered, **results_dict_ancestor}
    
    print(f"\n{'='*80}")
    print(f"Regular keys: {len(results_dict_regular_filtered)}")
    print(f"Ancestor keys: {len(results_dict_ancestor)}")
    print(f"Combined: {len(results_dict_combined)} total results (regular + ancestor)")
    print(f"{'='*80}\n")
    
    if not results_dict_combined:
        print("No results found for the specified keys!")
        return None
    
    # Create dataframe from combined results
    df = create_dataframe(results_dict_combined)
    print(f"\nModels: {sorted(df['model'].unique())}")
    print(f"Node counts: {sorted(df['node_count'].unique())}")
    
    # Collect all keys for summary
    all_keys = regular_keys + (ancestor_keys if ancestor_keys else [])
    
    # Generate plots
    plot_box_comparison_separate_variants(df, node_counts_to_compare=node_counts_to_compare, 
                                        title_suffix=title_suffix, output_prefix=output_prefix)
    print_summary_statistics(df, title_suffix=title_suffix, output_prefix=output_prefix,
                           checkpoint_keys=all_keys)
    
    return df


def main():
    """Main function to generate all plots."""
    print("="*80)
    print("Generating LinGaus Benchmark Plots")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # 5-node models
    print("\n" + "="*80)
    print("Processing 5-Node Models")
    print("="*80)
    
    regular_keys_5node = [
        "lingaus_50node_benchmarked_baseline_16715917.0",
    "lingaus_50node_benchmarked_gcn_16715920.0",
    "lingaus_50node_benchmarked_gcn_and_hartatt_16694430.0",
    "lingaus_50node_benchmarked_gcn_and_softatt_16715919.0",
    "lingaus_50node_benchmarked_hardatt_16715921.0",
    "lingaus_50node_benchmarked_softatt_16715922.0",
    ]
    
    ancestor_keys_5node = [
        "lingaus_ancestor_50node_baseline_16723333.0",
    "lingaus_ancestor_50node_gcn_16723348.0",
    "lingaus_ancestor_50node_gcn_and_hartatt_16726249.0",
    "lingaus_ancestor_50node_gcn_and_softatt_16723335.0",
    "lingaus_ancestor_50node_hardatt_16723337.0",
    "lingaus_ancestor_50node_softatt_16723338.0",
    ]
    
    df_5node = analyze_models(
        checkpoint_base=checkpoint_base,
        regular_keys=regular_keys_5node,
        ancestor_keys=ancestor_keys_5node,
        node_counts_to_compare=[2, 5, 20, 50],
        title_suffix="50-Node Training (Regular + Ancestor)",
        output_prefix="50node_analysis"
    )
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("="*80)


def main_cli():
    """Command line interface for custom analysis."""
    parser = argparse.ArgumentParser(description='Analyze benchmark results and save plots')
    parser.add_argument('--checkpoint-path', required=True, 
                       help='Path to checkpoint directory')
    parser.add_argument('--regular-keys', nargs='+', required=True,
                       help='List of regular model keys to analyze')
    parser.add_argument('--ancestor-keys', nargs='+', default=None,
                       help='List of ancestor model keys to analyze')
    parser.add_argument('--node-counts', nargs='+', type=int, default=[2, 5],
                       help='Node counts to compare in plots')
    parser.add_argument('--title', default="",
                       help='Title suffix for plots')
    parser.add_argument('--output-prefix', default="benchmark",
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Set up matplotlib
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (30, 8)
    
    print("="*80)
    print("Generating Benchmark Analysis Plots")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Run analysis
    df = analyze_models(
        checkpoint_base=args.checkpoint_path,
        regular_keys=args.regular_keys,
        ancestor_keys=args.ancestor_keys,
        node_counts_to_compare=args.node_counts,
        title_suffix=args.title,
        output_prefix=args.output_prefix
    )
    
    print(f"\nAll plots saved to: {output_dir}")
    
    return df


if __name__ == "__main__":
    # Check if command line arguments are provided
    import sys
    if len(sys.argv) > 1:
        # Use command line interface
        main_cli()
    else:
        # Use default main function with predefined keys
        main()