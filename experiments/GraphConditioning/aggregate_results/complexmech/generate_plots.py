#!/usr/bin/env python3
"""
Generate and save plots for ComplexMech benchmark results.

This script analyzes ComplexMech benchmark results with complex mechanisms
(XGBoost, MLPs, mixed noise types) and saves plots to timestamped folders.

The ComplexMech benchmark evaluates:
- Different node counts: 2, 5, 20, 50
- Different variants: base, path_TY, path_YT, path_independent_TY
- Different sample sizes for path variants: 500, 700, 800, 900, 950

Style matches the LinGaus IDK benchmark plots.
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
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import argparse
import pickle


# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Styling
sns.set_style("whitegrid")

# Define colors for sample sizes (similar to hide_fraction colors in LinGaus IDK)
SAMPLE_SIZE_COLORS = {
    500: '#440154',    # Dark purple
    700: '#31688e',    # Blue
    800: '#35b779',    # Green
    900: '#fde724',    # Yellow
    950: '#cc4778',    # Pink/red
    'base': '#808080', # Gray for base variant
}

# Define variant display names
VARIANT_DISPLAY_NAMES = {
    'base': 'Base Causal Structure',
    'path_YT': 'Y → T Path',
    'path_TY': 'T → Y Path',
    'path_independent_TY': 'T ⊥ Y (Independent)',
    'aggregated_all': 'Aggregated\n(All Variants + All Node Counts)'
}


def bootstrap_confidence_interval(data, confidence_level=0.90, n_bootstrap=1000, use_median=True):
    """Compute bootstrap confidence interval for the median (or mean) of data."""
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    if use_median:
        original_stat = np.median(data)
    else:
        original_stat = np.mean(data)
    
    np.random.seed(42)
    bootstrap_samples = np.random.choice(data, size=(n_bootstrap, n), replace=True)
    
    if use_median:
        bootstrap_stats = np.median(bootstrap_samples, axis=1)
    else:
        bootstrap_stats = np.mean(bootstrap_samples, axis=1)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return original_stat, ci_lower, ci_upper


def load_individual_results(results_dir: Path) -> Dict[str, Any]:
    """Load individual sample results from pickle files for recomputing CIs."""
    results = {}
    
    # Find all individual pickle files
    pkl_files = list(results_dir.glob("individual_*.pkl"))
    
    if not pkl_files:
        print(f"No individual pickle files found in {results_dir}")
        return results
    
    print(f"Found {len(pkl_files)} individual pickle files")
    
    for pkl_path in sorted(pkl_files):
        filename = pkl_path.stem  # e.g., "individual_5node_path_TY_ntest_500"
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
            continue
        
        # Parse filename to extract node_count, variant, sample_size
        parts = filename.replace("individual_", "").split("_")
        
        # Extract node count (e.g., "5node" -> 5)
        node_str = parts[0]
        if "node" in node_str:
            node_count = int(node_str.replace("node", ""))
        else:
            print(f"Could not parse node count from: {filename}")
            continue
        
        # Check if this is base or path variant
        if len(parts) == 1:
            # Base variant: individual_5node.pkl
            variant = "base"
            sample_size = None
        elif "path" in filename:
            # Path variant: individual_5node_path_TY_ntest_500.pkl
            variant_parts = []
            sample_size = None
            
            for i, part in enumerate(parts[1:]):
                if part == "ntest":
                    # Next part is sample size
                    sample_size = int(parts[i + 2])
                    break
                variant_parts.append(part)
            
            variant = "_".join(variant_parts)
        else:
            print(f"Could not parse variant from: {filename}")
            continue
        
        # Create key for this configuration
        if sample_size is not None:
            key = f"{node_count}nodes_{variant}_ntest{sample_size}"
        else:
            key = f"{node_count}nodes_{variant}"
        
        results[key] = {
            'node_count': node_count,
            'variant': variant,
            'sample_size': sample_size,
            'individual_results': data,
        }
        
        print(f"  Loaded: {key} ({len(data)} samples)")
    
    return results


def load_complexmech_results(results_dir: Path) -> Dict[str, Any]:
    """Load ComplexMech benchmark results from a directory.
    
    Looks for JSON files with patterns:
    - aggregated_{node_count}node.json (base variant)
    - aggregated_{node_count}node_{variant}_ntest_{sample_size}.json (path variants)
    
    Returns a dict with result keys -> metrics
    """
    results = {}
    
    # Find all aggregated JSON files
    json_files = list(results_dir.glob("aggregated_*.json"))
    
    if not json_files:
        print(f"No aggregated JSON files found in {results_dir}")
        return results
    
    print(f"Found {len(json_files)} aggregated JSON files")
    
    for json_path in sorted(json_files):
        filename = json_path.stem  # e.g., "aggregated_5node_path_TY_ntest_500"
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            continue
        
        # Parse filename to extract node_count, variant, sample_size
        parts = filename.replace("aggregated_", "").split("_")
        
        # Extract node count (e.g., "5node" -> 5)
        node_str = parts[0]
        if "node" in node_str:
            node_count = int(node_str.replace("node", ""))
        else:
            print(f"Could not parse node count from: {filename}")
            continue
        
        # Check if this is base or path variant
        if len(parts) == 1:
            # Base variant: aggregated_5node.json
            variant = "base"
            sample_size = None
        elif "path" in filename:
            # Path variant: aggregated_5node_path_TY_ntest_500.json
            variant_parts = []
            sample_size = None
            
            for i, part in enumerate(parts[1:]):
                if part == "ntest":
                    # Next part is sample size
                    sample_size = int(parts[i + 2])
                    break
                variant_parts.append(part)
            
            variant = "_".join(variant_parts)
        else:
            print(f"Could not parse variant from: {filename}")
            continue
        
        # Create key for this configuration
        if sample_size is not None:
            key = f"{node_count}nodes_{variant}_ntest{sample_size}"
        else:
            key = f"{node_count}nodes_{variant}"
        
        results[key] = {
            'node_count': node_count,
            'variant': variant,
            'sample_size': sample_size,
            'metrics': data,
        }
        
        n_samples = data.get('metadata', {}).get('n_samples', 'unknown')
        print(f"  Loaded: {key}, n_samples={n_samples}")
    
    return results


def create_dataframe(results: Dict, individual_results: Optional[Dict] = None, 
                     confidence_level: float = 0.90) -> pd.DataFrame:
    """Convert results dictionary to pandas DataFrame.
    
    If individual_results are provided, recompute CIs at the specified confidence level.
    Uses median statistics instead of mean for robustness against outliers.
    """
    data_rows = []
    
    for key, result_data in results.items():
        node_count = result_data['node_count']
        variant = result_data['variant']
        sample_size = result_data['sample_size']
        metrics = result_data['metrics']
        
        row = {
            'variant': variant,
            'node_count': node_count,
            'sample_size': sample_size,
        }
        
        # Extract metrics from pre-computed results
        for metric_name in ['mse', 'r2', 'nll']:
            if metric_name in metrics:
                metric_values = metrics[metric_name]
                if isinstance(metric_values, dict):
                    for stat_name, stat_value in metric_values.items():
                        if isinstance(stat_value, (int, float)):
                            row[f'{metric_name}_{stat_name}'] = stat_value
        
        # If we have individual results, recompute median and CIs
        if individual_results and key in individual_results:
            ind_data = individual_results[key]['individual_results']
            
            for metric_name in ['mse', 'r2', 'nll']:
                values = [r[metric_name] for r in ind_data if metric_name in r]
                if values:
                    # Use median instead of mean
                    median_val, ci_lower, ci_upper = bootstrap_confidence_interval(
                        np.array(values), confidence_level=confidence_level, use_median=True
                    )
                    row[f'{metric_name}_median'] = median_val
                    row[f'{metric_name}_median_ci_lower'] = ci_lower
                    row[f'{metric_name}_median_ci_upper'] = ci_upper
        
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    print(f"\nDataFrame shape: {df.shape}")
    if len(df) > 0:
        print(f"Node counts: {sorted(df['node_count'].unique())}")
        print(f"Variants: {sorted(df['variant'].unique())}")
        sample_sizes = [s for s in df['sample_size'].unique() if s is not None]
        if sample_sizes:
            print(f"Sample sizes: {sorted(sample_sizes)}")
    
    return df


def create_aggregated_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated dataframe combining all variants (except base) and all node counts.
    
    For each sample_size, aggregate across:
    - All path variants (path_TY, path_YT, path_independent_TY)
    - All node counts (2, 5, 20, 50)
    
    Returns a new dataframe with variant='aggregated_all' and node_count=0 (dummy).
    Uses median statistics for robustness.
    """
    # Filter to only path variants (not base)
    path_variants = ['path_TY', 'path_YT', 'path_independent_TY']
    df_paths = df[df['variant'].isin(path_variants)].copy()
    
    if len(df_paths) == 0:
        print("Warning: No path variant data found for aggregation")
        return pd.DataFrame()
    
    # Group by sample_size, aggregate median values across variants and node counts
    agg_rows = []
    
    for sample_size in df_paths['sample_size'].unique():
        if sample_size is None:
            continue
            
        size_data = df_paths[df_paths['sample_size'] == sample_size]
        
        if len(size_data) == 0:
            continue
        
        agg_row = {
            'variant': 'aggregated_all',
            'node_count': 0,  # Dummy value
            'sample_size': sample_size,
        }
        
        for metric in ['mse', 'r2', 'nll']:
            median_col = f'{metric}_median'
            ci_lower_col = f'{metric}_median_ci_lower'
            ci_upper_col = f'{metric}_median_ci_upper'
            
            if median_col in size_data.columns:
                # Use median of medians across all variant-node combinations
                agg_row[median_col] = size_data[median_col].median()
                
                # Simple CI approximation using median
                if ci_lower_col in size_data.columns and ci_upper_col in size_data.columns:
                    agg_row[ci_lower_col] = size_data[ci_lower_col].median()
                    agg_row[ci_upper_col] = size_data[ci_upper_col].median()
        
        agg_rows.append(agg_row)
    
    agg_df = pd.DataFrame(agg_rows)
    
    if len(agg_df) > 0:
        print(f"Created aggregated dataframe with {len(agg_df)} rows")
    
    return agg_df


def plot_box_comparison_separate_variants(df: pd.DataFrame, node_counts_to_compare: List[int],
                                          output_dir: Path, title_suffix: str = "", 
                                          output_prefix: str = ""):
    """Create separate box plot figures for each variant, showing sample sizes as separate boxes.
    
    Style matches LinGaus IDK plots.
    """
    variants = sorted(df['variant'].unique())
    metrics_to_plot = ['mse', 'r2', 'nll']
    metric_titles = ['MSE', 'R²', 'NLL']
    
    for variant in variants:
        variant_data = df[df['variant'] == variant].copy()
        
        if len(variant_data) == 0:
            print(f"No data for variant: {variant}")
            continue
        
        # Get unique sample sizes for this variant
        if variant == 'base':
            sample_sizes = [None]
        else:
            sample_sizes = sorted([s for s in variant_data['sample_size'].unique() if s is not None])
        
        # Special layout for aggregated_all: horizontal (1 row x 3 metrics)
        if variant == 'aggregated_all':
            fig, axes = plt.subplots(1, len(metrics_to_plot), 
                                    figsize=(5.5 * len(metrics_to_plot), 4.5))
            axes = axes.reshape(1, -1)
            node_counts_for_plot = [0]  # Dummy
        else:
            # Normal layout: metrics as rows, node counts as columns
            node_counts_for_plot = [n for n in node_counts_to_compare 
                                   if n in variant_data['node_count'].unique()]
            if not node_counts_for_plot:
                print(f"No matching node counts for variant {variant}")
                continue
                
            fig, axes = plt.subplots(len(metrics_to_plot), len(node_counts_for_plot), 
                                    figsize=(5.5 * len(node_counts_for_plot), 4.5 * len(metrics_to_plot)))
            
            # Handle single column case
            if len(node_counts_for_plot) == 1:
                axes = axes.reshape(-1, 1)
        
        # Plot each metric
        for metric_idx, (metric, metric_title) in enumerate(zip(metrics_to_plot, metric_titles)):
            for node_idx, node_count in enumerate(node_counts_for_plot):
                if variant == 'aggregated_all':
                    ax = axes[0, metric_idx]
                    node_data = variant_data.copy()
                else:
                    ax = axes[metric_idx, node_idx]
                    node_data = variant_data[variant_data['node_count'] == node_count].copy()
                
                if len(node_data) == 0:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=14)
                    continue
                
                # Use median columns
                metric_col = f'{metric}_median'
                ci_lower_col = f'{metric}_median_ci_lower'
                ci_upper_col = f'{metric}_median_ci_upper'
                
                if metric_col not in node_data.columns:
                    continue
                
                # Plot bars for each sample size
                positions = []
                heights = []
                colors = []
                error_lower = []
                error_upper = []
                
                for ss_idx, sample_size in enumerate(sample_sizes):
                    if sample_size is None:
                        ss_data = node_data
                    else:
                        ss_data = node_data[node_data['sample_size'] == sample_size]
                    
                    if len(ss_data) == 0:
                        continue
                    
                    median_val = ss_data[metric_col].values[0]
                    positions.append(ss_idx)
                    heights.append(median_val)
                    colors.append(SAMPLE_SIZE_COLORS.get(sample_size, '#808080'))
                    
                    # Get CI if available
                    if ci_lower_col in ss_data.columns and ci_upper_col in ss_data.columns:
                        ci_low = ss_data[ci_lower_col].values[0]
                        ci_high = ss_data[ci_upper_col].values[0]
                        if not np.isnan(ci_low) and not np.isnan(ci_high):
                            error_lower.append(median_val - ci_low)
                            error_upper.append(ci_high - median_val)
                        else:
                            error_lower.append(0)
                            error_upper.append(0)
                    else:
                        error_lower.append(0)
                        error_upper.append(0)
                
                if positions:
                    bars = ax.bar(positions, heights, color=colors, alpha=0.7, 
                                 edgecolor='black', linewidth=1.5)
                    
                    # Add error bars
                    if any(e > 0 for e in error_lower) or any(e > 0 for e in error_upper):
                        ax.errorbar(positions, heights, 
                                   yerr=[error_lower, error_upper],
                                   fmt='none', color='black', capsize=5, capthick=2)
                    
                    # Set x-axis labels
                    if variant == 'base':
                        ax.set_xticks([0])
                        ax.set_xticklabels(['Base'], fontsize=14)
                    else:
                        ax.set_xticks(positions)
                        ax.set_xticklabels([str(s) for s in sample_sizes], fontsize=14)
                        ax.set_xlabel('Sample Size (ntest)', fontsize=16)
                
                # Set titles and labels
                if variant == 'aggregated_all':
                    ax.set_title(metric_title, fontsize=18, fontweight='bold')
                else:
                    if metric_idx == 0:
                        ax.set_title(f'{node_count} Nodes', fontsize=18, fontweight='bold')
                    if node_idx == 0:
                        ax.set_ylabel(metric_title, fontsize=18, fontweight='bold')
                
                ax.tick_params(axis='y', labelsize=14)
                ax.grid(True, alpha=0.3, axis='y')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        # Add variant title
        variant_display = VARIANT_DISPLAY_NAMES.get(variant, variant)
        fig.suptitle(f'{variant_display}\n{title_suffix}', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        filename = f"{output_prefix}_boxplot_{variant}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)


def print_summary_statistics(df: pd.DataFrame, output_dir: Path, 
                            title_suffix: str = "", output_prefix: str = ""):
    """Print summary statistics and save to file."""
    output_file = output_dir / f"{output_prefix}_summary.txt"
    
    with open(output_file, 'w') as f:
        def write_line(line=""):
            print(line)
            f.write(line + "\n")
        
        write_line("="*80)
        if title_suffix:
            write_line(f"SUMMARY STATISTICS - {title_suffix}")
        else:
            write_line("SUMMARY STATISTICS")
        write_line("="*80)
        
        variants = sorted(df['variant'].unique())
        for variant in variants:
            write_line(f"\n{'='*80}")
            write_line(f"VARIANT: {VARIANT_DISPLAY_NAMES.get(variant, variant)}")
            write_line(f"{'='*80}")
            variant_data = df[df['variant'] == variant]
            
            # Get unique sample sizes for this variant
            if variant == 'base':
                sample_sizes = [None]
            else:
                sample_sizes = sorted([s for s in variant_data['sample_size'].unique() if s is not None])
            
            for sample_size in sample_sizes:
                if sample_size is None:
                    write_line(f"\nSAMPLE SIZE: Base (all samples)")
                    ss_data = variant_data[variant_data['sample_size'].isna()]
                else:
                    write_line(f"\nSAMPLE SIZE: {sample_size}")
                    ss_data = variant_data[variant_data['sample_size'] == sample_size]
                
                write_line("-"*60)
                
                for node_count in sorted(ss_data['node_count'].unique()):
                    node_data = ss_data[ss_data['node_count'] == node_count]
                    
                    if len(node_data) > 0:
                        # Use median columns
                        mse = node_data['mse_median'].values[0] if 'mse_median' in node_data.columns else np.nan
                        r2 = node_data['r2_median'].values[0] if 'r2_median' in node_data.columns else np.nan
                        nll = node_data['nll_median'].values[0] if 'nll_median' in node_data.columns else np.nan
                        
                        if variant == 'aggregated_all':
                            write_line(f"  Aggregated: MSE={mse:.6f}, R²={r2:.4f}, NLL={nll:.4f}")
                        else:
                            write_line(f"  {node_count:2d} nodes: MSE={mse:.6f}, R²={r2:.4f}, NLL={nll:.4f}")
        
        write_line("\n" + "="*80)
    
    print(f"Saved summary: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for ComplexMech benchmark results (LinGaus IDK style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing ComplexMech benchmark JSON results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: results_dir/plots/<timestamp>)"
    )
    parser.add_argument(
        "--node_counts",
        type=int,
        nargs="+",
        default=[2, 5, 20, 50],
        help="Node counts to include in plots"
    )
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.90,
        help="Confidence level for bootstrap CIs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ComplexMech Model",
        help="Name to display in plot titles"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "plots" / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ComplexMech Benchmark Visualization (LinGaus IDK Style)")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence level: {args.confidence_level*100:.0f}%")
    print()
    
    # Load aggregated results
    print("Loading aggregated results...")
    results = load_complexmech_results(results_dir)
    
    if not results:
        print("No results loaded!")
        return
    
    # Load individual results for CI recomputation
    print("\nLoading individual results...")
    individual_results = load_individual_results(results_dir)
    
    # Convert to dataframe with recomputed CIs
    print("\nConverting to DataFrame...")
    df = create_dataframe(results, individual_results, confidence_level=args.confidence_level)
    
    if len(df) == 0:
        print("No valid data in DataFrame!")
        return
    
    # Generate plots for each variant
    print("\nGenerating plots...")
    plot_box_comparison_separate_variants(
        df, 
        args.node_counts,
        output_dir,
        title_suffix=args.model_name,
        output_prefix="complexmech"
    )
    
    # Create aggregated plot
    print("\n" + "="*80)
    print("Creating aggregated plot (all path variants + all node counts)")
    print("="*80)
    df_aggregated = create_aggregated_dataframe(df)
    
    if len(df_aggregated) > 0:
        plot_box_comparison_separate_variants(
            df_aggregated,
            [0],  # Dummy node count for aggregated
            output_dir,
            title_suffix=f"{args.model_name} - Aggregated",
            output_prefix="complexmech_aggregated"
        )
    
    # Print summary statistics
    print("\n--- Summary ---")
    print_summary_statistics(df, output_dir, title_suffix=args.model_name, 
                            output_prefix="complexmech")
    
    if len(df_aggregated) > 0:
        print_summary_statistics(df_aggregated, output_dir, 
                                title_suffix=f"{args.model_name} - Aggregated",
                                output_prefix="complexmech_aggregated")
    
    print()
    print("="*80)
    print("Visualization complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
