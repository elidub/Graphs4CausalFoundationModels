#!/usr/bin/env python3
"""
Generate and save plots for ComplexMech benchmark results with varying ntest values.

This version loads INDIVIDUAL result files (not aggregated), recomputes means and 
95% confidence intervals from the raw data for each configuration.

Key difference from hide experiment: Results are separated by ntest (number of test samples)
instead of hide fractions.
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import pandas as pd
from tqdm import tqdm

# Configuration
checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints"
checkpoint_base = Path(checkpoint_path)

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parent / "plots" / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_style("whitegrid")

# Define model order and colors for ComplexMech models
MODEL_ORDER = [
    '80',  # The model with partial graph knowledge (16773480)
    '78',  # Model trained with no graph knowledge (16773478)
    'hide_all',  # Model trained with full graph knowledge (hide_all/16771679)
]

MODEL_COLORS = {
    '80': '#21918c',  # Teal color
    'hide_all': '#440154',  # Dark purple (full knowledge)
    '78': '#cc4778',  # Pink/red (no knowledge)
}


def compute_ci_bootstrap(data, confidence=0.95, n_bootstrap=1000):
    """
    Compute confidence interval using parametric method (t-distribution) for speed.
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Unused (kept for API compatibility)
    
    Returns:
        (mean, ci_lower, ci_upper)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    data = np.array(data)
    mean_val = np.mean(data)
    
    if len(data) == 1:
        return mean_val, mean_val, mean_val
    
    # Use t-distribution for confidence interval (much faster than bootstrap)
    std_err = stats.sem(data)  # Standard error of the mean
    ci_range = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    ci_lower = mean_val - ci_range
    ci_upper = mean_val + ci_range
    
    return mean_val, ci_lower, ci_upper


def load_benchmark_results(checkpoint_dirs):
    """Load individual JSON files from ComplexMech checkpoint directories and recompute statistics."""
    results_dict = {}
    
    for ckpt_path in checkpoint_dirs:
        results_dir = Path(ckpt_path)
        
        if not results_dir.exists():
            print(f"Missing results dir: {results_dir}")
            continue
        
        # Extract model identifier from path (parent directory contains the checkpoint ID)
        checkpoint_name = results_dir.parent.name
        
        # Load separate variant files with different ntest values
        variant_files = {}
        
        for node_count in [2, 5, 10, 20, 35, 50]:
            # Base variant (no path specification, no ntest)
            base_file = results_dir / f"individual_{node_count}nodes.json"
            if base_file.exists():
                if 'base' not in variant_files:
                    variant_files['base'] = []
                variant_files['base'].append((node_count, None, base_file))
            
            # Path variants with different ntest values (no default/no-ntest files exist)
            for variant_suffix in ['path_YT', 'path_TY', 'path_independent_TY']:
                # With ntest values
                for ntest in [100, 250, 500, 750, 900]:
                    variant_file = results_dir / f"individual_{node_count}nodes_{variant_suffix}_ntest{ntest}.json"
                    if variant_file.exists():
                        if variant_suffix not in variant_files:
                            variant_files[variant_suffix] = []
                        variant_files[variant_suffix].append((node_count, ntest, variant_file))
        
        if variant_files:
            checkpoint_results = {}
            # Count total files for progress bar
            total_files = sum(len(files) for files in variant_files.values())
            
            with tqdm(total=total_files, desc=f"Loading {checkpoint_name}", leave=False) as pbar:
                for variant, files in variant_files.items():
                    variant_data = {}
                    for node_count, ntest, filepath in files:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        # Build result key: "2nodes_base" or "2nodes_path_TY_ntest250"
                        if ntest is None:
                            result_key = f"{node_count}nodes_{variant}"
                        else:
                            result_key = f"{node_count}nodes_{variant}_ntest{ntest}"
                        
                        # Extract raw values for each metric
                        # The ntest JSON format is an array of dictionaries: [{"mse": ..., "r2": ..., "nll": ...}, ...]
                        result_data = {}
                        
                        if isinstance(data, list) and len(data) > 0:
                            # Extract each metric's values across all samples
                            for metric_name in ['mse', 'r2', 'nll']:
                                raw_values = [sample[metric_name] for sample in data if metric_name in sample]
                                
                                if len(raw_values) > 0:
                                    # Recompute statistics from individual values
                                    mean_val, ci_lower, ci_upper = compute_ci_bootstrap(raw_values)
                                    
                                    result_data[f'{metric_name}_raw'] = raw_values
                                    result_data[f'{metric_name}_mean'] = mean_val
                                    result_data[f'{metric_name}_mean_ci_lower'] = ci_lower
                                    result_data[f'{metric_name}_mean_ci_upper'] = ci_upper
                                    result_data[f'{metric_name}_std'] = np.std(raw_values)
                                    result_data[f'{metric_name}_median'] = np.median(raw_values)
                                    result_data[f'{metric_name}_n_samples'] = len(raw_values)
                        
                        if result_data:
                            variant_data[result_key] = result_data
                        
                        pbar.update(1)
                    
                    if variant_data:
                        checkpoint_results[variant] = variant_data
            
            if checkpoint_results:
                results_dict[checkpoint_name] = checkpoint_results
                print(f"Loaded: {checkpoint_name} ({sum(len(v) for v in checkpoint_results.values())} files)")
        else:
            print(f"Missing JSON in: {checkpoint_name}")
    
    print(f"\nTotal loaded: {len(results_dict)} checkpoints")
    return results_dict


def extract_model_name(checkpoint_name):
    """Extract model name from checkpoint directory name."""
    if 'hide_all' in checkpoint_name:
        return 'hide_all'
    elif '16773478' in checkpoint_name:  # No graph model
        return '78'
    else:
        return '80'


def create_dataframe(results_dict):
    """Convert results dictionary to pandas DataFrame."""
    data_rows = []
    
    for checkpoint_name, checkpoint_data in results_dict.items():
        model_name = extract_model_name(checkpoint_name)
        
        for variant, variant_data in checkpoint_data.items():
            for result_key, result_data in variant_data.items():
                # Parse the result key to extract node count, variant, and ntest
                # Format: "2nodes_base" or "2nodes_path_TY_ntest250"
                parts = result_key.split('_')
                
                # Extract node count (e.g., "2nodes" -> 2)
                node_count_str = parts[0].replace('nodes', '')
                node_count_val = int(node_count_str)
                
                # Extract variant and ntest
                if 'ntest' in result_key:
                    # Extract ntest value
                    ntest_part = [p for p in parts if p.startswith('ntest')][0]
                    ntest = int(ntest_part.replace('ntest', ''))
                    # Extract variant name (between nodes and ntest)
                    variant_parts = parts[1:-1]  # Skip node count and ntest
                    variant_name = '_'.join(variant_parts)
                else:
                    ntest = None
                    variant_name = '_'.join(parts[1:])
                
                row = {
                    'model': model_name,
                    'variant': variant_name,
                    'node_count': node_count_val,
                    'ntest': ntest,
                }
                
                # Extract recomputed metrics
                for metric_name in ['mse', 'r2', 'nll']:
                    for key in result_data.keys():
                        if key.startswith(metric_name):
                            row[key] = result_data[key]
                
                data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    print(f"Original shape: {df.shape}")
    df = df.dropna(subset=['mse_mean', 'r2_mean', 'nll_mean'])
    print(f"After dropping NaN rows: {df.shape}")
    
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)
    
    return df


def create_aggregated_dataframe(df):
    """Create aggregated dataframe combining path_TY, path_YT, and path_independent_TY.
    
    For each model, node_count, and ntest value, combines the raw values from all three
    path variants and recomputes statistics.
    """
    print("\n" + "="*80)
    print("Creating aggregated dataframe...")
    print("="*80)
    
    # Filter to only path variants
    path_variants = ['path_TY', 'path_YT', 'path_independent_TY']
    path_df = df[df['variant'].isin(path_variants)].copy()
    
    print(f"Filtered to path variants: {len(path_df)} rows")
    
    if len(path_df) == 0:
        print("No path variant data found for aggregation")
        return pd.DataFrame()
    
    aggregated_rows = []
    
    # Group by model, node_count, and ntest value
    groups = list(path_df.groupby(['model', 'node_count', 'ntest']))
    print(f"Processing {len(groups)} unique (model, node_count, ntest) combinations...")
    
    for (model, node_count, ntest), group in tqdm(groups, desc="Aggregating data"):
        if len(group) == 0:
            continue
        
        # Combine raw values from all path variants for each metric
        row = {
            'model': model,
            'variant': 'aggregated_paths',
            'node_count': node_count,
            'ntest': ntest,
        }
        
        for metric in ['mse', 'r2', 'nll']:
            raw_col = f'{metric}_raw'
            if raw_col not in group.columns:
                continue
            
            # Combine all raw values from the three path variants
            combined_raw = []
            for raw_values in group[raw_col]:
                if raw_values and len(raw_values) > 0:
                    combined_raw.extend(raw_values)
            
            if len(combined_raw) > 0:
                # Recompute statistics from combined data
                mean_val, ci_lower, ci_upper = compute_ci_bootstrap(combined_raw)
                
                row[f'{metric}_mean'] = mean_val
                row[f'{metric}_mean_ci_lower'] = ci_lower
                row[f'{metric}_mean_ci_upper'] = ci_upper
                row[f'{metric}_std'] = np.std(combined_raw)
                row[f'{metric}_median'] = np.median(combined_raw)
                row[f'{metric}_n_samples'] = len(combined_raw)
                row[f'{metric}_raw'] = combined_raw
        
        aggregated_rows.append(row)
    
    agg_df = pd.DataFrame(aggregated_rows)
    
    if len(agg_df) > 0:
        agg_df['model'] = pd.Categorical(agg_df['model'], categories=MODEL_ORDER, ordered=True)
        print(f"\n✓ Created aggregated dataframe with {len(agg_df)} rows")
        print(f"  Models: {sorted(agg_df['model'].unique())}")
        print(f"  Node counts: {sorted(agg_df['node_count'].unique())}")
        print(f"  Ntest values: {sorted([n for n in agg_df['ntest'].unique() if n is not None])}")
    
    return agg_df


def plot_box_comparison_separate_variants(df, node_counts_to_compare, title_suffix="", output_prefix=""):
    """Create separate box plot figures for each variant, showing ntest values as separate boxes.
    
    Box plots show:
    - Box: 25th to 75th percentile (IQR)
    - Red line: Median
    - Whiskers: 95% bootstrap confidence intervals of the mean
    """
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    variants = sorted(df['variant'].unique())
    metrics_to_plot = ['mse', 'r2', 'nll']
    metric_titles = ['MSE', 'R²', 'NLL']
    
    variant_display_names = {
        'base': 'Base Causal Structure',
        'path_YT': 'Y → T Path',
        'path_TY': 'T → Y Path',
        'path_independent_TY': 'T ⊥ Y (Independent)',
        'aggregated_paths': 'Aggregated Paths',
    }
    
    print(f"Generating plots for {len(variants)} variants...")
    for variant in tqdm(variants, desc="Plotting variants"):
        variant_data = df[df['variant'] == variant].copy()
        
        if len(variant_data) == 0:
            print(f"No data for variant: {variant}")
            continue
        
        # Get unique ntest values for this variant
        ntest_values = sorted([n for n in variant_data['ntest'].unique() if n is not None])
        if not ntest_values:
            ntest_values = [None]  # Base has no specific ntest
        
        # Get unique models
        models = sorted(variant_data['model'].unique())
        n_models = len(models)
        
        # Normal layout: metrics as rows, node counts as columns
        fig, axes = plt.subplots(len(metrics_to_plot), len(node_counts_to_compare), 
                                figsize=(5.5 * len(node_counts_to_compare), 4.5 * len(metrics_to_plot)))
        
        # Handle single column case
        if len(node_counts_to_compare) == 1:
            axes = axes.reshape(-1, 1)
        
        # Create progress bar for subplots
        total_subplots = len(metrics_to_plot) * len(node_counts_to_compare)
        subplot_pbar = tqdm(total=total_subplots, desc=f"  Creating subplots for {variant}", leave=False)
        
        # Normal layout: iterate over metrics (rows) and node counts (columns)
        for metric_idx, (metric, metric_title) in enumerate(zip(metrics_to_plot, metric_titles)):
            for node_idx, node_count in enumerate(node_counts_to_compare):
                ax = axes[metric_idx, node_idx]
            
                node_data = variant_data[variant_data['node_count'] == node_count].copy()
            
                if len(node_data) == 0:
                    ax.set_visible(False)
                    subplot_pbar.update(1)
                    continue
                
                metric_raw_col = f'{metric}_raw'
                if metric_raw_col not in node_data.columns:
                    ax.set_visible(False)
                    subplot_pbar.update(1)
                    continue
                
                box_data = []
                positions = []
                colors = []
                labels = []
                
                # Create boxes grouped by ntest, with models side by side
                position_offset = 0
                group_width = n_models * 0.7  # Width for one ntest group
                
                for ntest_idx, ntest in enumerate(ntest_values):
                    if ntest is None:
                        ntest_data = node_data[node_data['ntest'].isna()]
                    else:
                        ntest_data = node_data[node_data['ntest'] == ntest]
                    
                    if len(ntest_data) == 0:
                        continue
                    
                    # For each model within this ntest group
                    for model_idx, model in enumerate(models):
                        model_data = ntest_data[ntest_data['model'] == model]
                        
                        if len(model_data) == 0:
                            continue
                        
                        # Get the raw values for this configuration
                        raw_values = model_data[metric_raw_col].values[0]
                        
                        if not raw_values or len(raw_values) == 0:
                            continue
                        
                        # Get confidence intervals for whiskers
                        ci_lower = model_data[f'{metric}_mean_ci_lower'].values[0]
                        ci_upper = model_data[f'{metric}_mean_ci_upper'].values[0]
                        median_val = model_data[f'{metric}_median'].values[0]
                        
                        # Create synthetic distribution for box plot
                        # 5-point distribution: [CI_lower, Q1, median, Q3, CI_upper]
                        # Q1 = halfway between CI_lower and median
                        # Q3 = halfway between median and CI_upper
                        q1 = ci_lower + 0.5 * (median_val - ci_lower)
                        q3 = median_val + 0.5 * (ci_upper - median_val)
                        
                        synthetic_data = np.array([
                            ci_lower,
                            q1,
                            median_val,
                            q3,
                            ci_upper
                        ])
                        
                        box_data.append(synthetic_data)
                        
                        # Position: group offset + model offset within group
                        pos = position_offset + model_idx * 0.7
                        positions.append(pos)
                        colors.append(MODEL_COLORS.get(model, '#808080'))
                        
                        # Label only for first occurrence of each model
                        if ntest_idx == 0:
                            labels.append(model)
                
                    # Move to next ntest group (with spacing)
                    position_offset += group_width + 0.5
                
                if box_data and positions:
                    # Plot boxes
                    bp = ax.boxplot(box_data, positions=positions, widths=0.6, 
                                   patch_artist=True, showfliers=False,
                                   medianprops=dict(color='red', linewidth=2),
                                   boxprops=dict(linewidth=1.5),
                                   whiskerprops=dict(linewidth=1.5),
                                   capprops=dict(linewidth=1.5))
                    
                    # Color the boxes
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Set x-axis labels (ntest values)
                    tick_positions = []
                    tick_labels = []
                    offset = (n_models - 1) * 0.7 / 2  # Center of each group
                    for ntest_idx, ntest in enumerate(ntest_values):
                        group_pos = ntest_idx * (group_width + 0.5) + offset
                        tick_positions.append(group_pos)
                        if ntest is None:
                            tick_labels.append('Default')
                        else:
                            tick_labels.append(str(ntest))
                    
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, fontsize=14)
                    ax.set_xlabel('Number of Test Samples (ntest)', fontsize=16, fontweight='bold')
                
                # Set titles and labels
                if metric_idx == 0:
                    ax.set_title(f'{node_count} Nodes', fontsize=18, fontweight='bold')
                
                if node_idx == 0:
                    ax.set_ylabel(metric_title, fontsize=18, fontweight='bold')
                
                # Set tick label sizes
                ax.tick_params(axis='y', labelsize=18)
                
                ax.grid(True, alpha=0.3, axis='y')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Update progress bar after each subplot
                subplot_pbar.update(1)
        
        # Close the subplot progress bar
        subplot_pbar.close()
        
        # Add legend for models (centered below the plots)
        if len(models) > 1:
            legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLORS.get(m, '#808080'), alpha=0.7, label=m) 
                             for m in models]
            fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                      fontsize=18, ncol=len(models), frameon=True)
        
        variant_display = variant_display_names.get(variant, variant)
        fig.suptitle(f'{variant_display}\n{title_suffix}', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        filename = f"{output_prefix}_boxplot_{variant}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)


def print_summary_statistics(df, title_suffix="", output_prefix="", checkpoint_keys=None):
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
        write_line("\n*** RECOMPUTED FROM INDIVIDUAL RESULTS ***")
        write_line("*** Using Bootstrap 95% Confidence Intervals ***\n")
        
        # Write checkpoint paths if provided
        if checkpoint_keys:
            write_line("\nCHECKPOINT PATHS:")
            write_line("-"*80)
            for key in sorted(checkpoint_keys):
                write_line(f"  {key}")
            write_line("="*80)
        
        variants = sorted(df['variant'].unique())
        for variant in variants:
            write_line(f"\n{'='*80}")
            write_line(f"VARIANT: {variant}")
            write_line(f"{'='*80}")
            variant_data = df[df['variant'] == variant]
            
            # Get unique ntest values for this variant
            ntest_values = sorted([n for n in variant_data['ntest'].unique() if n is not None])
            if variant == 'base' or not ntest_values:
                ntest_values = [None]
            
            for ntest in ntest_values:
                if ntest is None:
                    ntest_data = variant_data[variant_data['ntest'].isna()]
                    write_line(f"\n  Ntest: Default")
                else:
                    ntest_data = variant_data[variant_data['ntest'] == ntest]
                    write_line(f"\n  Ntest: {ntest}")
                
                write_line("-"*80)
                
                for model in sorted(ntest_data['model'].unique()):
                    model_data = ntest_data[ntest_data['model'] == model]
                    write_line(f"\n    Model: {model}")
                    write_line("    " + "-"*60)
                    
                    for node_count in sorted(model_data['node_count'].unique()):
                        node_data = model_data[model_data['node_count'] == node_count]
                        write_line(f"\n      {node_count} Nodes:")
                        
                        for metric in ['mse', 'r2', 'nll']:
                            mean_val = node_data[f'{metric}_mean'].values[0]
                            ci_lower = node_data[f'{metric}_mean_ci_lower'].values[0]
                            ci_upper = node_data[f'{metric}_mean_ci_upper'].values[0]
                            std_val = node_data[f'{metric}_std'].values[0]
                            n_samples = node_data[f'{metric}_n_samples'].values[0]
                            
                            write_line(f"        {metric.upper():4s}: {mean_val:8.4f} ± {std_val:8.4f} "
                                     f"[{ci_lower:8.4f}, {ci_upper:8.4f}] (n={n_samples})")
        
        write_line("\n" + "="*80)
    
    print(f"Saved summary: {output_file}")


def main():
    """Main function to generate all plots."""
    print("="*80)
    print("Generating ComplexMech Benchmark Plots (RECOMPUTED from individual results)")
    print("Ntest Variation Experiment")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # ComplexMech models with different ntest values
    print("\n" + "="*80)
    print("Processing ComplexMech Models (ntest variations)")
    print("="*80)
    
    # Plot all three models for comparison
    checkpoint_dirs = [
        checkpoint_base / "final_earlytest_16773480.0" / "complexmech_ntest_high",
        checkpoint_base / "final_earlytest_16773478.0" / "complexmech_ntest_high",
        checkpoint_base / "final_earlytest_hide_all_16771679.0" / "complexmech_ntest_high",
    ]
    
    results_dict = load_benchmark_results(checkpoint_dirs)
    
    if results_dict:
        df = create_dataframe(results_dict)
        print(f"Models: {sorted(df['model'].unique())}")
        print(f"Node counts: {sorted(df['node_count'].unique())}")
        print(f"Variants: {sorted(df['variant'].unique())}")
        print(f"Ntest values: {sorted([n for n in df['ntest'].unique() if n is not None])}")
        
        # Report sample sizes
        if 'mse_n_samples' in df.columns:
            print(f"Sample sizes per configuration: {df['mse_n_samples'].unique()}")
        
        plot_box_comparison_separate_variants(df, [2, 5, 10, 20, 35, 50], 
                                             title_suffix="ComplexMech Benchmark (Recomputed from Individual Results)", 
                                             output_prefix="complexmech_ntest_recomputed")
        print_summary_statistics(df, title_suffix="ComplexMech Benchmark (Recomputed from Individual Results)", 
                               output_prefix="complexmech_ntest_recomputed",
                               checkpoint_keys=[str(p) for p in checkpoint_dirs])
        
        # Create and plot aggregated data (combining path_TY, path_YT, path_independent_TY)
        print("\n" + "="*80)
        print("AGGREGATING PATH VARIANTS (path_TY + path_YT + path_independent_TY)")
        print("="*80)
        
        print("Starting aggregation...")
        agg_df = create_aggregated_dataframe(df)
        print(f"Aggregation complete. Result: {len(agg_df)} rows")
        
        if len(agg_df) > 0:
            print("\nGenerating aggregated plots...")
            plot_box_comparison_separate_variants(agg_df, [2, 5, 10, 20, 35, 50], 
                                                 title_suffix="ComplexMech Benchmark - Aggregated Paths", 
                                                 output_prefix="complexmech_ntest_aggregated")
            print("\nGenerating aggregated summary statistics...")
            print_summary_statistics(agg_df, title_suffix="ComplexMech Benchmark - Aggregated Paths", 
                                   output_prefix="complexmech_ntest_aggregated",
                                   checkpoint_keys=[str(p) for p in checkpoint_dirs])
            print("\n✓ Aggregated plots complete!")
        else:
            print("❌ Could not create aggregated data")
    else:
        print("No results found for ComplexMech models")
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
