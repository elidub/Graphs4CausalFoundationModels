#!/usr/bin/env python3
"""
Generate and save plots for LinGaus IDK benchmark results.

This script analyzes LinGausIDK benchmark results with partial graph knowledge
(three-state adjacency matrices: {-1, 0, 1}) and saves plots to timestamped folders.
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
output_dir = Path(__file__).parent / "plots" / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (30, 8)

# Define model order and colors for IDK models
MODEL_ORDER = [
    'idk_gcn_and_softatt',  # The IDK model with partial graph knowledge
    'idk_gcn_and_softatt_hide_none',  # Model trained with full graph knowledge
    'idk_gcn_and_softatt_hide_all',  # Model trained with no graph knowledge
]

MODEL_COLORS = {
    'idk_gcn_and_softatt': '#21918c',  # Teal color
    'idk_gcn_and_softatt_hide_none': '#440154',  # Dark purple (full knowledge)
    'idk_gcn_and_softatt_hide_all': '#cc4778',  # Pink/red (no knowledge)
}


def load_benchmark_results(checkpoint_base, pattern, eval_step=None):
    """Load all benchmark JSON files matching the given pattern."""
    results_dict = {}
    
    for ckpt_dir in sorted(checkpoint_base.glob(pattern)):
        checkpoint_results = {}
        
        if eval_step is not None:
            results_dir = ckpt_dir / f"lingaus_eval_step{eval_step}"
        else:
            results_dir = ckpt_dir / "lingaus_final"
        
        if not results_dir.exists():
            print(f"Missing results dir: {results_dir}")
            continue
        
        # First, try loading lingaus_benchmark_final.json (contains hide fractions for IDK)
        json_path = results_dir / "lingaus_benchmark_final.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    checkpoint_results['base'] = data
                    results_dict[ckpt_dir.name] = checkpoint_results
                    print(f"Loaded: {ckpt_dir.name} (IDK format with {len(data)} result combinations)")
                continue
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
        
        # Fallback: try loading separate variant files
        has_variant_files = False
        variant_files = {}
        
        for node_count in [2, 5, 10, 20, 35, 50]:
            base_file = results_dir / f"aggregated_{node_count}nodes.json"
            if base_file.exists():
                has_variant_files = True
                if 'base' not in variant_files:
                    variant_files['base'] = []
                variant_files['base'].append((node_count, base_file))
            
            for variant_suffix in ['path_YT', 'path_TY', 'path_independent_TY']:
                variant_file = results_dir / f"aggregated_{node_count}nodes_{variant_suffix}.json"
                if variant_file.exists():
                    has_variant_files = True
                    if variant_suffix not in variant_files:
                        variant_files[variant_suffix] = []
                    variant_files[variant_suffix].append((node_count, variant_file))
        
        if has_variant_files:
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
            print(f"Missing JSON in: {ckpt_dir.name}")
    
    print(f"\nTotal loaded: {len(results_dict)} results")
    return results_dict


def extract_model_name(checkpoint_name):
    """Extract model name from checkpoint directory name."""
    parts = checkpoint_name.split('_')
    model_name_parts = []
    is_ancestor = False
    
    if "ancestor" in parts:
        is_ancestor = True
    
    start_collecting = False
    for i, part in enumerate(parts):
        # Start collecting after "benchmarked" or "idk"
        if part in ["benchmarked", "idk"]:
            start_collecting = True
            continue
        
        if "node" in part and part.replace("node", "").isdigit():
            start_collecting = True
            continue
        
        if start_collecting and not part.replace('.', '').replace('-', '').isdigit():
            model_name_parts.append(part)
    
    model_name = '_'.join(model_name_parts) if model_name_parts else 'unknown'
    
    # For IDK models, simplify the name (remove redundant "idk" suffix if present)
    if is_ancestor and model_name != 'unknown':
        model_name = f'idk_{model_name}'
    
    return model_name


def create_dataframe(results_dict, include_variants=True):
    """Convert results dictionary to pandas DataFrame.
    
    Parses hide fractions from keys like '2nodes_path_TY_hide0.25' and 
    extracts them as a separate column.
    """
    data_rows = []
    
    for checkpoint_name, checkpoint_data in results_dict.items():
        model_name = extract_model_name(checkpoint_name)
        
        for variant, variant_data in checkpoint_data.items():
            # Check if this is the legacy format (lingaus_benchmark_final.json)
            # In legacy format, variant_data is the full dict with keys like "2nodes_base", "2nodes_path_TY_hide0.0"
            if variant == 'base' and isinstance(variant_data, dict):
                # Check if the keys look like result keys (e.g., "2nodes_base")
                sample_key = next(iter(variant_data.keys())) if variant_data else ""
                if 'nodes' in sample_key and ('base' in sample_key or 'hide' in sample_key):
                    # This is the legacy format - variant_data contains result keys directly
                    for result_key, result_data in variant_data.items():
                        # Parse the result key to extract node count, variant, and hide fraction
                        parts = result_key.split('_')
                        
                        # Extract node count (e.g., "2nodes" -> 2)
                        node_count_str = parts[0].replace('nodes', '')
                        node_count_val = int(node_count_str)
                        
                        # Extract variant and hide fraction
                        if 'hide' in result_key:
                            # Format: "2nodes_path_TY_hide0.25"
                            hide_idx = result_key.index('hide')
                            hide_fraction = float(result_key[hide_idx+4:])
                            variant_name = '_'.join(parts[1:-1])  # Everything between node count and hide
                        else:
                            # Format: "2nodes_base" (base variant with random hide fraction)
                            hide_fraction = None  # Base variant uses uniform random hide fraction
                            variant_name = '_'.join(parts[1:])
                        
                        row = {
                            'model': model_name,
                            'variant': variant_name,
                            'node_count': node_count_val,
                            'hide_fraction': hide_fraction,
                        }
                        
                        # Extract metrics
                        for metric_name, metric_values in result_data.items():
                            if metric_name != 'metadata' and isinstance(metric_values, dict):
                                for stat_name, stat_value in metric_values.items():
                                    if isinstance(stat_value, (int, float)):
                                        row[f"{metric_name}_{stat_name}"] = stat_value
                        
                        data_rows.append(row)
                    continue
            
            # Standard format - variant_data is dict of node_count -> data
            for node_count, node_data in variant_data.items():
                row = {
                    'model': model_name,
                    'variant': variant,
                    'node_count': int(node_count),
                    'hide_fraction': None,  # No hide fraction in standard format
                }
                
                for metric_name, metric_values in node_data.items():
                    if metric_name != 'metadata' and isinstance(metric_values, dict):
                        for stat_name, stat_value in metric_values.items():
                            if isinstance(stat_value, (int, float)):
                                row[f"{metric_name}_{stat_name}"] = stat_value
                
                data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    print(f"Original shape: {df.shape}")
    df = df.dropna(subset=['mse_mean', 'r2_mean', 'nll_mean'])
    print(f"After dropping NaN rows: {df.shape}")
    
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)
    
    return df


def plot_box_comparison_separate_variants(df, node_counts_to_compare, title_suffix="", output_prefix=""):
    """Create separate box plot figures for each variant, showing hide fractions as separate boxes."""
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    variants = sorted(df['variant'].unique())
    metrics_to_plot = ['mse', 'r2', 'nll']
    metric_titles = ['MSE', 'R²', 'NLL']
    
    variant_display_names = {
        'base': 'Base Causal Structure (Uniform Hide Fraction)',
        'path_YT': 'Y → T Path',
        'path_TY': 'T → Y Path',
        'path_independent_TY': 'T ⊥ Y (Independent)'
    }
    
    # Colors for hide fractions
    hide_colors = {
        0.0: '#440154',    # Dark purple (full knowledge)
        0.25: '#31688e',   # Blue
        0.5: '#35b779',    # Green
        0.75: '#fde724',   # Yellow
        1.0: '#cc4778',    # Pink/red (no knowledge)
        None: '#808080'    # Gray for base variant
    }
    
    for variant in variants:
        variant_data = df[df['variant'] == variant].copy()
        
        if len(variant_data) == 0:
            print(f"No data for variant: {variant}")
            continue
        
        # Get unique hide fractions for this variant
        hide_fractions = sorted([h for h in variant_data['hide_fraction'].unique() if h is not None])
        if variant == 'base':
            hide_fractions = [None]  # Base has no specific hide fraction
        
        # Get unique models
        models = sorted(variant_data['model'].unique())
        n_models = len(models)
        
        fig, axes = plt.subplots(len(metrics_to_plot), len(node_counts_to_compare), 
                                figsize=(20, 12))
        
        # Handle single column case
        if len(node_counts_to_compare) == 1:
            axes = axes.reshape(-1, 1)
        
        for metric_idx, (metric, metric_title) in enumerate(zip(metrics_to_plot, metric_titles)):
            for node_idx, node_count in enumerate(node_counts_to_compare):
                ax = axes[metric_idx, node_idx]
                
                node_data = variant_data[variant_data['node_count'] == node_count].copy()
                
                if len(node_data) == 0:
                    ax.text(0.5, 0.5, f'No data', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{node_count} Nodes')
                    continue
                
                metric_col = f'{metric}_mean'
                if metric_col not in node_data.columns:
                    continue
                
                box_data = []
                positions = []
                colors = []
                labels = []
                mean_values = []
                
                # Create boxes grouped by hide fraction, with models side by side
                position_offset = 0
                group_width = n_models * 0.7  # Width for one hide fraction group
                
                for hide_idx, hide_frac in enumerate(hide_fractions):
                    if hide_frac is None:
                        hide_label = 'Uniform'
                    else:
                        hide_label = f'{hide_frac:.2f}'
                    
                    # Add each model for this hide fraction
                    for model_idx, model in enumerate(models):
                        if hide_frac is None:
                            hide_data = node_data[(node_data['hide_fraction'].isna()) & (node_data['model'] == model)]
                        else:
                            hide_data = node_data[(node_data['hide_fraction'] == hide_frac) & (node_data['model'] == model)]
                        
                        if len(hide_data) == 0:
                            continue
                        
                        pos = position_offset + model_idx * 0.7
                        positions.append(pos)
                        mean_val = hide_data[metric_col].values[0]
                        mean_values.append(mean_val)
                        
                        # Only add label for first hide fraction (for legend)
                        if hide_idx == 0:
                            labels.append(f'{model}')
                        
                        ci_lower_col = f'{metric}_mean_ci_lower'
                        ci_upper_col = f'{metric}_mean_ci_upper'
                        
                        if ci_lower_col in hide_data.columns and ci_upper_col in hide_data.columns:
                            ci_lower = hide_data[ci_lower_col].values[0]
                            ci_upper = hide_data[ci_upper_col].values[0]
                            
                            # Create synthetic data for box plot using confidence intervals
                            synthetic_data = [
                                ci_lower,
                                ci_lower + (mean_val - ci_lower) * 0.5,
                                mean_val,
                                mean_val + (ci_upper - mean_val) * 0.5,
                                ci_upper
                            ]
                        else:
                            synthetic_data = [mean_val] * 5
                        
                        box_data.append(synthetic_data)
                        colors.append(MODEL_COLORS.get(model, '#808080'))
                    
                    position_offset += group_width + 0.5  # Space between hide fraction groups
                
                if box_data and positions:
                    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                                   patch_artist=True, showmeans=False,
                                   medianprops=dict(color='red', linewidth=2),
                                   boxprops=dict(linewidth=1.5),
                                   whiskerprops=dict(linewidth=1.5),
                                   capprops=dict(linewidth=1.5))
                    
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Add mean values as text
                    for pos, mean_val in zip(positions, mean_values):
                        ax.text(pos, mean_val, f'{mean_val:.3f}', 
                               ha='center', va='bottom', fontsize=6, rotation=90,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                       edgecolor='none', alpha=0.8))
                    
                    # Set x-axis labels for hide fractions
                    hide_labels = [f'{h:.2f}' if h is not None else 'Uniform' for h in hide_fractions]
                    # Calculate center positions for each hide fraction group
                    group_centers = []
                    pos_offset = 0
                    for _ in hide_fractions:
                        center = pos_offset + (n_models - 1) * 0.7 / 2
                        group_centers.append(center)
                        pos_offset += group_width + 0.5
                    
                    ax.set_xticks(group_centers)
                    ax.set_xticklabels(hide_labels, rotation=0, fontsize=10)
                    ax.set_xlabel('Hide Fraction', fontsize=10)
                
                # Set titles and labels
                if metric_idx == 0:
                    ax.set_title(f'{node_count} Nodes', fontsize=12, fontweight='bold')
                
                if node_idx == 0:
                    ax.set_ylabel(metric_title, fontsize=11, fontweight='bold')
                
                ax.grid(True, alpha=0.3, axis='y')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
        # Add legend for models (only once, in the first subplot)
        if len(models) > 1:
            legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLORS.get(m, '#808080'), alpha=0.7, label=m) 
                             for m in models]
            axes[0, 0].legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        variant_display = variant_display_names.get(variant, variant)
        fig.suptitle(f'{variant_display} - {title_suffix}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
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
            
            # Get unique hide fractions for this variant
            hide_fractions = sorted([h for h in variant_data['hide_fraction'].unique() if h is not None])
            if variant == 'base':
                hide_fractions = [None]
            
            for hide_frac in hide_fractions:
                if hide_frac is None:
                    write_line(f"\nHIDE FRACTION: Uniform (Base Variant)")
                    hide_data = variant_data[variant_data['hide_fraction'].isna()]
                else:
                    write_line(f"\nHIDE FRACTION: {hide_frac:.2f}")
                    hide_data = variant_data[variant_data['hide_fraction'] == hide_frac]
                
                write_line("-"*80)
                
                for model in sorted(hide_data['model'].unique()):
                    model_data = hide_data[hide_data['model'] == model]
                    
                    if len(model_data) > 0:
                        write_line(f"\n  Model: {str(model).upper()}")
                        
                        if 'mse_mean' in model_data.columns:
                            write_line(f"    MSE   - Mean: {model_data['mse_mean'].mean():.6f} ± {model_data['mse_mean'].std():.6f}")
                        if 'r2_mean' in model_data.columns:
                            write_line(f"    R²    - Mean: {model_data['r2_mean'].mean():.6f} ± {model_data['r2_mean'].std():.6f}")
                        if 'nll_mean' in model_data.columns:
                            write_line(f"    NLL   - Mean: {model_data['nll_mean'].mean():.6f} ± {model_data['nll_mean'].std():.6f}")
        
        write_line("\n" + "="*80)
    
    print(f"Saved summary: {output_file}")


def main():
    """Main function to generate all plots."""
    print("="*80)
    print("Generating LinGaus IDK Benchmark Plots")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # 5-node IDK model
    print("\n" + "="*80)
    print("Processing 5-Node IDK Models")
    print("="*80)
    
    idk_keys_5node = [
        "lingaus_ancestor_5node_idk_gcn_and_softatt_16739007.0",
        "lingaus_ancestor_5node_idk_gcn_and_softatt_hide_none_16741967.0",
        "lingaus_ancestor_5node_idk_gcn_and_softatt_hide_all_16741968.0",
    ]
    
    results_dict_5node = {}
    for key in idk_keys_5node:
        results = load_benchmark_results(checkpoint_base, key)
        results_dict_5node.update(results)
    
    if results_dict_5node:
        df_5node = create_dataframe(results_dict_5node)
        print(f"Models: {sorted(df_5node['model'].unique())}")
        print(f"Node counts: {sorted(df_5node['node_count'].unique())}")
        print(f"Variants: {sorted(df_5node['variant'].unique())}")
        
        plot_box_comparison_separate_variants(df_5node, [2, 5], 
                                             title_suffix="5-Node IDK Training", 
                                             output_prefix="5node_idk")
        print_summary_statistics(df_5node, title_suffix="5-Node IDK Training", 
                               output_prefix="5node_idk",
                               checkpoint_keys=idk_keys_5node)
    else:
        print("No results found for 5-node IDK models")
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
