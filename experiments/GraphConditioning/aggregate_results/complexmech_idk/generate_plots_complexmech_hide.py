#!/usr/bin/env python3
"""
Generate and save plots for ComplexMech benchmark results with varying hide fractions.

This script analyzes ComplexMech benchmark results with different hide fractions
(hide: 0.0, 0.25, 0.5, 0.75, 1.0) and saves plots to timestamped folders.
Adapted from ComplexMech ntest plotting script to show hide variations.
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
import re

# Configuration
checkpoint_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints"
checkpoint_base = Path(checkpoint_path)

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parent / "plots" / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

# Styling
sns.set_style("whitegrid")

# Define model order and colors for ComplexMech IDK models
MODEL_ORDER = [
    '80',  # The IDK model with partial graph knowledge (16773480)
    '78',  # Model trained with no graph knowledge (16773478)
    'hide_all',  # Model trained with full graph knowledge (hide_all/16771679)
]

MODEL_COLORS = {
    '80': '#21918c',  # Teal color
    'hide_all': '#440154',  # Dark purple (full knowledge)
    '78': '#cc4778',  # Pink/red (no knowledge)
}


def load_benchmark_results(checkpoint_dirs):
    """Load aggregated JSON files from ComplexMech checkpoint directories."""
    results_dict = {}
    
    for ckpt_path in checkpoint_dirs:
        results_dir = Path(ckpt_path)
        
        if not results_dir.exists():
            print(f"Missing results dir: {results_dir}")
            continue
        
        # Extract model identifier from path
        checkpoint_name = results_dir.parent.name
        
        # Load separate variant files with different hide fractions
        variant_files = {}
        
        for node_count in [2, 5, 10, 20, 35, 50]:
            # Base variant (no path specification, no hide)
            base_file = results_dir / f"aggregated_{node_count}nodes.json"
            if base_file.exists():
                if 'base' not in variant_files:
                    variant_files['base'] = []
                variant_files['base'].append((node_count, None, base_file))
            
            # Path variants with different hide values
            for variant_suffix in ['path_YT', 'path_TY', 'path_independent_TY']:
                # No hide specified (full graph knowledge)
                variant_file_no_hide = results_dir / f"aggregated_{node_count}nodes_{variant_suffix}.json"
                if variant_file_no_hide.exists():
                    if variant_suffix not in variant_files:
                        variant_files[variant_suffix] = []
                    variant_files[variant_suffix].append((node_count, None, variant_file_no_hide))
                
                # With hide fractions
                for hide_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    variant_file = results_dir / f"aggregated_{node_count}nodes_{variant_suffix}_hide{hide_frac}.json"
                    if variant_file.exists():
                        if variant_suffix not in variant_files:
                            variant_files[variant_suffix] = []
                        variant_files[variant_suffix].append((node_count, hide_frac, variant_file))
        
        if variant_files:
            checkpoint_results = {}
            for variant, files in variant_files.items():
                variant_data = {}
                for node_count, hide_frac, filepath in files:
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            
                            # Extract hide from metadata (preferred) or use passed value
                            meta = data.get('metadata', {})
                            hide_value = meta.get('hide_fraction', hide_frac)
                            
                            # Store with hide info
                            key = f"{node_count}nodes_{variant}"
                            if hide_value is not None:
                                key += f"_hide{hide_value}"
                            
                            variant_data[key] = data
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                
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
                # Parse the result key to extract node count, variant, and hide
                # Format: "2nodes_base" or "2nodes_path_TY_hide0.5"
                parts = result_key.split('_')
                
                # Extract node count (e.g., "2nodes" -> 2)
                node_count_str = parts[0].replace('nodes', '')
                node_count_val = int(node_count_str)
                
                # Extract variant and hide
                if 'hide' in result_key:
                    # Format: "2nodes_path_TY_hide0.5"
                    # Find the part that starts with 'hide'
                    hide_part = None
                    hide_part_idx = None
                    for idx, part in enumerate(parts):
                        if part.startswith('hide'):
                            hide_part = part
                            hide_part_idx = idx
                            break
                    
                    if hide_part:
                        hide = float(hide_part.replace('hide', ''))
                        variant_name = '_'.join(parts[1:hide_part_idx])
                    else:
                        hide = None
                        variant_name = '_'.join(parts[1:])
                else:
                    # Format: "2nodes_base" or "2nodes_path_TY" (no hide)
                    hide = None
                    variant_name = '_'.join(parts[1:])
                
                row = {
                    'model': model_name,
                    'variant': variant_name,
                    'node_count': node_count_val,
                    'hide': hide,
                }
                
                # Extract metrics - prefer median over mean for robustness
                for metric_name in ['mse', 'r2', 'nll']:
                    metric_values = result_data.get(metric_name)
                    if metric_values and isinstance(metric_values, dict):
                        # Prefer median if available (high-fidelity results have it)
                        if 'median' in metric_values:
                            row[f"{metric_name}_mean"] = metric_values.get('median')
                            row[f"{metric_name}_mean_ci_lower"] = metric_values.get('median_ci_lower')
                            row[f"{metric_name}_mean_ci_upper"] = metric_values.get('median_ci_upper')
                        else:
                            # Fall back to mean (minimal results only have mean)
                            row[f"{metric_name}_mean"] = metric_values.get('mean')
                            row[f"{metric_name}_mean_ci_lower"] = metric_values.get('ci_lower')
                            row[f"{metric_name}_mean_ci_upper"] = metric_values.get('ci_upper')
                        
                        row[f"{metric_name}_std"] = metric_values.get('std')
                
                data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    
    print(f"Original shape: {df.shape}")
    df = df.dropna(subset=['mse_mean', 'r2_mean', 'nll_mean'])
    print(f"After dropping NaN rows: {df.shape}")
    
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)
    
    return df


def plot_box_comparison_separate_variants(df, node_counts_to_compare, title_suffix="", output_prefix=""):
    """Create separate box plot figures for each variant, showing hide values as separate boxes."""
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
    }
    
    for variant in variants:
        variant_data = df[df['variant'] == variant].copy()
        
        if len(variant_data) == 0:
            print(f"No data for variant: {variant}")
            continue
        
        # Get unique hide values for this variant
        hide_values = sorted([h for h in variant_data['hide'].unique() if h is not None])
        if not hide_values:
            hide_values = [None]  # Base has no specific hide
        
        # Get unique models
        models = sorted(variant_data['model'].unique())
        n_models = len(models)
        
        # Normal layout: metrics as rows, node counts as columns
        fig, axes = plt.subplots(len(metrics_to_plot), len(node_counts_to_compare), 
                                figsize=(5.5 * len(node_counts_to_compare), 4.5 * len(metrics_to_plot)))
        
        # Handle single column case
        if len(node_counts_to_compare) == 1:
            axes = axes.reshape(-1, 1)
        
        # Normal layout: iterate over metrics (rows) and node counts (columns)
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
                
                # Create boxes grouped by hide, with models side by side
                position_offset = 0
                group_width = n_models * 0.7  # Width for one hide group (max possible)
                
                for hide_idx, hide in enumerate(hide_values):
                    if hide is None:
                        hide_label = 'Full'
                    else:
                        hide_label = f'{hide:.2f}'
                    
                    # Show all models for all hide values
                    models_to_show = models
                    
                    n_models_to_show = len(models_to_show)
                    if n_models_to_show == 0:
                        position_offset += group_width + 0.5
                        continue
                    
                    # Calculate centering offset if we have fewer models than the maximum
                    centering_offset = (n_models - n_models_to_show) * 0.7 / 2
                    
                    # Add each model for this hide value
                    for local_model_idx, model in enumerate(models_to_show):
                        if hide is None:
                            hide_data = node_data[(node_data['hide'].isna()) & (node_data['model'] == model)]
                        else:
                            hide_data = node_data[(node_data['hide'] == hide) & (node_data['model'] == model)]
                        
                        if len(hide_data) == 0:
                            continue
                        
                        # Position with centering offset
                        pos = position_offset + centering_offset + local_model_idx * 0.7
                        positions.append(pos)
                        mean_val = hide_data[metric_col].values[0]
                        mean_values.append(mean_val)
                        
                        # Only add label for first hide (for legend)
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
                    
                    position_offset += group_width + 0.5  # Space between hide groups
                
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
                    
                    # Set x-axis labels for hide values
                    hide_labels = [f'{h:.2f}' if h is not None else 'Full' for h in hide_values]
                    # Calculate center positions for each hide group
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
                if metric_idx == 0:
                    ax.set_title(f'{node_count} Nodes', fontsize=18, fontweight='bold')
                
                if node_idx == 0:
                    ax.set_ylabel(metric_title, fontsize=18, fontweight='bold')
                
                # Set tick label sizes
                ax.tick_params(axis='y', labelsize=18)
                
                ax.grid(True, alpha=0.3, axis='y')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
        
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
            
            # Get unique hide values for this variant
            hide_values = sorted([h for h in variant_data['hide'].unique() if h is not None])
            if variant == 'base' or not hide_values:
                hide_values = [None]
            
            for hide in hide_values:
                if hide is None:
                    write_line(f"\nHIDE: Full Graph Knowledge")
                    hide_data = variant_data[variant_data['hide'].isna()]
                else:
                    write_line(f"\nHIDE: {hide}")
                    hide_data = variant_data[variant_data['hide'] == hide]
                
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
    print("Generating ComplexMech Benchmark Plots (hide fraction variations)")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # ComplexMech models with different hide fractions
    print("\n" + "="*80)
    print("Processing ComplexMech Models (hide variations)")
    print("="*80)
    
    # Plot all three models for comparison
    checkpoint_dirs = [
        checkpoint_base / "final_earlytest_16773480.0" / "complexmech_idk_high",
        checkpoint_base / "final_earlytest_16773478.0" / "complexmech_idk_high",
        checkpoint_base / "final_earlytest_hide_all_16771679.0" / "complexmech_idk_high",
    ]
    
    results_dict = load_benchmark_results(checkpoint_dirs)
    
    if results_dict:
        df = create_dataframe(results_dict)
        print(f"Models: {sorted(df['model'].unique())}")
        print(f"Node counts: {sorted(df['node_count'].unique())}")
        print(f"Variants: {sorted(df['variant'].unique())}")
        print(f"Hide values: {sorted([h for h in df['hide'].unique() if h is not None])}")
        
        plot_box_comparison_separate_variants(df, [2, 5, 10, 20, 35, 50], 
                                             title_suffix="ComplexMech Benchmark (Hide Fraction Variations)", 
                                             output_prefix="complexmech_hide")
        print_summary_statistics(df, title_suffix="ComplexMech Benchmark (Hide Fraction Variations)", 
                               output_prefix="complexmech_hide",
                               checkpoint_keys=[str(p) for p in checkpoint_dirs])
    else:
        print("No results found for ComplexMech models")
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
