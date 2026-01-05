#!/usr/bin/env python3
"""
Generate and save plots for LinGaus IDK benchmark results.

This script converts the v1.ipynb notebook analysis into a standalone script
that saves all plots to the 'plots' folder.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints"
checkpoint_base = Path(checkpoint_path)
output_dir = Path(__file__).parent / "plots"
output_dir.mkdir(exist_ok=True)

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
        
        if eval_step is not None:
            results_dir = ckpt_dir / f"lingaus_eval_step{eval_step}"
        else:
            results_dir = ckpt_dir / "lingaus_final"
        
        if not results_dir.exists():
            print(f"Missing results dir: {results_dir}")
            continue
        
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
            json_path = results_dir / "lingaus_benchmark_final.json"
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
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


def create_dataframe(results_dict, include_variants=True):
    """Convert results dictionary to pandas DataFrame."""
    data_rows = []
    
    for checkpoint_name, checkpoint_data in results_dict.items():
        model_name = extract_model_name(checkpoint_name)
        
        for variant, variant_data in checkpoint_data.items():
            for node_count, node_data in variant_data.items():
                row = {
                    'model': model_name,
                    'variant': variant,
                    'node_count': int(node_count),
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
    """Create separate box plot figures for each variant and save them."""
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    variants = sorted(df['variant'].unique())
    metrics_to_plot = ['mse', 'r2', 'nll']
    metric_titles = ['MSE', 'R²', 'NLL']
    
    variant_display_names = {
        'base': 'Base Causal Structure',
        'path_YT': 'Y → T Path',
        'path_TY': 'T → Y Path',
        'path_independent_TY': 'T ⊥ Y (Independent)'
    }
    
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
                
                node_data = variant_data[variant_data['node_count'] == node_count].copy()
                
                if len(node_data) == 0:
                    ax.text(0.5, 0.5, f'No data', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{node_count} Nodes')
                    continue
                
                node_data = node_data.sort_values('model')
                models = node_data['model'].unique()
                n_models = len(models)
                
                metric_col = f'{metric}_mean'
                if metric_col not in node_data.columns:
                    continue
                
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
                    mean_val = model_data[metric_col].values[0]
                    mean_values.append(mean_val)
                    
                    if model == 'baseline':
                        baseline_val = mean_val
                    
                    ci_lower_col = f'{metric}_mean_ci_lower'
                    ci_upper_col = f'{metric}_mean_ci_upper'
                    
                    if ci_lower_col in model_data.columns and ci_upper_col in model_data.columns:
                        ci_lower = model_data[ci_lower_col].values[0]
                        ci_upper = model_data[ci_upper_col].values[0]
                        
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
                    colors.append(MODEL_COLORS.get(str(model), '#808080'))
                
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
                    
                    for pos, mean_val in zip(positions, mean_values):
                        ax.text(pos, mean_val, f'{mean_val:.3f}', 
                               ha='center', va='bottom', fontsize=7, rotation=0,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor='none', alpha=0.7))
                
                if baseline_val is not None:
                    ax.axhline(y=baseline_val, color='gray', linestyle='--', 
                             linewidth=2, alpha=0.7, zorder=0)
                
                ax.set_xticks(range(n_models))
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
                
                if node_idx == 0:
                    ax.set_ylabel(metric_title, fontsize=11, fontweight='bold')
                
                if metric_idx == 0:
                    ax.set_title(f'{node_count} Nodes', fontsize=11, fontweight='bold')
                
                ax.grid(axis='y', alpha=0.3)
        
        variant_name = variant_display_names.get(variant, variant)
        main_title = f'Model Performance Box Plots: {variant_name} (from 95% CI)'
        if title_suffix:
            main_title = f'{main_title} - {title_suffix}'
        plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        filename = f"{output_prefix}_boxplot_{variant}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)


def print_summary_statistics(df, title_suffix="", output_prefix=""):
    """Print summary statistics and save to file."""
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


def main():
    """Main function to generate all plots."""
    print("="*80)
    print("Generating LinGaus IDK Benchmark Plots")
    print("="*80)
    
    # 5-node models
    print("\n" + "="*80)
    print("Processing 5-Node Models")
    print("="*80)
    
    new_keys_5node = [
       "lingaus_5node_benchmarked_baseline_16715905.0",
       "lingaus_5node_benchmarked_gcn_16715908.0",
        "lingaus_5node_benchmarked_gcn_and_hartatt_16715906.0",
        "lingaus_5node_benchmarked_gcn_and_softatt_16715907.0",
        "lingaus_5node_benchmarked_hardatt_16715909.0",
        "lingaus_5node_benchmarked_softatt_16715910.0",
    ]
    
    pattern_5node_regular = "lingaus_5node_benchmarked_*"
    results_dict_regular = load_benchmark_results(checkpoint_base, pattern_5node_regular)
    results_dict_filtered = {k: v for k, v in results_dict_regular.items() if k in new_keys_5node}
    
    df_5node = create_dataframe(results_dict_filtered)
    print(f"Models: {sorted(df_5node['model'].unique())}")
    print(f"Node counts: {sorted(df_5node['node_count'].unique())}")
    
    plot_box_comparison_separate_variants(df_5node, [2, 5], 
                                         title_suffix="5-Node Training", 
                                         output_prefix="5node")
    print_summary_statistics(df_5node, title_suffix="5-Node Training", 
                           output_prefix="5node")
    
    # 20-node models
    print("\n" + "="*80)
    print("Processing 20-Node Models")
    print("="*80)
    
    new_keys_20node = [
        "lingaus_20node_benchmarked_baseline_16715911.0",
        "lingaus_20node_benchmarked_gcn_16715914.0",
        "lingaus_20node_benchmarked_gcn_and_hartatt_16715912.0",
        "lingaus_20node_benchmarked_gcn_and_softatt_16715913.0",
        "lingaus_20node_benchmarked_hardatt_16715915.0",
        "lingaus_20node_benchmarked_softatt_16715916.0",
    ]
    
    pattern_20node_regular = "lingaus_20node_benchmarked_*"
    results_dict_regular = load_benchmark_results(checkpoint_base, pattern_20node_regular)
    results_dict_combined = {k: v for k, v in results_dict_regular.items() if k in new_keys_20node}
    
    df_20node = create_dataframe(results_dict_combined)
    print(f"Models: {sorted(df_20node['model'].unique())}")
    print(f"Node counts: {sorted(df_20node['node_count'].unique())}")
    
    plot_box_comparison_separate_variants(df_20node, [2, 5, 10, 20], 
                                         title_suffix="20-Node Training",
                                         output_prefix="20node")
    print_summary_statistics(df_20node, title_suffix="20-Node Training",
                           output_prefix="20node")
    
    # 20-node combined with fixed
    print("\n" + "="*80)
    print("Processing 20-Node Combined with Fixed")
    print("="*80)
    
    pids_to_include = ["16715911.0", "16715914.0", "16715912.0", "16715913.0", "16715915.0", "16715916.0"]
    regular_keys = [k for k in results_dict_regular.keys() 
                   if '20nodesfixed' not in k and any(pid in k for pid in pids_to_include)]
    results_dict_regular_filtered = {k: v for k, v in results_dict_regular.items() if k in regular_keys}
    
    pattern_20node_fixed = "lingaus_20node_benchmarked_gcn_and_softatt_20nodesfixed_*"
    results_dict_fixed = load_benchmark_results(checkpoint_base, pattern_20node_fixed)
    
    results_dict_fixed_renamed = {}
    for key, value in results_dict_fixed.items():
        new_key = key.replace("_20nodesfixed_", "_fixed_")
        results_dict_fixed_renamed[new_key] = value
    
    results_dict_combined_fixed = {**results_dict_regular_filtered, **results_dict_fixed_renamed}
    df_20node_combined = create_dataframe(results_dict_combined_fixed)
    
    plot_box_comparison_separate_variants(df_20node_combined, [2, 5, 10, 20],
                                         title_suffix="20-Node Training (Regular vs Fixed)",
                                         output_prefix="20node_combined")
    print_summary_statistics(df_20node_combined, 
                           title_suffix="20-Node Training (Regular vs Fixed)",
                           output_prefix="20node_combined")
    
    # 50-node models
    print("\n" + "="*80)
    print("Processing 50-Node Models")
    print("="*80)
    
    new_keys_50node = [
        "lingaus_50node_benchmarked_baseline_16715917.0",
        "lingaus_50node_benchmarked_gcn_16715920.0",
        "lingaus_50node_benchmarked_gcn_and_hartatt_16715918.0",
        "lingaus_50node_benchmarked_gcn_and_softatt_16715919.0",
        "lingaus_50node_benchmarked_hardatt_16715921.0",
        "lingaus_50node_benchmarked_softatt_16715922.0",
    ]
    
    pattern_50node_regular = "lingaus_50node_benchmarked_*"
    results_dict_regular = load_benchmark_results(checkpoint_base, pattern_50node_regular)
    results_dict_regular = {k: v for k, v in results_dict_regular.items() if k in new_keys_50node}
    
    df_50node = create_dataframe(results_dict_regular)
    print(f"Models: {sorted(df_50node['model'].unique())}")
    print(f"Node counts: {sorted(df_50node['node_count'].unique())}")
    
    plot_box_comparison_separate_variants(df_50node, [2, 5, 10, 20, 35, 50],
                                         title_suffix="50-Node Training",
                                         output_prefix="50node")
    print_summary_statistics(df_50node, title_suffix="50-Node Training",
                           output_prefix="50node")
    
    print("\n" + "="*80)
    print(f"All plots saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
