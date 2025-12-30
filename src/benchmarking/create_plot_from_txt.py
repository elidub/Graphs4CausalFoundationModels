"""
Create publication-quality plots from text-formatted benchmark results.

This script reads benchmark results from a text file and creates bar plots
comparing the performance of the custom PFN model against baselines.
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Path to the benchmark results text file
RESULTS_FILE = "results_10000_tabpfn_hacky.txt"


def parse_text_results(txt_path: str) -> pd.DataFrame:
    """
    Parse benchmark results from text file format.
    
    Args:
        txt_path: Path to the text file with benchmark results
    
    Returns:
        DataFrame with parsed model performance data
    """
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    data = []
    
    for line in lines:
        # Skip header and empty lines
        if not line.strip() or 'Detailed per-model' in line:
            continue
        
        # Parse line with format:
        # - model [metric] | mean=X (95% CI [low, high]), median=Y (95% CI [low, high]), ...
        match = re.match(
            r'\s*-\s*(\w+)\s+\[(\w+)\]\s+\|\s+mean=([-\d.]+)\s+\(95% CI \[([-\d.]+),\s*([-\d.]+)\]\)',
            line
        )
        
        if match:
            model, metric, mean, ci_low, ci_high = match.groups()
            
            # Also extract median if needed
            median_match = re.search(r'median=([-\d.]+)', line)
            median = float(median_match.group(1)) if median_match else None
            
            # Extract avg_rank_mse and avg_rank_r2
            rank_mse_match = re.search(r'avg_rank_mse=([-\d.]+)', line)
            rank_r2_match = re.search(r'avg_rank_r2=([-\d.]+)', line)
            
            avg_rank_mse = float(rank_mse_match.group(1)) if rank_mse_match else None
            avg_rank_r2 = float(rank_r2_match.group(1)) if rank_r2_match else None
            
            data.append({
                'model': model,
                'metric': metric,
                'mean': float(mean),
                'ci95_mean_low': float(ci_low),
                'ci95_mean_high': float(ci_high),
                'median': median,
                'avg_rank_mse': avg_rank_mse,
                'avg_rank_r2': avg_rank_r2
            })
    
    return pd.DataFrame(data)


def load_summary_data(txt_path: str, metric: str = "mse") -> pd.DataFrame:
    """
    Load summary model data from text file.
    
    Args:
        txt_path: Path to the text file
        metric: Metric to filter for ('mse' or 'r2')
    
    Returns:
        DataFrame with summary rows for the specified metric
    """
    df = parse_text_results(txt_path)
    
    # Filter for the specified metric
    summary_df = df[df['metric'] == metric].copy()
    
    # Rename models for better readability
    model_name_map = {
        'pfn': 'PFN (causal prior)',
        'tabpfn': 'TabPFN v2.5',
        'dtr': 'Decision Tree',
        'lr': 'Linear Regression',
        'rf': 'Random Forest',
        'gbrt': 'Gradient Boosting',
        'knn': 'KNN Regressor',
        'xgboost': 'XGBoost',
        'catboost': 'CatBoost',
        'adaboost': 'AdaBoost',
        'extratrees': 'Extra Trees',
        'ridge': 'Ridge Regression',
        'lasso': 'Lasso Regression',
        'bayesridge': 'Bayesian Ridge',
    }
    
    summary_df['model'] = summary_df['model'].replace(model_name_map)
    
    return summary_df


def plot_model_comparison(
    summary_df: pd.DataFrame,
    metric: str = "mse",
    output_path: str = None,
    figsize: tuple = (14, 8),
    highlight_model: str = "PFN (causal prior)",
):
    """
    Create a bar plot comparing model performance.
    
    Args:
        summary_df: DataFrame with summary statistics for each model
        metric: Metric being plotted ('mse' or 'r2')
        output_path: Path to save the figure (if None, displays instead)
        figsize: Figure size as (width, height)
        highlight_model: Model to highlight in a different color
    """
    # Sort by mean value (ascending for MSE, descending for R²)
    ascending = (metric.lower() == "mse")
    summary_df = summary_df.sort_values('mean', ascending=ascending)
    
    # Extract data for plotting
    models = summary_df['model'].values
    means = summary_df['mean'].values
    ci_low = summary_df['ci95_mean_low'].values
    ci_high = summary_df['ci95_mean_high'].values
    
    # Calculate error bars (distance from mean to CI bounds)
    err_low = means - ci_low
    err_high = ci_high - means
    errors = np.array([err_low, err_high])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create color array (highlight the custom model)
    colors = ['#d62728' if model == highlight_model else '#1f77b4' 
              for model in models]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, means, xerr=errors, capsize=8, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=2.5,
                   error_kw={'linewidth': 2.5, 'capthick': 2.5})
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=24)
    
    metric_label = "MSE" if metric.lower() == "mse" else "R²"
    ax.set_xlabel(f'{metric_label} (Mean ± 95% CI)', fontsize=24)
    ax.set_title(f'Model Performance Comparison ({metric_label})', 
                 fontsize=24, fontweight='bold')
    
    # Increase x-axis tick label size and make tick marks thicker
    ax.tick_params(axis='x', labelsize=24, width=2)
    ax.tick_params(axis='y', width=2)
    
    # Add grid for readability with thicker lines
    ax.grid(axis='x', alpha=0.5, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)
    
    # Make the spines (plot borders) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # Add legend to explain colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', linewidth=2, label='PFN (causal prior)'),
        Patch(facecolor='#1f77b4', edgecolor='black', linewidth=2, label='Baselines')
    ]
    legend = ax.legend(handles=legend_elements, loc='lower right' if metric.lower() == 'mse' else 'lower left', 
                       fontsize=24, edgecolor='black', fancybox=False, framealpha=1)
    legend.get_frame().set_linewidth(2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    return fig, ax


def main():
    """Main function to generate plots from benchmark results."""
    # Get the script directory and construct absolute path
    script_dir = Path(__file__).parent
    txt_path = script_dir / RESULTS_FILE
    
    if not txt_path.exists():
        print(f"Error: Results file not found at {txt_path}")
        print("Please update the RESULTS_FILE variable with the correct path.")
        return
    
    print(f"Loading benchmark results from: {txt_path}")
    
    # Load MSE summary data
    mse_summary = load_summary_data(txt_path, metric="mse")
    
    if mse_summary.empty:
        print("Error: No MSE data found in the text file.")
        return
    
    print(f"Found {len(mse_summary)} models in the benchmark results.")
    print("\nModel performance (MSE - lower is better):")
    print(mse_summary[['model', 'mean', 'median', 'ci95_mean_low', 'ci95_mean_high']].to_string(index=False))
    
    # Create output directory
    output_dir = txt_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Generate MSE plot
    output_path_mse = output_dir / "model_comparison_mse_from_txt.png"
    plot_model_comparison(mse_summary, metric="mse", output_path=str(output_path_mse))
    print(f"\nMSE plot created successfully!")
    
    # Load R² summary data
    r2_summary = load_summary_data(txt_path, metric="r2")
    
    if not r2_summary.empty:
        print("\nModel performance (R² - higher is better):")
        print(r2_summary[['model', 'mean', 'median', 'ci95_mean_low', 'ci95_mean_high']].to_string(index=False))
        
        # Generate R² plot
        output_path_r2 = output_dir / "model_comparison_r2_from_txt.png"
        plot_model_comparison(r2_summary, metric="r2", output_path=str(output_path_r2))
        print(f"R² plot created successfully!")


if __name__ == "__main__":
    main()
