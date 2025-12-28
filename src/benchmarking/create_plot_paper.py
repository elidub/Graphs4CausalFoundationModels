"""
Create publication-quality plots from benchmark results.

This script reads benchmark results CSV files and creates bar plots
comparing the performance of the custom PFN model against baselines.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Path to the benchmark results CSV file
RESULTS_FILE = "results/very_high_1000_samples_run_20251224_051835/benchmark_results.csv"


def load_summary_data(csv_path: str, metric: str = "mse") -> pd.DataFrame:
    """
    Load summary model data from benchmark results CSV.
    
    Args:
        csv_path: Path to the benchmark results CSV file
        metric: Metric to filter for ('mse' or 'r2')
    
    Returns:
        DataFrame with summary_model rows for the specified metric
    """
    df = pd.read_csv(csv_path)
    
    # Filter for summary_model rows with the specified metric
    summary_df = df[
        (df['process_id'] == 'summary_model') & 
        (df['metric'] == metric)
    ].copy()
    
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
    csv_path = script_dir / RESULTS_FILE
    
    if not csv_path.exists():
        print(f"Error: Results file not found at {csv_path}")
        print("Please update the RESULTS_FILE variable with the correct path.")
        return
    
    print(f"Loading benchmark results from: {csv_path}")
    
    # Load MSE summary data
    mse_summary = load_summary_data(csv_path, metric="mse")
    
    if mse_summary.empty:
        print("Error: No summary_model data found in the CSV file.")
        return
    
    print(f"Found {len(mse_summary)} models in the benchmark results.")
    print("\nModel performance (MSE - lower is better):")
    print(mse_summary[['model', 'mean', 'median', 'ci95_mean_low', 'ci95_mean_high']].to_string(index=False))
    
    # Create output directory
    output_dir = csv_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Generate MSE plot
    output_path = output_dir / "model_comparison_mse.png"
    plot_model_comparison(mse_summary, metric="mse", output_path=str(output_path))
    print(f"\nMSE plot created successfully!")


if __name__ == "__main__":
    main()
