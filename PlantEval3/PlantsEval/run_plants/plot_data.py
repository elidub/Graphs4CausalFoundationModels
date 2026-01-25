#!/usr/bin/env python3
"""
Simple script to load and visualize the plant CID benchmark data.
"""

from pathlib import Path
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
BASE_DIR = Path("/fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval3/PlantsEval")
DATA_DIR = BASE_DIR / "plant_data"
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_benchmark(filepath):
    """Load benchmark data."""
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
    
    # Check if it's a CID_Benchmark object with realizations
    if hasattr(data, 'realizations'):
        return data.realizations
    elif isinstance(data, dict) and 'realizations' in data:
        return data['realizations']
    else:
        raise ValueError(f"Unknown data format: {type(data)}")


def plot_realization(realization, realization_idx=0):
    """Create comprehensive plots for a single realization."""
    
    # Extract data
    X_train = realization.X_train
    t_train = realization.t_train
    y_train = realization.y_train
    X_test = realization.X_test
    t_test = realization.t_test
    true_cid = realization.true_cid
    
    # Get feature names from the dataframes if available
    # Note: 'X' and 'Y' in the dataframe are spatial position features (not the outcome)
    # The treatment is Light_Intensity_umols and the outcome is Biomass_g
    feature_names = ['X (position)', 'Y (position)', 'Bed_Position', 
                     'Red_Light_Intensity', 'Blue_Light_Intensity', 
                     'Far_Red_Light_Intensity']
    treatment_name = 'Light_Intensity'
    outcome_name = 'Biomass (g)'
    
    n_features = X_train.shape[1]
    treatment_range = (t_train.min(), t_train.max())
    
    print(f"\nRealization {realization_idx}:")
    print(f"  Train: X_train={X_train.shape}, t_train={t_train.shape}, y_train={y_train.shape}")
    print(f"  Test:  X_test={X_test.shape}, t_test={t_test.shape}, true_cid={true_cid.shape}")
    print(f"  Number of features: {n_features}")
    print(f"  Treatment range: [{treatment_range[0]:.3f}, {treatment_range[1]:.3f}]")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Treatment distribution (train vs test) - CONTINUOUS
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(t_train, bins=50, alpha=0.7, density=True, label='Train', color='blue')
    ax1.hist(t_test, bins=50, alpha=0.5, density=True, label='Test', color='orange')
    ax1.set_xlabel(treatment_name)
    ax1.set_ylabel('Density')
    ax1.set_title('Treatment Distribution (Continuous)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Outcome vs Treatment scatter (train) - CONTINUOUS
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(t_train, y_train, alpha=0.3, s=10, color='blue')
    ax2.set_xlabel(treatment_name)
    ax2.set_ylabel(outcome_name)
    ax2.set_title(f'{outcome_name} vs {treatment_name} (Train)')
    ax2.grid(True, alpha=0.3)
    
    # 3. True CID vs Treatment scatter (test) - CONTINUOUS
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(t_test, true_cid, alpha=0.3, s=10, color='orange')
    ax3.set_xlabel(treatment_name)
    ax3.set_ylabel('True CID')
    ax3.set_title(f'True CID vs {treatment_name} (Test)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature correlation matrix (train)
    ax4 = plt.subplot(3, 3, 4)
    # Sample subset if too many features
    n_features_to_show = min(10, n_features)
    corr_matrix = np.corrcoef(X_train[:, :n_features_to_show].T)
    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax4.set_title(f'Feature Correlation Matrix')
    ax4.set_xlabel('Feature Index')
    ax4.set_ylabel('Feature Index')
    # Add feature names as tick labels if available
    if len(feature_names) >= n_features_to_show:
        ax4.set_xticks(range(n_features_to_show))
        ax4.set_yticks(range(n_features_to_show))
        ax4.set_xticklabels([name[:15] for name in feature_names[:n_features_to_show]], rotation=45, ha='right', fontsize=8)
        ax4.set_yticklabels([name[:15] for name in feature_names[:n_features_to_show]], fontsize=8)
    plt.colorbar(im, ax=ax4)
    
    # 5. Feature distributions (first 4 features)
    for i in range(min(4, n_features)):
        ax = plt.subplot(3, 3, 5 + i)
        ax.hist(X_train[:, i], bins=30, alpha=0.7, label='Train', color='blue')
        ax.hist(X_test[:, i], bins=30, alpha=0.5, label='Test', color='orange')
        feature_label = feature_names[i] if i < len(feature_names) else f'Feature {i}'
        ax.set_xlabel(feature_label, fontsize=9)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{feature_label} Distribution', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. Mean outcome by treatment with binning (train) - CONTINUOUS
    ax9 = plt.subplot(3, 3, 9)
    # Create bins for continuous treatment
    n_bins = 20
    bins = np.linspace(t_train.min(), t_train.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(t_train, bins)
    
    mean_outcomes = []
    std_outcomes = []
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            mean_outcomes.append(np.mean(y_train[mask]))
            std_outcomes.append(np.std(y_train[mask]))
        else:
            mean_outcomes.append(np.nan)
            std_outcomes.append(np.nan)
    
    # Filter out NaN values
    valid_mask = ~np.isnan(mean_outcomes)
    bin_centers = bin_centers[valid_mask]
    mean_outcomes = np.array(mean_outcomes)[valid_mask]
    std_outcomes = np.array(std_outcomes)[valid_mask]
    
    ax9.errorbar(bin_centers, mean_outcomes, yerr=std_outcomes, fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax9.set_xlabel(treatment_name, fontsize=9)
    ax9.set_ylabel(f'Mean {outcome_name} ± Std')
    ax9.set_title(f'Mean {outcome_name} by {treatment_name} (Binned)', fontsize=10)
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"realization_{realization_idx}_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    # Additional plot: Scatter plots for first few features vs outcome
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(min(6, n_features)):
        ax = axes[i]
        # Color by treatment value (continuous colormap)
        scatter = ax.scatter(X_train[:, i], y_train, c=t_train, alpha=0.5, s=10, cmap='viridis')
        cbar = plt.colorbar(scatter, ax=ax, label=treatment_name)
        cbar.ax.tick_params(labelsize=8)
        feature_label = feature_names[i] if i < len(feature_names) else f'Feature {i}'
        ax.set_xlabel(feature_label, fontsize=9)
        ax.set_ylabel(outcome_name, fontsize=9)
        ax.set_title(f'{feature_label} vs {outcome_name}\n(colored by {treatment_name})', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"realization_{realization_idx}_feature_scatter.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()
    
    # Additional plot: Detailed correlation matrix with values
    fig3 = plt.figure(figsize=(12, 10))
    
    # Compute full correlation matrix including treatment and outcome
    # Combine all variables: features + treatment + outcome
    all_data_train = np.column_stack([X_train, t_train.reshape(-1, 1), y_train.reshape(-1, 1)])
    all_names = feature_names + [treatment_name, outcome_name]
    
    corr_matrix_full = np.corrcoef(all_data_train.T)
    
    # Create heatmap with annotations
    ax = plt.subplot(1, 1, 1)
    im = ax.imshow(corr_matrix_full, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_names)))
    ax.set_yticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(all_names, fontsize=9)
    
    # Add correlation values as text
    for i in range(len(all_names)):
        for j in range(len(all_names)):
            text = ax.text(j, i, f'{corr_matrix_full[i, j]:.2f}',
                          ha="center", va="center", color="black" if abs(corr_matrix_full[i, j]) < 0.5 else "white",
                          fontsize=8)
    
    ax.set_title('Full Correlation Matrix (Features + Treatment + Outcome)', fontsize=12, pad=20)
    plt.colorbar(im, ax=ax, label='Correlation')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"realization_{realization_idx}_correlation_detailed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    plt.close()


def main():
    """Main function to load and plot all data."""
    print("="*80)
    print("Plant CID Benchmark Data Visualization")
    print("="*80)
    
    # Find all data files
    data_files = list(DATA_DIR.glob("*.pkl"))
    print(f"\nFound {len(data_files)} data file(s):")
    for f in data_files:
        print(f"  - {f.name}")
    
    if not data_files:
        print("\nNo data files found!")
        return
    
    # Load and plot each dataset
    for data_file in data_files:
        print(f"\n{'='*80}")
        print(f"Processing: {data_file.name}")
        print(f"{'='*80}")
        
        try:
            realizations = load_benchmark(data_file)
            print(f"Loaded {len(realizations)} realizations")
            
            # Plot first 3 realizations (or all if fewer)
            n_to_plot = min(3, len(realizations))
            print(f"\nPlotting first {n_to_plot} realization(s)...")
            
            for i in range(n_to_plot):
                plot_realization(realizations[i], realization_idx=i)
            
        except Exception as e:
            print(f"Error processing {data_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
