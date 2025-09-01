#!/usr/bin/env python3
"""
Inspect Dataloader Samples from YAML Config

This script reads a YAML configuration file, creates a dataloader as in simple_run.py,
samples batches from the dataloader, and visualizes individual tabular datasets within 
each batch using pairwise scatterplots and correlation heatmaps.

Usage:
    python inspect_dataloader_samples.py --config path/to/config.yaml
    python inspect_dataloader_samples.py --config ../../../experiments/FirstTests/configs/early_test.yaml
"""

import sys
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from torch.utils.data import DataLoader

# Suppress specific warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message="Use subset (sliced data) of np.ndarray is not recommended")

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import required modules
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset


def load_yaml_config(config_path: str):
    """Load YAML configuration file."""
    # Convert to Path and make it relative to repository root
    if not Path(config_path).is_absolute():
        # Get repository root (3 levels up from src/training/checks/)
        repo_root = Path(__file__).parent.parent.parent.parent
        config_path = repo_root / config_path
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"[OK] Loaded config from: {config_path}")
    return config


def extract_config_values(config_dict):
    """Extract values from YAML config format (handles both 'value' and direct values)."""
    extracted = {}
    for key, value in config_dict.items():
        if isinstance(value, dict) and 'value' in value:
            extracted[key] = value['value']
        else:
            extracted[key] = value
    return extracted


class DataloaderDatasetVisualizer:
    """
    Visualizer for individual tabular datasets within dataloader batches.
    
    This class loads a YAML config, creates a dataloader, samples batches,
    and visualizes individual datasets within each batch using scatterplots
    and correlation heatmaps.
    """
    
    def __init__(self, config_path: str, seed: int = 42):
        """
        Initialize the visualizer with a config file.
        
        Args:
            config_path: Path to YAML configuration file
            seed: Random seed for reproducibility
        """
        self.config_path = Path(config_path)
        self.seed = seed
        self.config = None
        self.dataset_maker = None
        self.dataset = None
        self.dataloader = None
        self.current_output_file = None  # Current output file for logging
        self.main_log_file = None  # Main log file handle
        self.results_base_dir = None  # Base directory for all results
        
    def log(self, message: str):
        """Log message to both current output file and console."""
        # Write to log file
        if self.current_output_file:
            try:
                self.current_output_file.write(message + '\n')
                self.current_output_file.flush()
            except (ValueError, IOError) as e:
                # Handle case where file is closed or other IO error
                print(f"[WARNING] Could not write to log file: {e}")
        
        # Also print to console for better visibility
        print(message)
        
    def load_config(self):
        """Load and parse the YAML configuration."""
        self.log("\n[INFO] Loading configuration...")
        self.config = load_yaml_config(str(self.config_path))
        
        self.log("[OK] Configuration loaded successfully")
        self.log(f"[INFO] Experiment: {self.config.get('experiment_name', 'unnamed')}")
        self.log(f"[INFO] Description: {self.config.get('description', 'no description')}")
        
    def create_dataset_and_dataloader(self, n_datasets_per_batch=None):
        """Create dataset and dataloader from config (same as simple_run.py)."""
        self.log("\n[INFO] Creating dataset and dataloader...")
        
        # Extract config sections
        scm_config = self.config.get('scm_config', {})
        dataset_config = self.config.get('dataset_config', {})
        preprocessing_config = self.config.get('preprocessing_config', {})
        training_config = extract_config_values(self.config.get('training_config', {}))
        
        self.log(f"[INFO] SCM nodes: {scm_config.get('num_nodes', {}).get('value', 'unknown')}")
        self.log(f"[INFO] Dataset size: {dataset_config.get('dataset_size', {}).get('value', 'unknown')}")
        self.log(f"[INFO] Max features: {dataset_config.get('max_number_features', {}).get('value', 'unknown')}")
        
        # Create dataset maker (same as simple_run.py)
        self.dataset_maker = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config
        )
        
        # Create dataset
        self.log(f"[INFO] Creating dataset with seed {self.seed}...")
        self.dataset = self.dataset_maker.create_dataset(seed=self.seed)
        self.log(f"[OK] Dataset created with {len(self.dataset)} samples")
        
        # Create dataloader (same as simple_run.py)
        # Override batch_size with n_datasets_per_batch if provided
        if n_datasets_per_batch is not None:
            batch_size = n_datasets_per_batch
            self.log(f"[INFO] Overriding batch_size with N_DATASETS_PER_BATCH: {batch_size}")
        else:
            batch_size = training_config.get('batch_size', 4)
            
        num_workers = 0 # just use one worker
        
        self.log("[INFO] Creating dataloader:")
        self.log(f"[INFO]   Batch size: {batch_size}")
        self.log(f"[INFO]   Num workers: {num_workers}")
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0
        )
        self.log("[OK] Dataloader created successfully")
        
    def extract_tabular_data_from_batch(self, batch):
        """
        Extract individual tabular datasets from a batch.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            List of (X_train, y_train, X_test, y_test) tuples for each dataset in the batch
        """
        datasets = []
        
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            # SimplePFN format: [X_train, y_train, X_test, y_test]
            X_train, y_train, X_test, y_test = batch
            batch_size = X_train.shape[0]
            
            self.log("[INFO] SimplePFN format detected:")
            self.log(f"[INFO]   Batch size: {batch_size}")
            self.log(f"[INFO]   X_train shape: {X_train.shape}")
            self.log(f"[INFO]   y_train shape: {y_train.shape}")
            self.log(f"[INFO]   X_test shape: {X_test.shape}")
            self.log(f"[INFO]   y_test shape: {y_test.shape}")
            
            # Extract individual datasets from the batch
            for i in range(batch_size):
                # Get training data for this dataset
                X_train_i = X_train[i].detach().cpu().numpy()  # Shape: [n_samples, n_features]
                y_train_i = y_train[i].detach().cpu().numpy()  # Shape: [n_samples, 1] or [n_samples]
                X_test_i = X_test[i].detach().cpu().numpy()    # Shape: [n_samples, n_features]
                y_test_i = y_test[i].detach().cpu().numpy()    # Shape: [n_samples, 1] or [n_samples]
                
                # Combine X and y for visualization (squeeze if needed)
                if len(y_train_i.shape) > 1 and y_train_i.shape[1] == 1:
                    y_train_i = y_train_i.squeeze(1)
                if len(y_test_i.shape) > 1 and y_test_i.shape[1] == 1:
                    y_test_i = y_test_i.squeeze(1)
                
                datasets.append((X_train_i, y_train_i, X_test_i, y_test_i))
                
        else:
            self.log(f"[WARN] Unexpected batch format: {type(batch)}")
            if hasattr(batch, '__len__'):
                self.log(f"[WARN] Batch length: {len(batch)}")
            
        return datasets
    
    def visualize_dataset(self, X_train, y_train, X_test, y_test, dataset_idx, batch_idx, save_dir=None):
        """
        Visualize a single tabular dataset with scatterplots and correlation heatmap.
        All output is saved to files - nothing is displayed.
        
        Args:
            X_train: Training feature matrix [n_samples, n_features]
            y_train: Training target vector [n_samples]
            X_test: Test feature matrix [n_samples, n_features]
            y_test: Test target vector [n_samples]
            dataset_idx: Index of dataset within batch
            batch_idx: Index of batch
            save_dir: Directory to save plots and analysis
        """
        n_train_samples, n_features = X_train.shape
        n_test_samples = X_test.shape[0]
        
        self.log(f"[INFO] Visualizing dataset {dataset_idx} from batch {batch_idx}")
        
        # ============================================================================
        # COMPREHENSIVE DIMENSION ANALYSIS
        # ============================================================================
        self.log("="*80)
        self.log("                         DIMENSION ANALYSIS")
        self.log("="*80)
        
        # Original shapes from dataloader (including padding)
        self.log("[INFO] Original shapes from dataloader (with padding):")
        self.log(f"[INFO]   X_train: {X_train.shape} | X_test: {X_test.shape}")
        self.log(f"[INFO]   y_train: {y_train.shape} | y_test: {y_test.shape}")
        self.log(f"[INFO]   Total samples: Train={n_train_samples}, Test={n_test_samples}")
        self.log(f"[INFO]   Total features: {n_features}")
        
        # Config values for comparison
        self.log("[INFO] Config values:")
        self.log("[INFO]   max_number_samples: 100 (from early_test.yaml)")
        self.log("[INFO]   max_number_features: 10 (from early_test.yaml)")
        self.log("[INFO]   number_samples_per_dataset: uniform(50, 99)")
        
        # Data ranges
        self.log("[INFO] Data ranges:")
        self.log(f"[INFO]   X_train: [{X_train.min():.3f}, {X_train.max():.3f}]")
        self.log(f"[INFO]   y_train: [{y_train.min():.3f}, {y_train.max():.3f}]")
        self.log(f"[INFO]   X_test: [{X_test.min():.3f}, {X_test.max():.3f}]")
        self.log(f"[INFO]   y_test: [{y_test.min():.3f}, {y_test.max():.3f}]")
        
        # CRITICAL: Identify padded zeros from BasicProcessing._pad_data()
        # In BasicProcessing, data is padded to (max_num_samples, max_num_features) with zeros
        # We need to identify the actual data region vs padded zeros
        
        # Step 1: Identify active (non-padded) features
        # Features that are all zeros across all samples are padding
        train_feature_variances = np.var(X_train, axis=0)
        test_feature_variances = np.var(X_test, axis=0)
        combined_feature_variances = train_feature_variances + test_feature_variances
        active_features_mask = combined_feature_variances > 1e-10
        
        # Step 2: Identify active (non-padded) samples  
        # Samples that are all zeros across all active features are padding
        # First get active features to check sample activity properly
        if np.any(active_features_mask):
            X_train_active_features = X_train[:, active_features_mask]
            X_test_active_features = X_test[:, active_features_mask]
            
            # Check which samples have any non-zero values in active features
            train_sample_norms = np.linalg.norm(X_train_active_features, axis=1)
            test_sample_norms = np.linalg.norm(X_test_active_features, axis=1)
            active_train_samples_mask = train_sample_norms > 1e-10
            active_test_samples_mask = test_sample_norms > 1e-10
        else:
            # No active features at all
            active_train_samples_mask = np.zeros(n_train_samples, dtype=bool)
            active_test_samples_mask = np.zeros(n_test_samples, dtype=bool)
        
        # Calculate actual dimensions
        n_active_features = np.sum(active_features_mask)
        n_active_train_samples = np.sum(active_train_samples_mask)
        n_active_test_samples = np.sum(active_test_samples_mask)
        n_padded_features = n_features - n_active_features
        n_padded_train_samples = n_train_samples - n_active_train_samples
        n_padded_test_samples = n_test_samples - n_active_test_samples
        
        # Calculate fractions
        feature_padding_fraction = n_padded_features / n_features
        train_sample_padding_fraction = n_padded_train_samples / n_train_samples
        test_sample_padding_fraction = n_padded_test_samples / n_test_samples
        
        self.log("="*80)
        self.log("                    PADDING ANALYSIS SUMMARY")
        self.log("="*80)
        self.log("[INFO] FEATURES:")
        self.log(f"[INFO]   Total features (with padding): {n_features}")
        self.log(f"[INFO]   Active features (actual data): {n_active_features}")
        self.log(f"[INFO]   Padded features (zeros): {n_padded_features}")
        self.log(f"[INFO]   Feature padding fraction: {feature_padding_fraction:.1%}")
        
        self.log("[INFO] TRAIN SAMPLES:")
        self.log(f"[INFO]   Total train samples (with padding): {n_train_samples}")
        self.log(f"[INFO]   Active train samples (actual data): {n_active_train_samples}")
        self.log(f"[INFO]   Padded train samples (zeros): {n_padded_train_samples}")
        self.log(f"[INFO]   Train sample padding fraction: {train_sample_padding_fraction:.1%}")
        
        self.log("[INFO] TEST SAMPLES:")
        self.log(f"[INFO]   Total test samples (with padding): {n_test_samples}")
        self.log(f"[INFO]   Active test samples (actual data): {n_active_test_samples}")
        self.log(f"[INFO]   Padded test samples (zeros): {n_padded_test_samples}")
        self.log(f"[INFO]   Test sample padding fraction: {test_sample_padding_fraction:.1%}")
        
        # Original vs Config comparison
        total_original_samples = n_active_train_samples + n_active_test_samples
        self.log("="*80)
        self.log("                  CONFIG vs ACTUAL COMPARISON")
        self.log("="*80)
        self.log("[INFO] SAMPLE COUNT ANALYSIS:")
        self.log("[INFO]   Config max_number_samples: 100")
        self.log("[INFO]   Config expected range: 50-99 samples")
        self.log(f"[INFO]   Actual total samples: {total_original_samples}")
        self.log(f"[INFO]   Actual train samples: {n_active_train_samples}")
        self.log(f"[INFO]   Actual test samples: {n_active_test_samples}")
        
        self.log("[INFO] FEATURE COUNT ANALYSIS:")
        self.log("[INFO]   Config max_number_features: 10")
        self.log(f"[INFO]   Actual active features: {n_active_features}")
        self.log(f"[INFO]   Features lost to preprocessing: {10 - n_active_features}")
        
        # Try to understand what happened to the features
        # Based on BasicProcessing.py, features can be lost due to:
        # 1. Feature dropout (dropout_prob parameter)
        # 2. Target feature removal (one feature becomes the target)
        # 3. Original SCM having fewer features than max
        self.log("[INFO] PREPROCESSING IMPACT ANALYSIS:")
        self.log("[INFO]   Expected feature loss sources:")
        self.log("[INFO]     - Target feature removed: 1 feature")
        self.log(f"[INFO]     - Feature dropout (config 0.1): ~{int(0.1 * 10)} features expected")
        self.log("[INFO]     - Original SCM features: 3-8 (from scm_config.num_nodes)")
        expected_remaining = 10 - 1  # Target removed
        expected_after_dropout = expected_remaining * (1 - 0.1)  # 10% dropout
        self.log(f"[INFO]     - Expected remaining after target removal: {expected_remaining}")
        self.log(f"[INFO]     - Expected after 10% dropout: ~{expected_after_dropout:.1f}")
        self.log(f"[INFO]     - Actual remaining: {n_active_features}")
        
        # CRITICAL DEBUGGING: The shape inconsistency!
        self.log("[CRITICAL] SHAPE INCONSISTENCY DETECTED:")
        self.log("[CRITICAL]   Dataloader reported 9 features, config expects 10")
        self.log("[CRITICAL]   This suggests preprocessing already removed features BEFORE padding")
        self.log("[CRITICAL]   Likely causes:")
        self.log("[CRITICAL]     1. Original SCM generated < 10 features")
        self.log("[CRITICAL]     2. Feature dropout happened before padding")
        self.log("[CRITICAL]     3. Target feature was already removed")
        
        # Analyze the pattern of zeroed features
        zeroed_indices = np.where(~active_features_mask)[0]
        active_indices = np.where(active_features_mask)[0]
        self.log("[INFO] FEATURE PATTERN ANALYSIS:")
        self.log(f"[INFO]   Active feature positions: {active_indices.tolist()}")
        self.log(f"[INFO]   Zeroed feature positions: {zeroed_indices.tolist()}")
        if len(zeroed_indices) > 0:
            if np.all(zeroed_indices >= n_active_features):
                self.log("[INFO]   Pattern: All zeros at end → Likely padding-only")
                self.log(f"[CRITICAL]   CONCLUSION: Original data had {n_active_features} features")
                self.log("[CRITICAL]   Missing features were likely removed in BasicProcessing, not padding")
            elif np.all(zeroed_indices < n_active_features):
                self.log("[INFO]   Pattern: Zeros at beginning → Likely preprocessing dropout")
            else:
                self.log("[INFO]   Pattern: Mixed zeros → Combination of dropout + padding")
                
        # Additional analysis for test sample padding
        self.log("[CRITICAL] TEST SAMPLE PADDING ANALYSIS:")
        self.log(f"[CRITICAL]   Test samples show {test_sample_padding_fraction:.1%} padding ({n_padded_test_samples}/{n_test_samples} are zeros)")
        self.log(f"[CRITICAL]   This suggests the original dataset had {n_active_test_samples} test samples")
        self.log("[CRITICAL]   Expected from config: 30% of total samples should be test")
        self.log(f"[CRITICAL]   With {total_original_samples} total samples: 0.3 * {total_original_samples} = {int(0.3 * total_original_samples)} expected test samples")
        self.log(f"[CRITICAL]   But we have {n_active_test_samples} actual test samples!")
        if n_active_test_samples < int(0.3 * total_original_samples):
            self.log("[CRITICAL]   This indicates train_fraction = 0.7 was applied to a smaller dataset")
        else:
            self.log("[INFO]   Test sample count is reasonable for the dataset size")
        
        # Detailed feature analysis
        self.log("="*80)
        self.log("                    DETAILED FEATURE ANALYSIS")
        self.log("="*80)
        self.log(f"[DEBUG] Feature variances - Train: {train_feature_variances}")
        self.log(f"[DEBUG] Feature variances - Test: {test_feature_variances}")
        self.log(f"[DEBUG] Active features mask: {active_features_mask}")
        self.log(f"[DEBUG] Active feature indices: {np.where(active_features_mask)[0].tolist()}")
        self.log(f"[DEBUG] Active train samples: first 10 = {active_train_samples_mask[:10]}")
        self.log(f"[DEBUG] Active train samples: last 10 = {active_train_samples_mask[-10:]}")
        self.log(f"[DEBUG] Active test samples: first 10 = {active_test_samples_mask[:10]}")
        self.log(f"[DEBUG] Active test samples: last 10 = {active_test_samples_mask[-10:]}")
        
        self.log("="*80)
        
        if n_active_features == 0:
            self.log(f"[WARN] No active features for dataset {dataset_idx}, skipping visualization")
            return
        
        if n_active_train_samples == 0 or n_active_test_samples == 0:
            self.log(f"[WARN] No active samples for dataset {dataset_idx}, skipping visualization")
            return
        
        # Filter to only active (non-padded) data for analysis
        X_train_filtered = X_train[active_train_samples_mask][:, active_features_mask]
        y_train_filtered = y_train[active_train_samples_mask]
        X_test_filtered = X_test[active_test_samples_mask][:, active_features_mask]
        y_test_filtered = y_test[active_test_samples_mask]
        
        self.log("[INFO] After removing padding:")
        self.log(f"[INFO]   Filtered train: {X_train_filtered.shape[0]} samples × {X_train_filtered.shape[1]} features")
        self.log(f"[INFO]   Filtered test: {X_test_filtered.shape[0]} samples × {X_test_filtered.shape[1]} features")
        self.log(f"[INFO]   X_train_filtered range: [{X_train_filtered.min():.3f}, {X_train_filtered.max():.3f}]")
        self.log(f"[INFO]   X_test_filtered range: [{X_test_filtered.min():.3f}, {X_test_filtered.max():.3f}]")
        
        # Create DataFrame for easier plotting with only active features (use training data for viz)
        active_feature_names = [f'X{i}' for i in range(n_features) if active_features_mask[i]]
        df = pd.DataFrame(X_train_filtered, columns=active_feature_names)
        df['target'] = y_train_filtered
        
        # Predictability Analysis - Train on train, evaluate on test!
        self.log("[INFO] Analyzing target predictability (train->test)...")
        self._analyze_predictability_train_test(X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered, active_feature_names)
        
        # Create figure with subplots
        n_active_features = X_train_filtered.shape[1]
        
        # Simple Effective Scatterplot Matrix (replacing problematic seaborn pairplot)
        self.log("[INFO] Creating feature scatterplot matrix...")
        
        # Limit to first 6 features if too many for readability
        max_features_for_plots = min(6, n_active_features)
        selected_feature_names = active_feature_names[:max_features_for_plots]
        X_selected = X_train_filtered[:, :max_features_for_plots]
        
        self.log(f"[INFO] Plotting {max_features_for_plots} features")
        
        # Create scatterplot matrix: features vs features + features vs target
        n_plots = max_features_for_plots + 1  # +1 for target
        fig, axes = plt.subplots(n_plots, n_plots, figsize=(3*n_plots, 3*n_plots))
        
        # Handle single subplot case
        if n_plots == 1:
            axes = [[axes]]
        elif n_plots == 2:
            axes = axes.reshape(2, 2)
        
        # All data for plotting (features + target)
        plot_data = np.column_stack([X_selected, y_train_filtered])
        plot_labels = selected_feature_names + ['target']
        
        for i in range(n_plots):
            for j in range(n_plots):
                ax = axes[i][j]
                
                if i == j:
                    # Diagonal: histograms
                    ax.hist(plot_data[:, i], alpha=0.7, bins=20, color='skyblue')
                    ax.set_title(f'{plot_labels[i]}', fontsize=10)
                else:
                    # Off-diagonal: scatter plots
                    ax.scatter(plot_data[:, j], plot_data[:, i], alpha=0.6, s=15)
                    if i == n_plots - 1:  # Bottom row
                        ax.set_xlabel(plot_labels[j], fontsize=9)
                    if j == 0:  # Left column
                        ax.set_ylabel(plot_labels[i], fontsize=9)
                
                # Clean up axes
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
        
        plt.suptitle(f'Feature Relationships - Batch {batch_idx}, Dataset {dataset_idx}', 
                    fontsize=14, y=0.95)
        plt.tight_layout()
        
        # Save the figure to the specified directory
        if save_dir:
            save_path = Path(save_dir) / f'scatterplot_matrix_batch{batch_idx}_dataset{dataset_idx}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.log(f"[OK] Saved scatterplot matrix to: {save_path}")
        
        # Close the figure instead of displaying it
        plt.close(fig)
        
        # Simple correlation heatmap (only if we have multiple features)
        if n_active_features > 1:
            self.log("[INFO] Creating correlation heatmap...")
            
            fig2, ax = plt.subplots(figsize=(8, 6))
            
            # Calculate correlation matrix for selected features + target
            corr_data = pd.DataFrame(plot_data, columns=plot_labels)
            corr_matrix = corr_data.corr()
            
            # Create heatmap
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8},
                ax=ax
            )
            
            ax.set_title(f'Correlation Matrix - Batch {batch_idx}, Dataset {dataset_idx}', 
                        fontsize=14, pad=20)
            
            if save_dir:
                save_path = Path(save_dir) / f'correlation_batch{batch_idx}_dataset{dataset_idx}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.log(f"[OK] Saved correlation heatmap to: {save_path}")
            
            # Close the figure instead of displaying it
            plt.close(fig2)
            
            # Summary statistics
            self.log("[INFO] Dataset summary statistics:")
            self.log("[INFO]   Feature correlations with target:")
            target_corrs = corr_matrix['target'][:-1]  # Exclude target-target correlation
            for feat, corr in target_corrs.items():
                self.log(f"[INFO]     {feat}: {corr:.3f}")
        else:
            self.log("[INFO] Only 1 active feature, skipping correlation heatmap")
            
            # Simple statistics for single feature
            self.log("[INFO] Dataset summary statistics:")
            self.log(f"[INFO]   Single feature: {active_feature_names[0]}")
            self.log(f"[INFO]   Feature range: [{X_train_filtered.min():.3f}, {X_train_filtered.max():.3f}]")
            self.log(f"[INFO]   Feature std: {X_train_filtered.std():.3f}")
            correlation_with_target = np.corrcoef(X_train_filtered.flatten(), y_train_filtered)[0, 1]
            self.log(f"[INFO]   Correlation with target: {correlation_with_target:.3f}")
        
        self.log("")
    
    def _analyze_predictability(self, X, y, feature_names):
        """
        Analyze how predictable the target is using linear model and random forest.
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Target vector [n_samples]
            feature_names: List of feature names
        """
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            
            n_samples, n_features = X.shape
            
            # Skip if too few samples or features
            if n_samples < 10 or n_features == 0:
                self.log("[INFO] Too few samples/features for predictability analysis")
                return
            
            self.log(f"[INFO] Predictability analysis ({n_samples} samples, {n_features} features):")
            
            # Linear Model
            try:
                linear_model = LinearRegression()
                linear_model.fit(X, y)
                linear_r2 = linear_model.score(X, y)
                
                # Cross-validation if enough samples
                if n_samples >= 20:
                    cv_folds = min(5, n_samples // 4)
                    linear_cv_scores = cross_val_score(LinearRegression(), X, y, cv=cv_folds, scoring='r2')
                    linear_cv_mean = linear_cv_scores.mean()
                    linear_cv_std = linear_cv_scores.std()
                    self.log(f"[INFO]   Linear Model R²: {linear_r2:.3f} (train), {linear_cv_mean:.3f}±{linear_cv_std:.3f} (CV)")
                else:
                    self.log(f"[INFO]   Linear Model R²: {linear_r2:.3f} (train only)")
                    
            except Exception as e:
                self.log(f"[WARN]   Linear model failed: {e}")
            
            # Random Forest
            try:
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                rf_model.fit(X, y)
                rf_r2 = rf_model.score(X, y)
                
                # Cross-validation if enough samples
                if n_samples >= 20:
                    cv_folds = min(5, n_samples // 4)
                    rf_cv_scores = cross_val_score(
                        RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5), 
                        X, y, cv=cv_folds, scoring='r2'
                    )
                    rf_cv_mean = rf_cv_scores.mean()
                    rf_cv_std = rf_cv_scores.std()
                    self.log(f"[INFO]   Random Forest R²: {rf_r2:.3f} (train), {rf_cv_mean:.3f}±{rf_cv_std:.3f} (CV)")
                    
                    # Feature importance
                    if n_features <= 10:  # Only show for reasonable number of features
                        importances = rf_model.feature_importances_
                        self.log("[INFO]   Feature Importance (Random Forest):")
                        for feat, imp in zip(feature_names, importances):
                            self.log(f"[INFO]     {feat}: {imp:.3f}")
                else:
                    self.log(f"[INFO]   Random Forest R²: {rf_r2:.3f} (train only)")
                    
            except Exception as e:
                self.log(f"[WARN]   Random Forest failed: {e}")
                
        except ImportError:
            self.log("[WARN] sklearn not available for predictability analysis")
        except Exception as e:
            self.log(f"[WARN] Predictability analysis failed: {e}")
    
    def _analyze_predictability_train_test(self, X_train, y_train, X_test, y_test, feature_names):
        """
        Analyze how predictable the target is using train/test split.
        
        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training targets [n_samples]
            X_test: Test features [n_samples, n_features]
            y_test: Test targets [n_samples]
            feature_names: List of feature names
        """
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            
            n_train, n_features = X_train.shape
            n_test = X_test.shape[0]
            
            # Skip if too few samples or features
            if n_train < 5 or n_test < 5 or n_features == 0:
                self.log("[INFO] Too few samples/features for train-test analysis")
                return
            
            if n_features == 1:
                self.log("[INFO] Single feature analysis:")
                self.log("[INFO]   Note: Limited predictive analysis possible with only one feature")
            
            self.log(f"[INFO] Train-Test predictability analysis ({n_train} train, {n_test} test, {n_features} features):")
            
            # Linear Model
            try:
                linear_model = LinearRegression()
                linear_model.fit(X_train, y_train)
                
                # Get predictions
                train_pred = linear_model.predict(X_train)
                test_pred = linear_model.predict(X_test)
                
                # Evaluate on both train and test
                train_r2 = linear_model.score(X_train, y_train)
                test_r2 = linear_model.score(X_test, y_test)
                
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                # Debug: Show actual vs predicted values for a few samples
                self.log(f"[DEBUG] Linear Model - Train predictions (first 5): {train_pred[:5]}")
                self.log(f"[DEBUG] Linear Model - Train actual (first 5): {y_train[:5]}")
                self.log(f"[DEBUG] Linear Model - Test predictions (first 5): {test_pred[:5]}")
                self.log(f"[DEBUG] Linear Model - Test actual (first 5): {y_test[:5]}")
                
                self.log("[INFO]   Linear Model:")
                self.log(f"[INFO]     Train R²: {train_r2:.6f}, MSE: {train_mse:.8f}")
                self.log(f"[INFO]     Test R²: {test_r2:.6f}, MSE: {test_mse:.8f}")
                
                # Check for suspicious results
                if test_mse < 1e-6:
                    self.log(f"[WARN]     Suspiciously low test MSE: {test_mse:.2e}")
                if test_r2 < -10:
                    self.log(f"[WARN]     Extremely poor test R² (worse than baseline): {test_r2:.6f}")
                elif test_r2 > 0.99:
                    self.log(f"[WARN]     Suspiciously high test R² (possible overfitting): {test_r2:.6f}")
                    
            except Exception as e:
                self.log(f"[WARN]   Linear model failed: {e}")
            
            # Random Forest
            try:
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                rf_model.fit(X_train, y_train)
                
                # Get predictions
                train_pred = rf_model.predict(X_train)
                test_pred = rf_model.predict(X_test)
                
                # Evaluate on both train and test
                train_r2 = rf_model.score(X_train, y_train)
                test_r2 = rf_model.score(X_test, y_test)
                
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                self.log("[INFO]   Random Forest:")
                self.log(f"[INFO]     Train R²: {train_r2:.6f}, MSE: {train_mse:.8f}")
                self.log(f"[INFO]     Test R²: {test_r2:.6f}, MSE: {test_mse:.8f}")
                
                # Check for suspicious results  
                if test_mse < 1e-6:
                    self.log(f"[WARN]     Suspiciously low test MSE: {test_mse:.2e}")
                if test_r2 < -10:
                    self.log(f"[WARN]     Extremely poor test R² (worse than baseline): {test_r2:.6f}")
                elif test_r2 > 0.99:
                    self.log(f"[WARN]     Suspiciously high test R² (possible overfitting): {test_r2:.6f}")
                    
                # Feature importance
                if n_features <= 10:  # Only show for reasonable number of features
                    importances = rf_model.feature_importances_
                    self.log("[INFO]     Feature Importance:")
                    for feat, imp in zip(feature_names, importances):
                        self.log(f"[INFO]       {feat}: {imp:.3f}")
                    
            except Exception as e:
                self.log(f"[WARN]   Random Forest failed: {e}")
                
        except ImportError:
            self.log("[WARN] sklearn not available for train-test analysis")
        except Exception as e:
            self.log(f"[WARN] Train-test analysis failed: {e}")
    
    
    def sample_and_visualize_batches(self, n_batches=2, n_datasets_per_batch=3):
        """
        Sample batches from dataloader and visualize individual datasets.
        
        Args:
            n_batches: Number of batches to sample
            n_datasets_per_batch: Number of datasets to visualize per batch
        """
        self.log("\n[INFO] Sampling and visualizing batches...")
        self.log(f"[INFO] Target: {n_batches} batches, {n_datasets_per_batch} datasets per batch")
        
        batch_count = 0
        for batch_idx, batch in enumerate(self.dataloader):
            if batch_count >= n_batches:
                break
                
            # Create batch directory
            batch_dir = Path(self.results_base_dir) / f"batch_{batch_idx + 1}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            # Create batch log file
            batch_log_path = batch_dir / "batch_analysis.txt"
            batch_log_file = open(batch_log_path, 'w', encoding='utf-8')
            self.current_output_file = batch_log_file
            
            try:
                self.log("="*80)
                self.log(f"BATCH {batch_idx + 1} ANALYSIS OUTPUT")
                self.log(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.log("="*80)
                
                self.log("\n" + "="*50)
                self.log(f"BATCH {batch_idx + 1}")
                self.log("="*50)
                
                # Extract individual datasets from batch
                datasets = self.extract_tabular_data_from_batch(batch)
                
                if not datasets:
                    self.log(f"[WARN] No datasets extracted from batch {batch_idx}")
                    continue
                
                self.log(f"[OK] Extracted {len(datasets)} datasets from batch")
                
                # Visualize specified number of datasets
                n_to_visualize = min(n_datasets_per_batch, len(datasets))
                self.log(f"[INFO] Visualizing first {n_to_visualize} datasets...")
                
                for dataset_idx in range(n_to_visualize):
                    X_train, y_train, X_test, y_test = datasets[dataset_idx]
                    
                    # Create dataset directory
                    dataset_dir = batch_dir / f"dataset_{dataset_idx + 1}"
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create dataset log file and safely handle it
                    dataset_log_path = dataset_dir / "analysis.txt"
                    dataset_log_file = open(dataset_log_path, 'w', encoding='utf-8')
                    
                    # Store the previous output file to restore later
                    prev_output_file = self.current_output_file
                    self.current_output_file = dataset_log_file
                    
                    try:
                        self.log(f"\n--- Dataset {dataset_idx + 1} ---")
                        self.visualize_dataset(
                            X_train, y_train, X_test, y_test,
                            dataset_idx=dataset_idx + 1, 
                            batch_idx=batch_idx + 1,
                            save_dir=dataset_dir
                        )
                    finally:
                        # Make sure to close the dataset log file and restore previous file
                        if self.current_output_file:
                            self.current_output_file.close()
                        self.current_output_file = prev_output_file
                
                batch_count += 1
                
                self.log(f"\n[INFO] Completed analysis of {n_to_visualize} datasets in batch {batch_idx + 1}")
            finally:
                # Make sure to close the batch log file
                if batch_log_file:
                    batch_log_file.close()
        
        print(f"\n[OK] Completed visualization of {batch_count} batches")
    
    def run_visualization(self, n_batches=2, n_datasets_per_batch=3, results_base_dir=None):
        """
        Run complete visualization with output going to both files and console.
        Plots will be saved to files but NOT displayed interactively.
        
        Args:
            n_batches: Number of batches to sample
            n_datasets_per_batch: Number of datasets per batch to visualize
            results_base_dir: Base directory for all results (should be Path object)
        """
        # Store results base directory
        self.results_base_dir = results_base_dir
        
        # Create main log file in results directory
        main_log_path = Path(self.results_base_dir) / "analysis_log.txt"
        
        # Open the log file directly (not using 'with' to avoid premature closing)
        self.main_log_file = open(main_log_path, 'w', encoding='utf-8')
        self.current_output_file = self.main_log_file
        
        self.log("=" * 60)
        self.log("      DataLoader Dataset Visualization from YAML Config")
        self.log("=" * 60)
        
        try:
            # Step 1: Load config
            self.load_config()
            
            # Step 2: Create dataset and dataloader (override batch size)
            self.create_dataset_and_dataloader(n_datasets_per_batch=n_datasets_per_batch)
            
            # Step 3: Sample batches and visualize datasets
            self.sample_and_visualize_batches(
                n_batches=n_batches,
                n_datasets_per_batch=n_datasets_per_batch
            )
            
            self.log("=" * 60)
            self.log("[OK] Complete visualization finished successfully!")
            self.log("=" * 60)
            
        except Exception as e:
            self.log("=" * 60)
            self.log("[ERROR] Visualization failed!")
            self.log("=" * 60)
            self.log(f"Error: {e}")
            self.log(f"Error type: {type(e).__name__}")
            import traceback
            error_trace = traceback.format_exc()
            self.log("\nFull traceback:")
            self.log(error_trace)
            self.log("=" * 60)
            raise
        finally:
            # Make sure to close the main log file when done
            if hasattr(self, 'main_log_file') and self.main_log_file:
                self.main_log_file.close()
                self.main_log_file = None


def main():
    """Main entry point for dataloader dataset visualization."""
    
    # =============================================================
    # CONFIGURATION - Edit these variables to change behavior
    # =============================================================
    
    # Path to YAML config file (relative to repository root)
    CONFIG_PATH = 'experiments/FirstTests/configs/early_test.yaml'
    
    # Random seed for reproducibility
    SEED = 42
    
    # Number of batches to sample from dataloader
    N_BATCHES = 1
    
    # Number of datasets to visualize per batch
    N_DATASETS_PER_BATCH = 5
    
    # Results will be saved to organized folders (plots will NOT be displayed)
    # Format: checks/Results/run_YYYYMMDD_HHMMSS/batch_X/dataset_Y/
    RESULTS_BASE_DIR = 'src/training/checks/Results'
    
    # =============================================================
    # END CONFIGURATION
    # =============================================================
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(RESULTS_BASE_DIR) / f'run_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("      DataLoader Dataset Analysis - File Output Mode")
    print("=" * 60)
    print("Configuration:")
    print(f"  CONFIG_PATH = {CONFIG_PATH}")
    print(f"  SEED = {SEED}")
    print(f"  N_BATCHES = {N_BATCHES}")
    print(f"  N_DATASETS_PER_BATCH = {N_DATASETS_PER_BATCH}")
    print(f"  RESULTS_DIR = {results_dir}")
    print()
    print("Output will be saved to organized folders. Plots will NOT be displayed.")
    print("Analysis output will be written to both console and files.")
    print("=" * 60)
    
    # Create visualizer
    visualizer = DataloaderDatasetVisualizer(
        config_path=CONFIG_PATH,
        seed=SEED
    )
    
    # Run visualization with file output mode
    visualizer.run_visualization(
        n_batches=N_BATCHES,
        n_datasets_per_batch=N_DATASETS_PER_BATCH,
        results_base_dir=results_dir
    )
    
    print("=" * 60)
    print("Analysis complete! Check results in:")
    print(f"  {results_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
