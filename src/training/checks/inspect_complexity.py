#!/usr/bin/env python3
"""
Inspect Data Complexity and Learnability from YAML Config

This script analyzes the complexity and learnability of datasets sampled from
a prior distribution specified in a YAML config file.

Complexity: How well do Random Forests with different max_depth values perform?
Learnability: MSE ratio when training on quarter vs half of the data.

All metrics include bootstrap confidence intervals.

Usage:
    python inspect_complexity.py --config path/to/config.yaml
"""

import argparse
import sys
import yaml
import warnings
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

# Suppress specific warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

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
        # Get repository root path dynamically
        file_path = Path(__file__)
        # For src/training/checks/
        repo_root = file_path.parent.parent.parent.parent
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


def remove_outliers(data, lower_percentile=2.5, upper_percentile=97.5):
    """
    Remove outliers from data using percentile-based trimming.
    
    Args:
        data: Array or list of data points
        lower_percentile: Lower percentile threshold (default: 2.5)
        upper_percentile: Upper percentile threshold (default: 97.5)
        
    Returns:
        Filtered data array with outliers removed
    """
    data = np.asarray(data)
    
    if len(data) == 0:
        return data
    
    # Remove NaN values first
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        return data
    
    # Calculate percentile bounds
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    
    # Filter data
    mask = (data >= lower_bound) & (data <= upper_bound)
    filtered_data = data[mask]
    
    return filtered_data


def bootstrap_ci(data, statistic_fn=np.mean, n_bootstrap=1000, confidence_level=0.95, random_state=None):
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Args:
        data: Array of data points
        statistic_fn: Function to calculate statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (statistic, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    rng = np.random.RandomState(random_state)
    data = np.asarray(data)
    n = len(data)
    
    # Calculate observed statistic
    observed_stat = statistic_fn(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(bootstrap_stats, lower_percentile)
    upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return observed_stat, lower, upper


class DataComplexityAnalyzer:
    """
    Analyzer for dataset complexity and learnability.
    
    Focuses on:
    1. Complexity: RF performance with different max_depth values
    2. Learnability: MSE ratio between quarter-data and half-data training
    """
    
    def __init__(self, config_path: str, seed: int = 42, n_estimators: int = 500):
        """
        Initialize the analyzer with a config file.
        
        Args:
            config_path: Path to YAML configuration file
            seed: Random seed for reproducibility
            n_estimators: Number of trees in Random Forest models
        """
        self.config_path = Path(config_path)
        self.seed = seed
        self.n_estimators = n_estimators
        self.config = None
        self.dataset_maker = None
        self.dataset = None
        self.dataloader = None
        self.current_output_file = None
        self.results_base_dir = None
        
        # Statistics storage
        self.complexity_results = {
            'max_depths': [],  # Will be set once
            'mse_by_depth': {},  # depth -> list of MSE values
            'r2_by_depth': {},   # depth -> list of R2 values
        }
        
        self.learnability_results = {
            'quarter_data_mse': [],
            'half_data_mse': [],
            'learnability_ratio': [],  # quarter_mse / half_mse
        }
        
        # Dataset characteristics
        self.dataset_stats = {
            'n_features': [],
            'n_train_samples': [],
            'n_test_samples': [],
        }
        
    def log(self, message: str):
        """Log message to both file and console."""
        if self.current_output_file:
            self.current_output_file.write(message + '\n')
            self.current_output_file.flush()
        print(message)
        
    def load_config(self):
        """Load and parse the YAML configuration."""
        self.log("\n[INFO] Loading configuration...")
        self.config = load_yaml_config(str(self.config_path))
        
        self.log("[OK] Configuration loaded successfully")
        self.log(f"[INFO] Experiment: {self.config.get('experiment_name', 'unnamed')}")
        self.log(f"[INFO] Description: {self.config.get('description', 'no description')}")
        
    def create_dataset_and_dataloader(self, batch_size=None):
        """Create dataset and dataloader from config."""
        self.log("\n[INFO] Creating dataset and dataloader...")
        
        # Extract config sections
        scm_config = self.config.get('scm_config', {})
        dataset_config = self.config.get('dataset_config', {})
        preprocessing_config = self.config.get('preprocessing_config', {})
        training_config = extract_config_values(self.config.get('training_config', {}))
        
        # Create dataset maker
        self.dataset_maker = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config
        )
        
        # Create dataset
        self.log(f"[INFO] Creating dataset with seed {self.seed}...")
        self.dataset = self.dataset_maker.create_dataset(seed=self.seed)
        self.log(f"[OK] Dataset created with {len(self.dataset)} samples")
        
        # Create dataloader
        if batch_size is None:
            batch_size = training_config.get('batch_size', 4)
            
        num_workers = 8
        
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
        
    def extract_datasets_from_batch(self, batch):
        """
        Extract individual datasets from a batch.
        
        Returns:
            List of (X_train, y_train, X_test, y_test) tuples
        """
        datasets = []
        
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            X_train, y_train, X_test, y_test = batch
            batch_size = X_train.shape[0]
            
            for i in range(batch_size):
                X_train_i = X_train[i].cpu().numpy()
                y_train_i = y_train[i].cpu().numpy().squeeze()
                X_test_i = X_test[i].cpu().numpy()
                y_test_i = y_test[i].cpu().numpy().squeeze()
                
                datasets.append((X_train_i, y_train_i, X_test_i, y_test_i))
        
        return datasets
    
    def filter_padding(self, X_train, y_train, X_test, y_test):
        """
        Remove padding from data (zero-variance features and zero-norm samples).
        
        Returns:
            Filtered (X_train, y_train, X_test, y_test) and number of active features
        """
        # Identify active features (non-zero variance)
        train_var = np.var(X_train, axis=0)
        test_var = np.var(X_test, axis=0)
        active_features = (train_var + test_var) > 1e-10
        
        if not np.any(active_features):
            return None, None, None, None, 0
        
        # Identify active samples (non-zero norm)
        X_train_active_feat = X_train[:, active_features]
        X_test_active_feat = X_test[:, active_features]
        
        train_norms = np.linalg.norm(X_train_active_feat, axis=1)
        test_norms = np.linalg.norm(X_test_active_feat, axis=1)
        
        active_train = train_norms > 1e-10
        active_test = test_norms > 1e-10
        
        if not np.any(active_train) or not np.any(active_test):
            return None, None, None, None, 0
        
        # Filter data
        X_train_filtered = X_train[active_train][:, active_features]
        y_train_filtered = y_train[active_train]
        X_test_filtered = X_test[active_test][:, active_features]
        y_test_filtered = y_test[active_test]
        
        n_active_features = np.sum(active_features)
        
        return X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered, n_active_features
    
    def analyze_complexity(self, X_train, y_train, X_test, y_test, max_depths=[1, 2, 3, 5, 10, 20, None]):
        """
        Analyze complexity by training RF with different max_depth values.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            max_depths: List of max_depth values to try (None = unlimited)
            
        Returns:
            Dict mapping max_depth -> (mse, r2)
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        results = {}
        
        for depth in max_depths:
            try:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=depth,
                    random_state=self.seed,
                    n_jobs=1
                )
                
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[depth] = (mse, r2)
                
            except Exception as e:
                self.log(f"[WARN] Complexity analysis failed for max_depth={depth}: {e}")
                results[depth] = (np.nan, np.nan)
        
        return results
    
    def analyze_learnability(self, X_train, y_train, X_test, y_test):
        """
        Analyze learnability: MSE ratio between quarter-data and half-data training.
        
        Returns:
            Tuple of (quarter_mse, half_mse, ratio)
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        
        n_train = len(X_train)
        
        if n_train < 8:
            return np.nan, np.nan, np.nan
        
        # Split training data into quarters
        quarter_size = n_train // 4
        half_size = n_train // 2
        
        try:
            # Train on quarter of data
            X_quarter = X_train[:quarter_size]
            y_quarter = y_train[:quarter_size]
            
            rf_quarter = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=None,
                random_state=self.seed,
                n_jobs=1
            )
            rf_quarter.fit(X_quarter, y_quarter)
            y_pred_quarter = rf_quarter.predict(X_test)
            quarter_mse = mean_squared_error(y_test, y_pred_quarter)
            
            # Train on half of data
            X_half = X_train[:half_size]
            y_half = y_train[:half_size]
            
            rf_half = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=None,
                random_state=self.seed,
                n_jobs=1
            )
            rf_half.fit(X_half, y_half)
            y_pred_half = rf_half.predict(X_test)
            half_mse = mean_squared_error(y_test, y_pred_half)
            
            # Calculate ratio (higher = more learnable)
            ratio = quarter_mse / half_mse if half_mse > 1e-10 else np.nan
            
            return quarter_mse, half_mse, ratio
            
        except Exception as e:
            self.log(f"[WARN] Learnability analysis failed: {e}")
            return np.nan, np.nan, np.nan
    
    def analyze_dataset(self, X_train, y_train, X_test, y_test, dataset_idx, batch_idx):
        """
        Analyze a single dataset for complexity and learnability.
        """
        # Filter padding
        X_train_f, y_train_f, X_test_f, y_test_f, n_features = self.filter_padding(
            X_train, y_train, X_test, y_test
        )
        
        if X_train_f is None:
            self.log(f"[WARN] Dataset {dataset_idx} (batch {batch_idx}): No active data after filtering")
            return
        
        n_train = len(X_train_f)
        n_test = len(X_test_f)
        
        self.log(f"\n[INFO] Analyzing dataset {dataset_idx} (batch {batch_idx}):")
        self.log(f"[INFO]   Features: {n_features}, Train: {n_train}, Test: {n_test}")
        
        # Store dataset characteristics
        self.dataset_stats['n_features'].append(n_features)
        self.dataset_stats['n_train_samples'].append(n_train)
        self.dataset_stats['n_test_samples'].append(n_test)
        
        # Complexity analysis
        max_depths = [1, 2, 3, 5, 10, 20, None]
        complexity_results = self.analyze_complexity(X_train_f, y_train_f, X_test_f, y_test_f, max_depths)
        
        self.log("[INFO]   Complexity (MSE by max_depth):")
        for depth, (mse, r2) in complexity_results.items():
            depth_str = 'unlimited' if depth is None else str(depth)
            self.log(f"[INFO]     max_depth={depth_str:>9}: MSE={mse:.4f}, R²={r2:.4f}")
            
            # Store results
            if depth not in self.complexity_results['mse_by_depth']:
                self.complexity_results['mse_by_depth'][depth] = []
                self.complexity_results['r2_by_depth'][depth] = []
            
            self.complexity_results['mse_by_depth'][depth].append(mse)
            self.complexity_results['r2_by_depth'][depth].append(r2)
        
        # Store max_depths list (only once)
        if not self.complexity_results['max_depths']:
            self.complexity_results['max_depths'] = max_depths
        
        # Learnability analysis
        quarter_mse, half_mse, ratio = self.analyze_learnability(X_train_f, y_train_f, X_test_f, y_test_f)
        
        self.log("[INFO]   Learnability:")
        self.log(f"[INFO]     Quarter-data MSE: {quarter_mse:.4f}")
        self.log(f"[INFO]     Half-data MSE:    {half_mse:.4f}")
        self.log(f"[INFO]     Ratio:            {ratio:.4f} (higher = more learnable)")
        
        # Store results
        self.learnability_results['quarter_data_mse'].append(quarter_mse)
        self.learnability_results['half_data_mse'].append(half_mse)
        self.learnability_results['learnability_ratio'].append(ratio)
    
    def process_batches(self, n_batches, n_datasets_per_batch):
        """
        Process multiple batches and analyze datasets.
        """
        self.log(f"\n[INFO] Processing {n_batches} batches with {n_datasets_per_batch} datasets each...")
        
        batch_count = 0
        total_datasets = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            if batch_count >= n_batches:
                break
            
            self.log(f"\n{'='*60}")
            self.log(f"BATCH {batch_idx + 1}")
            self.log(f"{'='*60}")
            
            datasets = self.extract_datasets_from_batch(batch)
            self.log(f"[INFO] Extracted {len(datasets)} datasets from batch")
            
            for dataset_idx, (X_train, y_train, X_test, y_test) in enumerate(datasets):
                if dataset_idx >= n_datasets_per_batch:
                    break
                
                self.analyze_dataset(X_train, y_train, X_test, y_test, dataset_idx, batch_idx)
                total_datasets += 1
            
            batch_count += 1
        
        self.log(f"\n[OK] Processed {batch_count} batches, {total_datasets} datasets total")
    
    def generate_summary_and_plots(self):
        """
        Generate summary statistics with bootstrap CIs and create plots.
        """
        self.log("\n" + "="*80)
        self.log("                    SUMMARY STATISTICS WITH CONFIDENCE INTERVALS")
        self.log("="*80)
        
        # Create summary directory
        summary_dir = Path(self.results_base_dir) / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        n_datasets = len(self.dataset_stats['n_features'])
        self.log(f"\n[INFO] Total datasets analyzed: {n_datasets}")
        
        # Dataset characteristics
        self.log("\n[INFO] Dataset Characteristics:")
        
        feat_mean, feat_lower, feat_upper = bootstrap_ci(self.dataset_stats['n_features'])
        self.log(f"[INFO]   Features: {feat_mean:.1f} [{feat_lower:.1f}, {feat_upper:.1f}] 95% CI")
        
        train_mean, train_lower, train_upper = bootstrap_ci(self.dataset_stats['n_train_samples'])
        self.log(f"[INFO]   Train samples: {train_mean:.1f} [{train_lower:.1f}, {train_upper:.1f}] 95% CI")
        
        test_mean, test_lower, test_upper = bootstrap_ci(self.dataset_stats['n_test_samples'])
        self.log(f"[INFO]   Test samples: {test_mean:.1f} [{test_lower:.1f}, {test_upper:.1f}] 95% CI")
        
        # Complexity Analysis Summary
        self.log("\n" + "="*80)
        self.log("COMPLEXITY ANALYSIS (MSE by max_depth)")
        self.log("="*80)
        
        max_depths = self.complexity_results['max_depths']
        mse_means = []
        mse_lowers = []
        mse_uppers = []
        r2_means = []
        r2_lowers = []
        r2_uppers = []
        
        for depth in max_depths:
            depth_str = 'unlimited' if depth is None else str(depth)
            
            # MSE statistics - remove outliers before analysis
            mse_data = [x for x in self.complexity_results['mse_by_depth'][depth] if not np.isnan(x)]
            if mse_data:
                mse_data_clean = remove_outliers(mse_data)
                n_removed = len(mse_data) - len(mse_data_clean)
                
                mse_mean, mse_lower, mse_upper = bootstrap_ci(mse_data_clean)
                mse_means.append(mse_mean)
                mse_lowers.append(mse_lower)
                mse_uppers.append(mse_upper)
                
                self.log(f"[INFO]   max_depth={depth_str:>9}: MSE = {mse_mean:.4f} [{mse_lower:.4f}, {mse_upper:.4f}] 95% CI (removed {n_removed} outliers)")
            else:
                mse_means.append(np.nan)
                mse_lowers.append(np.nan)
                mse_uppers.append(np.nan)
            
            # R² statistics - remove outliers before analysis
            r2_data = [x for x in self.complexity_results['r2_by_depth'][depth] if not np.isnan(x)]
            if r2_data:
                r2_data_clean = remove_outliers(r2_data)
                n_removed = len(r2_data) - len(r2_data_clean)
                
                r2_mean, r2_lower, r2_upper = bootstrap_ci(r2_data_clean)
                r2_means.append(r2_mean)
                r2_lowers.append(r2_lower)
                r2_uppers.append(r2_upper)
                
                self.log(f"[INFO]                      R² = {r2_mean:.4f} [{r2_lower:.4f}, {r2_upper:.4f}] 95% CI (removed {n_removed} outliers)")
            else:
                r2_means.append(np.nan)
                r2_lowers.append(np.nan)
                r2_uppers.append(np.nan)
        
        # Create complexity plots
        self.log("\n[INFO] Creating complexity plots...")
        
        # Plot 1: MSE vs max_depth with error bars
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Convert None to a large number for plotting
        depth_labels = ['1', '2', '3', '5', '10', '20', 'unlimited']
        x_positions = list(range(len(max_depths)))
        
        # MSE plot
        ax1.errorbar(x_positions, mse_means, 
                    yerr=[np.array(mse_means) - np.array(mse_lowers),
                          np.array(mse_uppers) - np.array(mse_means)],
                    marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                    color='steelblue', ecolor='darkblue', alpha=0.8)
        ax1.set_xlabel('max_depth', fontsize=12)
        ax1.set_ylabel('MSE (lower is better)', fontsize=12)
        ax1.set_title('Complexity: MSE vs max_depth with 95% CI\n(outliers removed)', fontsize=14)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(depth_labels)
        ax1.grid(alpha=0.3)
        
        # R² plot
        ax2.errorbar(x_positions, r2_means,
                    yerr=[np.array(r2_means) - np.array(r2_lowers),
                          np.array(r2_uppers) - np.array(r2_means)],
                    marker='s', capsize=5, capthick=2, linewidth=2, markersize=8,
                    color='forestgreen', ecolor='darkgreen', alpha=0.8)
        ax2.set_xlabel('max_depth', fontsize=12)
        ax2.set_ylabel('R² (higher is better)', fontsize=12)
        ax2.set_title('Complexity: R² vs max_depth with 95% CI\n(outliers removed)', fontsize=14)
        ax2.set_xticks(x_positions)
        ax2.set_xticklabels(depth_labels)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        complexity_path = summary_dir / 'complexity_analysis.png'
        plt.savefig(complexity_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"[OK] Complexity plot saved to: {complexity_path}")
        
        # Learnability Analysis Summary
        self.log("\n" + "="*80)
        self.log("LEARNABILITY ANALYSIS")
        self.log("="*80)
        
        # Filter out NaN values and remove outliers
        quarter_mse_data = [x for x in self.learnability_results['quarter_data_mse'] if not np.isnan(x)]
        half_mse_data = [x for x in self.learnability_results['half_data_mse'] if not np.isnan(x)]
        ratio_data = [x for x in self.learnability_results['learnability_ratio'] if not np.isnan(x)]
        
        if quarter_mse_data:
            quarter_mse_clean = remove_outliers(quarter_mse_data)
            n_removed = len(quarter_mse_data) - len(quarter_mse_clean)
            qmse_mean, qmse_lower, qmse_upper = bootstrap_ci(quarter_mse_clean)
            self.log(f"[INFO]   Quarter-data MSE: {qmse_mean:.4f} [{qmse_lower:.4f}, {qmse_upper:.4f}] 95% CI (removed {n_removed} outliers)")
        
        if half_mse_data:
            half_mse_clean = remove_outliers(half_mse_data)
            n_removed = len(half_mse_data) - len(half_mse_clean)
            hmse_mean, hmse_lower, hmse_upper = bootstrap_ci(half_mse_clean)
            self.log(f"[INFO]   Half-data MSE:    {hmse_mean:.4f} [{hmse_lower:.4f}, {hmse_upper:.4f}] 95% CI (removed {n_removed} outliers)")
        
        if ratio_data:
            ratio_clean = remove_outliers(ratio_data)
            n_removed = len(ratio_data) - len(ratio_clean)
            ratio_mean, ratio_lower, ratio_upper = bootstrap_ci(ratio_clean)
            self.log(f"[INFO]   Learnability Ratio: {ratio_mean:.4f} [{ratio_lower:.4f}, {ratio_upper:.4f}] 95% CI (removed {n_removed} outliers)")
            self.log(f"[INFO]   (Ratio = Quarter_MSE / Half_MSE, higher = more learnable)")
        
        # Create learnability plots
        self.log("\n[INFO] Creating learnability plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # MSE comparison plot (using cleaned data from above)
        mse_categories = ['Quarter Data', 'Half Data']
        mse_values = [qmse_mean if quarter_mse_data else np.nan,
                     hmse_mean if half_mse_data else np.nan]
        mse_errors = [[qmse_mean - qmse_lower if quarter_mse_data else 0,
                       hmse_mean - hmse_lower if half_mse_data else 0],
                      [qmse_upper - qmse_mean if quarter_mse_data else 0,
                       hmse_upper - hmse_mean if half_mse_data else 0]]
        
        ax1.bar(mse_categories, mse_values, yerr=mse_errors, capsize=10,
               color=['coral', 'lightblue'], edgecolor='black', alpha=0.8)
        ax1.set_ylabel('MSE', fontsize=12)
        ax1.set_title('Learnability: MSE Comparison with 95% CI\n(outliers removed)', fontsize=14)
        ax1.grid(alpha=0.3, axis='y')
        
        # Learnability ratio distribution (using cleaned data)
        if ratio_data:
            ax2.hist(ratio_clean, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax2.axvline(ratio_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {ratio_mean:.3f}')
            ax2.axvline(ratio_lower, color='orange', linestyle=':', linewidth=2, label=f'95% CI: [{ratio_lower:.3f}, {ratio_upper:.3f}]')
            ax2.axvline(ratio_upper, color='orange', linestyle=':', linewidth=2)
            ax2.set_xlabel('Learnability Ratio', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Learnability Ratio Distribution\n(outliers removed)', fontsize=14)
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        learnability_path = summary_dir / 'learnability_analysis.png'
        plt.savefig(learnability_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.log(f"[OK] Learnability plot saved to: {learnability_path}")
        
        self.log("\n" + "="*80)
        self.log("[OK] Summary analysis complete")
        self.log("="*80)
    
    def run_analysis(self, n_batches, n_datasets_per_batch, results_base_dir):
        """
        Run complete complexity and learnability analysis.
        """
        self.results_base_dir = results_base_dir
        
        # Create main log file
        main_log_path = Path(self.results_base_dir) / "analysis_log.txt"
        main_log = open(main_log_path, 'w', encoding='utf-8')
        self.current_output_file = main_log
        
        self.log("=" * 80)
        self.log("      Data Complexity and Learnability Analysis")
        self.log("=" * 80)
        
        try:
            # Load config
            self.load_config()
            
            # Create dataset and dataloader
            self.create_dataset_and_dataloader(batch_size=n_datasets_per_batch)
            
            # Process batches
            self.process_batches(n_batches, n_datasets_per_batch)
            
            # Generate summary and plots
            self.generate_summary_and_plots()
            
            self.log("\n" + "=" * 80)
            self.log("[OK] Analysis completed successfully!")
            self.log("=" * 80)
            
        except Exception as e:
            self.log("\n" + "=" * 80)
            self.log("[ERROR] Analysis failed!")
            self.log("=" * 80)
            self.log(f"Error: {e}")
            import traceback
            self.log(traceback.format_exc())
            raise
            
        finally:
            if self.current_output_file and not self.current_output_file.closed:
                self.current_output_file.close()


def main():
    """Main entry point for complexity analysis."""
    
    # =============================================================
    # CONFIGURATION - Edit these variables to change behavior
    # =============================================================
    
    # Path to YAML config file (relative to repository root)
    CONFIG_PATH = 'experiments/FirstTests/configs/early_test2_experiments.yaml'
    
    # Random seed for reproducibility
    SEED = 42
    
    # Number of trees in Random Forest models
    N_ESTIMATORS = 500
    
    # Number of batches to sample from dataloader
    N_BATCHES = 2
    
    # Number of datasets to analyze per batch
    N_DATASETS_PER_BATCH = 50
    
    # Results base directory
    RESULTS_BASE_DIR = 'src/training/checks/ResultsComplexity'
    
    # =============================================================
    # END CONFIGURATION
    # =============================================================
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(RESULTS_BASE_DIR) / f'run_{timestamp}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("      Data Complexity and Learnability Analysis")
    print("=" * 80)
    print("Configuration:")
    print(f"  CONFIG_PATH = {CONFIG_PATH}")
    print(f"  SEED = {SEED}")
    print(f"  N_ESTIMATORS = {N_ESTIMATORS}")
    print(f"  N_BATCHES = {N_BATCHES}")
    print(f"  N_DATASETS_PER_BATCH = {N_DATASETS_PER_BATCH}")
    print(f"  RESULTS_DIR = {results_dir}")
    print()
    print("Analyzing:")
    print("  - Complexity: RF performance with different max_depth values")
    print("  - Learnability: MSE ratio (quarter-data / half-data training)")
    print("  - All metrics include bootstrap 95% confidence intervals")
    print("=" * 80)
    
    # Create analyzer
    analyzer = DataComplexityAnalyzer(
        config_path=CONFIG_PATH,
        seed=SEED,
        n_estimators=N_ESTIMATORS
    )
    
    # Run analysis
    analyzer.run_analysis(
        n_batches=N_BATCHES,
        n_datasets_per_batch=N_DATASETS_PER_BATCH,
        results_base_dir=results_dir
    )
    
    print("\n" + "=" * 80)
    print("Analysis complete! Results saved in:")
    print(f"  {results_dir}")
    print(f"  {results_dir}/summary/  <- Summary plots and statistics")
    print("=" * 80)


if __name__ == "__main__":
    main()
