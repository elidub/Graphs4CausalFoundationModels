#!/usr/bin/env python3
"""
Inspect Dataloader Samples from YAML Config

This script reads a YAML configuration file, creates a dataloader as in simple_run.py,
samples batches from the dataloader, and visualizes individual tabular datasets within 
each batch using pairwise scatterplots and correlation heatmaps.

Usage:
    You can directly run this file, or: 
    python inspect_dataloader_samples.py --config path/to/config.yaml
    python inspect_dataloader_samples.py --config ../../../experiments/FirstTests/configs/early_test.yaml
"""

import sys
from typing import Optional
import yaml
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message="Use subset (sliced data) of np.ndarray is not recommended")
from torch.utils.data import DataLoader
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import required modules
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
from priordata_processing.Datasets.MakeInterpolatedPurelyObservationalDataset import (
    MakeInterpolatedPurelyObservationalDataset,
)


def load_yaml_config(config_path: str):
    """Load YAML configuration file."""
    # Convert to Path and make it relative to repository root
    if not Path(config_path).is_absolute():
        # Get repository root path dynamically
        file_path = Path(__file__)
        # Check if we're in src/priors/training/checks/ or src/training/checks/
        if 'priors' in str(file_path):
            # For src/priors/training/checks/
            repo_root = file_path.parent.parent.parent.parent.parent
        else:
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


class DataloaderDatasetVisualizer:
    """
    Visualizer for individual tabular datasets within dataloader batches.
    
    This class loads a YAML config, creates a dataloader, samples batches,
    and visualizes individual datasets within each batch using scatterplots
    and correlation heatmaps.
    """
    
    @staticmethod
    def trim_for_plotting(data, lower_percentile=2.5, upper_percentile=97.5):
        """
        Trim data to remove extreme values for better visualization.
        Removes lower_percentile and (100 - upper_percentile) from both tails.
        
        Args:
            data: numpy array or list of values
            lower_percentile: Lower percentile to trim (default 2.5)
            upper_percentile: Upper percentile to trim (default 97.5)
            
        Returns:
            Tuple of (trimmed_data, lower_bound, upper_bound)
        """
        data_array = np.asarray(data)
        if len(data_array) == 0:
            return data_array, None, None
        
        lower_bound = np.percentile(data_array, lower_percentile)
        upper_bound = np.percentile(data_array, upper_percentile)
        
        mask = (data_array >= lower_bound) & (data_array <= upper_bound)
        trimmed = data_array[mask]
        
        return trimmed, lower_bound, upper_bound
    
    def __init__(self, config_path: str, seed: int = 42, fixed_t: Optional[float] = None):
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
        self.results_base_dir = None  # Base directory for all results
        self.fixed_t = fixed_t  # If set, fix curriculum time to this value across samples
        
        # Dictionary to collect statistics across all datasets for summary
        self.all_stats = {
            'linear_train_r2': [],
            'linear_test_r2': [],
            'linear_train_mse': [],
            'linear_test_mse': [],
            'rf_train_r2': [],
            'rf_test_r2': [],
            'rf_train_mse': [],
            'rf_test_mse': [],
            'linear_half_data_test_mse': [],  # For learnability metrics
            'rf_half_data_test_mse': [],      # For learnability metrics
            'linear_learnability': [],        # Ratio of full data MSE to half data MSE
            'rf_learnability': [],            # Ratio of full data MSE to half data MSE
            'target_ranges': [],
            'feature_counts': [],
            'train_sample_counts': [],
            'test_sample_counts': [],
            'feature_importances': {},
            'target_correlations': []
        }
        
    def log(self, message: str):
        """Log message to both current output file and console."""
        # Write to log file
        if self.current_output_file:
            self.current_output_file.write(message + '\n')
            self.current_output_file.flush()
        
        # Also print to console for better visibility
        print(message)
        
    def load_config(self):
        """Load and parse the YAML configuration."""
        self.log("\n[INFO] Loading configuration...")
        self.config = load_yaml_config(str(self.config_path))
        
        self.log("[OK] Configuration loaded successfully")
        self.log(f"[INFO] Experiment: {self.config.get('experiment_name', 'unnamed')}")
        self.log(f"[INFO] Description: {self.config.get('description', 'no description')}")
        
        # Write the complete YAML config to the log file
        self.log("\n" + "="*80)
        self.log("                    COMPLETE YAML CONFIGURATION")
        self.log("="*80)
        # Convert config back to YAML format for pretty printing
        config_yaml = yaml.dump(self.config, default_flow_style=False, sort_keys=False)
        self.log(config_yaml)
        self.log("="*80 + "\n")
        
    def create_dataset_and_dataloader(self, n_datasets_per_batch=None):
        """Create dataset and dataloader from config (same as simple_run.py)."""
        self.log("\n[INFO] Creating dataset and dataloader...")
        
        training_config = extract_config_values(self.config.get('training_config', {}))

        is_curriculum = (
            ('scm_config_t0' in self.config and 'scm_config_t1' in self.config)
            and ('dataset_config_t0' in self.config and 'dataset_config_t1' in self.config)
            and (
                ('preprocessing_config_t0' in self.config and 'preprocessing_config_t1' in self.config)
                or 'preprocessing_config' in self.config  # allow shared preprocessing config
            )
        )

        if is_curriculum:
            self.log("[INFO] Curriculum config detected (t0/t1). Building interpolated dataset…")
            scm_config_t0 = self.config.get('scm_config_t0', {})
            scm_config_t1 = self.config.get('scm_config_t1', {})
            dataset_config_t0 = self.config.get('dataset_config_t0', {})
            dataset_config_t1 = self.config.get('dataset_config_t1', {})
            # If only one preprocessing_config provided, reuse for both
            if 'preprocessing_config_t0' in self.config and 'preprocessing_config_t1' in self.config:
                preprocessing_config_t0 = self.config.get('preprocessing_config_t0', {})
                preprocessing_config_t1 = self.config.get('preprocessing_config_t1', {})
            else:
                preprocessing_shared = self.config.get('preprocessing_config', {})
                preprocessing_config_t0 = preprocessing_shared
                preprocessing_config_t1 = preprocessing_shared

            interpolation_function = str(self.config.get('interpolation_function', 'linear')).strip().lower().rstrip('.')

            # Log some basics
            self.log(f"[INFO] Interpolation schedule: {interpolation_function}")
            ds_size0 = dataset_config_t0.get('dataset_size', {}).get('value', 'unknown')
            ds_size1 = dataset_config_t1.get('dataset_size', {}).get('value', 'unknown')
            self.log(f"[INFO] Dataset sizes t0/t1: {ds_size0}/{ds_size1}")

            self.dataset_maker = MakeInterpolatedPurelyObservationalDataset(
                scm_config_t0=scm_config_t0,
                scm_config_t1=scm_config_t1,
                preprocessing_config_t0=preprocessing_config_t0,
                preprocessing_config_t1=preprocessing_config_t1,
                dataset_config_t0=dataset_config_t0,
                dataset_config_t1=dataset_config_t1,
                interpolation_function=interpolation_function,
                seed=self.seed,
            )

            self.log(f"[INFO] Creating curriculum dataset with seed {self.seed}…")
            self.dataset = self.dataset_maker.create_dataset()

            # If a fixed t is requested, override the dataset's time mapping
            if self.fixed_t is not None:
                fixed_t_val = float(self.fixed_t)
                self.log(f"[INFO] Forcing fixed curriculum time t={fixed_t_val:.3f} for all samples")

                # Override time function; to avoid multiprocessing pickling issues, we'll use num_workers=0
                def _fixed_time_for_index(_self, _idx: int) -> float:  # noqa: ANN001
                    return fixed_t_val

                # Bind to instance
                import types as _types
                self.dataset._time_for_index = _types.MethodType(_fixed_time_for_index, self.dataset)

        else:
            # Fallback to classic single-config dataset
            scm_config = self.config.get('scm_config', {})
            dataset_config = self.config.get('dataset_config', {})
            preprocessing_config = self.config.get('preprocessing_config', {})

            self.log(f"[INFO] SCM nodes: {scm_config.get('num_nodes', {}).get('value', 'unknown')}")
            self.log(f"[INFO] Dataset size: {dataset_config.get('dataset_size', {}).get('value', 'unknown')}")
            self.log(f"[INFO] Max features: {dataset_config.get('max_number_features', {}).get('value', 'unknown')}")

            self.dataset_maker = MakePurelyObservationalDataset(
                scm_config=scm_config,
                preprocessing_config=preprocessing_config,
                dataset_config=dataset_config,
            )
            self.log(f"[INFO] Creating dataset with seed {self.seed}…")
            self.dataset = self.dataset_maker.create_dataset(seed=self.seed)
            self.log(f"[OK] Dataset created with {len(self.dataset)} samples")
        
        # Create dataloader (same as simple_run.py)
        # Override batch_size with n_datasets_per_batch if provided
        if n_datasets_per_batch is not None:
            batch_size = n_datasets_per_batch
            self.log(f"[INFO] Overriding batch_size with N_DATASETS_PER_BATCH: {batch_size}")
        else:
            batch_size = training_config.get('batch_size', 4)

        # Use single-process DataLoader when forcing fixed t to avoid pickling local bound method
        num_workers = 0 if (is_curriculum and self.fixed_t is not None) else 4

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
        
        if isinstance(batch, (list, tuple)) and len(batch) in (4, 6):
            # SimplePFN format: [X_train, y_train, X_test, y_test]
            if len(batch) == 6:
                X_train, y_train, X_test, y_test, t_vals, a_vals = batch
                try:
                    # Log the first t/alpha in the batch for visibility
                    t0 = float(t_vals[0].item()) if hasattr(t_vals, 'shape') else float(t_vals)
                    a0 = float(a_vals[0].item()) if hasattr(a_vals, 'shape') else float(a_vals)
                    self.log(f"[INFO] Batch metadata: t[0]={t0:.3f}, alpha[0]={a0:.3f}")
                except Exception:
                    pass
            else:
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
        dataset_config = self.config.get('dataset_config', {})
        max_samples = dataset_config.get('max_number_samples', {}).get('value', 'unknown')
        max_features = dataset_config.get('max_number_features', {}).get('value', 'unknown')
        
        # Get distribution info for number of samples per dataset
        samples_dist_info = dataset_config.get('number_samples_per_dataset', {})
        dist_type = samples_dist_info.get('distribution', 'unknown')
        dist_params = samples_dist_info.get('distribution_parameters', {})
        
        # Format distribution parameters for display
        dist_params_str = ""
        if dist_type == "uniform" and 'low' in dist_params and 'high' in dist_params:
            dist_params_str = f"({dist_params['low']}, {dist_params['high']})"
        elif dist_params:
            # Generic parameter formatting for other distribution types
            param_strings = [f"{k}={v}" for k, v in dist_params.items()]
            dist_params_str = f"({', '.join(param_strings)})"
            
        self.log("[INFO] Config values:")
        self.log(f"[INFO]   max_number_samples: {max_samples}")
        self.log(f"[INFO]   max_number_features: {max_features}")
        self.log(f"[INFO]   number_samples_per_dataset: {dist_type}{dist_params_str}")
        
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
        
        # Counts for non-zero elements are calculated inline in the log statements below
        
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
        self.log(f"[INFO]   Total elements in X_train: {X_train.size}")
        self.log(f"[INFO]   Non-zero elements in X_train: {np.count_nonzero(X_train)} ({np.count_nonzero(X_train)/X_train.size:.1%})")
        self.log(f"[INFO]   Total elements in y_train: {y_train.size}")
        self.log(f"[INFO]   Non-zero elements in y_train: {np.count_nonzero(y_train)} ({np.count_nonzero(y_train)/y_train.size:.1%})")
        
        self.log("[INFO] TEST SAMPLES:")
        self.log(f"[INFO]   Total test samples (with padding): {n_test_samples}")
        self.log(f"[INFO]   Active test samples (actual data): {n_active_test_samples}")
        self.log(f"[INFO]   Padded test samples (zeros): {n_padded_test_samples}")
        self.log(f"[INFO]   Test sample padding fraction: {test_sample_padding_fraction:.1%}")
        self.log(f"[INFO]   Total elements in X_test: {X_test.size}")
        self.log(f"[INFO]   Non-zero elements in X_test: {np.count_nonzero(X_test)} ({np.count_nonzero(X_test)/X_test.size:.1%})")
        self.log(f"[INFO]   Total elements in y_test: {y_test.size}")
        self.log(f"[INFO]   Non-zero elements in y_test: {np.count_nonzero(y_test)} ({np.count_nonzero(y_test)/y_test.size:.1%})")
        self.log(f"[INFO]   Padded test samples (zeros): {n_padded_test_samples}")
        self.log(f"[INFO]   Test sample padding fraction: {test_sample_padding_fraction:.1%}")
        
        # Original vs Config comparison
        total_original_samples = n_active_train_samples + n_active_test_samples
        self.log("="*80)
        self.log("                  CONFIG vs ACTUAL COMPARISON")
        self.log("="*80)
        self.log("[INFO] SAMPLE COUNT ANALYSIS:")
        self.log(f"[INFO]   Config max_number_samples: {max_samples}")
        
        # Display expected range based on distribution parameters
        if dist_type == "uniform" and 'low' in dist_params and 'high' in dist_params:
            self.log(f"[INFO]   Config expected range: {dist_params['low']}-{dist_params['high']} samples")
        else:
            # Format distribution parameters for display similar to original code
            dist_desc = f"{dist_type}"
            if dist_params:
                param_strings = [f"{k}={v}" for k, v in dist_params.items()]
                dist_desc = f"{dist_type}({', '.join(param_strings)})"
            self.log(f"[INFO]   Config samples distribution: {dist_desc}")
            
        self.log(f"[INFO]   Actual total samples: {total_original_samples}")
        self.log(f"[INFO]   Actual train samples: {n_active_train_samples}")
        self.log(f"[INFO]   Actual test samples: {n_active_test_samples}")
        
        self.log("[INFO] FEATURE COUNT ANALYSIS:")
        self.log(f"[INFO]   Config max_number_features: {max_features}")
        self.log(f"[INFO]   Actual active features: {n_active_features}")
        self.log(f"[INFO]   Features lost to preprocessing: {int(max_features) - n_active_features if isinstance(max_features, (int, float)) else 'unknown'}")
        
        # Try to understand what happened to the features
        # Based on BasicProcessing.py, features can be lost due to:
        # 1. Feature dropout (dropout_prob parameter)
        # 2. Target feature removal (one feature becomes the target)
        # 3. Original SCM having fewer features than max
        # Get relevant preprocessing parameters
        preprocessing_config = self.config.get('preprocessing_config', {})
        feature_dropout_prob = preprocessing_config.get('feature_dropout_prob', {}).get('value', 0.0)
        try:
            feature_dropout_prob = float(feature_dropout_prob)
        except Exception:
            feature_dropout_prob = 0.0
        
        # Get SCM config for node info
        scm_config = self.config.get('scm_config', {})
        num_nodes = scm_config.get('num_nodes', {}).get('value', 'unknown')
        
        self.log("[INFO] PREPROCESSING IMPACT ANALYSIS:")
        self.log("[INFO]   Expected feature loss sources:")
        self.log("[INFO]     - Target feature removed: 1 feature")
        # Safe conversions for reporting
        try:
            max_features_int = int(max_features)
        except Exception:
            max_features_int = None
        if max_features_int is not None:
            drop_exp = int(feature_dropout_prob * max_features_int)
            self.log(f"[INFO]     - Feature dropout (config {feature_dropout_prob}): ~{drop_exp} features expected")
        else:
            self.log(f"[INFO]     - Feature dropout (config {feature_dropout_prob}): expected features unknown (max_number_features not provided)")
        self.log(f"[INFO]     - Original SCM features: {num_nodes} (from scm_config.num_nodes)")
        if max_features_int is not None:
            expected_remaining = max_features_int - 1  # Target removed
            self.log(f"[INFO]     - Expected remaining after target removal: {expected_remaining}")
            expected_after_dropout = expected_remaining * (1 - feature_dropout_prob)
            self.log(f"[INFO]     - Expected after dropout: ~{expected_after_dropout:.1f}")
        else:
            self.log("[INFO]     - Expected remaining after target removal: unknown")
            self.log("[INFO]     - Expected after dropout: unknown")
        self.log(f"[INFO]     - Actual remaining: {n_active_features}")
        
        # CRITICAL DEBUGGING: The shape inconsistency!
        self.log("[CRITICAL] SHAPE INCONSISTENCY DETECTED:")
        self.log(f"[CRITICAL]   Dataloader reported {n_features} features, config expects {max_features}")
        self.log("[CRITICAL]   This suggests preprocessing already removed features BEFORE padding")
        self.log("[CRITICAL]   Likely causes:")
        self.log("[CRITICAL]     1. Original SCM generated fewer features than configured")
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
        # Get the train_fraction from the dataset config
        train_fraction = dataset_config.get('train_fraction', {}).get('value', 0.7)
        test_fraction = 1 - train_fraction
        
        self.log(f"[CRITICAL]   Expected from config: {test_fraction:.1%} of total samples should be test")
        self.log(f"[CRITICAL]   With {total_original_samples} total samples: {test_fraction} * {total_original_samples} = {int(test_fraction * total_original_samples)} expected test samples")
        self.log(f"[CRITICAL]   But we have {n_active_test_samples} actual test samples!")
        if n_active_test_samples < int(test_fraction * total_original_samples):
            self.log("[CRITICAL]   This indicates train_fraction was applied to a smaller dataset")
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
        print(f"[DEBUG] Zeroed feature indices: {np.where(~active_features_mask)[0].tolist()}")
        
        # Detailed sample analysis (show first/last few)
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
        
        # Store dataset statistics for summary
        y_range = (y_train_filtered.min(), y_train_filtered.max())
        self.all_stats['feature_counts'].append(X_train_filtered.shape[1])
        self.all_stats['train_sample_counts'].append(X_train_filtered.shape[0])
        self.all_stats['test_sample_counts'].append(X_test_filtered.shape[0])
        self.all_stats['target_ranges'].append(y_range)
        
        # Create DataFrame for easier plotting with only active features (use training data for viz)
        active_feature_names = [f'X{i}' for i in range(n_features) if active_features_mask[i]]
        df = pd.DataFrame(X_train_filtered, columns=active_feature_names)
        df['target'] = y_train_filtered
        
        # ORIGINAL DATA PREDICTABILITY ANALYSIS (before filtering)
        # ========================================================
        self.log("="*80)
        self.log("               ORIGINAL DATA PREDICTABILITY ANALYSIS")
        self.log("               (BEFORE padding/filtering removal)")
        self.log("="*80)
        
        # Create feature names for original data (all features)
        original_feature_names = [f'X{i}' for i in range(n_features)]
        
        # Flatten y arrays to 1D for analysis (remove any extra dimensions)
        y_train_orig = y_train.squeeze() if len(y_train.shape) > 1 else y_train
        y_test_orig = y_test.squeeze() if len(y_test.shape) > 1 else y_test
        
        self.log(f"[INFO] Original data analysis using ALL data (including padding):")
        self.log(f"[INFO]   X_train: {X_train.shape}, y_train: {y_train_orig.shape}")
        self.log(f"[INFO]   X_test: {X_test.shape}, y_test: {y_test_orig.shape}")
        self.log(f"[INFO] This includes any padding that was added during data processing.")
        self.log(f"[INFO] Features with zero variance will still be included in the analysis.")
        
        # Analyze predictability using original unfiltered data
        self._analyze_predictability_train_test(X_train, y_train_orig, X_test, y_test_orig, original_feature_names)
        
        # FILTERED DATA PREDICTABILITY ANALYSIS (current approach)
        # =========================================================
        self.log("="*80)
        self.log("                FILTERED DATA PREDICTABILITY ANALYSIS")
        self.log("                (AFTER padding/filtering removal)")
        self.log("="*80)
        
        # Predictability Analysis - Train on train, evaluate on test!
        self.log("[INFO] Analyzing target predictability (train->test) on filtered data...")
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
        
        # Trim data for plotting (remove 2.5% extremes from each end)
        plot_data_trimmed = np.copy(plot_data)
        for col_idx in range(plot_data.shape[1]):
            trimmed_col, _, _ = self.trim_for_plotting(plot_data[:, col_idx])
            if len(trimmed_col) > 0:
                # Find indices that remain after trimming
                lower = np.percentile(plot_data[:, col_idx], 2.5)
                upper = np.percentile(plot_data[:, col_idx], 97.5)
                mask = (plot_data[:, col_idx] >= lower) & (plot_data[:, col_idx] <= upper)
                # Keep only rows that fall within bounds for ALL columns
                if col_idx == 0:
                    combined_mask = mask
                else:
                    combined_mask = combined_mask & mask
        
        # Apply combined mask to all columns
        plot_data_trimmed = plot_data[combined_mask]
        
        self.log(f"[INFO] Trimmed data for visualization: {len(plot_data)} -> {len(plot_data_trimmed)} samples (removed {len(plot_data) - len(plot_data_trimmed)} extreme values)")
        
        for i in range(n_plots):
            for j in range(n_plots):
                ax = axes[i][j]
                
                if i == j:
                    # Diagonal: histograms with trimmed data
                    ax.hist(plot_data_trimmed[:, i], alpha=0.7, bins=20, color='skyblue')
                    ax.set_title(f'{plot_labels[i]}', fontsize=10)
                else:
                    # Off-diagonal: scatter plots with trimmed data
                    ax.scatter(plot_data_trimmed[:, j], plot_data_trimmed[:, i], alpha=0.6, s=15)
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
                # Store correlation data for summary
                self.all_stats['target_correlations'].append((feat, corr))
        else:
            self.log("[INFO] Only 1 active feature, skipping correlation heatmap")
            
            # Simple statistics for single feature
            self.log("[INFO] Dataset summary statistics:")
            self.log(f"[INFO]   Single feature: {active_feature_names[0]}")
            self.log(f"[INFO]   Feature range: [{X_train_filtered.min():.3f}, {X_train_filtered.max():.3f}]")
            self.log(f"[INFO]   Feature std: {X_train_filtered.std():.3f}")
            correlation_with_target = np.corrcoef(X_train_filtered.flatten(), y_train_filtered)[0, 1]
            self.log(f"[INFO]   Correlation with target: {correlation_with_target:.3f}")
            
            # Store correlation data for summary
            self.all_stats['target_correlations'].append((active_feature_names[0], correlation_with_target))
        
        print()
    
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
            from sklearn.metrics import r2_score
            
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
                # Use sklearn r2_score instead of model.score()
                y_pred = linear_model.predict(X)
                linear_r2 = r2_score(y, y_pred)
                
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
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                rf_model.fit(X, y)
                # Use sklearn r2_score instead of model.score()
                y_pred = rf_model.predict(X)
                rf_r2 = r2_score(y, y_pred)
                
                # Cross-validation if enough samples
                if n_samples >= 20:
                    cv_folds = min(5, n_samples // 4)
                    rf_cv_scores = cross_val_score(
                        RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10), 
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
            from sklearn.metrics import mean_squared_error, r2_score
            import numpy as np
            
            n_train, n_features = X_train.shape
            n_test = X_test.shape[0]
            
            # Skip if too few samples or features
            if n_train < 5 or n_test < 5 or n_features == 0:
                self.log("[INFO] Too few samples/features for train-test analysis")
                return
            
            # Need minimum samples for half-data learnability calculation
            if n_train < 10:
                self.log("[INFO] Too few training samples for learnability calculation")
                compute_learnability = False
            else:
                compute_learnability = True
            
            if n_features == 1:
                self.log("[INFO] Single feature analysis:")
                self.log("[INFO]   Note: Limited predictive analysis possible with only one feature")
            
            self.log(f"[INFO] Train-Test predictability analysis ({n_train} train, {n_test} test, {n_features} features):")
            
            # Add detailed shape information
            self.log("[INFO]   Data shapes:")
            self.log(f"[INFO]     X_train: {X_train.shape} - [{X_train.dtype}]")
            self.log(f"[INFO]     y_train: {y_train.shape} - [{y_train.dtype}]")
            self.log(f"[INFO]     X_test: {X_test.shape} - [{X_test.dtype}]")
            self.log(f"[INFO]     y_test: {y_test.shape} - [{y_test.dtype}]")
            
            # Add shape consistency check
            if X_train.shape[1] != X_test.shape[1]:
                self.log(f"[WARN]     Shape inconsistency: Train features ({X_train.shape[1]}) != Test features ({X_test.shape[1]})")
                
            # Add additional statistics about the data tables
            self.log("[INFO]   Data statistics:")
            self.log(f"[INFO]     X_train range: [{X_train.min():.4f}, {X_train.max():.4f}], mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
            self.log(f"[INFO]     y_train range: [{y_train.min():.4f}, {y_train.max():.4f}], mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
            self.log(f"[INFO]     X_test range: [{X_test.min():.4f}, {X_test.max():.4f}], mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")
            self.log(f"[INFO]     y_test range: [{y_test.min():.4f}, {y_test.max():.4f}], mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")
            
            # Linear Model
            try:
                linear_model = LinearRegression()
                linear_model.fit(X_train, y_train)
                
                # Get predictions
                train_pred = linear_model.predict(X_train)
                test_pred = linear_model.predict(X_test)
                
                # Evaluate on both train and test using sklearn r2_score
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                # Debug: Show actual vs predicted values for a few samples
                self.log(f"[DEBUG] Linear Model - Train predictions (first 5): {train_pred[:5]}")
                self.log(f"[DEBUG] Linear Model - Train actual (first 5): {y_train[:5]}")
                self.log(f"[DEBUG] Linear Model - Test predictions (first 5): {test_pred[:5]}")
                self.log(f"[DEBUG] Linear Model - Test actual (first 5): {y_test[:5]}")
                
                # Add prediction statistics
                train_pred_stats = f"min={train_pred.min():.4f}, max={train_pred.max():.4f}, mean={train_pred.mean():.4f}, std={train_pred.std():.4f}"
                test_pred_stats = f"min={test_pred.min():.4f}, max={test_pred.max():.4f}, mean={test_pred.mean():.4f}, std={test_pred.std():.4f}"
                self.log(f"[DEBUG] Linear Model - Train predictions stats: {train_pred_stats}")
                self.log(f"[DEBUG] Linear Model - Test predictions stats: {test_pred_stats}")
                
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
                
                # Store statistics for summary
                self.all_stats['linear_train_r2'].append(train_r2)
                self.all_stats['linear_test_r2'].append(test_r2)
                self.all_stats['linear_train_mse'].append(train_mse)
                self.all_stats['linear_test_mse'].append(test_mse)
                
                # Calculate learnability (train on half the data)
                if compute_learnability:
                    try:
                        # Use half of the training data (randomly selected)
                        half_size = n_train // 2
                        if half_size >= 5:  # Ensure we have enough samples
                            # Generate random indices without replacement
                            np.random.seed(42)  # For reproducibility
                            half_indices = np.random.choice(n_train, half_size, replace=False)
                            X_train_half = X_train[half_indices]
                            y_train_half = y_train[half_indices]
                            
                            # Train on half data
                            linear_half = LinearRegression()
                            linear_half.fit(X_train_half, y_train_half)
                            
                            # Test on the same test set
                            half_test_pred = linear_half.predict(X_test)
                            half_test_mse = mean_squared_error(y_test, half_test_pred)
                            
                            # Calculate learnability ratio (half data MSE / full data MSE)
                            # Higher ratio means the model benefits more from additional data
                            if test_mse > 0:  # Avoid division by zero
                                learnability = half_test_mse / test_mse
                                self.all_stats['linear_half_data_test_mse'].append(half_test_mse)
                                self.all_stats['linear_learnability'].append(learnability)
                                
                                self.log("[INFO]     Linear Learnability:")
                                self.log(f"[INFO]       Half-data test MSE: {half_test_mse:.8f}")
                                self.log(f"[INFO]       Half-to-full data MSE ratio: {learnability:.4f} (>1 means benefit from more data)")
                                if learnability < 0.9:
                                    self.log("[WARN]       Unexpected learnability < 1: Half data performs better than full data")
                                elif learnability > 1.5:
                                    self.log("[INFO]       High learnability: Model benefits significantly from additional data")
                    except Exception as e:
                        self.log(f"[WARN]     Linear learnability calculation failed: {e}")
                    
            except Exception as e:
                self.log(f"[WARN]   Linear model failed: {e}")
            
            # Random Forest
            try:
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                rf_model.fit(X_train, y_train)
                
                # Get predictions
                train_pred = rf_model.predict(X_train)
                test_pred = rf_model.predict(X_test)
                
                # Evaluate on both train and test using sklearn r2_score
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                # Debug: Show actual vs predicted values for a few samples
                self.log(f"[DEBUG] Random Forest - Train predictions (first 5): {train_pred[:5]}")
                self.log(f"[DEBUG] Random Forest - Train actual (first 5): {y_train[:5]}")
                self.log(f"[DEBUG] Random Forest - Test predictions (first 5): {test_pred[:5]}")
                self.log(f"[DEBUG] Random Forest - Test actual (first 5): {y_test[:5]}")
                
                # Add prediction statistics
                train_pred_stats = f"min={train_pred.min():.4f}, max={train_pred.max():.4f}, mean={train_pred.mean():.4f}, std={train_pred.std():.4f}"
                test_pred_stats = f"min={test_pred.min():.4f}, max={test_pred.max():.4f}, mean={test_pred.mean():.4f}, std={test_pred.std():.4f}"
                self.log(f"[DEBUG] Random Forest - Train predictions stats: {train_pred_stats}")
                self.log(f"[DEBUG] Random Forest - Test predictions stats: {test_pred_stats}")
                
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
                
                # Store statistics for summary
                self.all_stats['rf_train_r2'].append(train_r2)
                self.all_stats['rf_test_r2'].append(test_r2)
                self.all_stats['rf_train_mse'].append(train_mse)
                self.all_stats['rf_test_mse'].append(test_mse)
                
                # Calculate learnability (train on half the data)
                if compute_learnability:
                    try:
                        # Use half of the training data (randomly selected)
                        half_size = n_train // 2
                        if half_size >= 5:  # Ensure we have enough samples
                            # Generate random indices without replacement
                            np.random.seed(42)  # For reproducibility
                            half_indices = np.random.choice(n_train, half_size, replace=False)
                            X_train_half = X_train[half_indices]
                            y_train_half = y_train[half_indices]
                            
                            # Train on half data
                            rf_half = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
                            rf_half.fit(X_train_half, y_train_half)
                            
                            # Test on the same test set
                            half_test_pred = rf_half.predict(X_test)
                            half_test_mse = mean_squared_error(y_test, half_test_pred)
                            
                            # Calculate learnability ratio (half data MSE / full data MSE)
                            # Higher ratio means the model benefits more from additional data
                            if test_mse > 0:  # Avoid division by zero
                                learnability = half_test_mse / test_mse
                                self.all_stats['rf_half_data_test_mse'].append(half_test_mse)
                                self.all_stats['rf_learnability'].append(learnability)
                                
                                self.log("[INFO]     Random Forest Learnability:")
                                self.log(f"[INFO]       Half-data test MSE: {half_test_mse:.8f}")
                                self.log(f"[INFO]       Half-to-full data MSE ratio: {learnability:.4f} (>1 means benefit from more data)")
                                if learnability < 0.9:
                                    self.log("[WARN]       Unexpected learnability < 1: Half data performs better than full data")
                                elif learnability > 1.5:
                                    self.log("[INFO]       High learnability: Model benefits significantly from additional data")
                    except Exception as e:
                        self.log(f"[WARN]     Random Forest learnability calculation failed: {e}")
                    
                # Feature importance
                if n_features <= 10:  # Only show for reasonable number of features
                    importances = rf_model.feature_importances_
                    self.log("[INFO]     Feature Importance:")
                    for feat, imp in zip(feature_names, importances):
                        self.log(f"[INFO]       {feat}: {imp:.3f}")
                        
                        # Store feature importance for summary
                        if feat not in self.all_stats['feature_importances']:
                            self.all_stats['feature_importances'][feat] = []
                        self.all_stats['feature_importances'][feat].append(imp)
                    
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
            with open(batch_log_path, 'w', encoding='utf-8') as batch_log:
                self.current_output_file = batch_log
                
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
                    
                    # Create dataset log file
                    dataset_log_path = dataset_dir / "analysis.txt"
                    with open(dataset_log_path, 'w', encoding='utf-8') as dataset_log:
                        self.current_output_file = dataset_log
                        
                        self.log(f"\n--- Dataset {dataset_idx + 1} ---")
                        self.visualize_dataset(
                            X_train, y_train, X_test, y_test,
                            dataset_idx=dataset_idx + 1, 
                            batch_idx=batch_idx + 1,
                            save_dir=dataset_dir
                        )
                
                batch_count += 1
                
                # Restore output to batch log
                self.current_output_file = batch_log
                self.log(f"\n[INFO] Completed analysis of {n_to_visualize} datasets in batch {batch_idx + 1}")
        
        print(f"\n[OK] Completed visualization of {batch_count} batches")
    
    def generate_summary_statistics(self):
        """
        Generate overall summary statistics across all datasets and create histograms.
        
        This method creates summary plots and outputs overall statistics based on all 
        the datasets that were analyzed. It includes histograms of key metrics and 
        summary statistics for predictability across datasets.
        """
        # Skip if no datasets were processed
        if not self.all_stats['linear_test_r2'] and not self.all_stats['rf_test_r2']:
            self.log("[WARN] No dataset statistics available for summary")
            return
        
        # Print to console regardless of file state
        print("\n" + "="*80)
        print("                OVERALL SUMMARY STATISTICS ACROSS ALL DATASETS")
        print("="*80)
        
        # Create summary directory
        summary_dir = Path(self.results_base_dir) / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Store the current output file
        original_output_file = self.current_output_file
        
        # Create and open a new summary log file
        summary_log_path = summary_dir / "summary_statistics.txt"
        summary_log = open(summary_log_path, 'w', encoding='utf-8')
        self.current_output_file = summary_log
        
        # 1. DATASET CHARACTERISTICS SUMMARY
        self.log("\n" + "="*60)
        self.log("DATASET CHARACTERISTICS SUMMARY")
        self.log("="*60)
        
        # Calculate statistics
        n_datasets = len(self.all_stats['feature_counts'])
        
        if n_datasets > 0:
            avg_features = np.mean(self.all_stats['feature_counts'])
            std_features = np.std(self.all_stats['feature_counts'])
            min_features = np.min(self.all_stats['feature_counts'])
            max_features = np.max(self.all_stats['feature_counts'])
            
            avg_train = np.mean(self.all_stats['train_sample_counts'])
            std_train = np.std(self.all_stats['train_sample_counts'])
            min_train = np.min(self.all_stats['train_sample_counts'])
            max_train = np.max(self.all_stats['train_sample_counts'])
            
            avg_test = np.mean(self.all_stats['test_sample_counts'])
            std_test = np.std(self.all_stats['test_sample_counts'])
            min_test = np.min(self.all_stats['test_sample_counts'])
            max_test = np.max(self.all_stats['test_sample_counts'])
            
            # Output statistics
            self.log(f"[INFO] Total datasets analyzed: {n_datasets}")
            self.log(f"[INFO] Feature counts: {avg_features:.1f} ± {std_features:.1f} (min={min_features}, max={max_features})")
            self.log(f"[INFO] Train samples: {avg_train:.1f} ± {std_train:.1f} (min={min_train}, max={max_train})")
            self.log(f"[INFO] Test samples: {avg_test:.1f} ± {std_test:.1f} (min={min_test}, max={max_test})")
            
            # Create histogram of feature counts with trimmed data
            feature_counts_trimmed, _, _ = self.trim_for_plotting(self.all_stats['feature_counts'])
            plt.figure(figsize=(10, 6))
            plt.hist(feature_counts_trimmed, bins=min(10, int(np.max(feature_counts_trimmed) - np.min(feature_counts_trimmed) + 1)), 
                    alpha=0.7, color='steelblue', edgecolor='black')
            plt.title('Feature Count Distribution Across Datasets (2.5%-97.5% range)', fontsize=14)
            plt.xlabel('Number of Features', fontsize=12)
            plt.ylabel('Number of Datasets', fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save histogram
            feature_hist_path = summary_dir / 'feature_count_histogram.png'
            plt.savefig(feature_hist_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"[INFO] Feature count histogram saved to: {feature_hist_path}")
        
        # 2. PREDICTABILITY SUMMARY
        self.log("\n" + "="*60)
        self.log("PREDICTABILITY SUMMARY")
        self.log("="*60)
        
        if self.all_stats['linear_test_r2']:
            # Linear Model statistics
            linear_test_r2_mean = np.mean(self.all_stats['linear_test_r2'])
            linear_test_r2_std = np.std(self.all_stats['linear_test_r2'])
            linear_test_r2_median = np.median(self.all_stats['linear_test_r2'])
            linear_test_r2_min = np.min(self.all_stats['linear_test_r2'])
            linear_test_r2_max = np.max(self.all_stats['linear_test_r2'])
            
            # Linear Model MSE statistics
            linear_test_mse_mean = np.mean(self.all_stats['linear_test_mse']) if 'linear_test_mse' in self.all_stats and self.all_stats['linear_test_mse'] else np.nan
            linear_test_mse_std = np.std(self.all_stats['linear_test_mse']) if 'linear_test_mse' in self.all_stats and self.all_stats['linear_test_mse'] else np.nan
            linear_test_mse_median = np.median(self.all_stats['linear_test_mse']) if 'linear_test_mse' in self.all_stats and self.all_stats['linear_test_mse'] else np.nan
            linear_test_mse_min = np.min(self.all_stats['linear_test_mse']) if 'linear_test_mse' in self.all_stats and self.all_stats['linear_test_mse'] else np.nan
            linear_test_mse_max = np.max(self.all_stats['linear_test_mse']) if 'linear_test_mse' in self.all_stats and self.all_stats['linear_test_mse'] else np.nan
            
            # Random Forest statistics
            rf_test_r2_mean = np.mean(self.all_stats['rf_test_r2'])
            rf_test_r2_std = np.std(self.all_stats['rf_test_r2'])
            rf_test_r2_median = np.median(self.all_stats['rf_test_r2'])
            rf_test_r2_min = np.min(self.all_stats['rf_test_r2'])
            rf_test_r2_max = np.max(self.all_stats['rf_test_r2'])
            
            # Random Forest MSE statistics
            rf_test_mse_mean = np.mean(self.all_stats['rf_test_mse']) if 'rf_test_mse' in self.all_stats and self.all_stats['rf_test_mse'] else np.nan
            rf_test_mse_std = np.std(self.all_stats['rf_test_mse']) if 'rf_test_mse' in self.all_stats and self.all_stats['rf_test_mse'] else np.nan
            rf_test_mse_median = np.median(self.all_stats['rf_test_mse']) if 'rf_test_mse' in self.all_stats and self.all_stats['rf_test_mse'] else np.nan
            rf_test_mse_min = np.min(self.all_stats['rf_test_mse']) if 'rf_test_mse' in self.all_stats and self.all_stats['rf_test_mse'] else np.nan
            rf_test_mse_max = np.max(self.all_stats['rf_test_mse']) if 'rf_test_mse' in self.all_stats and self.all_stats['rf_test_mse'] else np.nan
            
            # Learnability statistics
            linear_learnability_mean = np.mean(self.all_stats['linear_learnability']) if self.all_stats['linear_learnability'] else np.nan
            linear_learnability_std = np.std(self.all_stats['linear_learnability']) if self.all_stats['linear_learnability'] else np.nan
            linear_learnability_median = np.median(self.all_stats['linear_learnability']) if self.all_stats['linear_learnability'] else np.nan
            linear_learnability_min = np.min(self.all_stats['linear_learnability']) if self.all_stats['linear_learnability'] else np.nan
            linear_learnability_max = np.max(self.all_stats['linear_learnability']) if self.all_stats['linear_learnability'] else np.nan
            
            # Output statistics
            self.log("[INFO] LINEAR MODEL (Test R²):")
            self.log(f"[INFO]   Mean:   {linear_test_r2_mean:.4f}")
            self.log(f"[INFO]   Median: {linear_test_r2_median:.4f}")
            self.log(f"[INFO]   Std:    {linear_test_r2_std:.4f}")
            self.log(f"[INFO]   Range:  [{linear_test_r2_min:.4f}, {linear_test_r2_max:.4f}]")
            
            # Output learnability statistics
            if not np.isnan(linear_learnability_mean):
                self.log("[INFO] LINEAR MODEL (Learnability - ratio of full data MSE to half data MSE):")
                self.log(f"[INFO]   Mean:   {linear_learnability_mean:.4f}")
                self.log(f"[INFO]   Median: {linear_learnability_median:.4f}")
                self.log(f"[INFO]   Std:    {linear_learnability_std:.4f}")
                self.log(f"[INFO]   Range:  [{linear_learnability_min:.4f}, {linear_learnability_max:.4f}]")
            
            if 'linear_test_mse' in self.all_stats and self.all_stats['linear_test_mse']:
                self.log("[INFO] LINEAR MODEL (Test MSE):")
                self.log(f"[INFO]   Mean:   {linear_test_mse_mean:.4f}")
                self.log(f"[INFO]   Median: {linear_test_mse_median:.4f}")
                self.log(f"[INFO]   Std:    {linear_test_mse_std:.4f}")
                self.log(f"[INFO]   Range:  [{linear_test_mse_min:.4f}, {linear_test_mse_max:.4f}]")
            
            # Random Forest learnability statistics
            rf_learnability_mean = np.mean(self.all_stats['rf_learnability']) if self.all_stats['rf_learnability'] else np.nan
            rf_learnability_std = np.std(self.all_stats['rf_learnability']) if self.all_stats['rf_learnability'] else np.nan
            rf_learnability_median = np.median(self.all_stats['rf_learnability']) if self.all_stats['rf_learnability'] else np.nan
            rf_learnability_min = np.min(self.all_stats['rf_learnability']) if self.all_stats['rf_learnability'] else np.nan
            rf_learnability_max = np.max(self.all_stats['rf_learnability']) if self.all_stats['rf_learnability'] else np.nan
            
            self.log("[INFO] RANDOM FOREST (Test R²):")
            self.log(f"[INFO]   Mean:   {rf_test_r2_mean:.4f}")
            self.log(f"[INFO]   Median: {rf_test_r2_median:.4f}")
            self.log(f"[INFO]   Std:    {rf_test_r2_std:.4f}")
            self.log(f"[INFO]   Range:  [{rf_test_r2_min:.4f}, {rf_test_r2_max:.4f}]")
            
            # Output learnability statistics
            if not np.isnan(rf_learnability_mean):
                self.log("[INFO] RANDOM FOREST (Learnability - ratio of full data MSE to half data MSE):")
                self.log(f"[INFO]   Mean:   {rf_learnability_mean:.4f}")
                self.log(f"[INFO]   Median: {rf_learnability_median:.4f}")
                self.log(f"[INFO]   Std:    {rf_learnability_std:.4f}")
                self.log(f"[INFO]   Range:  [{rf_learnability_min:.4f}, {rf_learnability_max:.4f}]")
            
            if 'rf_test_mse' in self.all_stats and self.all_stats['rf_test_mse']:
                self.log("[INFO] RANDOM FOREST (Test MSE):")
                self.log(f"[INFO]   Mean:   {rf_test_mse_mean:.4f}")
                self.log(f"[INFO]   Median: {rf_test_mse_median:.4f}")
                self.log(f"[INFO]   Std:    {rf_test_mse_std:.4f}")
                self.log(f"[INFO]   Range:  [{rf_test_mse_min:.4f}, {rf_test_mse_max:.4f}]")
            
            # Create histogram of R² values with trimmed data
            linear_r2_trimmed, _, _ = self.trim_for_plotting(self.all_stats['linear_test_r2'])
            rf_r2_trimmed, _, _ = self.trim_for_plotting(self.all_stats['rf_test_r2'])
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist(linear_r2_trimmed, bins=10, alpha=0.7, 
                     color='cornflowerblue', edgecolor='black')
            plt.title('Linear Model Test R² Distribution\n(2.5%-97.5% range)', fontsize=12)
            plt.xlabel('Test R²', fontsize=10)
            plt.ylabel('Number of Datasets', fontsize=10)
            plt.grid(alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(rf_r2_trimmed, bins=10, alpha=0.7,
                     color='forestgreen', edgecolor='black')
            plt.title('Random Forest Test R² Distribution\n(2.5%-97.5% range)', fontsize=12)
            plt.xlabel('Test R²', fontsize=10)
            plt.ylabel('Number of Datasets', fontsize=10)
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            r2_hist_path = summary_dir / 'r2_distribution_histogram.png'
            plt.savefig(r2_hist_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"[INFO] R² distribution histogram saved to: {r2_hist_path}")
            
            # Create histogram of MSE values (if available) with trimmed data
            if ('linear_test_mse' in self.all_stats and self.all_stats['linear_test_mse'] and 
                'rf_test_mse' in self.all_stats and self.all_stats['rf_test_mse']):
                
                linear_mse_trimmed, _, _ = self.trim_for_plotting(self.all_stats['linear_test_mse'])
                rf_mse_trimmed, _, _ = self.trim_for_plotting(self.all_stats['rf_test_mse'])
                
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.hist(linear_mse_trimmed, bins=10, alpha=0.7, 
                         color='lightcoral', edgecolor='black')
                plt.title('Linear Model Test MSE Distribution\n(2.5%-97.5% range)', fontsize=12)
                plt.xlabel('Test MSE', fontsize=10)
                plt.ylabel('Number of Datasets', fontsize=10)
                plt.grid(alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.hist(rf_mse_trimmed, bins=10, alpha=0.7,
                         color='mediumseagreen', edgecolor='black')
                plt.title('Random Forest Test MSE Distribution\n(2.5%-97.5% range)', fontsize=12)
                plt.xlabel('Test MSE', fontsize=10)
                plt.ylabel('Number of Datasets', fontsize=10)
                plt.grid(alpha=0.3)
                
                plt.tight_layout()
                mse_hist_path = summary_dir / 'mse_distribution_histogram.png'
                plt.savefig(mse_hist_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.log(f"[INFO] MSE distribution histogram saved to: {mse_hist_path}")
            
            # Create comparison plot: Linear vs RF R² with trimmed data
            # Trim both arrays and keep only pairs where both values are within range
            linear_r2_array = np.array(self.all_stats['linear_test_r2'])
            rf_r2_array = np.array(self.all_stats['rf_test_r2'])
            
            # Calculate percentile bounds for each
            linear_lower = np.percentile(linear_r2_array, 2.5)
            linear_upper = np.percentile(linear_r2_array, 97.5)
            rf_lower = np.percentile(rf_r2_array, 2.5)
            rf_upper = np.percentile(rf_r2_array, 97.5)
            
            # Keep only pairs where both are in range
            mask = ((linear_r2_array >= linear_lower) & (linear_r2_array <= linear_upper) &
                    (rf_r2_array >= rf_lower) & (rf_r2_array <= rf_upper))
            
            linear_r2_trimmed_pairs = linear_r2_array[mask]
            rf_r2_trimmed_pairs = rf_r2_array[mask]
            
            plt.figure(figsize=(8, 6))
            plt.scatter(linear_r2_trimmed_pairs, rf_r2_trimmed_pairs,
                       alpha=0.7, s=50, edgecolors='black', c='teal')
            
            plt.axline([0, 0], [1, 1], color='gray', linestyle='--', alpha=0.7)
            
            plt.xlim(-0.1, 1.0)
            plt.ylim(-0.1, 1.0)
            plt.title('Linear Model vs Random Forest Test R² Comparison\n(2.5%-97.5% range)', fontsize=14)
            plt.xlabel('Linear Model Test R²', fontsize=12)
            plt.ylabel('Random Forest Test R²', fontsize=12)
            plt.grid(alpha=0.3)
            
            r2_comparison_path = summary_dir / 'r2_model_comparison.png'
            plt.savefig(r2_comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"[INFO] Model comparison plot saved to: {r2_comparison_path}")
            
            # Create learnability histograms if data is available with trimmed data
            if self.all_stats['linear_learnability'] and self.all_stats['rf_learnability']:
                linear_learn_trimmed, _, _ = self.trim_for_plotting(self.all_stats['linear_learnability'])
                rf_learn_trimmed, _, _ = self.trim_for_plotting(self.all_stats['rf_learnability'])
                
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.hist(linear_learn_trimmed, bins=10, alpha=0.7, 
                         color='cornflowerblue', edgecolor='black')
                plt.title('Linear Model Learnability Distribution\n(2.5%-97.5% range)', fontsize=12)
                plt.xlabel('Learnability (Half/Full MSE ratio)', fontsize=10)
                plt.ylabel('Number of Datasets', fontsize=10)
                plt.grid(alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.hist(rf_learn_trimmed, bins=10, alpha=0.7,
                         color='forestgreen', edgecolor='black')
                plt.title('Random Forest Learnability Distribution\n(2.5%-97.5% range)', fontsize=12)
                plt.xlabel('Learnability (Half/Full MSE ratio)', fontsize=10)
                plt.ylabel('Number of Datasets', fontsize=10)
                plt.grid(alpha=0.3)
                
                plt.tight_layout()
                learnability_hist_path = summary_dir / 'learnability_distribution_histogram.png'
                plt.savefig(learnability_hist_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.log(f"[INFO] Learnability distribution histogram saved to: {learnability_hist_path}")
                
                # Create learnability comparison plot with trimmed data
                linear_learn_array = np.array(self.all_stats['linear_learnability'])
                rf_learn_array = np.array(self.all_stats['rf_learnability'])
                
                # Calculate percentile bounds for each
                linear_lower = np.percentile(linear_learn_array, 2.5)
                linear_upper = np.percentile(linear_learn_array, 97.5)
                rf_lower = np.percentile(rf_learn_array, 2.5)
                rf_upper = np.percentile(rf_learn_array, 97.5)
                
                # Keep only pairs where both are in range
                mask = ((linear_learn_array >= linear_lower) & (linear_learn_array <= linear_upper) &
                        (rf_learn_array >= rf_lower) & (rf_learn_array <= rf_upper))
                
                linear_learn_trimmed_pairs = linear_learn_array[mask]
                rf_learn_trimmed_pairs = rf_learn_array[mask]
                
                plt.figure(figsize=(8, 6))
                plt.scatter(linear_learn_trimmed_pairs, rf_learn_trimmed_pairs,
                           alpha=0.7, s=50, edgecolors='black', c='purple')
                
                # Add a reference line at x=y
                max_value = max(
                    np.max(linear_learn_trimmed_pairs) if len(linear_learn_trimmed_pairs) > 0 else 1,
                    np.max(rf_learn_trimmed_pairs) if len(rf_learn_trimmed_pairs) > 0 else 1
                )
                plt.axline([0, 0], [max_value, max_value], color='gray', linestyle='--', alpha=0.7)
                
                # Add a reference line at 1.0 (no learning benefit from more data)
                plt.axhline(1.0, color='red', linestyle='--', alpha=0.5)
                plt.axvline(1.0, color='red', linestyle='--', alpha=0.5)
                
                plt.title('Linear Model vs Random Forest Learnability Comparison\n(2.5%-97.5% range)', fontsize=14)
                plt.xlabel('Linear Model Learnability', fontsize=12)
                plt.ylabel('Random Forest Learnability', fontsize=12)
                plt.grid(alpha=0.3)
                
                learnability_comparison_path = summary_dir / 'learnability_model_comparison.png'
                plt.savefig(learnability_comparison_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.log(f"[INFO] Learnability comparison plot saved to: {learnability_comparison_path}")
            
            # Create comparison plot: Linear vs RF MSE with trimmed data
            if ('linear_test_mse' in self.all_stats and self.all_stats['linear_test_mse'] and 
                'rf_test_mse' in self.all_stats and self.all_stats['rf_test_mse']):
                
                linear_mse_array = np.array(self.all_stats['linear_test_mse'])
                rf_mse_array = np.array(self.all_stats['rf_test_mse'])
                
                # Calculate percentile bounds for each
                linear_lower = np.percentile(linear_mse_array, 2.5)
                linear_upper = np.percentile(linear_mse_array, 97.5)
                rf_lower = np.percentile(rf_mse_array, 2.5)
                rf_upper = np.percentile(rf_mse_array, 97.5)
                
                # Keep only pairs where both are in range
                mask = ((linear_mse_array >= linear_lower) & (linear_mse_array <= linear_upper) &
                        (rf_mse_array >= rf_lower) & (rf_mse_array <= rf_upper))
                
                linear_mse_trimmed_pairs = linear_mse_array[mask]
                rf_mse_trimmed_pairs = rf_mse_array[mask]
                
                plt.figure(figsize=(8, 6))
                
                # Get max value to set equal plot limits
                max_mse = max(
                    np.max(linear_mse_trimmed_pairs) if len(linear_mse_trimmed_pairs) > 0 else 1,
                    np.max(rf_mse_trimmed_pairs) if len(rf_mse_trimmed_pairs) > 0 else 1
                )
                
                plt.scatter(linear_mse_trimmed_pairs, rf_mse_trimmed_pairs,
                           alpha=0.7, s=50, edgecolors='black', c='crimson')
                
                plt.axline([0, 0], [max_mse, max_mse], color='gray', linestyle='--', alpha=0.7)
                
                plt.xlim(-0.05 * max_mse, 1.05 * max_mse)
                plt.ylim(-0.05 * max_mse, 1.05 * max_mse)
                plt.title('Linear Model vs Random Forest Test MSE Comparison\n(2.5%-97.5% range)', fontsize=14)
                plt.xlabel('Linear Model Test MSE', fontsize=12)
                plt.ylabel('Random Forest Test MSE', fontsize=12)
                plt.grid(alpha=0.3)
                
                mse_comparison_path = summary_dir / 'mse_model_comparison.png'
                plt.savefig(mse_comparison_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.log(f"[INFO] MSE model comparison plot saved to: {mse_comparison_path}")
        
        # 3. FEATURE IMPORTANCE SUMMARY
        self.log("\n" + "="*60)
        self.log("FEATURE IMPORTANCE SUMMARY")
        self.log("="*60)
        
        if self.all_stats['feature_importances']:
            self.log("[INFO] Average feature importance across datasets:")
            
            # Calculate mean importance for each feature
            avg_importances = {}
            for feat, values in self.all_stats['feature_importances'].items():
                avg_importances[feat] = np.mean(values)
            
            # Sort features by average importance
            sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
            
            for feat, imp in sorted_features:
                count = len(self.all_stats['feature_importances'][feat])
                self.log(f"[INFO]   {feat}: {imp:.4f} (used in {count} datasets)")
            
            # Create feature importance bar chart (for top 10 features)
            top_features = sorted_features[:10]
            if top_features:
                plt.figure(figsize=(10, 6))
                
                features = [x[0] for x in top_features]
                importances = [x[1] for x in top_features]
                
                plt.barh(features, importances, alpha=0.7, color='slateblue', edgecolor='black')
                plt.title('Average Feature Importance Across Datasets (Top Features)', fontsize=14)
                plt.xlabel('Average Importance', fontsize=12)
                plt.ylabel('Feature', fontsize=12)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                importance_path = summary_dir / 'feature_importance_summary.png'
                plt.savefig(importance_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.log(f"[INFO] Feature importance summary saved to: {importance_path}")
        
        # 4. CORRELATION ANALYSIS SUMMARY
        self.log("\n" + "="*60)
        self.log("CORRELATION ANALYSIS SUMMARY")
        self.log("="*60)
        
        if self.all_stats['target_correlations']:
            self.log("[INFO] Target correlation analysis:")
            
            # Organize correlations by feature
            feature_corrs = {}
            for feat, corr in self.all_stats['target_correlations']:
                if feat not in feature_corrs:
                    feature_corrs[feat] = []
                feature_corrs[feat].append(corr)
            
            # Calculate statistics per feature
            self.log("[INFO] Average absolute correlation with target by feature:")
            
            feature_avg_abs_corrs = {}
            for feat, corrs in feature_corrs.items():
                avg_abs_corr = np.mean(np.abs(corrs))
                feature_avg_abs_corrs[feat] = avg_abs_corr
                count = len(corrs)
                self.log(f"[INFO]   {feat}: {avg_abs_corr:.4f} (used in {count} datasets)")
            
            # Create correlation distribution histogram with trimmed data
            all_corrs = [corr for feat, corr in self.all_stats['target_correlations']]
            all_corrs_trimmed, _, _ = self.trim_for_plotting(all_corrs)
            
            plt.figure(figsize=(10, 6))
            plt.hist(all_corrs_trimmed, bins=20, alpha=0.7, color='tomato', edgecolor='black')
            plt.title('Distribution of Feature-Target Correlations Across All Datasets\n(2.5%-97.5% range)', fontsize=14)
            plt.xlabel('Correlation with Target', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            corr_hist_path = summary_dir / 'correlation_distribution.png'
            plt.savefig(corr_hist_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log(f"[INFO] Correlation distribution histogram saved to: {corr_hist_path}")
        
        self.log("\n" + "="*80)
        self.log("[OK] Summary analysis complete")
        self.log("="*80)
        
        # Close the summary log file
        if self.current_output_file and not self.current_output_file.closed:
            self.current_output_file.close()
            
        # Restore original output file if it's still open
        if original_output_file and not original_output_file.closed:
            self.current_output_file = original_output_file
            self.log(f"\n[INFO] Generated overall summary statistics in: {summary_dir}")
    
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
        main_log = open(main_log_path, 'w', encoding='utf-8')
        self.current_output_file = main_log
        
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
            
            # Step 4: Generate summary statistics and histograms
            self.generate_summary_statistics()
            
            self.log("=" * 60)
            self.log("[OK] Complete visualization finished successfully!")
            self.log("=" * 60)
            
        except Exception as e:
            # Ensure we have a valid log file handle (dataset-level "with" blocks may have closed it)
            if (self.current_output_file is None) or getattr(self.current_output_file, "closed", False):
                try:
                    # Try append to the main log file
                    self.current_output_file = open(main_log_path, 'a', encoding='utf-8')
                except Exception:
                    self.current_output_file = None
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
            # Make sure to close the file in case of an error
            if self.current_output_file and not self.current_output_file.closed:
                self.current_output_file.close()
            raise
            
        # Make sure to close the file when done
        if self.current_output_file and not self.current_output_file.closed:
            self.current_output_file.close()


def main():
    """Main entry point for dataloader dataset visualization."""
    
    # =============================================================
    # CONFIGURATION - Edit these variables to change behavior
    # =============================================================
    
    # Path to YAML config file (relative to repository root)
    CONFIG_PATH = 'experiments/FirstTests/configs/early_test_curriculum.yaml'
    
    # Random seed for reproducibility
    SEED = 42
    
    # Number of batches to sample from dataloader
    N_BATCHES = 1
    
    # Number of datasets to visualize per batch
    N_DATASETS_PER_BATCH = 10

    # Fixed curriculum time t in [0, 1]; when using curriculum configs, all samples will be drawn at this timepoint
    FIXED_T = 1.0
    
    # ResultsDataloaderSamples will be saved to organized folders (plots will NOT be displayed)
    # Format: checks/ResultsDataloaderSamples/run_YYYYMMDD_HHMMSS/batch_X/dataset_Y/
    # A summary folder with overall statistics will also be created
    
    # Detect which directory we're running from to set correct results path
    if 'priors' in str(Path(__file__)):
        # We're running from src/priors/training/checks/
        RESULTS_BASE_DIR = 'src/priors/training/checks/ResultsDataloaderSamples'
    else:
        # We're running from src/training/checks/
        RESULTS_BASE_DIR = 'src/training/checks/ResultsDataloaderSamples'
    
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
    
    # Debug information
    script_path = Path(__file__)
    script_dir = script_path.parent
    repo_root = script_dir.parent.parent.parent.parent
    expected_config_path = repo_root / CONFIG_PATH
    
    print("\nDebug Information:")
    print(f"  Script Path: {script_path}")
    print(f"  Script Directory: {script_dir}")
    print(f"  Repository Root: {repo_root}")
    print(f"  Expected Config Path: {expected_config_path}")
    print(f"  Config Path Exists: {expected_config_path.exists()}")
    print()
    print("Output will be saved to organized folders. Plots will NOT be displayed.")
    print("Analysis output will be written to both console and files.")
    print("An overall summary with statistics and histograms will be created in the summary/ folder.")
    print("=" * 60)
    
    # Create visualizer
    visualizer = DataloaderDatasetVisualizer(
        config_path=CONFIG_PATH,
        seed=SEED,
        fixed_t=FIXED_T,
    )
    
    # Run visualization with interactive mode
    visualizer.run_visualization(
        n_batches=N_BATCHES,
        n_datasets_per_batch=N_DATASETS_PER_BATCH,
        results_base_dir=results_dir
    )
    
    print("=" * 60)
    print("Analysis complete! Results are saved in:")
    print(f"  {results_dir}")
    print(f"  {results_dir}/summary/  <- Overall statistics summary and histograms")
    print("=" * 60)


if __name__ == "__main__":
    main()
