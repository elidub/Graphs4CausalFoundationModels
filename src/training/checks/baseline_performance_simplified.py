#!/usr/bin/env python3
"""
Baseline Performance Evaluation (Simplified)

This script evaluates the performance of various baseline machine learning models
on samples generated from the causal prior. It loads a YAML configuration file,
creates a dataloader, samples batches, and evaluates multiple baseline models
(Linear Regression, Random Forest, XGBoost, etc.) on the data.

Usage:
    python baseline_performance_simplified.py

Note: Configuration parameters are defined as constants in the main() function.
Modify the ALL_CAPS constants in main() to change configuration settings.
"""

import numpy as np
import time
import yaml
import warnings
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import project modules
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types by converting them to Python native types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message="Use subset (sliced data) of np.ndarray is not recommended")


def load_yaml_config(config_path: str):
    """Load YAML configuration file."""
    # First, resolve special path prefixes
    config_path_str = str(config_path)
    
    # Handle project-relative paths starting with 'experiments/'
    if config_path_str.startswith('experiments/') or config_path_str.startswith('experiments\\'):
        # Get repository root path dynamically
        file_path = Path(__file__)
        repo_root = file_path.parent.parent.parent.parent  # Go up to repo root from src/training/checks
        config_path = repo_root / config_path_str
    # Convert to Path and make it relative to repository root
    elif not Path(config_path).is_absolute():
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
        print(f"[ERROR] Could not find config file at: {config_path}")
        print(f"[INFO] Original config path: {config_path_str}")
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


class BaselineModelEvaluator:
    """
    Simplified evaluator for baseline model performance on causal prior datasets.
    
    This class loads a YAML config, creates a dataloader, samples batches,
    and evaluates various baseline models on the datasets.
    """
    
    def __init__(self, config_path: str, seed: int = 42):
        """Initialize the evaluator with a config file."""
        self.config_path = Path(config_path)
        self.seed = seed
        self.config = None
        self.dataset_maker = None
        self.dataset = None
        self.dataloader = None
        self.results_dir = None
        
        # Dictionary to collect statistics across all datasets
        self.all_stats = {
            'dataset_indices': [],
            'feature_counts': [],
            'train_sample_counts': [],
            'test_sample_counts': [],
            'model_performances': {},
            'model_runtimes': {}
        }
        
        # Models to evaluate
        self.models_to_evaluate = [
            'linear',
            'ridge',
            'random_forest',
            'xgboost',
            'knn'
        ]
        
        # Initialize stats for each model
        for model in self.models_to_evaluate:
            self.all_stats['model_performances'][model] = {
                'train_r2': [],
                'test_r2': [],
                'train_mse': [],
                'test_mse': []
            }
            self.all_stats['model_runtimes'][model] = []
    
    def load_config(self):
        """Load and parse the YAML configuration."""
        print(f"\n[INFO] Loading configuration from {self.config_path}...")
        self.config = load_yaml_config(str(self.config_path))
        print("[OK] Configuration loaded successfully")
        
    def create_dataset_and_dataloader(self, n_datasets_per_batch=None):
        """Create dataset and dataloader from config."""
        print("\n[INFO] Creating dataset and dataloader...")
        
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
        print(f"[INFO] Creating dataset with seed {self.seed}...")
        self.dataset = self.dataset_maker.create_dataset(seed=self.seed)
        print(f"[OK] Dataset created with {len(self.dataset)} samples")
        
        # Create dataloader
        # Override batch_size with n_datasets_per_batch if provided
        if n_datasets_per_batch is not None:
            batch_size = n_datasets_per_batch
            print(f"[INFO] Using specified batch size: {batch_size}")
        else:
            batch_size = training_config.get('batch_size', 4)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=0  # For easier debugging
        )
        
        print(f"[OK] Created dataloader with batch size {batch_size}")
    
    def evaluate_batch(self, batch, batch_idx=0):
        """Evaluate a batch of datasets."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            # Format: [X_batch, y_batch] for standard observational data
            X_batch, y_batch = batch
            batch_size = X_batch.shape[0]
            
            print(f"\n[INFO] Processing batch {batch_idx} with {batch_size} datasets")
            print(f"[INFO] X shape: {X_batch.shape}, y shape: {y_batch.shape}")
            
            # Process each dataset in the batch
            for dataset_idx in range(batch_size):
                # Extract individual dataset
                X = X_batch[dataset_idx].numpy()
                y = y_batch[dataset_idx].numpy()
                
                # Track dataset index
                self.all_stats['dataset_indices'].append(f"B{batch_idx}_D{dataset_idx}")
                
                print(f"\n{'='*80}")
                print(f"DATASET {dataset_idx} (Batch {batch_idx})")
                print(f"{'='*80}")
                
                # Evaluate models on this dataset
                self.evaluate_dataset(X, y, dataset_idx)
        
        elif isinstance(batch, (list, tuple)) and len(batch) == 4:
            # Format: [X_train, y_train, X_test, y_test] for train-test split data
            X_train_batch, y_train_batch, X_test_batch, y_test_batch = batch
            batch_size = X_train_batch.shape[0]
            
            print(f"\n[INFO] Processing batch {batch_idx} with {batch_size} datasets (train-test split format)")
            
            # Process each dataset in the batch
            for dataset_idx in range(batch_size):
                # Extract individual dataset with train-test split
                X_train = X_train_batch[dataset_idx].detach().cpu().numpy()
                y_train = y_train_batch[dataset_idx].detach().cpu().numpy()
                X_test = X_test_batch[dataset_idx].detach().cpu().numpy()
                y_test = y_test_batch[dataset_idx].detach().cpu().numpy()
                
                # Track dataset index
                self.all_stats['dataset_indices'].append(f"B{batch_idx}_D{dataset_idx}")
                
                print(f"\n{'='*80}")
                print(f"DATASET {dataset_idx} (Batch {batch_idx})")
                print(f"{'='*80}")
                
                # Store dataset statistics
                self.all_stats['feature_counts'].append(X_train.shape[1])
                self.all_stats['train_sample_counts'].append(X_train.shape[0])
                self.all_stats['test_sample_counts'].append(X_test.shape[0])
                
                # Reshape y if needed
                if y_train.ndim > 1 and y_train.shape[1] == 1:
                    y_train = y_train.reshape(-1)
                if y_test.ndim > 1 and y_test.shape[1] == 1:
                    y_test = y_test.reshape(-1)
                
                # Evaluate models directly on train-test split
                self.evaluate_models(X_train, y_train, X_test, y_test, dataset_idx)
                
        else:
            print(f"[WARN] Unsupported batch format: {type(batch)} with {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'} elements")
    
    def evaluate_dataset(self, X, y, dataset_idx):
        """Evaluate dataset by splitting into train-test and running models."""
        # Get basic dataset info
        n_samples, n_features = X.shape
        
        # Reshape y if needed
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.reshape(-1)
        
        # Check for NaN/inf values
        if np.isnan(X).any() or np.isnan(y).any() or np.isinf(X).any() or np.isinf(y).any():
            print("[WARN] Dataset contains NaN or infinite values - skipping")
            return
        
        # Check for active features (non-zero variance)
        feature_variances = np.var(X, axis=0)
        active_features_mask = feature_variances > 1e-10
        n_active_features = np.sum(active_features_mask)
        
        if n_active_features == 0:
            print("[WARN] Dataset has no features with non-zero variance - skipping")
            return
        
        # Filter to active features
        X = X[:, active_features_mask]
        
        # Split into train/test
        train_fraction = 0.7
        n_train = int(train_fraction * n_samples)
        
        # Extract train and test data
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:]
        y_test = y[n_train:]
        
        print(f"[INFO] Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
        print(f"[INFO] Test: {X_test.shape[0]} samples × {X_test.shape[1]} features")
        
        # Store dataset statistics
        self.all_stats['feature_counts'].append(X_train.shape[1])
        self.all_stats['train_sample_counts'].append(X_train.shape[0])
        self.all_stats['test_sample_counts'].append(X_test.shape[0])
        
        # Evaluate models
        self.evaluate_models(X_train, y_train, X_test, y_test, dataset_idx)
    
    def evaluate_models(self, X_train, y_train, X_test, y_test, dataset_idx):
        """Evaluate multiple baseline models on the dataset."""
        from sklearn.metrics import mean_squared_error, r2_score
        
        print("\n[INFO] Evaluating baseline models...")
        n_train, n_features = X_train.shape
        n_test = X_test.shape[0]
        
        # Skip if too few samples
        if n_train < 5 or n_test < 5 or n_features == 0:
            print("[WARN] Too few samples/features for model evaluation - skipping")
            return
        
        # Dictionary to store model results for this dataset
        dataset_results = {}
        
        # 1. Linear Regression
        if 'linear' in self.models_to_evaluate:
            try:
                from sklearn.linear_model import LinearRegression
                
                print("\n[INFO] Linear Regression:")
                
                start_time = time.time()
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Get predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                total_time = time.time() - start_time
                
                # Evaluate
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                print(f"[INFO]   Train R²: {train_r2:.6f}, MSE: {train_mse:.8f}")
                print(f"[INFO]   Test R²: {test_r2:.6f}, MSE: {test_mse:.8f}")
                print(f"[INFO]   Total time: {total_time:.6f}s")
                
                # Store results
                self.all_stats['model_performances']['linear']['train_r2'].append(train_r2)
                self.all_stats['model_performances']['linear']['test_r2'].append(test_r2)
                self.all_stats['model_performances']['linear']['train_mse'].append(train_mse)
                self.all_stats['model_performances']['linear']['test_mse'].append(test_mse)
                self.all_stats['model_runtimes']['linear'].append(total_time)
                
                dataset_results['linear'] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'time': total_time
                }
                
            except Exception as e:
                print(f"[WARN] Linear Regression failed: {e}")
        
        # 2. Ridge Regression
        if 'ridge' in self.models_to_evaluate:
            try:
                from sklearn.linear_model import Ridge
                
                print("\n[INFO] Ridge Regression:")
                
                start_time = time.time()
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                
                # Get predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                total_time = time.time() - start_time
                
                # Evaluate
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                print(f"[INFO]   Train R²: {train_r2:.6f}, MSE: {train_mse:.8f}")
                print(f"[INFO]   Test R²: {test_r2:.6f}, MSE: {test_mse:.8f}")
                print(f"[INFO]   Total time: {total_time:.6f}s")
                
                # Store results
                self.all_stats['model_performances']['ridge']['train_r2'].append(train_r2)
                self.all_stats['model_performances']['ridge']['test_r2'].append(test_r2)
                self.all_stats['model_performances']['ridge']['train_mse'].append(train_mse)
                self.all_stats['model_performances']['ridge']['test_mse'].append(test_mse)
                self.all_stats['model_runtimes']['ridge'].append(total_time)
                
                dataset_results['ridge'] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'time': total_time
                }
                
            except Exception as e:
                print(f"[WARN] Ridge Regression failed: {e}")
        
        # 3. Random Forest
        if 'random_forest' in self.models_to_evaluate:
            try:
                from sklearn.ensemble import RandomForestRegressor
                
                print("\n[INFO] Random Forest:")
                
                start_time = time.time()
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                model.fit(X_train, y_train)
                
                # Get predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                total_time = time.time() - start_time
                
                # Evaluate
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                print(f"[INFO]   Train R²: {train_r2:.6f}, MSE: {train_mse:.8f}")
                print(f"[INFO]   Test R²: {test_r2:.6f}, MSE: {test_mse:.8f}")
                print(f"[INFO]   Total time: {total_time:.6f}s")
                
                # Store results
                self.all_stats['model_performances']['random_forest']['train_r2'].append(train_r2)
                self.all_stats['model_performances']['random_forest']['test_r2'].append(test_r2)
                self.all_stats['model_performances']['random_forest']['train_mse'].append(train_mse)
                self.all_stats['model_performances']['random_forest']['test_mse'].append(test_mse)
                self.all_stats['model_runtimes']['random_forest'].append(total_time)
                
                dataset_results['random_forest'] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'time': total_time
                }
                
            except Exception as e:
                print(f"[WARN] Random Forest failed: {e}")
        
        # 4. XGBoost
        if 'xgboost' in self.models_to_evaluate:
            try:
                import xgboost as xgb
                
                print("\n[INFO] XGBoost:")
                
                start_time = time.time()
                model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
                model.fit(X_train, y_train)
                
                # Get predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                total_time = time.time() - start_time
                
                # Evaluate
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                print(f"[INFO]   Train R²: {train_r2:.6f}, MSE: {train_mse:.8f}")
                print(f"[INFO]   Test R²: {test_r2:.6f}, MSE: {test_mse:.8f}")
                print(f"[INFO]   Total time: {total_time:.6f}s")
                
                # Store results
                self.all_stats['model_performances']['xgboost']['train_r2'].append(train_r2)
                self.all_stats['model_performances']['xgboost']['test_r2'].append(test_r2)
                self.all_stats['model_performances']['xgboost']['train_mse'].append(train_mse)
                self.all_stats['model_performances']['xgboost']['test_mse'].append(test_mse)
                self.all_stats['model_runtimes']['xgboost'].append(total_time)
                
                dataset_results['xgboost'] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'time': total_time
                }
                
            except Exception as e:
                print(f"[WARN] XGBoost failed: {e}")
                
        # 5. K-Nearest Neighbors
        if 'knn' in self.models_to_evaluate:
            try:
                from sklearn.neighbors import KNeighborsRegressor
                
                print("\n[INFO] K-Nearest Neighbors:")
                
                # Limit k to n_train to avoid errors
                k = min(5, n_train - 1)
                
                start_time = time.time()
                model = KNeighborsRegressor(n_neighbors=k)
                model.fit(X_train, y_train)
                
                # Get predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                total_time = time.time() - start_time
                
                # Evaluate
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                print(f"[INFO]   Train R²: {train_r2:.6f}, MSE: {train_mse:.8f}")
                print(f"[INFO]   Test R²: {test_r2:.6f}, MSE: {test_mse:.8f}")
                print(f"[INFO]   Total time: {total_time:.6f}s")
                
                # Store results
                self.all_stats['model_performances']['knn']['train_r2'].append(train_r2)
                self.all_stats['model_performances']['knn']['test_r2'].append(test_r2)
                self.all_stats['model_performances']['knn']['train_mse'].append(train_mse)
                self.all_stats['model_performances']['knn']['test_mse'].append(test_mse)
                self.all_stats['model_runtimes']['knn'].append(total_time)
                
                dataset_results['knn'] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'time': total_time
                }
                
            except Exception as e:
                print(f"[WARN] K-Nearest Neighbors failed: {e}")
    
    def create_results_dir(self):
        """Create results directory with timestamp."""
        # Create base results directory in the checks/ResultsBaselines folder
        file_path = Path(__file__)
        checks_dir = file_path.parent  # This is the 'checks' directory
        base_dir = checks_dir / 'ResultsBaselines'
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = base_dir / timestamp
        results_dir.mkdir(exist_ok=True)
        
        self.results_dir = results_dir
        print(f"[INFO] Results will be saved to: {results_dir}")
        
        return results_dir
    
    def generate_performance_visualizations(self):
        """Generate visualizations of model performance."""
        if not self.results_dir:
            self.create_results_dir()
            
        n_datasets = len(self.all_stats['dataset_indices'])
        if n_datasets == 0:
            print("[WARN] No datasets were analyzed, skipping visualizations")
            return
            
        print(f"\n[INFO] Generating performance visualizations for {n_datasets} datasets...")
        
        # Prepare data for plotting
        model_names = []
        test_r2_means = []
        test_r2_stds = []
        test_mse_means = []
        test_mse_stds = []
        runtime_means = []
        runtime_stds = []
        
        # Model display names (prettier than keys)
        model_display_names = {
            'linear': 'Linear Regression',
            'ridge': 'Ridge Regression',
            'random_forest': 'Random Forest',
            'xgboost': 'XGBoost',
            'knn': 'K-Nearest Neighbors'
        }
        
        # Calculate statistics for each model
        for model in self.models_to_evaluate:
            if model in self.all_stats['model_performances'] and len(self.all_stats['model_performances'][model]['test_r2']) > 0:
                test_r2_values = self.all_stats['model_performances'][model]['test_r2']
                test_mse_values = self.all_stats['model_performances'][model]['test_mse']
                runtime_values = self.all_stats['model_runtimes'][model]
                
                model_names.append(model_display_names.get(model, model))
                test_r2_means.append(np.mean(test_r2_values))
                test_r2_stds.append(np.std(test_r2_values))
                test_mse_means.append(np.mean(test_mse_values))
                test_mse_stds.append(np.std(test_mse_values))
                runtime_means.append(np.mean(runtime_values))
                runtime_stds.append(np.std(runtime_values))
        
        # Set plot style
        plt.style.use('ggplot')
        
        # 1. Test R² Comparison (Mean values)
        plt.figure(figsize=(10, 6))
        indices = np.arange(len(model_names))
        
        # Sort by mean test R²
        sort_idx = np.argsort(test_r2_means)[::-1]  # Descending order
        sorted_model_names = [model_names[i] for i in sort_idx]
        sorted_test_r2_means = [test_r2_means[i] for i in sort_idx]
        sorted_test_r2_stds = [test_r2_stds[i] for i in sort_idx]
        
        plt.bar(indices, sorted_test_r2_means, yerr=sorted_test_r2_stds, 
                capsize=10, color='steelblue', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('Test R² Score')
        plt.title('Model Comparison - Mean Test R² Score (higher is better)')
        plt.xticks(indices, sorted_model_names, rotation=45, ha='right')
        plt.ylim(max(0, min(sorted_test_r2_means) - 0.1), min(1.0, max(sorted_test_r2_means) + 0.1))
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(sorted_test_r2_means):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
            
        plt.savefig(self.results_dir / 'test_r2_comparison_mean.png', dpi=300)
        plt.close()
        
        # 1B. Test R² Comparison (Median values)
        plt.figure(figsize=(10, 6))
        
        # Calculate and sort by median test R²
        test_r2_medians = []
        model_keys = list(self.models_to_evaluate)  # Use the original model keys
        
        # Create a mapping from display names back to keys
        display_to_key = {model_display_names.get(key, key): key for key in model_keys}
        
        # Calculate medians using the correct keys
        for i, display_name in enumerate(model_names):
            key = display_to_key.get(display_name, display_name)  # Get the original key
            values = [v for v in self.all_stats['model_performances'][key]['test_r2'] if not np.isnan(v)]
            test_r2_medians.append(np.median(values) if values else np.nan)
            
        # Sort by median test R²
        sort_idx = np.argsort(test_r2_medians)[::-1]  # Descending order
        sorted_model_names = [model_names[i] for i in sort_idx]
        sorted_test_r2_medians = [test_r2_medians[i] for i in sort_idx]
        
        plt.bar(indices, sorted_test_r2_medians, 
                capsize=10, color='darkblue', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('Test R² Score')
        plt.title('Model Comparison - Median Test R² Score (higher is better)')
        plt.xticks(indices, sorted_model_names, rotation=45, ha='right')
        plt.ylim(max(0, min(sorted_test_r2_medians) - 0.1), min(1.0, max(sorted_test_r2_medians) + 0.1))
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(sorted_test_r2_medians):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
            
        plt.savefig(self.results_dir / 'test_r2_comparison_median.png', dpi=300)
        plt.close()
        
        # 2. Test MSE Comparison (Mean values)
        plt.figure(figsize=(10, 6))
        
        # Sort by test MSE (ascending order for MSE - lower is better)
        sort_idx = np.argsort(test_mse_means)
        sorted_model_names = [model_names[i] for i in sort_idx]
        sorted_test_mse_means = [test_mse_means[i] for i in sort_idx]
        sorted_test_mse_stds = [test_mse_stds[i] for i in sort_idx]
        
        plt.bar(indices, sorted_test_mse_means, yerr=sorted_test_mse_stds, 
                capsize=10, color='lightcoral', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('Test MSE')
        plt.title('Model Comparison - Mean Test MSE (lower is better)')
        plt.xticks(indices, sorted_model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(sorted_test_mse_means):
            plt.text(i, v + max(sorted_test_mse_means) * 0.01, f"{v:.3g}", ha='center', va='bottom', fontweight='bold')
            
        plt.savefig(self.results_dir / 'test_mse_comparison_mean.png', dpi=300)
        plt.close()
        
        # 2B. Test MSE Comparison (Median values)
        plt.figure(figsize=(10, 6))
        
        # Calculate and sort by median test MSE
        test_mse_medians = []
        
        # Use mapping from display names to keys
        for i, display_name in enumerate(model_names):
            key = display_to_key.get(display_name, display_name)  # Get the original key
            values = [v for v in self.all_stats['model_performances'][key]['test_mse'] if not np.isnan(v)]
            test_mse_medians.append(np.median(values) if values else np.nan)
            
        # Sort by median test MSE (ascending order - lower is better)
        sort_idx = np.argsort(test_mse_medians)
        sorted_model_names = [model_names[i] for i in sort_idx]
        sorted_test_mse_medians = [test_mse_medians[i] for i in sort_idx]
        
        plt.bar(indices, sorted_test_mse_medians, 
                capsize=10, color='darkred', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('Test MSE')
        plt.title('Model Comparison - Median Test MSE (lower is better)')
        plt.xticks(indices, sorted_model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(sorted_test_mse_medians):
            plt.text(i, v + max(sorted_test_mse_medians) * 0.01, f"{v:.3g}", ha='center', va='bottom', fontweight='bold')
            
        plt.savefig(self.results_dir / 'test_mse_comparison_median.png', dpi=300)
        plt.close()
        
        # 3. Runtime Comparison (Mean values)
        plt.figure(figsize=(10, 6))
        
        # Sort by runtime (ascending order - faster is better)
        sort_idx = np.argsort(runtime_means)
        sorted_model_names = [model_names[i] for i in sort_idx]
        sorted_runtime_means = [runtime_means[i] for i in sort_idx]
        sorted_runtime_stds = [runtime_stds[i] for i in sort_idx]
        
        plt.bar(indices, sorted_runtime_means, yerr=sorted_runtime_stds, 
                capsize=10, color='mediumseagreen', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('Runtime (seconds)')
        plt.title('Model Comparison - Mean Runtime (lower is better)')
        plt.xticks(indices, sorted_model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(sorted_runtime_means):
            plt.text(i, v + max(sorted_runtime_means) * 0.01, f"{v:.3f}s", ha='center', va='bottom', fontweight='bold')
            
        plt.savefig(self.results_dir / 'runtime_comparison_mean.png', dpi=300)
        plt.close()
        
        # 3B. Runtime Comparison (Median values)
        plt.figure(figsize=(10, 6))
        
        # Calculate and sort by median runtime
        runtime_medians = []
        
        # Use mapping from display names to keys
        for i, display_name in enumerate(model_names):
            key = display_to_key.get(display_name, display_name)  # Get the original key
            values = [v for v in self.all_stats['model_runtimes'][key] if not np.isnan(v)]
            runtime_medians.append(np.median(values) if values else np.nan)
            
        # Sort by median runtime (ascending order - faster is better)
        sort_idx = np.argsort(runtime_medians)
        sorted_model_names = [model_names[i] for i in sort_idx]
        sorted_runtime_medians = [runtime_medians[i] for i in sort_idx]
        
        plt.bar(indices, sorted_runtime_medians, 
                capsize=10, color='darkgreen', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('Runtime (seconds)')
        plt.title('Model Comparison - Median Runtime (lower is better)')
        plt.xticks(indices, sorted_model_names, rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, v in enumerate(sorted_runtime_medians):
            plt.text(i, v + max(sorted_runtime_medians) * 0.01, f"{v:.3f}s", ha='center', va='bottom', fontweight='bold')
            
        plt.savefig(self.results_dir / 'runtime_comparison_median.png', dpi=300)
        plt.close()
        
        # 4. Performance-Runtime Tradeoff Scatter Plot (means)
        plt.figure(figsize=(10, 8))
        
        # Use unsorted data for scatter plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        
        plt.figure(figsize=(10, 8))
        for i, model in enumerate(model_names):
            plt.scatter(runtime_means[i], test_r2_means[i], s=300, c=[colors[i]], 
                        alpha=0.7, edgecolors='w', linewidth=2, label=model)
                        
            # Add error bars for both dimensions
            plt.errorbar(runtime_means[i], test_r2_means[i], 
                        xerr=runtime_stds[i], yerr=test_r2_stds[i], 
                        fmt='none', ecolor=colors[i], alpha=0.5, capsize=5)
        
        # Annotate points with model names
        for i, model in enumerate(model_names):
            plt.annotate(model, (runtime_means[i], test_r2_means[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Test R² Score')
        plt.title('Performance-Runtime Tradeoff (Mean Values)')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_runtime_tradeoff_mean.png', dpi=300)
        plt.close()
        
        # 5. Performance-Runtime Tradeoff Scatter Plot (medians)
        plt.figure(figsize=(10, 8))
        
        # Calculate median values for each model
        test_r2_medians = []
        runtime_medians = []
        
        # Use the model keys directly since we're iterating through self.models_to_evaluate
        for i, model_key in enumerate(self.models_to_evaluate):
            if model_key in self.all_stats['model_performances'] and len(self.all_stats['model_performances'][model_key]['test_r2']) > 0:
                test_r2_values = self.all_stats['model_performances'][model_key]['test_r2']
                runtime_values = self.all_stats['model_runtimes'][model_key]
                
                # Get corresponding display name for this model key
                display_name = model_display_names.get(model_key, model_key)
                
                # Calculate medians
                test_r2_median = np.median(test_r2_values)
                runtime_median = np.median(runtime_values)
                
                # Find the index of this display name in the model_names list
                # If it exists in model_names, use that color, otherwise use a default
                if display_name in model_names:
                    idx = model_names.index(display_name)
                    color = colors[idx]
                else:
                    color = colors[i % len(colors)]  # Fallback color
                
                # Plot the point
                plt.scatter(runtime_median, test_r2_median, s=300, c=[color], 
                            alpha=0.7, edgecolors='w', linewidth=2, label=display_name)
                
                # Annotate with display name
                plt.annotate(display_name, (runtime_median, test_r2_median),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=10, fontweight='bold')
        
        plt.xlabel('Runtime (seconds)')
        plt.ylabel('Test R² Score')
        plt.title('Performance-Runtime Tradeoff (Median Values)')
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_runtime_tradeoff_median.png', dpi=300)
        plt.close()
        
        # 5. Generate detailed summary tables including mean and median metrics
        results_data = []
        
        # Process each model using its original key
        for i, display_name in enumerate(model_names):
            # Get the original model key
            key = display_to_key.get(display_name, display_name)
            
            # Calculate median values for each metric using the key
            test_r2_values = self.all_stats['model_performances'][key]['test_r2']
            test_mse_values = self.all_stats['model_performances'][key]['test_mse']
            runtime_values = self.all_stats['model_runtimes'][key]
            
            test_r2_median = np.median(test_r2_values)
            test_mse_median = np.median(test_mse_values)
            runtime_median = np.median(runtime_values)
            
            results_data.append({
                'Model': model,
                'Test_R2_Mean': f"{test_r2_means[i]:.4f} ± {test_r2_stds[i]:.4f}",
                'Test_R2_Median': f"{test_r2_median:.4f}",
                'Test_MSE_Mean': f"{test_mse_means[i]:.4g} ± {test_mse_stds[i]:.4g}",
                'Test_MSE_Median': f"{test_mse_median:.4g}",
                'Runtime_Mean': f"{runtime_means[i]:.4f}s ± {runtime_stds[i]:.4f}s",
                'Runtime_Median': f"{runtime_median:.4f}s",
                # Store raw values for sorting
                'Test_R2_Mean_Value': test_r2_means[i],
                'Test_R2_Median_Value': test_r2_median,
                'Test_MSE_Mean_Value': test_mse_means[i],
                'Test_MSE_Median_Value': test_mse_median,
                'Runtime_Mean_Value': runtime_means[i],
                'Runtime_Median_Value': runtime_median
            })
        
        # Sort by Median Test R² (descending)
        results_data.sort(key=lambda x: x['Test_R2_Median_Value'], reverse=True)
        
        # Save as CSV with all details
        df = pd.DataFrame(results_data)
        columns_to_display = [
            'Model', 
            'Test_R2_Mean', 'Test_R2_Median', 
            'Test_MSE_Mean', 'Test_MSE_Median', 
            'Runtime_Mean', 'Runtime_Median'
        ]
        df_display = df[columns_to_display]  # Display columns only
        df_display.to_csv(self.results_dir / 'model_performance_summary.csv', index=False)
        
        # Save as Markdown with more detailed focus on median performance
        with open(self.results_dir / 'model_performance_summary.md', 'w') as f:
            f.write("# Model Performance Summary\n\n")
            f.write(f"Total datasets analyzed: {n_datasets}\n\n")
            f.write("## Performance Metrics (sorted by Median Test R²)\n\n")
            f.write("| Model | Median Test R² | Mean Test R² | Median MSE | Mean MSE | Median Runtime | Mean Runtime |\n")
            f.write("|-------|--------------|------------|------------|---------|---------------|-------------|\n")
            for row in results_data:
                f.write(f"| {row['Model']} | {row['Test_R2_Median']} | {row['Test_R2_Mean']} | " +
                        f"{row['Test_MSE_Median']} | {row['Test_MSE_Mean']} | " +
                        f"{row['Runtime_Median']} | {row['Runtime_Mean']} |\n")
        
        print(f"[OK] Visualizations saved to {self.results_dir}")
        
        # 6. Save raw statistics and summary metrics as JSON for further analysis
        with open(self.results_dir / 'raw_statistics.json', 'w') as f:
            # Create a copy of the stats that can be serialized to JSON
            stats_copy = {
                'dataset_indices': self.all_stats['dataset_indices'],
                'feature_counts': [int(x) for x in self.all_stats['feature_counts']],
                'train_sample_counts': [int(x) for x in self.all_stats['train_sample_counts']],
                'test_sample_counts': [int(x) for x in self.all_stats['test_sample_counts']],
                'model_performances': {},
                'model_runtimes': {},
                'summary_metrics': {}  # Add summary metrics section
            }
            
                # Add model performance data
                # With our custom NumpyEncoder, we don't need to manually convert everything,
                # but let's be safe and ensure lists are properly handled
            for model in self.all_stats['model_performances']:
                stats_copy['model_performances'][model] = {}
                for metric in self.all_stats['model_performances'][model]:
                    values = self.all_stats['model_performances'][model][metric]
                    if isinstance(values, list):
                        # Convert all values to native Python float to ensure JSON serialization works
                        stats_copy['model_performances'][model][metric] = [float(v) for v in values]
                    else:
                        # Convert single values too
                        stats_copy['model_performances'][model][metric] = float(values) if isinstance(values, (np.number, np.floating)) else values
                        
                if model in self.all_stats['model_runtimes']:
                    stats_copy['model_runtimes'][model] = [float(v) for v in self.all_stats['model_runtimes'][model]]            # Add summary metrics with mean and median for each model and metric
            stats_copy['summary_metrics'] = {}
            for model_key in self.all_stats['model_performances']:
                # Get display name for this model
                display_name = model_display_names.get(model_key, model_key)
                
                # Store metrics under both the key and display name for flexibility
                stats_copy['summary_metrics'][model_key] = {}
                
                # Add performance metrics (test/train R², MSE)
                for metric in self.all_stats['model_performances'][model_key]:
                    values = self.all_stats['model_performances'][model_key][metric]
                    if values:
                        # Use Python's native float for all metrics to ensure JSON compatibility
                        stats_copy['summary_metrics'][model_key][f"{metric}_mean"] = float(np.mean(values))
                        stats_copy['summary_metrics'][model_key][f"{metric}_median"] = float(np.median(values))
                        stats_copy['summary_metrics'][model_key][f"{metric}_std"] = float(np.std(values))
                
                # Add runtime metrics
                if model_key in self.all_stats['model_runtimes'] and self.all_stats['model_runtimes'][model_key]:
                    runtime_values = self.all_stats['model_runtimes'][model_key]
                    stats_copy['summary_metrics'][model_key]["runtime_mean"] = float(np.mean(runtime_values))
                    stats_copy['summary_metrics'][model_key]["runtime_median"] = float(np.median(runtime_values))
                    stats_copy['summary_metrics'][model_key]["runtime_std"] = float(np.std(runtime_values))
                
                # Also store a mapping from display names to keys for reference
                stats_copy['model_display_names'] = model_display_names
            
            # Use our custom encoder that handles NumPy types
            json.dump(stats_copy, f, indent=2, cls=NumpyEncoder)
            
        # Also save just the summary metrics in a separate, easy-to-read file
        with open(self.results_dir / 'performance_summary.txt', 'w') as f:
            f.write("===== MODEL PERFORMANCE SUMMARY =====\n\n")
            f.write(f"Total datasets analyzed: {n_datasets}\n\n")
            
            # Sort models by median test R²
            model_metrics = []
            for model_key in self.all_stats['model_performances']:
                if len(self.all_stats['model_performances'][model_key]['test_r2']) > 0:
                    test_r2_values = self.all_stats['model_performances'][model_key]['test_r2']
                    median_r2 = np.median(test_r2_values)
                    
                    # Get display name for this model
                    display_name = model_display_names.get(model_key, model_key)
                    
                    # Store both key and display name for later use
                    model_metrics.append((model_key, median_r2, display_name))
            
            # Sort by median R² (descending)
            model_metrics.sort(key=lambda x: x[1], reverse=True)
            
            for model_key, _, display_name in model_metrics:
                test_r2_values = self.all_stats['model_performances'][model_key]['test_r2']
                test_mse_values = self.all_stats['model_performances'][model_key]['test_mse']
                runtime_values = self.all_stats['model_runtimes'][model_key]
                
                # Calculate statistics
                r2_mean = np.mean(test_r2_values)
                r2_median = np.median(test_r2_values)
                r2_std = np.std(test_r2_values)
                
                mse_mean = np.mean(test_mse_values)
                mse_median = np.median(test_mse_values)
                mse_std = np.std(test_mse_values)
                
                runtime_mean = np.mean(runtime_values)
                runtime_median = np.median(runtime_values)
                runtime_std = np.std(runtime_values)
                
                f.write(f"MODEL: {display_name}\n")
                f.write(f"  Test R² (median): {r2_median:.6f}\n")
                f.write(f"  Test R² (mean):   {r2_mean:.6f} ± {r2_std:.6f}\n")
                f.write(f"  Test MSE (median): {mse_median:.8f}\n")
                f.write(f"  Test MSE (mean):   {mse_mean:.8f} ± {mse_std:.8f}\n")
                f.write(f"  Runtime (median): {runtime_median:.6f}s\n")
                f.write(f"  Runtime (mean):   {runtime_mean:.6f}s ± {runtime_std:.6f}s\n")
                f.write("\n")
                
                # Calculate statistics
                r2_mean = np.mean(test_r2_values)
                r2_median = np.median(test_r2_values)
                r2_std = np.std(test_r2_values)
                
                mse_mean = np.mean(test_mse_values)
                mse_median = np.median(test_mse_values)
                mse_std = np.std(test_mse_values)
                
                runtime_mean = np.mean(runtime_values)
                runtime_median = np.median(runtime_values)
                runtime_std = np.std(runtime_values)
                
                f.write(f"MODEL: {model}\n")
                f.write(f"  Test R² (median): {r2_median:.6f}\n")
                f.write(f"  Test R² (mean):   {r2_mean:.6f} ± {r2_std:.6f}\n")
                f.write(f"  Test MSE (median): {mse_median:.8f}\n")
                f.write(f"  Test MSE (mean):   {mse_mean:.8f} ± {mse_std:.8f}\n")
                f.write(f"  Runtime (median): {runtime_median:.6f}s\n")
                f.write(f"  Runtime (mean):   {runtime_mean:.6f}s ± {runtime_std:.6f}s\n")
                f.write("\n")
        
        print(f"[OK] Raw statistics and performance summary saved to {self.results_dir}")
        
    def run_evaluation(self, n_batches=1, datasets_per_batch=4, visualize=True):
        """Run the complete evaluation pipeline."""
        # Create results directory
        self.create_results_dir()
        
        # 1. Load configuration
        self.load_config()
        
        # 2. Create dataset and dataloader
        self.create_dataset_and_dataloader(datasets_per_batch)
        
        # 3. Iterate over batches
        print(f"\n[INFO] Starting evaluation with {n_batches} batches of {datasets_per_batch} datasets each")
        
        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx >= n_batches:
                break
                
            self.evaluate_batch(batch, batch_idx)
        
        # 4. Generate visualizations and summary
        if visualize:
            self.generate_performance_visualizations()
            
        print(f"\n[INFO] Evaluation complete! Results saved to {self.results_dir}")
        return self.all_stats


def main():
    # Configuration constants
    CONFIG_PATH = "experiments/FirstTests/configs/early_test.yaml"  # Path to configuration YAML file
    SEED = 42                 # Random seed for reproducibility
    NUM_BATCHES = 1           # Number of batches to process
    DATASETS_PER_BATCH = 1000    # Number of datasets per batch
    VISUALIZE = True          # Generate performance visualizations
    
    # Create evaluator
    evaluator = BaselineModelEvaluator(config_path=CONFIG_PATH, seed=SEED)
    
    # Run evaluation
    evaluator.run_evaluation(
        n_batches=NUM_BATCHES,
        datasets_per_batch=DATASETS_PER_BATCH,
        visualize=VISUALIZE
    )


if __name__ == "__main__":
    main()
