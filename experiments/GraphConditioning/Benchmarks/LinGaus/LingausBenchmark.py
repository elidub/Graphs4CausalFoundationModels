"""
Linear-Gaussian Benchmark for Graph-Conditioned Interventional Models.

This benchmark creates synthetic datasets with varying numbers of nodes to test
graph-conditioned causal inference models. It samples data from multiple configurations
and stores them for evaluation.
"""

import os
import sys
import yaml
import torch
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import json

from sklearn.metrics import r2_score, mean_squared_error

# Add src to path - get the repo root directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
src_path = os.path.join(repo_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset
from models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn
from models.InterventionalPFN_sklearn import InterventionalPFNSklearn


class LinGausBenchmark:
    """
    Benchmark class for graph-conditioned interventional models.
    
    This class handles:
    1. Loading configuration files for different node counts
    2. Sampling synthetic datasets from InterventionalDataset
    3. Storing sampled data to disk for evaluation
    4. Loading stored data for benchmarking
    
    The benchmark uses Linear-Gaussian SCMs with varying complexity (2, 5, 10, 20, 35, 50 nodes).
    """
    
    # Config file names mapping
    CONFIG_FILES = {
        2: "Benchmark_config_2_nodes.yaml",
        5: "Benchmark_config_5_nodes.yaml",
        10: "Benchmark_config_10_nodes.yaml",
        20: "Benchmark_config_20_nodes.yaml",
        35: "Benchmark_config_35_nodes.yaml",
        50: "Benchmark_config_50_nodes.yaml",
        "variable": "Benchmark_config_variable_nodes.yaml",
    }
    
    def __init__(
        self,
        benchmark_dir: str = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/GraphConditioning/Benchmarks/LinGaus",
        cache_dir: Optional[str] = None,
        verbose: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the benchmark.
        
        Args:
            benchmark_dir: Directory containing config files
            cache_dir: Directory to store/load cached data (default: benchmark_dir/data_cache)
            verbose: Whether to print progress messages
            max_samples: Maximum number of samples to evaluate per dataset (default: None = all samples)
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.benchmark_dir / "data_cache"
        self.verbose = verbose
        self.max_samples = max_samples
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store loaded configs
        self.configs: Dict[str, Dict[str, Any]] = {}
        
        # Store loaded model
        self.model: Optional[Any] = None  # Can be either GraphConditionedInterventionalPFNSklearn or InterventionalPFNSklearn
        
        if self.verbose:
            print(f"LinGausBenchmark initialized")
            print(f"  Benchmark dir: {self.benchmark_dir}")
            print(f"  Cache dir: {self.cache_dir}")
            if max_samples is not None:
                print(f"  Max samples per dataset: {max_samples}")
    
    def load_config(self, node_count: int) -> Dict[str, Any]:
        """
        Load a configuration file for a specific node count.
        
        Args:
            node_count: Number of nodes (2, 5, 10, 20, 35, 50, or "variable")
            
        Returns:
            Dictionary containing scm_config, dataset_config, and preprocessing_config
        """
        if node_count not in self.CONFIG_FILES:
            raise ValueError(
                f"No config for {node_count} nodes. Available: {list(self.CONFIG_FILES.keys())}"
            )
        
        # Check if already loaded
        config_key = str(node_count)
        if config_key in self.configs:
            return self.configs[config_key]
        
        # Load from file
        config_path = self.benchmark_dir / self.CONFIG_FILES[node_count]
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store for later reuse
        self.configs[config_key] = config
        
        if self.verbose:
            print(f"Loaded config for {node_count} nodes from {config_path.name}")
        
        return config
    
    def sample_and_save_data(
        self,
        node_count: int,
        num_samples: int,
        dataset_seed: Optional[int] = None,
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Sample data from InterventionalDataset and save to disk.
        
        Args:
            node_count: Number of nodes (2, 5, 10, 20, 35, 50, or "variable")
            num_samples: Number of dataset samples to generate
            dataset_seed: Random seed for dataset (default: None)
            filename: Custom filename for saved data (default: auto-generated)
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to saved file
        """
        # Load configuration
        config = self.load_config(node_count)
        
        # Extract the three main configs
        scm_config = config['scm_config']
        dataset_config = config['dataset_config']
        preprocessing_config = config['preprocessing_config']
        
        # Ensure adjacency matrix is returned
        if 'return_adjacency_matrix' not in dataset_config:
            dataset_config['return_adjacency_matrix'] = {'value': True}
        else:
            dataset_config['return_adjacency_matrix']['value'] = True
        
        # Generate filename if not provided
        if filename is None:
            seed_str = f"_seed{dataset_seed}" if dataset_seed is not None else ""
            filename = f"lingaus_{node_count}nodes_{num_samples}samples{seed_str}.pkl"
        
        save_path = self.cache_dir / filename
        
        # Check if file exists
        if save_path.exists() and not overwrite:
            if self.verbose:
                print(f"File already exists: {save_path}")
                print(f"  Use overwrite=True to regenerate")
            return str(save_path)
        
        if self.verbose:
            print(f"\nSampling data for {node_count} nodes...")
            print(f"  Num samples: {num_samples}")
            print(f"  Seed: {dataset_seed}")
        
        # Create dataset
        dataset = InterventionalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config,
            seed=dataset_seed,
            return_scm=True,  # Include SCM for debugging/analysis
        )
        
        if self.verbose:
            print(f"  Dataset created with size {len(dataset)}")
            print(f"  Sampling {num_samples} elements...")
        
        # Sample data
        sampled_data = []
        iterator = tqdm(range(num_samples), desc="Sampling") if self.verbose else range(num_samples)
        
        for idx in iterator:
            try:
                # Get data tuple: (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adj, scm, processor, intervention_node)
                data_item = dataset[idx]
                
                # Store everything except potentially large SCM object (optional)
                # You can adjust what to save based on your needs
                sampled_data.append({
                    'X_obs': data_item[0],
                    'T_obs': data_item[1],
                    'Y_obs': data_item[2],
                    'X_intv': data_item[3],
                    'T_intv': data_item[4],
                    'Y_intv': data_item[5],
                    'adjacency_matrix': data_item[6],
                    'intervention_node': data_item[9] if len(data_item) > 9 else None,
                    'metadata': {
                        'intervened_feature': data_item[8].intervened_feature if len(data_item) > 8 else None,
                        'selected_target_feature': data_item[8].selected_target_feature if len(data_item) > 8 else None,
                        'kept_feature_indices': data_item[8].kept_feature_indices if len(data_item) > 8 else None,
                    }
                })
            except Exception as e:
                if self.verbose:
                    print(f"\n  Warning: Failed to sample item {idx}: {e}")
                continue
        
        # Create metadata
        metadata = {
            'node_count': node_count,
            'num_samples': len(sampled_data),
            'dataset_seed': dataset_seed,
            'config': config,
            'scm_config': scm_config,
            'dataset_config': dataset_config,
            'preprocessing_config': preprocessing_config,
        }
        
        # Save to disk
        save_data = {
            'data': sampled_data,
            'metadata': metadata,
        }
        
        if self.verbose:
            print(f"\nSaving to {save_path}...")
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        if self.verbose:
            print(f"  Saved {len(sampled_data)} samples successfully")
            print(f"  File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(save_path)
    
    def sample_all_configs(
        self,
        num_samples_per_config: int = 100,
        dataset_seed: Optional[int] = 42,
        node_counts: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> Dict[int, str]:
        """
        Sample data for all (or specified) node configurations.
        
        Args:
            num_samples_per_config: Number of samples to generate per config
            dataset_seed: Base seed for datasets (each config gets seed + offset)
            node_counts: List of node counts to sample (default: all available)
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary mapping node_count -> saved file path
        """
        if node_counts is None:
            node_counts = [2, 5, 10, 20, 35, 50]  # Exclude "variable" by default
        
        saved_paths = {}
        
        for i, node_count in enumerate(node_counts):
            # Use different seed for each config if base seed provided
            seed = dataset_seed + i * 1000 if dataset_seed is not None else None
            
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Processing config {i+1}/{len(node_counts)}: {node_count} nodes")
                print(f"{'='*80}")
            
            try:
                path = self.sample_and_save_data(
                    node_count=node_count,
                    num_samples=num_samples_per_config,
                    dataset_seed=seed,
                    overwrite=overwrite,
                )
                saved_paths[node_count] = path
            except Exception as e:
                if self.verbose:
                    print(f"ERROR: Failed to sample {node_count}-node config: {e}")
                continue
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Completed sampling for {len(saved_paths)}/{len(node_counts)} configs")
            print(f"{'='*80}")
        
        return saved_paths
    
    def load_data(self, filename: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Load sampled data from disk.
        
        Args:
            filename: Name of the saved file
            
        Returns:
            Tuple of (sampled_data_list, metadata_dict)
        """
        load_path = self.cache_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"Data file not found: {load_path}")
        
        if self.verbose:
            print(f"Loading data from {load_path}...")
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        data = save_data['data']
        metadata = save_data['metadata']
        
        if self.verbose:
            print(f"  Loaded {len(data)} samples")
            print(f"  Node count: {metadata.get('node_count')}")
            print(f"  Dataset seed: {metadata.get('dataset_seed')}")
        
        return data, metadata


    def load_model(
        self,
        config_path: str,
        checkpoint_path: str,
        verbose: Optional[bool] = None,
        **model_kwargs,
    ) -> Any:
        """
        Load a graph-conditioned or standard interventional PFN model.
        
        Automatically detects the model type by checking the config file for
        model_config.use_graph_conditioning flag.
        
        Args:
            config_path: Path to the model config YAML file
            checkpoint_path: Path to the model checkpoint (.pt file)
            verbose: Whether to print loading messages (default: use self.verbose)
            **model_kwargs: Additional keyword arguments to pass to the model constructor
                          (e.g., n_estimators, max_n_train for InterventionalPFNSklearn)
            
        Returns:
            Loaded model (either GraphConditionedInterventionalPFNSklearn or InterventionalPFNSklearn)
        """
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print(f"\nLoading model...")
            print(f"  Config: {config_path}")
            print(f"  Checkpoint: {checkpoint_path}")
        
        # Load config to check model type
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store config for later use (e.g., in preprocessing)
        self._current_config = config
        self._current_config_path = config_path
        
        # Generate default model name from config path
        # Extract last two path components and join them
        # e.g., ".../simple_pfn_XXX/step_50000_config.yaml" -> "simple_pfn_XXX_step_50000_config"
        config_path_obj = Path(config_path)
        parent_dir = config_path_obj.parent.name
        config_filename = config_path_obj.stem  # filename without extension
        self._default_model_name = f"{parent_dir}_{config_filename}"
        
        # Check if graph conditioning is used
        use_graph_conditioning = config.get('model_config', {}).get('use_graph_conditioning', {}).get('value', False)
        
        if verbose:
            print(f"  Model type: {'Graph-Conditioned' if use_graph_conditioning else 'Standard'} InterventionalPFN")
        
        # Create and load the appropriate model
        if use_graph_conditioning:
            model = GraphConditionedInterventionalPFNSklearn(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                verbose=verbose,
                **model_kwargs,
            )
        else:
            model = InterventionalPFNSklearn(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                verbose=verbose,
                **model_kwargs,
            )
        
        model.load()
        
        # Store the model
        self.model = model
        
        if verbose:
            print(f"  Model loaded successfully!")
        
        return model
    
    def evaluate_sample(
        self,
        data_item: Dict[str, Any],
        model: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a single sample and return metrics.
        
        Assumes the data is already preprocessed and stored in the expected format.
        
        Args:
            data_item: Dictionary containing:
                - 'X_obs': Observational covariates (tensor)
                - 'T_obs': Observational treatment (tensor)
                - 'Y_obs': Observational outcome (tensor)
                - 'X_intv': Interventional covariates (tensor)
                - 'T_intv': Interventional treatment (tensor)
                - 'Y_intv': Interventional outcome (tensor)
                - 'adjacency_matrix': Graph adjacency matrix (tensor)
            model: Model to evaluate (default: use self.model)
            
        Returns:
            Dictionary with 'mse', 'r2', 'nll' metrics
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model loaded. Use load_model() first or pass a model.")
        
        # Extract data components (already preprocessed)
        X_obs = data_item['X_obs']
        T_obs = data_item['T_obs']
        Y_obs = data_item['Y_obs']
        X_intv = data_item['X_intv']
        T_intv = data_item['T_intv']
        Y_intv = data_item['Y_intv']
        adj = data_item['adjacency_matrix']
        
        # Convert to numpy if needed
        if torch.is_tensor(X_obs):
            X_obs = X_obs.numpy()
            T_obs = T_obs.numpy()
            Y_obs = Y_obs.numpy()
            X_intv = X_intv.numpy()
            T_intv = T_intv.numpy()
            Y_intv = Y_intv.numpy()
            adj = adj.numpy()
        
        # Get expected number of features from model config
        # The model expects num_features columns (X, T, Y format)
        if hasattr(self, '_current_config') and self._current_config:
            expected_features = self._current_config.get('model_config', {}).get('num_features', {}).get('value', 3)
        else:
            expected_features = 3  # Default to 3 features
        
        # Trim feature matrices to match expected number of features
        # Keep only the first expected_features columns from the right side
        if X_obs.shape[1] > expected_features:
            X_obs = X_obs[:, :expected_features]
            X_intv = X_intv[:, :expected_features]
        
        # Trim adjacency matrix to match actual number of nodes
        # num_nodes = num_features (covariates) + 1 (target) + 1 (intervention node)
        num_nodes = expected_features + 2
        if adj.shape[0] > num_nodes:
            adj = adj[:num_nodes, :num_nodes]
        
        # Check if model uses graph conditioning
        use_graph = hasattr(model, 'predict') and 'adjacency_matrix' in model.predict.__code__.co_varnames
        
        # Get predictions
        if use_graph:
            pred = model.predict(
                X_obs=X_obs,
                T_obs=T_obs,
                Y_obs=Y_obs,
                X_intv=X_intv,
                T_intv=T_intv,
                adjacency_matrix=adj,
            )
        else:
            pred = model.predict(
                X_obs=X_obs,
                T_obs=T_obs,
                Y_obs=Y_obs,
                X_intv=X_intv,
                T_intv=T_intv,
            )
        
        # Compute MSE using sklearn
        mse = mean_squared_error(Y_intv, pred)
        
        # Compute R2 using sklearn
        r2 = r2_score(Y_intv, pred)
        
        # Compute NLL
        if use_graph:
            log_likelihood = model.predict_log_likelihood(
                X_obs=X_obs,
                T_obs=T_obs,
                Y_obs=Y_obs,
                X_intv=X_intv,
                T_intv=T_intv,
                Y_intv=Y_intv,
                adjacency_matrix=adj,
            )
        else:
            log_likelihood = model.predict_log_likelihood(
                X_obs=X_obs,
                T_obs=T_obs,
                Y_obs=Y_obs,
                X_intv=X_intv,
                T_intv=T_intv,
                Y_intv=Y_intv,
            )
        
        nll = -np.mean(log_likelihood)
        
        return {
            'mse': float(mse),
            'r2': float(r2),
            'nll': float(nll),
        }
    
    def evaluate_dataset(
        self,
        data: List[Dict[str, Any]],
        model: Optional[Any] = None,
        verbose: Optional[bool] = None,
    ) -> List[Dict[str, float]]:
        """
        Evaluate model on a full dataset.
        
        Args:
            data: List of data items
            model: Model to evaluate (default: use self.model)
            verbose: Whether to show progress (default: use self.verbose)
            
        Returns:
            List of metric dictionaries, one per sample
        """
        if verbose is None:
            verbose = self.verbose
        
        # Limit to max_samples if specified
        if self.max_samples is not None and len(data) > self.max_samples:
            data = data[:self.max_samples]
            if verbose:
                print(f"  Limiting evaluation to first {self.max_samples} samples")
        
        results = []
        iterator = tqdm(data, desc="Evaluating") if verbose else data
        
        for data_item in iterator:
            try:
                metrics = self.evaluate_sample(data_item, model=model)
                results.append(metrics)
            except Exception as e:
                if verbose:
                    print(f"\nWarning: Failed to evaluate sample: {e}")
                continue
        
        return results
    
    def bootstrap_ci(
        self,
        values: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            values: Array of values
            n_bootstrap: Number of bootstrap samples
            ci: Confidence interval (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_means = []
        n = len(values)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - ci
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return float(lower), float(upper)
    
    def aggregate_results(
        self,
        results: List[Dict[str, float]],
        n_bootstrap: int = 1000,
    ) -> Dict[str, Any]:
        """
        Aggregate results with bootstrap confidence intervals.
        
        Args:
            results: List of metric dictionaries
            n_bootstrap: Number of bootstrap samples for CI
            
        Returns:
            Dictionary with aggregated statistics
        """
        if len(results) == 0:
            return {}
        
        metrics = ['mse', 'r2', 'nll']
        aggregated = {}
        
        for metric in metrics:
            values = np.array([r[metric] for r in results])
            
            mean_val = np.mean(values)
            median_val = np.median(values)
            
            # Bootstrap CIs for mean
            mean_ci_lower, mean_ci_upper = self.bootstrap_ci(values, n_bootstrap=n_bootstrap)
            
            # Bootstrap CIs for median
            bootstrap_medians = []
            n = len(values)
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=n, replace=True)
                bootstrap_medians.append(np.median(sample))
            
            median_ci_lower = np.percentile(bootstrap_medians, 2.5)
            median_ci_upper = np.percentile(bootstrap_medians, 97.5)
            
            aggregated[metric] = {
                'mean': float(mean_val),
                'mean_ci_lower': float(mean_ci_lower),
                'mean_ci_upper': float(mean_ci_upper),
                'median': float(median_val),
                'median_ci_lower': float(median_ci_lower),
                'median_ci_upper': float(median_ci_upper),
                'std': float(np.std(values)),
                'n_samples': len(values),
            }
        
        return aggregated
    
    def run_benchmark(
        self,
        data_filename: str,
        model: Optional[Any] = None,
        output_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        n_bootstrap: int = 1000,
        save_individual_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run benchmark on a dataset and save results.
        
        Args:
            data_filename: Name of the data file to load
            model: Model to evaluate (default: use self.model)
            output_dir: Directory to save results (default: benchmark_dir/benchmark_res/model_name)
            model_name: Name for the model (for folder organization)
            n_bootstrap: Number of bootstrap samples
            save_individual_results: Whether to save per-sample results
            
        Returns:
            Dictionary with aggregated results
        """
        # Load data
        data, metadata = self.load_data(data_filename)
        
        node_count = metadata['node_count']
        
        if self.verbose:
            print(f"\nRunning benchmark on {node_count}-node dataset...")
            print(f"  Number of samples: {len(data)}")
        
        # Evaluate
        results = self.evaluate_dataset(data, model=model)
        
        if self.verbose:
            print(f"  Evaluated {len(results)}/{len(data)} samples successfully")
        
        # Aggregate
        aggregated = self.aggregate_results(results, n_bootstrap=n_bootstrap)
        
        # Add metadata
        aggregated['metadata'] = {
            'node_count': node_count,
            'n_samples': len(results),
            'data_filename': data_filename,
            'model_name': model_name,
            'n_bootstrap': n_bootstrap,
        }
        
        # Setup output directory structure
        if output_dir is None:
            base_output_dir = self.benchmark_dir / "benchmark_res"
            if model_name:
                output_dir = base_output_dir / model_name
            elif hasattr(self, '_default_model_name'):
                # Use auto-generated name from config path
                output_dir = base_output_dir / self._default_model_name
            else:
                output_dir = base_output_dir / "unnamed_model"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames (simpler now that we have dedicated folders)
        aggregated_filename = f"aggregated_{node_count}nodes.json"
        
        aggregated_path = output_dir / aggregated_filename
        with open(aggregated_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        if self.verbose:
            print(f"  Saved aggregated results to: {aggregated_path}")
        
        # Save individual results if requested
        if save_individual_results:
            individual_filename = f"individual_{node_count}nodes.json"
            individual_path = output_dir / individual_filename
            
            with open(individual_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            if self.verbose:
                print(f"  Saved individual results to: {individual_path}")
        
        return aggregated
    
    def run_full_benchmark(
        self,
        node_counts: Optional[List[int]] = None,
        model: Optional[Any] = None,
        output_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
        n_bootstrap: int = 1000,
        num_samples: int = 1000,
        base_seed: int = 42,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run benchmark on all node count configurations.
        
        Args:
            node_counts: List of node counts to evaluate (default: all available)
            model: Model to evaluate (default: use self.model)
            output_dir: Directory to save results (default: benchmark_dir/benchmark_res/model_name)
            model_name: Name for the model (for folder organization)
            config_path: Path to the model config file to copy to results folder
            n_bootstrap: Number of bootstrap samples
            num_samples: Number of samples in each dataset
            base_seed: Base seed used for dataset generation
            
        Returns:
            Dictionary mapping node_count -> aggregated results
        """
        if node_counts is None:
            node_counts = [2, 5, 10, 20, 35, 50]
        
        # Setup output directory structure
        if output_dir is None:
            base_output_dir = self.benchmark_dir / "benchmark_res"
            if model_name:
                output_dir = base_output_dir / model_name
            elif hasattr(self, '_default_model_name'):
                # Use auto-generated name from config path
                output_dir = base_output_dir / self._default_model_name
            else:
                output_dir = base_output_dir / "unnamed_model"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model config file to output directory if provided
        if config_path:
            import shutil
            config_source = Path(config_path)
            if config_source.exists():
                config_dest = output_dir / "model_config.yaml"
                shutil.copy2(config_source, config_dest)
                if self.verbose:
                    print(f"\nCopied model config to: {config_dest}")
        elif hasattr(self, '_current_config_path'):
            # Use the config path from load_model if not explicitly provided
            import shutil
            config_source = Path(self._current_config_path)
            if config_source.exists():
                config_dest = output_dir / "model_config.yaml"
                shutil.copy2(config_source, config_dest)
                if self.verbose:
                    print(f"\nCopied model config to: {config_dest}")
        
        # Map node counts to their seeds based on generate_benchmark_data.py
        seed_map = {
            2: base_seed,
            5: base_seed + 1000,
            10: base_seed + 2000,
            20: base_seed + 3000,
            35: base_seed + 4000,
            50: base_seed + 5000,
        }
        
        all_results = {}
        
        for node_count in node_counts:
            # Generate expected filename with correct seed
            dataset_seed = seed_map.get(node_count, base_seed)
            data_filename = f"lingaus_{node_count}nodes_{num_samples}samples_seed{dataset_seed}.pkl"
            
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Benchmarking {node_count}-node configuration")
                print(f"{'='*80}")
            
            try:
                results = self.run_benchmark(
                    data_filename=data_filename,
                    model=model,
                    output_dir=output_dir,
                    model_name=model_name,
                    n_bootstrap=n_bootstrap,
                )
                all_results[node_count] = results
            except Exception as e:
                if self.verbose:
                    print(f"ERROR: Failed to benchmark {node_count}-node config: {e}")
                continue
        
        # Save summary in the model's folder
        summary_filename = "summary_all_nodes.json"
        summary_path = output_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Benchmark complete!")
            print(f"  Evaluated {len(all_results)}/{len(node_counts)} configurations")
            print(f"  Results folder: {output_dir}")
            print(f"  Summary saved to: {summary_path}")
            print(f"{'='*80}")
        
        return all_results
    
    def run(
        self,
        fidelity: str,
        checkpoint_path: str,
        config_path: Optional[str] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Run LinGaus benchmark with specified fidelity level.
        
        This method provides a unified interface compatible with the Trainer's benchmark
        integration pattern. It maps fidelity levels to evaluation configurations:
        
        - "low": 100 samples from all node counts (2, 5, 10, 20, 35, 50)
        - "high": 1000 samples from all node counts (2, 5, 10, 20, 35, 50)
        
        Args:
            fidelity: Fidelity level ("low" or "high")
            checkpoint_path: Path to the model checkpoint .pt file
            config_path: Path to the model config YAML file (uses self._current_config_path if None)
            
        Returns:
            Dictionary mapping node_count -> aggregated_results
            
        Raises:
            ValueError: If fidelity is not "low" or "high"
            RuntimeError: If no model is loaded and config_path is not provided
        """
        # Normalize fidelity
        fidelity = fidelity.strip().lower()
        
        # Map fidelity to max_samples
        fidelity_map = {
            "low": 100,
            "high": 1000,
        }
        
        if fidelity not in fidelity_map:
            raise ValueError(
                f"Invalid fidelity '{fidelity}' for LinGaus benchmark. "
                f"Choose 'low' (100 samples) or 'high' (1000 samples)."
            )
        
        max_samples = fidelity_map[fidelity]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"LinGaus Benchmark - Fidelity: {fidelity.upper()}")
            print(f"  Max samples per dataset: {max_samples}")
            print(f"  Node counts: [2, 5, 10, 20, 35, 50]")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"{'='*80}\n")
        
        # Determine config path
        if config_path is None:
            if hasattr(self, '_current_config_path'):
                config_path = self._current_config_path
            else:
                raise RuntimeError(
                    "No config_path provided and no model loaded. "
                    "Either provide config_path or call load_model() first."
                )
        
        # Load model if not already loaded or if checkpoint changed
        if self.model is None:
            self.load_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                verbose=self.verbose,
            )
        
        # Store original max_samples and override
        original_max_samples = self.max_samples
        self.max_samples = max_samples
        
        try:
            # Run full benchmark
            results = self.run_full_benchmark(
                node_counts=[2, 5, 10, 20, 35, 50],
                model=self.model,
                config_path=config_path,
                n_bootstrap=1000,
                num_samples=1000,
                base_seed=42,
            )
            
            return results
            
        finally:
            # Restore original max_samples
            self.max_samples = original_max_samples
    
    def quick_benchmark(
        self,
        config_path: str,
        checkpoint_path: str,
        max_samples: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Convenience function to quickly run a full benchmark on a model.
        
        This is a simplified interface that runs the benchmark with sensible defaults:
        - All node counts: [2, 5, 10, 20, 35, 50]
        - Bootstrap samples: 1000
        - Automatic folder naming from config path
        - Automatic model config copying
        
        Args:
            config_path: Path to the model config YAML file
            checkpoint_path: Path to the model checkpoint .pt file
            max_samples: Maximum number of samples to evaluate per dataset (default: None = all samples)
            model_kwargs: Optional additional kwargs for model constructor (e.g., {'n_estimators': 1})
            
        Returns:
            Dictionary mapping node_count -> aggregated_results
            
        Example:
            >>> benchmark = LinGausBenchmark(max_samples=100)
            >>> results = benchmark.quick_benchmark(
            ...     config_path='/path/to/config.yaml',
            ...     checkpoint_path='/path/to/checkpoint.pt',
            ...     max_samples=100,
            ... )
        """
        # Store original max_samples setting
        original_max_samples = self.max_samples
        
        # Override max_samples if provided
        if max_samples is not None:
            self.max_samples = max_samples
        
        try:
            # Load model
            if model_kwargs is None:
                model_kwargs = {}
            
            model = self.load_model(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                verbose=self.verbose,
                **model_kwargs,
            )
            
            # Run full benchmark with defaults
            results = self.run_full_benchmark(
                node_counts=[2, 5, 10, 20, 35, 50],
                model=model,
                config_path=config_path,
                n_bootstrap=1000,
                num_samples=1000,
                base_seed=42,
            )
            
            return results
            
        finally:
            # Restore original max_samples setting
            self.max_samples = original_max_samples
    
    def run_all_models_benchmark(
        self,
        model_configs: List[Dict[str, Any]],
        node_counts: Optional[List[int]] = None,
        n_bootstrap: int = 1000,
        num_samples: int = 1000,
        base_seed: int = 42,
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Run benchmark on multiple models across all node count configurations.
        
        This is a convenience method to benchmark multiple models at once.
        Each model gets its own folder with results for all node counts.
        
        Args:
            model_configs: List of model configuration dictionaries, each containing:
                - 'config_path': Path to model config YAML file
                - 'checkpoint_path': Path to model checkpoint .pt file
                - 'model_kwargs': Optional dict of additional kwargs for model constructor
                - 'model_name': Optional custom name (default: auto-generated from config path)
            node_counts: List of node counts to evaluate (default: [2, 5, 10, 20, 35, 50])
            n_bootstrap: Number of bootstrap samples for confidence intervals
            num_samples: Number of samples in each dataset
            base_seed: Base seed used for dataset generation
            
        Returns:
            Dictionary mapping model_name -> {node_count -> aggregated_results}
            
        Example:
            >>> model_configs = [
            ...     {
            ...         'config_path': '/path/to/graph_conditioned/config.yaml',
            ...         'checkpoint_path': '/path/to/graph_conditioned/checkpoint.pt',
            ...         'model_kwargs': {},
            ...     },
            ...     {
            ...         'config_path': '/path/to/baseline/config.yaml',
            ...         'checkpoint_path': '/path/to/baseline/checkpoint.pt',
            ...         'model_kwargs': {'n_estimators': 1, 'max_n_train': 10000},
            ...     },
            ... ]
            >>> results = benchmark.run_all_models_benchmark(model_configs)
        """
        if node_counts is None:
            node_counts = [2, 5, 10, 20, 35, 50]
        
        all_model_results = {}
        
        for i, model_config in enumerate(model_configs):
            config_path = model_config['config_path']
            checkpoint_path = model_config['checkpoint_path']
            model_kwargs = model_config.get('model_kwargs', {})
            custom_model_name = model_config.get('model_name', None)
            
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Benchmarking model {i+1}/{len(model_configs)}")
                print(f"  Config: {config_path}")
                print(f"{'='*80}")
            
            try:
                # Load model
                model = self.load_model(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    verbose=self.verbose,
                    **model_kwargs,
                )
                
                # Determine model name (use custom if provided, otherwise use auto-generated)
                if custom_model_name:
                    model_name = custom_model_name
                elif hasattr(self, '_default_model_name'):
                    model_name = self._default_model_name
                else:
                    model_name = f"model_{i}"
                
                # Run full benchmark for this model
                results = self.run_full_benchmark(
                    node_counts=node_counts,
                    model=model,
                    model_name=model_name,
                    config_path=config_path,
                    n_bootstrap=n_bootstrap,
                    num_samples=num_samples,
                    base_seed=base_seed,
                )
                
                all_model_results[model_name] = results
                
                if self.verbose:
                    print(f"\n  ✓ Completed benchmark for {model_name}")
                    print(f"    Evaluated {len(results)}/{len(node_counts)} node configurations")
                
            except Exception as e:
                if self.verbose:
                    print(f"\n  ✗ ERROR: Failed to benchmark model: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        # Save overall summary
        overall_summary_path = self.benchmark_dir / "benchmark_res" / "all_models_summary.json"
        with open(overall_summary_path, 'w') as f:
            json.dump(all_model_results, f, indent=2)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"All models benchmark complete!")
            print(f"  Total models evaluated: {len(all_model_results)}/{len(model_configs)}")
            print(f"  Results directory: {self.benchmark_dir / 'benchmark_res'}")
            print(f"  Overall summary: {overall_summary_path}")
            print(f"{'='*80}")
        
        return all_model_results


if __name__ == "__main__":
    # Run full benchmark on graph-conditioned model across all node configurations
    print("LinGausBenchmark - Full Benchmark")
    print("="*80)
    
    MAX_SAMPLES = 3  # Limit to 3 samples per dataset for quick testing
    benchmark = LinGausBenchmark(verbose=True, max_samples=MAX_SAMPLES)
    
    # Test loading a config
    print("\nTesting config loading...")
    config = benchmark.load_config(5)
    print(f"  Loaded config keys: {list(config.keys())}")
    print(f"  Num nodes: {config['scm_config']['num_nodes']}")
    
    # Load and run full benchmark on graph-conditioned model
    print("\nRunning full benchmark on graph-conditioned model...")
    print("  Node configurations: 2, 5, 10, 20, 35, 50")
    print("  Max samples per dataset: 100")
    print("="*80)
    
    config_path_gc = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/GraphConditioning/checkpoints/simple_pfn_16678198.0_five_node_lingaus_gcn_and_hard_attention/step_50000_config.yaml"
    checkpoint_path_gc = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/GraphConditioning/checkpoints/simple_pfn_16678198.0_five_node_lingaus_gcn_and_hard_attention/step_50000.pt"
    
    try:
        # Run full benchmark using the convenience function
        all_results = benchmark.quick_benchmark(
            config_path=config_path_gc,
            checkpoint_path=checkpoint_path_gc,
            max_samples=MAX_SAMPLES,  # Limit to 100 samples per dataset for quick testing
        )
        breakpoint()
        
        twonode_mse = all_results[2]['mse']['mean']
        fivenode_mse = all_results[5]['mse']['mean']
        tennode_mse = all_results[10]['mse']['mean']
        twentynode_mse = all_results[20]['mse']['mean']
        thirtyfivenode_mse = all_results[35]['mse']['mean']
        fiftynode_mse = all_results[50]['mse']['mean']

        print(f"\nMSE Results:")
        print(f"  2 nodes: {twonode_mse:.4f}")
        print(f"  5 nodes: {fivenode_mse:.4f}")
        print(f"  10 nodes: {tennode_mse:.4f}")
        print(f"  20 nodes: {twentynode_mse:.4f}")
        print(f"  35 nodes: {thirtyfivenode_mse:.4f}")
        print(f"  50 nodes: {fiftynode_mse:.4f}")

        
        print(f"\n{'='*80}")
        print("Benchmark Results Summary")
        print(f"{'='*80}")
        for node_count, result in all_results.items():
            print(f"\n{node_count} nodes:")
            for metric in ['mse', 'r2', 'nll']:
                if metric in result:
                    stats = result[metric]
                    print(f"  {metric.upper()}:")
                    print(f"    Mean: {stats['mean']:.4f} [{stats['mean_ci_lower']:.4f}, {stats['mean_ci_upper']:.4f}]")
                    print(f"    Median: {stats['median']:.4f} [{stats['median_ci_lower']:.4f}, {stats['median_ci_upper']:.4f}]")
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Convenience function usage:")
    print("  results = benchmark.quick_benchmark(")
    print("      config_path='/path/to/config.yaml',")
    print("      checkpoint_path='/path/to/checkpoint.pt',")
    print("      max_samples=100,  # Optional: limit samples per dataset")
    print("  )")
    print("\nTo run benchmark on multiple models:")
    print("  python run_benchmark.py")
    print("="*80)
