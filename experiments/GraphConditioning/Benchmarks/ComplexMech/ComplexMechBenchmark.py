"""
Complex Mechanism Benchmark for Graph-Conditioned Interventional Models.

This benchmark creates synthetic datasets with varying numbers of nodes and complex mechanisms
(XGBoost, MLPs with varying architectures, mixed noise types) to test graph-conditioned causal 
inference models. It samples data from multiple configurations and stores them for evaluation.
"""

import os
import sys
import yaml
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from sklearn.metrics import r2_score, mean_squared_error

# Add both repo root and src to path - get the repo root directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
src_path = os.path.join(repo_root, 'src')

# Debug logging for import issues
import_debug_log = "/tmp/complexmech_import_debug.log"
with open(import_debug_log, "w") as f:
    f.write(f"ComplexMech Benchmark Import Debug\n")
    f.write(f"="*80 + "\n")
    f.write(f"repo_root: {repo_root}\n")
    f.write(f"src_path: {src_path}\n")
    f.write(f"sys.path before: {sys.path}\n")

# Add repo root FIRST so 'src.models' imports work
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# Then add src path so direct imports work
if src_path not in sys.path:
    sys.path.insert(0, src_path)

with open(import_debug_log, "a") as f:
    f.write(f"sys.path after: {sys.path}\n")
    f.write(f"="*80 + "\n")

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset
from models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn
# Import the new batched version
try:
    from models.InterventionalPFN_sklearn_batched import InterventionalPFNSklearn
except ImportError:
    # Fallback to old version if new one not available
    from models.InterventionalPFN_sklearn import InterventionalPFNSklearn


class ComplexMechBenchmark:
    """
    Benchmark class for complex mechanism interventional models with partial graph knowledge.
    
    This class handles the ComplexMech benchmark with complex SCM mechanisms including:
    - XGBoost mechanisms with varying probability
    - MLP mechanisms with tabicl nonlinearities
    - Mixed noise types (Normal, Laplace, StudentT)
    - Varying architectures and parameters
    
    Features:
    1. Loading configuration files from nested directory structure
    2. Support for base configs and path variants with different sample sizes
    3. Sampling synthetic datasets with complex mechanisms
    4. Storing sampled data to disk for evaluation
    5. Loading stored data for benchmarking
    
    The benchmark uses Complex SCMs with varying complexity (2, 5, 20, 50 nodes).
    
    Directory structure:
        configs/{N}node/base.yaml                           # Base configuration
        configs/{N}node/path_TY/ntest_{size}.yaml           # Path variants with different sample sizes
        configs/{N}node/path_YT/ntest_{size}.yaml
        configs/{N}node/path_independent_TY/ntest_{size}.yaml
    """
    
    # Node counts available
    NODE_COUNTS = [2, 5, 20, 50]
    
    # Path variants (excluding base)
    PATH_VARIANTS = ["path_TY", "path_YT", "path_independent_TY"]
    
    # All variants including base
    VARIANTS = ["base"] + PATH_VARIANTS
    
    # Sample size variants for path configs
    SAMPLE_SIZES = [500, 700, 800, 900, 950]
    
    def __init__(
        self,
        benchmark_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
        max_samples: Optional[int] = None,
        use_ancestor_matrix: bool = False,
    ):
        """
        Initialize the benchmark.
        
        Args:
            benchmark_dir: Directory containing config files
            cache_dir: Directory to store/load cached data (default: benchmark_dir/data_cache)
            verbose: Whether to print progress messages
            max_samples: Maximum number of samples to evaluate per dataset (default: None = all samples)
            use_ancestor_matrix: If True, compute and use ancestor matrix instead of adjacency matrix (default: False)
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.benchmark_dir / "data_cache"
        self.verbose = verbose
        self.max_samples = max_samples
        self.use_ancestor_matrix = use_ancestor_matrix
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store loaded configs
        self.configs: Dict[str, Dict[str, Any]] = {}
        
        # Store loaded model
        self.model: Optional[Any] = None  # Can be either GraphConditionedInterventionalPFNSklearn or InterventionalPFNSklearn
        self.batch_size: Optional[int] = None  # Batch size for inference
        
        if self.verbose:
            print(f"ComplexMech Benchmark initialized")
            print(f"  Benchmark dir: {self.benchmark_dir}")
            print(f"  Cache dir: {self.cache_dir}")
            print(f"  Use ancestor matrix: {self.use_ancestor_matrix}")
            print(f"  Max samples per dataset: {self.max_samples}")
    
    def load_config(
        self, 
        node_count: int, 
        variant: str = "base",
        sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load a configuration file for a specific node count and variant.
        
        Args:
            node_count: Number of nodes (2, 5, 20, 50)
            variant: Config variant ("base", "path_TY", "path_YT", "path_independent_TY")
            sample_size: For path variants, which sample size to load (500, 700, 800, 900, 950)
                        Not needed for "base" variant
            
        Returns:
            Dictionary containing scm_config, dataset_config, and preprocessing_config
            
        Examples:
            >>> # Load base config
            >>> config = benchmark.load_config(5, "base")
            
            >>> # Load path variant with specific sample size
            >>> config = benchmark.load_config(5, "path_TY", sample_size=500)
        """
        if node_count not in self.NODE_COUNTS:
            raise ValueError(f"Node count {node_count} not supported. Available: {self.NODE_COUNTS}")
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Variant {variant} not supported. Available: {self.VARIANTS}")
        
        # For path variants, sample_size is required
        if variant != "base" and sample_size is None:
            raise ValueError(f"sample_size must be specified for path variant '{variant}'. "
                           f"Available sizes: {self.SAMPLE_SIZES}")
        
        # For base variant, sample_size should not be specified
        if variant == "base" and sample_size is not None:
            raise ValueError(f"sample_size should not be specified for base variant")
        
        # Validate sample_size if provided
        if sample_size is not None and sample_size not in self.SAMPLE_SIZES:
            raise ValueError(f"Sample size {sample_size} not supported. Available: {self.SAMPLE_SIZES}")
        
        # Create config key for caching
        if variant == "base":
            config_key = f"{node_count}node_base"
        else:
            config_key = f"{node_count}node_{variant}_ntest_{sample_size}"
        
        # Check if already loaded
        if config_key in self.configs:
            return self.configs[config_key]
        
        # Construct config path based on structure
        configs_dir = self.benchmark_dir / "configs"
        node_dir = configs_dir / f"{node_count}node"
        
        if variant == "base":
            config_path = node_dir / "base.yaml"
        else:
            config_path = node_dir / variant / f"ntest_{sample_size}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store for later reuse
        self.configs[config_key] = config
        
        if self.verbose:
            print(f"Loaded config: {config_key}")
            print(f"  Path: {config_path}")
            print(f"  Experiment: {config.get('experiment_name', 'N/A')}")
        
        return config
    
    def sample_and_save_data(
        self,
        node_count: int,
        num_samples: int,
        variant: str = "base",
        sample_size: Optional[int] = None,
        dataset_seed: Optional[int] = None,
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Sample data from InterventionalDataset and save to disk.
        
        Args:
            node_count: Number of nodes (2, 5, 20, 50)
            num_samples: Number of dataset samples to generate
            variant: Config variant ("base", "path_TY", "path_YT", "path_independent_TY")
            sample_size: For path variants, sample size parameter (500, 700, 800, 900, 950)
            dataset_seed: Random seed for dataset (default: None)
            filename: Custom filename for saved data (default: auto-generated)
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to saved file
        """
        # Load configuration
        config = self.load_config(node_count, variant=variant, sample_size=sample_size)
        
        # Extract the three main configs
        scm_config = config['scm_config']
        dataset_config = config['dataset_config']
        preprocessing_config = config['preprocessing_config']
        
        # Ensure adjacency matrix is returned (we always need it)
        # If use_ancestor_matrix is True, we'll compute ancestor matrix from adjacency
        if 'return_adjacency_matrix' not in dataset_config:
            dataset_config['return_adjacency_matrix'] = {'value': True}
        else:
            dataset_config['return_adjacency_matrix']['value'] = True
        
        # Ensure ancestor matrix is not directly requested from dataset
        # (we'll compute it ourselves if needed)
        if 'return_ancestor_matrix' in dataset_config:
            dataset_config['return_ancestor_matrix']['value'] = False
        
        # Generate filename if not provided
        if filename is None:
            size_str = f"_ntest_{sample_size}" if variant != "base" else ""
            filename = f"complexmech_{node_count}node_{variant}{size_str}_{num_samples}samples.pkl"
        
        save_path = self.cache_dir / filename
        
        # Check if file exists
        if save_path.exists() and not overwrite:
            if self.verbose:
                print(f"File already exists: {save_path}")
                print("Use overwrite=True to regenerate")
            return str(save_path)
        
        if self.verbose:
            print(f"Sampling {num_samples} datasets from ComplexMech config:")
            print(f"  Node count: {node_count}")
            print(f"  Variant: {variant}")
            if sample_size is not None:
                print(f"  Sample size: {sample_size}")
        
        # Create dataset
        dataset = InterventionalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config,
            seed=dataset_seed,
            return_scm=True,  # We need this to access the adjacency matrix
        )
        
        if self.verbose:
            print("Dataset created, starting sampling...")
        
        # Sample data
        sampled_data = []
        iterator = tqdm(range(num_samples), desc="Sampling") if self.verbose else range(num_samples)
        
        for idx in iterator:
            try:
                # Sample a single dataset instance
                sample = next(iter(dataset))
                
                # Extract components
                X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, scm_instance = sample
                
                # Get adjacency matrix from SCM
                adj_matrix = scm_instance.adjacency_matrix
                
                # Convert to ancestor matrix if requested
                if self.use_ancestor_matrix:
                    # Compute ancestor matrix: A_ij = 1 if there's a directed path from j to i
                    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
                    n_nodes = adj_tensor.shape[0]
                    # Compute (I + A)^{n-1} and check for paths
                    eye = torch.eye(n_nodes)
                    reachability = eye + adj_tensor
                    for _ in range(n_nodes - 1):
                        reachability = torch.matmul(reachability, eye + adj_tensor)
                    # Ancestor matrix: 1 if reachable (excluding self-connections)
                    ancestor_matrix = (reachability > 0).float() - eye
                    graph_matrix = ancestor_matrix.numpy()
                else:
                    graph_matrix = adj_matrix
                
                # Store the data
                data_dict = {
                    'X_obs': X_obs,
                    'T_obs': T_obs, 
                    'Y_obs': Y_obs,
                    'X_intv': X_intv,
                    'T_intv': T_intv,
                    'Y_intv': Y_intv,
                    'adjacency_matrix': graph_matrix,  # Note: Always stores the matrix type requested
                }
                
                sampled_data.append(data_dict)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error sampling dataset {idx}: {e}")
                continue
        
        # Create metadata
        metadata = {
            'node_count': node_count,
            'variant': variant,
            'sample_size': sample_size,
            'num_samples': len(sampled_data),
            'dataset_seed': dataset_seed,
            'use_ancestor_matrix': self.use_ancestor_matrix,
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
            print(f"Saving {len(sampled_data)} samples to: {save_path}")
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        if self.verbose:
            print(f"Successfully saved data to: {save_path}")
        
        return str(save_path)
    
    def sample_all_configs(
        self,
        num_samples_per_config: int = 100,
        dataset_seed: Optional[int] = 42,
        node_counts: Optional[List[int]] = None,
        variants: Optional[List[str]] = None,
        sample_sizes: Optional[List[int]] = None,
        overwrite: bool = False,
    ) -> Dict[str, str]:
        """
        Sample data for all (or specified) configurations.
        
        Args:
            num_samples_per_config: Number of samples to generate per config
            dataset_seed: Base seed for datasets (each config gets seed + offset)
            node_counts: List of node counts to sample (default: all available)
            variants: List of variants to sample (default: all available)
            sample_sizes: List of sample sizes for path variants (default: all available)
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary mapping config_key -> saved file path
        """
        if node_counts is None:
            node_counts = self.NODE_COUNTS.copy()
        
        if variants is None:
            variants = self.VARIANTS.copy()
            
        if sample_sizes is None:
            sample_sizes = self.SAMPLE_SIZES.copy()
        
        saved_paths = {}
        
        config_idx = 0
        total_configs = len(node_counts) * (1 + len([v for v in variants if v != "base"]) * len(sample_sizes))
        
        for node_count in node_counts:
            for variant in variants:
                if variant == "base":
                    # Base variant - no sample size needed
                    config_key = f"{node_count}node_base"
                    seed = dataset_seed + config_idx if dataset_seed is not None else None
                    
                    saved_path = self.sample_and_save_data(
                        node_count=node_count,
                        num_samples=num_samples_per_config,
                        variant=variant,
                        dataset_seed=seed,
                        overwrite=overwrite,
                    )
                    
                    saved_paths[config_key] = saved_path
                    config_idx += 1
                    
                    if self.verbose:
                        print(f"Completed config {config_idx}/{total_configs}: {config_key}")
                
                else:
                    # Path variants - need sample size
                    for sample_size in sample_sizes:
                        config_key = f"{node_count}node_{variant}_ntest_{sample_size}"
                        seed = dataset_seed + config_idx if dataset_seed is not None else None
                        
                        saved_path = self.sample_and_save_data(
                            node_count=node_count,
                            num_samples=num_samples_per_config,
                            variant=variant,
                            sample_size=sample_size,
                            dataset_seed=seed,
                            overwrite=overwrite,
                        )
                        
                        saved_paths[config_key] = saved_path
                        config_idx += 1
                        
                        if self.verbose:
                            print(f"Completed config {config_idx}/{total_configs}: {config_key}")
        
        if self.verbose:
            print(f"\nCompleted sampling for all {len(saved_paths)} configurations!")
            print(f"Files saved in: {self.cache_dir}")
        
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
            print(f"Loading data from: {load_path}")
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        data = save_data['data']
        metadata = save_data.get('metadata', {})
        
        if self.verbose:
            print(f"Loaded {len(data)} samples")
            print(f"Metadata: node_count={metadata.get('node_count')}, variant={metadata.get('variant')}")
        
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
            print(f"Loading model from:")
            print(f"  Config: {config_path}")
            print(f"  Checkpoint: {checkpoint_path}")
        
        # Load config to check model type
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store config for later use (e.g., in preprocessing)
        self._current_config = config
        self._current_config_path = config_path
        
        # Generate default model name from config path
        config_path_obj = Path(config_path)
        parent_dir = config_path_obj.parent.name
        config_filename = config_path_obj.stem
        self._default_model_name = f"{parent_dir}_{config_filename}"
        
        # Check if graph conditioning is used
        use_graph_conditioning = config.get('model_config', {}).get('use_graph_conditioning', {}).get('value', False)
        
        # Extract batch size from config
        def _get_cfg_value(cfg_dict, key, default=None):
            val = cfg_dict.get(key, default)
            if isinstance(val, dict) and 'value' in val:
                return val['value']
            return val
        
        training_batch_size = _get_cfg_value(config.get('training_config', {}), 'batch_size', None)
        if training_batch_size is None:
            training_batch_size = _get_cfg_value(config.get('model_config', {}), 'batch_size', None)
        
        if training_batch_size is not None:
            self.batch_size = max(1, training_batch_size // 2)
        else:
            self.batch_size = 32
        
        if verbose:
            print(f"Model type: {'Graph-conditioned' if use_graph_conditioning else 'Standard'}")
            print(f"Using batch size: {self.batch_size}")
        
        # Create and load the appropriate model
        if use_graph_conditioning:
            model = GraphConditionedInterventionalPFNSklearn(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                **model_kwargs
            )
        else:
            model = InterventionalPFNSklearn(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                **model_kwargs
            )
        
        model.load()
        
        # Store the model
        self.model = model
        
        if verbose:
            print("Model loaded successfully!")
        
        return model
    
    def evaluate_sample(
        self,
        data_item: Dict[str, Any],
        model: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a single sample and return metrics.
        
        Args:
            data_item: Dictionary containing preprocessed data
            model: Model to evaluate (default: use self.model)
            
        Returns:
            Dictionary with 'mse', 'r2', 'nll' metrics
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model available. Load a model first using load_model().")
        
        # Extract data components (already preprocessed)
        X_obs = data_item['X_obs']
        T_obs = data_item['T_obs']
        Y_obs = data_item['Y_obs']
        X_intv = data_item['X_intv']
        T_intv = data_item['T_intv']
        Y_intv = data_item['Y_intv']
        
        # Choose which graph matrix to use based on use_ancestor_matrix flag
        # This matches the behavior in LinGausBenchmarkIDK
        if self.use_ancestor_matrix:
            # Try to get ancestor matrix, fall back to graph_matrix
            adj = data_item.get('ancestor_matrix', None)
            if adj is None:
                adj = data_item.get('graph_matrix', None)
                if adj is not None and self.verbose:
                    print(f"Warning: ancestor_matrix not found, using graph_matrix instead")
        else:
            # Try to get adjacency/graph matrix
            adj = data_item.get('adjacency_matrix', None)
            if adj is None:
                adj = data_item.get('graph_matrix', None)
        
        # If still None, try any available matrix as fallback
        if adj is None:
            for key in ['ancestor_matrix', 'graph_matrix', 'adjacency_matrix']:
                if key in data_item and data_item[key] is not None:
                    adj = data_item[key]
                    if self.verbose:
                        print(f"Warning: Using fallback matrix '{key}'")
                    break
        
        if adj is None:
            raise ValueError(
                f"No graph matrix found in data_item. Available keys: {list(data_item.keys())}. "
                f"use_ancestor_matrix={self.use_ancestor_matrix}"
            )
        
        # Convert to numpy if needed (but keep tensor for any additional computations)
        adj_tensor = adj if torch.is_tensor(adj) else torch.tensor(adj)
        
        # Convert tensors to numpy if needed
        if torch.is_tensor(adj):
            adj = adj.numpy()
        
        # Convert other tensors to numpy
        if torch.is_tensor(X_obs):
            X_obs, T_obs, Y_obs = X_obs.numpy(), T_obs.numpy(), Y_obs.numpy()
            X_intv, T_intv, Y_intv = X_intv.numpy(), T_intv.numpy(), Y_intv.numpy()
        
        # Get expected number of features from model config
        if hasattr(self, '_current_config') and self._current_config:
            expected_features = self._current_config.get('model_config', {}).get('num_features', {}).get('value', X_obs.shape[1])
        else:
            expected_features = X_obs.shape[1]
        
        # Trim feature matrices to match expected number of features
        if X_obs.shape[1] > expected_features:
            X_obs, X_intv = X_obs[:, :expected_features], X_intv[:, :expected_features]
        
        # Trim adjacency matrix to match actual number of nodes
        num_nodes = expected_features + 2
        if adj.shape[0] > num_nodes:
            adj = adj[:num_nodes, :num_nodes]
        
        # Check if model uses graph conditioning
        use_graph = hasattr(model, 'predict') and 'adjacency_matrix' in model.predict.__code__.co_varnames
        
        # Get predictions
        if use_graph:
            pred = model.predict(
                X_obs=X_obs, T_obs=T_obs, Y_obs=Y_obs,
                X_intv=X_intv, T_intv=T_intv,
                adjacency_matrix=adj
            )
        else:
            pred = model.predict(
                X_obs=X_obs, T_obs=T_obs, Y_obs=Y_obs,
                X_intv=X_intv, T_intv=T_intv
            )
        
        # Compute MSE using sklearn
        mse = mean_squared_error(Y_intv, pred)
        
        # Compute R2 using sklearn
        r2 = r2_score(Y_intv, pred)
        
        # Compute NLL
        if use_graph:
            log_likelihood = model.log_likelihood(
                X_obs=X_obs, T_obs=T_obs, Y_obs=Y_obs,
                X_intv=X_intv, T_intv=T_intv, Y_intv=Y_intv,
                adjacency_matrix=adj
            )
        else:
            log_likelihood = model.log_likelihood(
                X_obs=X_obs, T_obs=T_obs, Y_obs=Y_obs,
                X_intv=X_intv, T_intv=T_intv, Y_intv=Y_intv
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
        
        if model is None:
            model = self.model
        
        # Limit to max_samples if specified
        if self.max_samples is not None and len(data) > self.max_samples:
            data = data[:self.max_samples]
            if verbose:
                print(f"Limited to {self.max_samples} samples")
        
        results = []
        iterator = tqdm(data, desc="Evaluating") if verbose else data
        
        for data_item in iterator:
            try:
                result = self.evaluate_sample(data_item, model=model)
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Error evaluating sample: {e}")
                    import traceback
                    traceback.print_exc()
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
            bootstrap_sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - ci
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return lower, upper
    
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
            values = np.array([r[metric] for r in results if metric in r])
            
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Bootstrap CI
                if len(values) > 1:
                    ci_lower, ci_upper = self.bootstrap_ci(values, n_bootstrap)
                else:
                    ci_lower, ci_upper = mean_val, mean_val
                
                aggregated[metric] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'ci_lower': float(ci_lower),
                    'ci_upper': float(ci_upper),
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
            output_dir: Directory to save results
            model_name: Name for the model (for folder organization)
            n_bootstrap: Number of bootstrap samples
            save_individual_results: Whether to save per-sample results
            
        Returns:
            Dictionary with aggregated results
        """
        # Load data
        data, metadata = self.load_data(data_filename)
        
        node_count = metadata['node_count']
        variant = metadata.get('variant', 'unknown')
        sample_size = metadata.get('sample_size', None)
        
        if self.verbose:
            print(f"Running benchmark on {node_count}node {variant} dataset")
            if sample_size is not None:
                print(f"Sample size: {sample_size}")
        
        # Evaluate
        results = self.evaluate_dataset(data, model=model)
        
        if self.verbose:
            print(f"Evaluated {len(results)} samples")
        
        # Aggregate
        aggregated = self.aggregate_results(results, n_bootstrap=n_bootstrap)
        
        # Add metadata
        aggregated['metadata'] = {
            'node_count': node_count,
            'variant': variant,
            'sample_size': sample_size,
            'n_samples': len(results),
            'data_filename': data_filename,
            'model_name': model_name,
            'n_bootstrap': n_bootstrap,
            'use_ancestor_matrix': metadata.get('use_ancestor_matrix', False),
        }
        
        # Setup output directory structure
        if output_dir is None:
            if model_name is None:
                model_name = getattr(self, '_default_model_name', 'unknown_model')
            output_dir = self.benchmark_dir / "benchmark_results" / model_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        variant_str = f"_{variant}" if variant != "base" else ""
        size_str = f"_ntest_{sample_size}" if sample_size is not None else ""
        aggregated_filename = f"aggregated_{node_count}node{variant_str}{size_str}.json"
        
        aggregated_path = output_dir / aggregated_filename
        with open(aggregated_path, 'w') as f:
            import json
            json.dump(aggregated, f, indent=2)
        
        if self.verbose:
            print(f"Saved aggregated results to: {aggregated_path}")
        
        # Save individual results if requested
        if save_individual_results:
            individual_filename = f"individual_{node_count}node{variant_str}{size_str}.pkl"
            individual_path = output_dir / individual_filename
            with open(individual_path, 'wb') as f:
                pickle.dump(results, f)
            if self.verbose:
                print(f"Saved individual results to: {individual_path}")
        
        return aggregated
    
    def run_full_benchmark(
        self,
        node_counts: Optional[List[int]] = None,
        variants: Optional[List[str]] = None,
        sample_sizes: Optional[List[int]] = None,
        model: Optional[Any] = None,
        output_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        n_bootstrap: int = 1000,
        num_samples: int = 1000,
        base_seed: int = 42,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run full benchmark across all specified configurations.
        
        Args:
            node_counts: Node counts to benchmark (default: all)
            variants: Variants to benchmark (default: all)
            sample_sizes: Sample sizes for path variants (default: all)
            model: Model to use (default: self.model)
            output_dir: Output directory (default: auto)
            model_name: Model name for organization
            n_bootstrap: Bootstrap samples for CI
            num_samples: Samples per config (for data generation)
            base_seed: Base seed for data generation
            
        Returns:
            Nested dict of results by config
        """
        if node_counts is None:
            node_counts = self.NODE_COUNTS.copy()
        if variants is None:
            variants = self.VARIANTS.copy()
        if sample_sizes is None:
            sample_sizes = self.SAMPLE_SIZES.copy()
        
        if self.verbose:
            print("Running ComplexMech full benchmark...")
            print(f"Node counts: {node_counts}")
            print(f"Variants: {variants}")
            print(f"Sample sizes: {sample_sizes}")
        
        # First ensure data exists for all configs
        print("Ensuring all datasets exist...")
        self.sample_all_configs(
            num_samples_per_config=num_samples,
            dataset_seed=base_seed,
            node_counts=node_counts,
            variants=variants,
            sample_sizes=sample_sizes,
            overwrite=False,
        )
        
        # Now run benchmarks
        all_results = {}
        
        for node_count in node_counts:
            for variant in variants:
                if variant == "base":
                    # Base variant
                    filename = f"complexmech_{node_count}node_{variant}_{num_samples}samples.pkl"
                    config_key = f"{node_count}node_base"
                    
                    try:
                        result = self.run_benchmark(
                            data_filename=filename,
                            model=model,
                            output_dir=output_dir,
                            model_name=model_name,
                            n_bootstrap=n_bootstrap,
                        )
                        all_results[config_key] = result
                        
                        if self.verbose:
                            print(f"✓ Completed benchmark: {config_key}")
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"✗ Failed benchmark {config_key}: {e}")
                        continue
                
                else:
                    # Path variants
                    for sample_size in sample_sizes:
                        filename = f"complexmech_{node_count}node_{variant}_ntest_{sample_size}_{num_samples}samples.pkl"
                        config_key = f"{node_count}node_{variant}_ntest_{sample_size}"
                        
                        try:
                            result = self.run_benchmark(
                                data_filename=filename,
                                model=model,
                                output_dir=output_dir,
                                model_name=model_name,
                                n_bootstrap=n_bootstrap,
                            )
                            all_results[config_key] = result
                            
                            if self.verbose:
                                print(f"✓ Completed benchmark: {config_key}")
                                
                        except Exception as e:
                            if self.verbose:
                                print(f"✗ Failed benchmark {config_key}: {e}")
                            continue
        
        if self.verbose:
            print(f"\nCompleted benchmarking on {len(all_results)} configurations!")
        
        return all_results
    
    def run(
        self,
        fidelity: str,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[Tuple[int, str, Optional[int]], Dict[str, Any]]:
        """
        Run ComplexMech benchmark with specified fidelity level on all variants.
        
        This method provides a unified interface compatible with the Trainer's benchmark
        integration pattern. It maps fidelity levels to evaluation configurations and runs
        on ALL variants (base, path_TY, path_YT, path_independent_TY) with all sample sizes.
        
        - "minimal": 3 samples from all node counts (2, 5, 20, 50) and all variants
        - "low": 100 samples from all node counts (2, 5, 20, 50) and all variants
        - "high": 1000 samples from all node counts (2, 5, 20, 50) and all variants
        
        Args:
            fidelity: Fidelity level ("minimal", "low", or "high")
            checkpoint_path: Path to the model checkpoint .pt file
            config_path: Path to the model config YAML file (optional, for model loading)
            output_dir: Directory to save results (default: self.benchmark_dir)
            
        Returns:
            Dictionary mapping (node_count, variant, sample_size) -> aggregated_results
            where sample_size is None for base variant
            
        Raises:
            ValueError: If fidelity is not "minimal", "low", or "high"
        """
        # Normalize fidelity
        fidelity = fidelity.strip().lower()
        
        # Map fidelity to max_samples
        fidelity_map = {
            "minimal": 3,
            "low": 100,
            "high": 1000,
        }
        
        if fidelity not in fidelity_map:
            raise ValueError(
                f"Invalid fidelity '{fidelity}' for ComplexMech benchmark. "
                f"Choose 'minimal' (3 samples), 'low' (100 samples), or 'high' (1000 samples)."
            )
        
        max_samples = fidelity_map[fidelity]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ComplexMech Benchmark - Fidelity: {fidelity.upper()}")
            print(f"  Max samples per dataset: {max_samples}")
            print(f"  Node counts: {self.NODE_COUNTS}")
            print(f"  Variants: {self.VARIANTS}")
            print(f"  Sample sizes: {self.SAMPLE_SIZES}")
            print(f"  Checkpoint: {checkpoint_path}")
            print(f"{'='*80}\n")
        
        # Load model from checkpoint
        if config_path is not None:
            self.load_model(config_path, checkpoint_path)
        else:
            raise RuntimeError(
                "config_path is required for ComplexMech benchmark. "
                "Provide config_path when calling run()."
            )
        
        # Temporarily override max_samples for this run
        original_max_samples = self.max_samples
        self.max_samples = max_samples
        
        try:
            # Run benchmark on all node counts, variants, and sample sizes
            all_results = {}
            
            # Count total configurations for progress reporting
            total_configs = len(self.NODE_COUNTS) * (1 + len([v for v in self.VARIANTS if v != "base"]) * len(self.SAMPLE_SIZES))
            config_idx = 0
            
            for node_count in self.NODE_COUNTS:
                for variant in self.VARIANTS:
                    if variant == "base":
                        # Base variant - no sample size parameter
                        config_idx += 1
                        data_filename = f"complexmech_{node_count}nodes_{variant}_1000samples_seed42.pkl"
                        config_key = (node_count, variant, None)
                        
                        if self.verbose:
                            print(f"\n[{config_idx}/{total_configs}] Running: {node_count} nodes, {variant}")
                            print(f"  Data file: {data_filename}")
                            print(f"  Max samples: {max_samples}")
                        
                        try:
                            result = self.run_benchmark(
                                data_filename=data_filename,
                                model=self.model,
                                output_dir=output_dir,
                                model_name=None,
                                n_bootstrap=1000,
                                save_individual_results=False,
                            )
                            all_results[config_key] = result
                            
                            if self.verbose:
                                print(f"  ✓ MSE={result.get('mse', {}).get('mean', float('nan')):.6f}")
                        
                        except Exception as e:
                            if self.verbose:
                                print(f"  ✗ Failed: {e}")
                            # Store NaN results for failed configs
                            all_results[config_key] = {
                                'mse': {'mean': float('nan'), 'median': float('nan'), 'std': float('nan')},
                                'r2': {'mean': float('nan'), 'median': float('nan'), 'std': float('nan')},
                                'nll': {'mean': float('nan'), 'median': float('nan'), 'std': float('nan')},
                            }
                    
                    else:
                        # Path variants - loop through all sample sizes
                        for sample_size in self.SAMPLE_SIZES:
                            config_idx += 1
                            data_filename = f"complexmech_{node_count}nodes_{variant}_ntest{sample_size}_1000samples_seed42.pkl"
                            config_key = (node_count, variant, sample_size)
                            
                            if self.verbose:
                                print(f"\n[{config_idx}/{total_configs}] Running: {node_count} nodes, {variant}, ntest={sample_size}")
                                print(f"  Data file: {data_filename}")
                                print(f"  Max samples: {max_samples}")
                            
                            try:
                                result = self.run_benchmark(
                                    data_filename=data_filename,
                                    model=self.model,
                                    output_dir=output_dir,
                                    model_name=None,
                                    n_bootstrap=1000,
                                    save_individual_results=False,
                                )
                                all_results[config_key] = result
                                
                                if self.verbose:
                                    print(f"  ✓ MSE={result.get('mse', {}).get('mean', float('nan')):.6f}")
                            
                            except Exception as e:
                                if self.verbose:
                                    print(f"  ✗ Failed: {e}")
                                # Store NaN results for failed configs
                                all_results[config_key] = {
                                    'mse': {'mean': float('nan'), 'median': float('nan'), 'std': float('nan')},
                                    'r2': {'mean': float('nan'), 'median': float('nan'), 'std': float('nan')},
                                    'nll': {'mean': float('nan'), 'median': float('nan'), 'std': float('nan')},
                                }
            
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"ComplexMech Benchmark Complete!")
                print(f"  Evaluated {len(all_results)}/{total_configs} configurations")
                print(f"{'='*80}\n")
            
            return all_results
            
        finally:
            # Restore original max_samples
            self.max_samples = original_max_samples
    
    def quick_benchmark(
        self,
        config_path: str,
        checkpoint_path: str,
        max_samples: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Quick benchmark with automatic model loading and limited samples.
        
        Args:
            config_path: Path to model config
            checkpoint_path: Path to model checkpoint
            max_samples: Max samples per dataset (default: self.max_samples)
            model_kwargs: Additional model parameters
            
        Returns:
            Nested dict of results by config
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        # Temporarily set max_samples if provided
        original_max_samples = self.max_samples
        if max_samples is not None:
            self.max_samples = max_samples
        
        try:
            # Load model
            self.load_model(config_path, checkpoint_path, **model_kwargs)
            
            # Run benchmark on small subset for quick testing
            results = self.run_full_benchmark(
                node_counts=[2, 5],  # Just test small configs
                variants=["base", "path_TY"],  # Just test subset
                sample_sizes=[500],  # Just one sample size
                num_samples=50,  # Small dataset for quick testing
            )
            
            return results
            
        finally:
            # Restore original max_samples
            self.max_samples = original_max_samples


if __name__ == "__main__":
    # Run quick test of the ComplexMech benchmark
    print("ComplexMechBenchmark - Quick Test")
    print("="*80)
    
    MAX_SAMPLES = 10  # Limit for quick testing
    USE_ANCESTOR_MATRIX = False
    
    benchmark = ComplexMechBenchmark(
        verbose=True, 
        max_samples=MAX_SAMPLES,
        use_ancestor_matrix=USE_ANCESTOR_MATRIX,
    )
    
    # Test loading a config
    print("\nTesting config loading...")
    try:
        config = benchmark.load_config(2, "base")
        print(f"  Loaded base config keys: {list(config.keys())}")
        print(f"  Num nodes: {config['scm_config']['num_nodes']}")
        
        config_path = benchmark.load_config(2, "path_TY", sample_size=500)
        print(f"  Loaded path_TY config with ntest_500")
        
    except Exception as e:
        print(f"Config loading failed: {e}")
    
    # Test data sampling
    print("\nTesting data sampling...")
    try:
        saved_path = benchmark.sample_and_save_data(
            node_count=2,
            num_samples=5,
            variant="base",
            overwrite=True,
        )
        print(f"  Saved test data to: {saved_path}")
        
        # Test loading the data
        data, metadata = benchmark.load_data(Path(saved_path).name)
        print(f"  Loaded {len(data)} samples")
        print(f"  Metadata keys: {list(metadata.keys())}")
        
    except Exception as e:
        print(f"Data sampling failed: {e}")
    
    print("\n" + "="*80)
    print("ComplexMech Benchmark ready for use!")
    print("Example usage:")
    print("  benchmark = ComplexMechBenchmark(max_samples=100)")
    print("  results = benchmark.quick_benchmark(")
    print("      config_path='/path/to/config.yaml',")
    print("      checkpoint_path='/path/to/checkpoint.pt',")
    print("  )")
    print("="*80)