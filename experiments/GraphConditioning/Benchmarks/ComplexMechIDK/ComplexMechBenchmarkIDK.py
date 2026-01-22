"""
Complex Mechanism Benchmark for Graph-Conditioned Interventional Models.

This benchmark creates synthetic datasets with complex (non-linear) mechanisms
and varying numbers of nodes to test graph-conditioned causal inference models.
It uses MLP mechanisms with TabICL-style nonlinearities and optional XGBoost mechanisms.
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



class ComplexMechBenchmarkIDK:
    """
    Benchmark class for graph-conditioned interventional models with partial graph knowledge.
    
    This class handles the ComplexMechIDK benchmark with three-state adjacency matrices {-1, 0, 1}
    where 0 represents unknown edges (I Don't Know).
    
    Features:
    1. Loading configuration files from nested directory structure
    2. Support for base configs (uniform hide_fraction) and path variants (fixed hide_fraction)
    3. Sampling synthetic datasets with partial graph knowledge
    4. Storing sampled data to disk for evaluation
    5. Loading stored data for benchmarking
    
    The benchmark uses Complex Mechanism SCMs (MLP with TabICL nonlinearities) with varying 
    complexity (2, 5, 10, 20, 35, 50 nodes).
    
    Directory structure:
        configs/{N}node/base.yaml                           # Uniform(0.0, 1.0)
        configs/{N}node/path_TY/hide_{fraction}.yaml        # Fixed hide fractions
        configs/{N}node/path_YT/hide_{fraction}.yaml
        configs/{N}node/path_independent_TY/hide_{fraction}.yaml
    """
    
    # Node counts available
    NODE_COUNTS = [2, 5, 10, 20, 35, 50]
    
    # Path variants (excluding base)
    PATH_VARIANTS = ["path_TY", "path_YT", "path_independent_TY"]
    
    # All variants including base
    VARIANTS = ["base"] + PATH_VARIANTS
    
    # Hide fractions for path variants
    HIDE_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    def __init__(
        self,
        benchmark_dir: str = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/GraphConditioning/Benchmarks/ComplexMechIDK",
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
            print(f"ComplexMechBenchmark initialized")
            print(f"  Benchmark dir: {self.benchmark_dir}")
            print(f"  Cache dir: {self.cache_dir}")
            if max_samples is not None:
                print(f"  Max samples per dataset: {max_samples}")
            print(f"  Use ancestor matrix: {use_ancestor_matrix}")
    
    def load_config(
        self, 
        node_count: int, 
        variant: str = "base",
        hide_fraction: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Load a configuration file for a specific node count and variant.
        
        Args:
            node_count: Number of nodes (2, 5, 10, 20, 35, 50)
            variant: Config variant ("base", "path_TY", "path_YT", "path_independent_TY")
            hide_fraction: For path variants, which hide fraction to load (0.0, 0.25, 0.5, 0.75, 1.0)
                          Not needed for "base" variant (uses uniform distribution)
            
        Returns:
            Dictionary containing scm_config, dataset_config, and preprocessing_config
            
        Examples:
            >>> # Load base config with uniform hide_fraction
            >>> config = benchmark.load_config(5, "base")
            
            >>> # Load path variant with specific hide fraction
            >>> config = benchmark.load_config(5, "path_TY", hide_fraction=0.5)
        """
        if node_count not in self.NODE_COUNTS:
            raise ValueError(
                f"No config for {node_count} nodes. Available: {self.NODE_COUNTS}"
            )
        
        if variant not in self.VARIANTS:
            raise ValueError(
                f"Invalid variant '{variant}'. Available: {self.VARIANTS}"
            )
        
        # For path variants, hide_fraction is required
        if variant != "base" and hide_fraction is None:
            raise ValueError(
                f"hide_fraction must be specified for variant '{variant}'. "
                f"Available: {self.HIDE_FRACTIONS}"
            )
        
        # For base variant, hide_fraction should not be specified
        if variant == "base" and hide_fraction is not None:
            raise ValueError(
                f"hide_fraction should not be specified for 'base' variant (uses uniform distribution)"
            )
        
        # Validate hide_fraction if provided
        if hide_fraction is not None and hide_fraction not in self.HIDE_FRACTIONS:
            raise ValueError(
                f"Invalid hide_fraction {hide_fraction}. Available: {self.HIDE_FRACTIONS}"
            )
        
        # Create config key for caching
        if variant == "base":
            config_key = f"{node_count}_base"
        else:
            config_key = f"{node_count}_{variant}_hide{hide_fraction}"
        
        # Check if already loaded
        if config_key in self.configs:
            return self.configs[config_key]
        
        # Construct config path based on structure
        configs_dir = self.benchmark_dir / "configs"
        node_dir = configs_dir / f"{node_count}node"
        
        if variant == "base":
            config_path = node_dir / "base.yaml"
        else:
            variant_dir = node_dir / variant
            config_path = variant_dir / f"hide_{hide_fraction}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store for later reuse
        self.configs[config_key] = config
        
        if self.verbose:
            if variant == "base":
                print(f"Loaded config: {node_count}node/base.yaml")
            else:
                print(f"Loaded config: {node_count}node/{variant}/hide_{hide_fraction}.yaml")
        
        return config
    
    def sample_and_save_data(
        self,
        node_count: int,
        num_samples: int,
        variant: str = "base",
        dataset_seed: Optional[int] = None,
        filename: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Sample data from InterventionalDataset and save to disk.
        
        Args:
            node_count: Number of nodes (2, 5, 10, 20, 35, 50)
            num_samples: Number of dataset samples to generate
            variant: Config variant ("base", "path_TY", "path_YT", "path_independent_TY")
            dataset_seed: Random seed for dataset (default: None)
            filename: Custom filename for saved data (default: auto-generated)
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path to saved file
        """
        # Load configuration
        config = self.load_config(node_count, variant=variant)
        
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
            seed_str = f"_seed{dataset_seed}" if dataset_seed is not None else ""
            variant_str = f"_{variant}" if variant != "base" else "_base"
            filename = f"complexmech_{node_count}nodes{variant_str}_{num_samples}samples{seed_str}.pkl"
        
        save_path = self.cache_dir / filename
        
        # Check if file exists
        if save_path.exists() and not overwrite:
            if self.verbose:
                print(f"File already exists: {save_path}")
                print(f"  Use overwrite=True to regenerate")
            return str(save_path)
        
        if self.verbose:
            print(f"\nSampling data for {node_count} nodes (variant: {variant})...")
            print(f"  Num samples: {num_samples}")
            print(f"  Seed: {dataset_seed}")
            print(f"  Use ancestor matrix: {self.use_ancestor_matrix}")
        
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
                
                # Get adjacency matrix
                adjacency_matrix = data_item[6]
                
                # NOTE: We always store the adjacency matrix in the data cache.
                # If use_ancestor_matrix is True, the ancestor matrix will be computed
                # on-the-fly during evaluation (in evaluate_sample and _evaluate_dataset_batched).
                # This avoids storing duplicate data and ensures flexibility.
                
                # Store everything except potentially large SCM object (optional)
                # You can adjust what to save based on your needs
                sampled_data.append({
                    'X_obs': data_item[0],
                    'T_obs': data_item[1],
                    'Y_obs': data_item[2],
                    'X_intv': data_item[3],
                    'T_intv': data_item[4],
                    'Y_intv': data_item[5],
                    'adjacency_matrix': adjacency_matrix,  # Always store adjacency matrix
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
            'variant': variant,
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
        variants: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Dict[Tuple[int, str], str]:
        """
        Sample data for all (or specified) node configurations and variants.
        
        The graph matrices in the sampled data will be either adjacency matrices
        or ancestor matrices depending on the use_ancestor_matrix flag set during
        initialization.
        
        Args:
            num_samples_per_config: Number of samples to generate per config
            dataset_seed: Base seed for datasets (each config gets seed + offset)
            node_counts: List of node counts to sample (default: all available)
            variants: List of variants to sample (default: all available)
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary mapping (node_count, variant) -> saved file path
        """
        if node_counts is None:
            node_counts = [2, 5, 10, 20, 35, 50]
        
        if variants is None:
            variants = self.VARIANTS
        
        saved_paths = {}
        
        config_idx = 0
        total_configs = len(node_counts) * len(variants)
        
        for node_count in node_counts:
            for variant in variants:
                config_idx += 1
                # Use different seed for each config if base seed provided
                seed = dataset_seed + config_idx * 1000 if dataset_seed is not None else None
                
                if self.verbose:
                    print(f"\n{'='*80}")
                    print(f"Processing config {config_idx}/{total_configs}: {node_count} nodes, variant: {variant}")
                    print(f"{'='*80}")
                
                try:
                    path = self.sample_and_save_data(
                        node_count=node_count,
                        num_samples=num_samples_per_config,
                        variant=variant,
                        dataset_seed=seed,
                        overwrite=overwrite,
                    )
                    saved_paths[(node_count, variant)] = path
                except Exception as e:
                    if self.verbose:
                        print(f"ERROR: Failed to sample {node_count}-node {variant} config: {e}")
                    continue
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Completed sampling for {len(saved_paths)}/{total_configs} configs")
            print(f"{'='*80}")
        
        return saved_paths
    
    def load_data(self, filename: str) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Load sampled data from disk.
        
        Handles both dictionary format (from sample_and_save_data) and tuple format
        (from generate_all_variants_data.py).
        
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
        metadata = save_data.get('metadata', {})
        
        # Handle both tuple and dict formats
        # Tuple format: (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv)
        # Dict format: {'X_obs': ..., 'T_obs': ..., etc.}
        if len(data) > 0 and isinstance(data[0], tuple):
            if self.verbose:
                print(f"  Converting tuple format to dict format...")
            # Convert tuples to dicts
            converted_data = []
            for item in data:
                if len(item) == 6:
                    # Standard format without adjacency matrix
                    converted_data.append({
                        'X_obs': item[0],
                        'T_obs': item[1],
                        'Y_obs': item[2],
                        'X_intv': item[3],
                        'T_intv': item[4],
                        'Y_intv': item[5],
                        'adjacency_matrix': None,  # Will be loaded from metadata if available
                    })
                elif len(item) >= 7:
                    # Format with adjacency matrix
                    converted_data.append({
                        'X_obs': item[0],
                        'T_obs': item[1],
                        'Y_obs': item[2],
                        'X_intv': item[3],
                        'T_intv': item[4],
                        'Y_intv': item[5],
                        'adjacency_matrix': item[6],
                    })
            data = converted_data
        
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
        
        # Extract batch size from config (for batched inference)
        # Helper to get value from wandb-style or flat config
        def _get_cfg_value(cfg_dict, key, default=None):
            if key in cfg_dict:
                val = cfg_dict[key]
                return val['value'] if isinstance(val, dict) and 'value' in val else val
            return default
        
        # Try to get batch_size from training_config first, then model_config
        training_batch_size = _get_cfg_value(config.get('training_config', {}), 'batch_size', None)
        if training_batch_size is None:
            training_batch_size = _get_cfg_value(config.get('model_config', {}), 'batch_size', None)
        
        # Use half the training batch size for evaluation to reduce memory pressure
        if training_batch_size is not None:
            self.batch_size = max(1, training_batch_size // 2)
        else:
            self.batch_size = None
        
        if verbose:
            print(f"  Model type: {'Graph-Conditioned' if use_graph_conditioning else 'Standard'} InterventionalPFN")
            if training_batch_size:
                print(f"  Training batch size from config: {training_batch_size}")
                print(f"  Evaluation batch size (half): {self.batch_size}")
        
        
        # Create and load the appropriate model
        if use_graph_conditioning:
            model = GraphConditionedInterventionalPFNSklearn(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                verbose=verbose,
                **model_kwargs,
            )
        else:
            # InterventionalPFNSklearn (batched version) only accepts:
            # config_path, checkpoint_path, device, verbose
            # Filter model_kwargs to only include these supported parameters
            supported_kwargs = {k: v for k, v in model_kwargs.items() if k in ['device']}
            model = InterventionalPFNSklearn(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                verbose=verbose,
                **supported_kwargs,
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
        
        NOTE: The stored data always contains adjacency matrices. If use_ancestor_matrix
        flag is True, this method will compute the ancestor matrix on-the-fly from the
        adjacency matrix before passing it to the model.
        
        Args:
            data_item: Dictionary containing:
                - 'X_obs': Observational covariates (tensor)
                - 'T_obs': Observational treatment (tensor)
                - 'Y_obs': Observational outcome (tensor)
                - 'X_intv': Interventional covariates (tensor)
                - 'T_intv': Interventional treatment (tensor)
                - 'Y_intv': Interventional outcome (tensor)
                - 'adjacency_matrix': Graph adjacency matrix (tensor)
                  NOTE: Always stores adjacency matrix; ancestor matrix computed on-the-fly if needed
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
        
        # Convert to numpy if needed (but keep tensor for ancestor matrix computation if needed)
        adj_tensor = adj if torch.is_tensor(adj) else torch.tensor(adj)
        
        # Compute ancestor matrix if flag is set
        
        if torch.is_tensor(adj):
            adj = adj.numpy()
        
        # Convert other tensors to numpy
        if torch.is_tensor(X_obs):
            X_obs = X_obs.numpy()
            T_obs = T_obs.numpy()
            Y_obs = Y_obs.numpy()
            X_intv = X_intv.numpy()
            T_intv = T_intv.numpy()
            Y_intv = Y_intv.numpy()
        
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
        Evaluate model on a full dataset with optional batched inference.
        
        Uses batched inference if self.batch_size is set, otherwise processes one sample at a time.
        
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
                print(f"  Limiting evaluation to first {self.max_samples} samples")
        
        # Check if model supports batched inference
        use_graph = hasattr(model, 'predict') and 'adjacency_matrix' in model.predict.__code__.co_varnames
        has_batched_param = hasattr(model, 'predict') and 'batched' in model.predict.__code__.co_varnames
        
        # Use batched inference if batch_size is set and model supports it
        if self.batch_size is not None and self.batch_size > 1 and has_batched_param:
            return self._evaluate_dataset_batched(data, model, use_graph, verbose)
        else:
            # Fall back to single-sample evaluation
            return self._evaluate_dataset_sequential(data, model, verbose)
    
    def _evaluate_dataset_sequential(
        self,
        data: List[Dict[str, Any]],
        model: Any,
        verbose: bool,
    ) -> List[Dict[str, float]]:
        """Evaluate dataset one sample at a time (original behavior)."""
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
    
    def _evaluate_dataset_batched(
        self,
        data: List[Dict[str, Any]],
        model: Any,
        use_graph: bool,
        verbose: bool,
    ) -> List[Dict[str, float]]:
        """
        Evaluate dataset using batched inference for efficiency.
        
        Groups samples into batches and processes them together.
        """
        results = []
        n_samples = len(data)
        
        # Get expected number of features from model config
        if hasattr(self, '_current_config') and self._current_config:
            expected_features = self._current_config.get('model_config', {}).get('num_features', {}).get('value', 3)
        else:
            expected_features = 3
        
        num_nodes = expected_features + 2
        
        # Process in batches
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        iterator = tqdm(range(n_batches), desc=f"Evaluating (batch_size={self.batch_size})") if verbose else range(n_batches)
        
        for batch_idx in iterator:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_data = data[start_idx:end_idx]
            
            try:
                # Prepare batched arrays
                batch_X_obs = []
                batch_T_obs = []
                batch_Y_obs = []
                batch_X_intv = []
                batch_T_intv = []
                batch_Y_intv = []
                batch_adj = []
                
                for data_item in batch_data:
                    X_obs = data_item['X_obs']
                    T_obs = data_item['T_obs']
                    Y_obs = data_item['Y_obs']
                    X_intv = data_item['X_intv']
                    T_intv = data_item['T_intv']
                    Y_intv = data_item['Y_intv']
                    adj = data_item['adjacency_matrix']
                    
                    # Keep tensor version for ancestor matrix computation if needed
                    adj_tensor = adj if torch.is_tensor(adj) else torch.tensor(adj)
                    
                    # Compute ancestor matrix if flag is set
                    # IMPORTANT: Stored data contains adjacency matrices, compute ancestor on-the-fly
       
                    if torch.is_tensor(adj):
                        adj = adj.numpy()
                    
                    # Convert other tensors to numpy
                    if torch.is_tensor(X_obs):
                        X_obs = X_obs.numpy()
                        T_obs = T_obs.numpy()
                        Y_obs = Y_obs.numpy()
                        X_intv = X_intv.numpy()
                        T_intv = T_intv.numpy()
                        Y_intv = Y_intv.numpy()
                    
                    # Trim features to match expected dimensions
                    if X_obs.shape[1] > expected_features:
                        X_obs = X_obs[:, :expected_features]
                        X_intv = X_intv[:, :expected_features]
                    
                    # Trim adjacency matrix
                    if adj.shape[0] > num_nodes:
                        adj = adj[:num_nodes, :num_nodes]
                    
                    # Squeeze Y arrays if they have an extra dimension
                    if Y_obs.ndim == 2 and Y_obs.shape[1] == 1:
                        Y_obs = Y_obs.squeeze(-1)
                    if Y_intv.ndim == 2 and Y_intv.shape[1] == 1:
                        Y_intv = Y_intv.squeeze(-1)
                    
                    batch_X_obs.append(X_obs)
                    batch_T_obs.append(T_obs)
                    batch_Y_obs.append(Y_obs)
                    batch_X_intv.append(X_intv)
                    batch_T_intv.append(T_intv)
                    batch_Y_intv.append(Y_intv)
                    if use_graph:
                        batch_adj.append(adj)
                
                # Stack into batched arrays
                batch_X_obs = np.stack(batch_X_obs, axis=0)  # (B, N, L)
                batch_T_obs = np.stack(batch_T_obs, axis=0)  # (B, N, 1)
                batch_Y_obs = np.stack(batch_Y_obs, axis=0)  # (B, N)
                batch_X_intv = np.stack(batch_X_intv, axis=0)  # (B, M, L)
                batch_T_intv = np.stack(batch_T_intv, axis=0)  # (B, M, 1)
                batch_Y_intv = np.stack(batch_Y_intv, axis=0)  # (B, M)
                if use_graph:
                    batch_adj = np.stack(batch_adj, axis=0)  # (B, num_nodes, num_nodes)
                
                # Get batched predictions
                if use_graph:
                    preds_batch = model.predict(
                        X_obs=batch_X_obs,
                        T_obs=batch_T_obs,
                        Y_obs=batch_Y_obs,
                        X_intv=batch_X_intv,
                        T_intv=batch_T_intv,
                        adjacency_matrix=batch_adj,
                        batched=True,
                    )
                    log_likelihood_batch = model.predict_log_likelihood(
                        X_obs=batch_X_obs,
                        T_obs=batch_T_obs,
                        Y_obs=batch_Y_obs,
                        X_intv=batch_X_intv,
                        T_intv=batch_T_intv,
                        Y_intv=batch_Y_intv,
                        adjacency_matrix=batch_adj,
                        batched=True,
                    )
                else:
                    preds_batch = model.predict(
                        X_obs=batch_X_obs,
                        T_obs=batch_T_obs,
                        Y_obs=batch_Y_obs,
                        X_intv=batch_X_intv,
                        T_intv=batch_T_intv,
                        batched=True,
                    )
                    log_likelihood_batch = model.predict_log_likelihood(
                        X_obs=batch_X_obs,
                        T_obs=batch_T_obs,
                        Y_obs=batch_Y_obs,
                        X_intv=batch_X_intv,
                        T_intv=batch_T_intv,
                        Y_intv=batch_Y_intv,
                        batched=True,
                    )
                
                # Process each sample in the batch
                for i in range(len(batch_data)):
                    pred_i = preds_batch[i]  # (M,)
                    Y_intv_i = batch_Y_intv[i]  # (M,)
                    log_likelihood_i = log_likelihood_batch[i]  # (M,)
                    
                    # Compute metrics
                    mse = mean_squared_error(Y_intv_i, pred_i)
                    r2 = r2_score(Y_intv_i, pred_i)
                    nll = -np.mean(log_likelihood_i)
                    
                    results.append({
                        'mse': float(mse),
                        'r2': float(r2),
                        'nll': float(nll),
                    })
                    
            except Exception as e:
                if verbose:
                    print(f"\nWarning: Failed to evaluate batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                # Fall back to sequential evaluation for this batch
                for data_item in batch_data:
                    try:
                        metrics = self.evaluate_sample(data_item, model=model)
                        results.append(metrics)
                    except Exception as e2:
                        if verbose:
                            print(f"\nWarning: Failed to evaluate sample: {e2}")
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
        
        # Try to get variant from metadata, otherwise parse from filename or config_file
        variant = metadata.get('variant', None)
        if variant is None:
            # Try to parse from config_file in metadata
            config_file = metadata.get('config_file', '')
            if 'path_TY' in config_file or 'path_TY' in data_filename:
                variant = 'path_TY'
            elif 'path_YT' in config_file or 'path_YT' in data_filename:
                variant = 'path_YT'
            elif 'path_independent_TY' in config_file or 'path_independent_TY' in data_filename:
                variant = 'path_independent_TY'
            else:
                variant = 'base'
        
        if self.verbose:
            print(f"\nRunning benchmark on {node_count}-node dataset (variant: {variant})...")
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
            'variant': variant,
            'n_samples': len(results),
            'data_filename': data_filename,
            'model_name': model_name,
            'n_bootstrap': n_bootstrap,
            'use_ancestor_matrix': metadata.get('use_ancestor_matrix', False),
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
        
        # Generate filenames including variant information
        variant_str = f"_{variant}" if variant != "base" else ""
        aggregated_filename = f"aggregated_{node_count}nodes{variant_str}.json"
        
        aggregated_path = output_dir / aggregated_filename
        with open(aggregated_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        if self.verbose:
            print(f"  Saved aggregated results to: {aggregated_path}")
        
        # Save individual results if requested
        if save_individual_results:
            individual_filename = f"individual_{node_count}nodes{variant_str}.json"
            individual_path = output_dir / individual_filename
            
            with open(individual_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            if self.verbose:
                print(f"  Saved individual results to: {individual_path}")
        
        return aggregated
    
    def run_full_benchmark(
        self,
        node_counts: Optional[List[int]] = None,
        variants: Optional[List[str]] = None,
        model: Optional[Any] = None,
        output_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
        n_bootstrap: int = 1000,
        num_samples: int = 1000,
        base_seed: int = 42,
    ) -> Dict[Tuple[int, str], Dict[str, Any]]:
        """
        Run benchmark on all node count configurations and variants.
        
        Args:
            node_counts: List of node counts to evaluate (default: all available)
            variants: List of variants to evaluate (default: all available)
            model: Model to evaluate (default: use self.model)
            output_dir: Directory to save results (default: benchmark_dir/benchmark_res/model_name)
            model_name: Name for the model (for folder organization)
            config_path: Path to the model config file to copy to results folder
            n_bootstrap: Number of bootstrap samples
            num_samples: Number of samples in each dataset
            base_seed: Base seed used for dataset generation
            
        Returns:
            Dictionary mapping (node_count, variant) -> aggregated results
        """
        if node_counts is None:
            node_counts = [2, 5, 10, 20, 35, 50]
        
        if variants is None:
            variants = self.VARIANTS
        
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
        
        # Map node counts and variants to their seeds 
        # NOTE: All variants use the SAME seed (base_seed) per the generate_all_variants_data.py script
        all_results = {}
        
        # Calculate total configs: base variants + (path variants * hide fractions)
        config_idx = 0
        base_count = len([v for v in variants if v == "base"])
        path_count = len([v for v in variants if v != "base"])
        total_configs = len(node_counts) * (base_count + path_count * len(self.HIDE_FRACTIONS))
        
        for node_count in node_counts:
            for variant in variants:
                # For path variants, loop through all hide fractions
                if variant in self.PATH_VARIANTS:
                    hide_fractions_to_test = self.HIDE_FRACTIONS
                else:
                    # For base variant, no hide fraction suffix needed
                    hide_fractions_to_test = [None]
                
                for hide_frac in hide_fractions_to_test:
                    config_idx += 1
                    # Generate expected filename with correct seed
                    # All files use base_seed (typically 42) regardless of variant
                    dataset_seed = base_seed
                    
                    # Build variant string to match actual data filenames
                    if hide_frac is not None:
                        # Path variant with hide fraction
                        variant_str = f"_{variant}_hide{hide_frac}"
                        variant_display = f"{variant} (hide={hide_frac})"
                    else:
                        # Base variant
                        variant_str = "_base"
                        variant_display = variant
                    
                    data_filename = f"complexmech_{node_count}nodes{variant_str}_{num_samples}samples_seed{dataset_seed}.pkl"
                    
                    if self.verbose:
                        print(f"\n{'='*80}")
                        print(f"Benchmarking config {config_idx}/{total_configs}: {node_count}-node, variant: {variant_display}")
                        print(f"{'='*80}")
                    
                    try:
                        results = self.run_benchmark(
                            data_filename=data_filename,
                            model=model,
                            output_dir=output_dir,
                            model_name=model_name,
                            n_bootstrap=n_bootstrap,
                        )
                        # Use string key for JSON serialization
                        if hide_frac is not None:
                            key = f"{node_count}nodes_{variant}_hide{hide_frac}"
                        else:
                            key = f"{node_count}nodes_{variant}"
                        all_results[key] = results
                    except Exception as e:
                        if self.verbose:
                            print(f"ERROR: Failed to benchmark {node_count}-node {variant_display} config: {e}")
                        continue
        
        # Save summary in the model's folder
        summary_filename = "summary_all_nodes.json"
        summary_path = output_dir / summary_filename
        
        # Convert tuple keys to strings for JSON serialization
        # JSON doesn't support tuple keys, so convert (node_count, variant) -> "node_count_variant"
        all_results_json = {}
        for key, results in all_results.items():
            if isinstance(key, tuple) and len(key) == 2:
                node_count, variant = key
                all_results_json[f"{node_count}_{variant}"] = results
            else:
                # Handle unexpected key format gracefully
                all_results_json[str(key)] = results
        
        with open(summary_path, 'w') as f:
            json.dump(all_results_json, f, indent=2)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Benchmark complete!")
            print(f"  Evaluated {len(all_results)}/{total_configs} configurations")
            print(f"  Results folder: {output_dir}")
            print(f"  Summary saved to: {summary_path}")
            print(f"{'='*80}")
        
        return all_results
    
    def run(
        self,
        fidelity: str,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[Tuple[int, str], Dict[str, Any]]:
        """
        Run ComplexMech benchmark with specified fidelity level on all variants.
        
        This method provides a unified interface compatible with the Trainer's benchmark
        integration pattern. It maps fidelity levels to evaluation configurations:
        
        - "minimal": 3 samples from all node counts (2, 5, 10, 20, 35, 50) and all variants
        - "low": 100 samples from all node counts (2, 5, 10, 20, 35, 50) and all variants
        - "high": 1000 samples from all node counts (2, 5, 10, 20, 35, 50) and all variants
        
        Args:
            fidelity: Fidelity level ("minimal", "low", or "high")
            checkpoint_path: Path to the model checkpoint .pt file
            config_path: Path to the model config YAML file (uses self._current_config_path if None)
            output_dir: Directory to save results (default: benchmark_dir/benchmark_res/model_name)
            
        Returns:
            Dictionary mapping (node_count, variant) -> aggregated_results
            
        Raises:
            ValueError: If fidelity is not "minimal", "low", or "high"
            RuntimeError: If no model is loaded and config_path is not provided
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
            print(f"  Node counts: [2, 5, 10, 20, 35, 50]")
            print(f"  Variants: {self.VARIANTS}")
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
        
        # Always reload the model with the provided checkpoint
        # This ensures we use the latest checkpoint for each benchmark run
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
                output_dir=output_dir,
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
    ) -> Dict[Tuple[int, str], Dict[str, Any]]:
        """
        Convenience function to quickly run a full benchmark on a model across all variants.
        
        This is a simplified interface that runs the benchmark with sensible defaults:
        - All node counts: [2, 5, 10, 20, 35, 50]
        - All variants: ["base", "path_TY", "path_YT", "path_independent_TY"]
        - Bootstrap samples: 1000
        - Automatic folder naming from config path
        - Automatic model config copying
        
        Args:
            config_path: Path to the model config YAML file
            checkpoint_path: Path to the model checkpoint .pt file
            max_samples: Maximum number of samples to evaluate per dataset (default: None = all samples)
            model_kwargs: Optional additional kwargs for model constructor (e.g., {'n_estimators': 1})
            
        Returns:
            Dictionary mapping (node_count, variant) -> aggregated_results
            
        Example:
            >>> benchmark = ComplexMechBenchmarkIDK(max_samples=100)
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
    print("ComplexMechBenchmarkIDK - Full Benchmark")
    print("="*80)
    
    MAX_SAMPLES = 25  # Limit to 3 samples per dataset for quick testing
    USE_ANCESTOR_MATRIX = False  # Set to True to use ancestor matrix instead of adjacency
    benchmark = ComplexMechBenchmarkIDK(
        verbose=True, 
        max_samples=MAX_SAMPLES,
        use_ancestor_matrix=USE_ANCESTOR_MATRIX,
    )
    
    # Test loading a config
    print("\nTesting config loading...")
    config = benchmark.load_config(5)
    print(f"  Loaded config keys: {list(config.keys())}")
    print(f"  Num nodes: {config['scm_config']['num_nodes']}")
    
    # Load and run benchmark on graph-conditioned model (2-node case only)
    print("\nRunning benchmark on graph-conditioned model...")
    print("  Node configurations: 2")
    print(f"  Max samples per dataset: {MAX_SAMPLES}")
    print("="*80)
    
    config_path_gc = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/lingaus_50node_benchmarked_baseline_16694707.0/best_model_config.yaml"
    checkpoint_path_gc = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/lingaus_50node_benchmarked_baseline_16694707.0/best_model.pt"
    
    try:
        # Load model first
        benchmark.load_model(
            config_path=config_path_gc,
            checkpoint_path=checkpoint_path_gc,
        )
        
        # Run benchmark only on 2-node case
        all_results = benchmark.run_full_benchmark(
            node_counts=[2],
            model=benchmark.model,
            config_path=config_path_gc,
            n_bootstrap=1000,
            num_samples=1000,
            base_seed=42,
        )
        
        twonode_mse = all_results[2]['mse']['mean']

        print(f"\nMSE Results:")
        print(f"  2 nodes: {twonode_mse:.4f}")

        
        print(f"\n{'='*80}")
        print("Benchmark Results Summary")
        print(f"{'='*80}")
        print(f"\n2 nodes:")
        result = all_results[2]
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
    print("  # Using adjacency matrix (default)")
    print("  benchmark = ComplexMechBenchmarkIDK(max_samples=100, use_ancestor_matrix=False)")
    print("  results = benchmark.quick_benchmark(")
    print("      config_path='/path/to/config.yaml',")
    print("      checkpoint_path='/path/to/checkpoint.pt',")
    print("  )")
    print("\n  # Using ancestor matrix instead")
    print("  benchmark_anc = ComplexMechBenchmarkIDK(max_samples=100, use_ancestor_matrix=True)")
    print("  results_anc = benchmark_anc.quick_benchmark(")
    print("      config_path='/path/to/config.yaml',")
    print("      checkpoint_path='/path/to/checkpoint.pt',")
    print("  )")
    print("\nTo run benchmark on multiple models:")
    print("  python run_benchmark.py")
    print("="*80)
