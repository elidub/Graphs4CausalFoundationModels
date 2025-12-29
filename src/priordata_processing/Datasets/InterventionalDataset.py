from __future__ import annotations
from typing import Dict, Any, Optional, Union
from torch.utils.data import Dataset
import torch
import torch.distributions as dist
import sys
import os
from copy import deepcopy

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from priors.causal_prior.noise_distributions.ResamplingDist import ResamplingDist
from priors.causal_prior.noise_distributions.RescaledResamplingDist import RescaledResamplingDist
from priors.causal_prior.noise_distributions.UniformResamplingDist import UniformResamplingDist
from priors.causal_prior.noise_distributions.ScaledUniformResamplingDist import ScaledUniformResamplingDist

from priors.causal_prior.scm.SCMSampler import SCMSampler
from priordata_processing.BasicProcessing import BasicProcessing
from utils import FixedSampler, TorchDistributionSampler, CategoricalSampler, DiscreteUniformSampler

# Import ancestor matrix computation function
try:
    from utils.graph_utils import adjacency_to_ancestor_matrix
except ImportError:
    # Try alternative import path
    try:
        from src.utils.graph_utils import adjacency_to_ancestor_matrix
    except ImportError:
        # Final fallback
        import sys
        from pathlib import Path
        utils_path = Path(__file__).resolve().parents[2] / "utils"
        if str(utils_path) not in sys.path:
            sys.path.insert(0, str(utils_path))
        from graph_utils import adjacency_to_ancestor_matrix


class InterventionalDataset(Dataset):
    """
    Dataset for interventional causal data with optional adjacency or ancestor matrix output.
    
    This class directly accepts SCM, preprocessing, and dataset configurations,
    internally creating SCMSampler and sampling hyperparameters on-the-fly.
    Supports returning the causal graph's adjacency matrix or ancestor matrix alongside the data.
    
    Parameters
    ----------
    scm_config : Dict[str, Any]
        Configuration for SCM hyperparameters (same format as SCMSampler).
        Supports all SCMSampler parameters including:
        - num_nodes, graph_edge_prob, graph_seed
        - mechanism parameters (mlp_*, xgb_*)
        - noise distribution parameters (exo_std_*, endo_std_*)
        - endo_p_zero: probability of endogenous noise being exactly zero
    preprocessing_config : Dict[str, Any]
        Configuration for data preprocessing hyperparameters
    dataset_config : Dict[str, Any]
        Configuration for dataset parameters (size, max samples, etc.)
        Special keys:
        - return_adjacency_matrix (bool): If True, include adjacency matrix in output
        - return_ancestor_matrix (bool): If True, include ancestor matrix in output
        Note: Only one of return_adjacency_matrix or return_ancestor_matrix can be True
    seed : Optional[int], default None
        Random seed for reproducibility
    
    Returns
    -------
    When return_adjacency_matrix=False and return_ancestor_matrix=False (default):
        (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv) or
        (X_obs, Y_obs, X_intv, Y_intv) depending on treatment variable inclusion
        
    When return_adjacency_matrix=True:
        Same as above plus adjacency matrix as the last element:
        (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adj_matrix) or
        (X_obs, Y_obs, X_intv, Y_intv, adj_matrix)
        
    When return_ancestor_matrix=True:
        Same as above plus ancestor matrix as the last element:
        (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, anc_matrix) or
        (X_obs, Y_obs, X_intv, Y_intv, anc_matrix)
        
        The adjacency matrix has a specific node ordering that aligns with the data:
        - Position 0: Treatment variable T (the intervened node)
        - Position 1: Outcome variable Y (the target feature)
        - Positions 2 to L+1: Feature variables corresponding to X[:,0], X[:,1], ..., X[:,L-1]
        
        This ordering matches the model's internal representation: [T, Y, X_0, ..., X_{L-1}]
        
        CRITICAL: The features at positions 2 to L+1 in the adjacency matrix correspond 
        to the REAL (non-padded) columns of X in the EXACT SAME ORDER. That is:
        - adj_matrix[0, :] and adj_matrix[:, 0] refer to treatment T
        - adj_matrix[1, :] and adj_matrix[:, 1] refer to outcome Y
        - adj_matrix[2, :] and adj_matrix[:, 2] refer to the same variable as X[:, 0]
        - adj_matrix[3, :] and adj_matrix[:, 3] refer to the same variable as X[:, 1]
        - adj_matrix[k+2, :] and adj_matrix[:, k+2] refer to X[:, k] for k < L
        
        The feature ordering in the adjacency matrix MATCHES the column ordering in X.
        Features are NOT necessarily in sorted order - they are in whatever order the
        BasicProcessing class outputs them (after dropout, but before padding).
        
        IMPORTANT: X may contain zero-padded columns to reach max_n_features.
        The adjacency matrix ONLY covers real nodes from the SCM, not padded columns!
        If adjacency matrix is (N x N) and X has M columns:
        - If M == N-2: No padding, perfect correspondence
        - If M > N-2: X contains (M - N + 2) padded columns at the end
          - X[:, :N-2] are real features (correspond to adjacency positions 2:N)
          - X[:, N-2:] are zero-padded columns (no adjacency correspondence)
        
        Example with padding:
          - SCM has 32 nodes: 1 treatment + 1 outcome + 30 others
          - After dropout: 25 features kept
          - Adjacency matrix: 27x27 (treatment + outcome + 25 features)
          - X is padded to max_n_features=50, so X.shape = (N, 50)
          - Adjacency position 0 = treatment T
          - Adjacency position 1 = outcome Y
          - X[:, :25] = real features (correspond to adjacency positions 2-26)
          - X[:, 25:] = zero padding (no adjacency correspondence)
        
        The ordering of features in X follows sorted(scm.dag.nodes()) after excluding
        the treatment and outcome nodes. Feature dropout may reduce the number of 
        features, but the ordering of kept features is preserved.
        
        Matrix entry adj_matrix[i,j] = 1.0 indicates a directed edge from the 
        variable at position i to the variable at position j.
        
        Examples:
        - adj_matrix[0, 1] = 1.0 means treatment T causes outcome Y (direct edge)
        - adj_matrix[2, 1] = 1.0 means feature X[:,0] causes outcome Y (direct edge)
        - adj_matrix[0, 2] = 1.0 means treatment T causes feature X[:,0] (direct edge)
        - adj_matrix[2, 3] = 1.0 means feature X[:,0] causes feature X[:,1] (direct edge)
        
        The ancestor matrix follows the same node ordering as the adjacency matrix,
        but represents the transitive closure of the adjacency matrix.
        Matrix entry anc_matrix[i,j] = 1.0 indicates that variable i is an ancestor
        of variable j (there exists a directed path from i to j, not necessarily direct).
        
        Examples:
        - anc_matrix[0, 1] = 1.0 means treatment T is an ancestor of outcome Y
        - anc_matrix[2, 1] = 1.0 means feature X[:,0] is an ancestor of outcome Y
        - If there's a path T -> X[:,0] -> Y, then anc_matrix[0, 1] = 1.0 even if
          there's no direct edge from T to Y (i.e., adj_matrix[0, 1] = 0.0)
        
    Examples
    --------
    >>> scm_config = {
    ...     "num_nodes": {"value": 5},
    ...     "endo_p_zero": {"value": 0.3},  # 30% of endogenous noise is zero
    ...     # ... other parameters
    ... }
    >>> # Example 1: Return adjacency matrix
    >>> dataset_config = {
    ...     "dataset_size": {"value": 100},
    ...     "return_adjacency_matrix": {"value": True},  # Enable adjacency matrix output
    ...     # ... other parameters
    ... }
    >>> dataset = InterventionalDataset(scm_config, preprocessing_config, dataset_config)
    >>> X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adj = dataset[0]
    >>> print(adj.shape)  # torch.Size([num_nodes, num_nodes])
    >>> # adj[i, j] = 1 means direct edge from i to j
    >>> 
    >>> # Example 2: Return ancestor matrix
    >>> dataset_config = {
    ...     "dataset_size": {"value": 100},
    ...     "return_ancestor_matrix": {"value": True},  # Enable ancestor matrix output
    ...     # ... other parameters
    ... }
    >>> dataset = InterventionalDataset(scm_config, preprocessing_config, dataset_config)
    >>> X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, anc = dataset[0]
    >>> print(anc.shape)  # torch.Size([num_nodes, num_nodes])
    >>> # If L features kept, adj is (L+2) x (L+2)
    >>> # adj[0, 1] tells if treatment causes outcome
    >>> # adj[2, 1] tells if first feature (X[:,0]) causes outcome  
    >>> # adj[0, 2] tells if treatment causes first feature (X[:,0])
    """
    
    # Expected preprocessing hyperparameters
    EXPECTED_PREPROCESSING_HYPERPARAMETERS = {
        "dropout_prob": float,
        "shuffle_data": bool,
        "target_feature": (int, type(None)),
        "random_seed": (int, type(None)),
        "negative_one_one_scaling": bool,
        "remove_outliers": bool,
        "outlier_quantile": float,
        "yeo_johnson": bool,
        "standardize": bool,
        "y_clip_quantile": (float, type(None)),
        "eps": float,
        "increase_treatment_scale": bool,
        "distribution_rescale_factor": float,
        "interventional_distribution_type": str,
        "test_feature_mask_fraction": float,
    }
    
    # Expected dataset configuration parameters
    EXPECTED_DATASET_HYPERPARAMETERS = {
        "dataset_size": int,
        "max_number_features": int,
        # New config scheme: caps and total per dataset
        "max_number_samples_per_dataset": int,
        "max_number_train_samples_per_dataset": int,
        "max_number_test_samples_per_dataset": int,
        # Per-dataset sampling distributions (classic naming kept for compatibility)
        "number_train_samples_per_dataset": (torch.distributions.Distribution, int),
        "number_test_samples_per_dataset": (torch.distributions.Distribution, int),
        # Optional new key to drive collator; we accept it but don't use directly here
        "n_test_samples_per_dataset": (torch.distributions.Distribution, int),
    }
    
    # Distribution factories for building samplers
    DISTRIBUTION_FACTORIES = {
        "fixed": lambda params: FixedSampler(params["value"]),
        "uniform": lambda params: TorchDistributionSampler(
            dist.Uniform(low=params["low"], high=params["high"])
        ),
        "normal": lambda params: TorchDistributionSampler(
            dist.Normal(loc=params["mean"], scale=params["std"])
        ),
        "lognormal": lambda params: TorchDistributionSampler(
            dist.LogNormal(loc=params["mean"], scale=params["std"])
        ),
        "exponential": lambda params: TorchDistributionSampler(
            dist.Exponential(rate=params["lambd"])
        ),
        "gamma": lambda params: TorchDistributionSampler(
            dist.Gamma(concentration=params["alpha"], rate=params["beta"])
        ),
        "beta": lambda params: TorchDistributionSampler(
            dist.Beta(concentration1=params["alpha"], concentration0=params["beta"])
        ),
        "categorical": lambda params: CategoricalSampler(
            params["choices"], params.get("probabilities")
        ),
        "discrete_uniform": lambda params: DiscreteUniformSampler(params["low"], params["high"]),
    }
    
    # PyTorch distribution factories
    TORCH_DISTRIBUTION_FACTORIES = {
        "uniform": lambda params: dist.Uniform(low=params["low"], high=params["high"]),
        "normal": lambda params: dist.Normal(loc=params["mean"], scale=params["std"]),
        "lognormal": lambda params: dist.LogNormal(loc=params["mean"], scale=params["std"]),
        "exponential": lambda params: dist.Exponential(rate=params["lambd"]),
        "gamma": lambda params: dist.Gamma(concentration=params["alpha"], rate=params["beta"]),
        "beta": lambda params: dist.Beta(concentration1=params["alpha"], concentration0=params["beta"]),
    }

    def __init__(self, 
                 scm_config: Dict[str, Any],
                 preprocessing_config: Dict[str, Any],
                 dataset_config: Dict[str, Any],
                 seed: Optional[int] = None,
                 return_scm: bool = False):
        """
        Initialize the dataset with configuration dictionaries.
        
        Args:
            scm_config: SCM hyperparameter configuration
            preprocessing_config: Preprocessing hyperparameter configuration
            dataset_config: Dataset configuration (size, max samples, etc.)
            seed: Random seed for reproducibility
            return_scm: If True, also 
              the SCM object in __getitem__ (for debugging)
        """
        self.scm_config = scm_config
        self.preprocessing_config = preprocessing_config
        self.dataset_config = dataset_config
        self.seed = seed
        self.return_scm = return_scm
        
        # Helper to extract value from config entries that may be plain or dicts with {'value': ...}
        def _get_cfg_value(cfg: Dict[str, Any], key: str, default: Any):
            raw = cfg.get(key, default)
            if isinstance(raw, dict) and "value" in raw:
                return raw["value"]
            return raw

        # Rejection sampling settings (optional)
        self.min_target_variance = _get_cfg_value(self.dataset_config, "min_target_variance", None)
        # Minimum fraction of unique target values (relative to number of samples)
        # e.g., 0.2 means at least 20% of samples must have unique target values
        self.min_unique_target_fraction = _get_cfg_value(self.dataset_config, "min_unique_target_fraction", None)
        # Prevent infinite loops: cap the number of re-sampling attempts
        self.max_resample_attempts = int(_get_cfg_value(self.dataset_config, "max_resample_attempts", 10) or 10)
        
        # Options to return graph matrices (adjacency or ancestor)
        self.return_adjacency_matrix = _get_cfg_value(self.dataset_config, "return_adjacency_matrix", False)
        self.return_ancestor_matrix = _get_cfg_value(self.dataset_config, "return_ancestor_matrix", False)
        
        # Validate that both flags are not True simultaneously
        if self.return_adjacency_matrix and self.return_ancestor_matrix:
            raise ValueError(
                "Cannot return both adjacency matrix and ancestor matrix. "
                "Please set only one of 'return_adjacency_matrix' or 'return_ancestor_matrix' to True."
            )
        
        # Path constraint settings (optional)
        self.ensure_treatment_outcome_path = _get_cfg_value(self.preprocessing_config, "ensure_treatment_outcome_path", False)
        self.ensure_outcome_treatment_path = _get_cfg_value(self.preprocessing_config, "ensure_outcome_treatment_path", False)
        self.ensure_no_connection_treatment_outcome = _get_cfg_value(self.preprocessing_config, "ensure_no_connection_treatment_outcome", False)
        
        # Validate that contradictory path constraints are not both enabled
        if self.ensure_treatment_outcome_path and self.ensure_no_connection_treatment_outcome:
            raise ValueError(
                "Cannot simultaneously require treatment->outcome path AND no connection. "
                "Please set only one of 'ensure_treatment_outcome_path' or 'ensure_no_connection_treatment_outcome' to True."
            )
        if self.ensure_outcome_treatment_path and self.ensure_no_connection_treatment_outcome:
            raise ValueError(
                "Cannot simultaneously require outcome->treatment path AND no connection. "
                "Please set only one of 'ensure_outcome_treatment_path' or 'ensure_no_connection_treatment_outcome' to True."
            )
        
        # Build samplers for preprocessing and dataset parameters
        self.preprocessing_samplers = self._build_samplers(
            self.preprocessing_config, 
            self.EXPECTED_PREPROCESSING_HYPERPARAMETERS, 
            "preprocessing"
        )
        
        # Filter out 'seed' from dataset_config since we handle it as a constructor parameter
        dataset_config_filtered = {k: v for k, v in self.dataset_config.items() if k != 'seed'}
        self.dataset_samplers = self._build_samplers(
            dataset_config_filtered, 
            self.EXPECTED_DATASET_HYPERPARAMETERS, 
            "dataset"
        )
        
        # Create SCMSampler with the SCM config
        scm_seed = None
        if seed is not None:
            scm_seed = (seed * 31 + 17) % (2**32)
        self.scm_sampler = SCMSampler(scm_config, seed=scm_seed)
        
        # Sample dataset parameters once to get the size
        # (size is the only parameter that should be fixed for the whole dataset)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        
        dataset_params = self._sample_parameters(
            self.dataset_samplers,
            self.EXPECTED_DATASET_HYPERPARAMETERS,
            "dataset",
            generator
        )
        
        # Extract only the dataset size (fixed for whole dataset)
        self.size = dataset_params["dataset_size"]

        # Store max values as attributes (these should be fixed)
        self.max_number_features = dataset_params.get("max_number_features")
        # Backward compatibility: accept both old and new keys
        self.max_number_train_samples = dataset_params.get(
            "max_number_train_samples_per_dataset",
            dataset_params.get("max_number_train_samples", 0),
        )
        self.max_number_test_samples = dataset_params.get(
            "max_number_test_samples_per_dataset",
            dataset_params.get("max_number_test_samples", 0),
        )
        # New total cap (not strictly needed inside this Dataset, but recorded for reference)
        self.max_number_samples_per_dataset = dataset_params.get(
            "max_number_samples_per_dataset",
            self.max_number_train_samples + self.max_number_test_samples,
        )

        # Don't store the distributions - they will be sampled per item in __getitem__
    
    def _build_samplers(self, config: Dict[str, Any], expected_params: Dict[str, Any], 
                       config_name: str) -> Dict[str, Any]:
        """Build sampler objects from configuration."""
        samplers = {}
        
        for param_name, param_config in config.items():
            # Handle shorthand fixed value notation
            if "value" in param_config and "distribution" not in param_config:
                if param_name in ["number_train_samples_per_dataset", "number_test_samples_per_dataset"]:
                    sampler = FixedSampler(param_config["value"])
                else:
                    sampler = FixedSampler(param_config["value"])
            elif "distribution" in param_config:
                dist_type = param_config["distribution"]
                
                if param_name in ["number_train_samples_per_dataset", "number_test_samples_per_dataset"]:
                    # Create PyTorch distribution directly
                    if dist_type not in self.TORCH_DISTRIBUTION_FACTORIES:
                        raise ValueError(f"Unknown distribution type for {param_name}: {dist_type}")
                    
                    dist_params = param_config.get("distribution_parameters", {})
                    try:
                        sampler = self.TORCH_DISTRIBUTION_FACTORIES[dist_type](dist_params)
                    except Exception as e:
                        raise ValueError(f"Error creating distribution for {config_name}.{param_name}: {e}")
                else:
                    # Regular parameter: create sampler
                    if dist_type not in self.DISTRIBUTION_FACTORIES:
                        raise ValueError(f"Unknown distribution type: {dist_type}")
                    
                    dist_params = param_config.get("distribution_parameters", {})
                    if dist_type == "fixed":
                        if "value" not in param_config:
                            raise ValueError(f"Fixed distribution for {param_name} requires 'value' key")
                        dist_params = {"value": param_config["value"]}
                    
                    try:
                        sampler = self.DISTRIBUTION_FACTORIES[dist_type](dist_params)
                    except Exception as e:
                        raise ValueError(f"Error creating sampler for {config_name}.{param_name}: {e}")
            else:
                raise ValueError(f"Configuration for {config_name}.{param_name} must specify 'distribution' or 'value'")
            
            samplers[param_name] = sampler
        
        return samplers
    
    def _sample_parameters(self, samplers: Dict[str, Any], expected_types: Dict[str, Any],
                          config_name: str, generator: torch.Generator) -> Dict[str, Any]:
        """Sample parameters from samplers with type validation."""
        sampled_params = {}
        
        for param_name, sampler in samplers.items():
            # If the parameter is not part of the expected types, sample it loosely and skip validation.
            if param_name not in expected_types:
                # Handle samplers that are FixedSampler-like or torch distributions
                if hasattr(sampler, 'sample'):
                    try:
                        value = sampler.sample(generator)
                    except TypeError:
                        # Some samplers may not accept a generator
                        value = sampler.sample()
                else:
                    value = sampler
                sampled_params[param_name] = value
                continue

            if param_name in ["number_train_samples_per_dataset", "number_test_samples_per_dataset"]:
                # Special case: handle FixedSampler, PyTorch distribution, or int directly
                if isinstance(sampler, FixedSampler):
                    value = sampler.sample(generator)
                elif isinstance(sampler, torch.distributions.Distribution):
                    value = sampler
                elif isinstance(sampler, int):
                    value = sampler
                else:
                    raise ValueError(f"Parameter {config_name}.{param_name} has invalid type")
            else:
                value = sampler.sample(generator)
            
            # Type validation
            expected_type = expected_types[param_name]
            if not isinstance(value, expected_type):
                if param_name in ["number_train_samples_per_dataset", "number_test_samples_per_dataset"]:
                    if isinstance(expected_type, tuple):
                        if not isinstance(value, expected_type):
                            raise ValueError(f"Parameter {config_name}.{param_name} has invalid type")
                    elif isinstance(value, torch.distributions.Distribution):
                        pass  # This is correct
                    else:
                        raise ValueError(f"Parameter {config_name}.{param_name} has invalid type")
                elif isinstance(expected_type, type) and expected_type is int and isinstance(value, float):
                    value = int(value)
                elif isinstance(expected_type, type) and expected_type is float and isinstance(value, int):
                    value = float(value)
                elif isinstance(expected_type, type) and expected_type is tuple and isinstance(value, list):
                    value = tuple(value)
                elif isinstance(expected_type, tuple) and type(None) in expected_type:
                    allowed_types = [t for t in expected_type if t is not type(None)]
                    if value is not None and not isinstance(value, tuple(allowed_types)):
                        raise ValueError(f"Parameter {config_name}.{param_name} has invalid type")
                else:
                    raise ValueError(f"Parameter {config_name}.{param_name} has invalid type")
            
            sampled_params[param_name] = value
        
        return sampled_params

    def __len__(self):
        return self.size
    
    def _contains_nan(self, *tensors):
        """Check if any of the provided tensors contains NaN values.
        Skips non-tensor objects (e.g., SCM, processor when return_scm=True)."""
        for tensor in tensors:
            if tensor is not None and torch.is_tensor(tensor) and torch.isnan(tensor).any():
                return True
        return False
    
    def __getitem__(self, idx):
        """Get item with retry logic if NaN values are detected."""
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
        
        max_retries = 10
        for attempt in range(max_retries):
            result = self._get_item_internal(idx, attempt)
            
            # Check if result contains NaN
            if not self._contains_nan(*result):
                return result
            
            # NaN detected, print warning and retry
            print(f"[InterventionalDataset] Warning: NaN detected in sample {idx} (attempt {attempt + 1}/{max_retries}). Resampling...")
        
        # If we've exhausted retries, raise an error
        raise RuntimeError(f"Failed to generate valid sample for index {idx} after {max_retries} attempts (NaN persists)")
    
    def _get_item_internal(self, idx, attempt=0):
        """Internal method to generate a single item."""
        # Use attempt number to vary the seed for retries
        seed = self.seed + idx if self.seed is not None else idx
        seed = seed + attempt * 1000000  # Offset seed by attempt number
        torch.manual_seed(seed)
        
        # Create a generator for this specific item
        item_generator = torch.Generator()
        item_generator.manual_seed(seed)
        
        # Sample preprocessing parameters for this item
        preprocessing_params = self._sample_parameters(
            self.preprocessing_samplers,
            self.EXPECTED_PREPROCESSING_HYPERPARAMETERS,
            "preprocessing",
            item_generator
        )
        
        # Sample dataset parameters for this item (except size and max values which are fixed)
        dataset_params = self._sample_parameters(
            self.dataset_samplers,
            self.EXPECTED_DATASET_HYPERPARAMETERS,
            "dataset",
            item_generator
        )

        if "number_train_samples_per_dataset" not in dataset_params: 
            train_dist = self.dataset_config["max_number_train_samples_per_dataset"]["value"]
            dataset_params["number_train_samples_per_dataset"] = train_dist
        if "number_test_samples_per_dataset" not in dataset_params:
            test_dist = self.dataset_config["max_number_test_samples_per_dataset"]["value"]
            dataset_params["number_test_samples_per_dataset"] = test_dist
        
        # Extract sample counts from dataset params
        train_dist = dataset_params["number_train_samples_per_dataset"]
        test_dist = dataset_params["number_test_samples_per_dataset"]
        
        # Sample train and test sample counts
        if isinstance(train_dist, torch.distributions.Distribution):
            number_train_samples = int(train_dist.sample().item())
        elif isinstance(train_dist, int):
            number_train_samples = train_dist
        else:
            number_train_samples = int(train_dist.sample(item_generator) if hasattr(train_dist, 'sample') else train_dist)
        
        if isinstance(test_dist, torch.distributions.Distribution):
            number_test_samples = int(test_dist.sample().item())
        elif isinstance(test_dist, int):
            number_test_samples = test_dist
        else:
            number_test_samples = int(test_dist.sample(item_generator) if hasattr(test_dist, 'sample') else test_dist)
        
        # Rejection strategy: resample if target variances are too small or path constraints not met
        # We re-run SCM sampling up to max_resample_attempts
        retry_attempt = 0
        last_result = None
        while True:
            # Sample an SCM
            scm = self.scm_sampler.sample(seed=seed + retry_attempt)

            if self.return_scm:
                org_scm = deepcopy(scm)
            
            # sample the observational data first
            scm.sample_exogenous(num_samples=number_train_samples)
            scm.sample_endogenous(num_samples=number_train_samples)

            obs0_raw = scm.propagate(num_samples=number_train_samples)
            # Reshape from (N,) to (N,1) for BasicProcessing compatibility
            obs0 = {k: v.reshape(-1, 1) if v.dim() == 1 else v for k, v in obs0_raw.items()}

            # now, do interventional sampling
            # for interventional sampling, first determine intervention node
            intervention_node = torch.randint(0, len(scm.dag.nodes()), (1,)).item() # select random node for intervention
            
            # CRITICAL: Check path constraints on the ORIGINAL SCM before intervention
            # We need to determine the target node first to check constraints
            # Use the same logic as BasicProcessing to select target
            all_nodes = list(scm.dag.nodes())
            if preprocessing_params["target_feature"] is not None:
                target_node = preprocessing_params["target_feature"]
            else:
                # BasicProcessing selects a random target different from intervention_node
                available_targets = [n for n in all_nodes if n != intervention_node]
                if len(available_targets) > 0:
                    target_idx = torch.randint(0, len(available_targets), (1,)).item()
                    target_node = available_targets[target_idx]
                else:
                    # Edge case: only one node in SCM
                    target_node = intervention_node
            
            # Check path constraints on ORIGINAL (non-intervened) SCM
            path_constraint_ok = True
            if any([self.ensure_treatment_outcome_path, 
                   self.ensure_outcome_treatment_path, 
                   self.ensure_no_connection_treatment_outcome]):
                
                treatment_node = intervention_node
                
                # Check 3a: Ensure treatment->outcome path exists
                if self.ensure_treatment_outcome_path:
                    path_exists = scm.exists_treatment_outcome_path(treatment_node, target_node)
                    if not path_exists:
                        path_constraint_ok = False
                        
                
                # Check 3b: Ensure outcome->treatment path exists
                if self.ensure_outcome_treatment_path:
                    path_exists = scm.exists_outcome_treatment_path(treatment_node, target_node)
                    if not path_exists:
                        path_constraint_ok = False
        
                
                # Check 3c: Ensure no connection between treatment and outcome
                if self.ensure_no_connection_treatment_outcome:
                    no_connection = scm.exists_no_connection_treatment_outcome(treatment_node, target_node)
                    if not no_connection:
                        path_constraint_ok = False
            
            # If path constraints fail, skip the rest and resample
            if not path_constraint_ok:
                retry_attempt += 1
                if retry_attempt >= self.max_resample_attempts:
                    print(f"[InterventionalDataset] Warning: Max resample attempts ({self.max_resample_attempts}) reached for path constraints. Using last sample.")
                    # Continue with current SCM despite constraint violation
                    pass
                else:
                    continue  # Resample a new SCM

            scm.sample_exogenous(num_samples=number_test_samples) #sample observational data again. 
            scm.sample_endogenous(num_samples=number_test_samples) #sample observational data again. 

            obs1_raw = scm.propagate(num_samples=number_test_samples)  # fresh observational batch (test-size)

            # Collect observational samples for the chosen intervention node (marginal)
            intervention_samples = obs1_raw[intervention_node]

            # Determine distribution type and rescaling based on preprocessing parameters
            dist_type = preprocessing_params.get("interventional_distribution_type", "resampling")
            increase_scale = preprocessing_params.get("increase_treatment_scale", False)
            rescale_factor = preprocessing_params.get("distribution_rescale_factor", 0.0)
            scale_factor = preprocessing_params.get("interventional_scale_factor", 3.0)  # Default to 3.0 for scaled_uniform
            
            # Create intervention distribution based on type
            if dist_type == "uniform":
                # Use uniform distribution over [min, max] of observational samples
                interventional_dist = UniformResamplingDist(intervention_samples)
            elif dist_type == "scaled_uniform":
                # Use scaled uniform distribution: U(mean - a, mean + a) where a = scale_factor * std
                # P(T_int) = U(-a, a) + b where a = scale_factor * std(P(T_obs)) and b = mean(P(T_obs))
                interventional_dist = ScaledUniformResamplingDist(
                    intervention_samples,
                    scale_factor=scale_factor
                )
            elif dist_type == "rescaled" or (increase_scale and rescale_factor != 0.0):
                # Use rescaled distribution to increase treatment variable scale
                interventional_dist = RescaledResamplingDist(
                    intervention_samples,
                    rescale_factor=rescale_factor
                )
            else:
                # Default: use standard resampling distribution (discrete sampling without replacement)
                interventional_dist = ResamplingDist(intervention_samples)

            scm.intervene(node = intervention_node) # intervene on the chosen node

            # Replace the noise distribution for the intervened node with its observational marginal
            if intervention_node in scm.dag.endogenous_variables():
                scm.endogenous_noise[intervention_node] = interventional_dist
            if intervention_node in scm.dag.exogenous_variables():
                scm.exogenous_noise[intervention_node] = interventional_dist

            # Sample new noise for interventional scenario
            scm.sample_exogenous(num_samples=number_test_samples)
            scm.sample_endogenous(num_samples=number_test_samples)

            interv1_raw = scm.propagate(num_samples=number_test_samples)  # interventional data (post-intervention)
            # Reshape from (N,) to (N,1) for BasicProcessing compatibility
            interv1 = {k: v.reshape(-1, 1) if v.dim() == 1 else v for k, v in interv1_raw.items()}
            
            # Create BasicProcessing instance with sampled preprocessing parameters
            # Map legacy combined flags to new split flags with sensible defaults
            _feature_standardize = preprocessing_params.get("feature_standardize",
                                                           preprocessing_params.get("standardize", True))
            _feature_neg11 = preprocessing_params.get("feature_negative_one_one_scaling",
                                                      False if _feature_standardize else preprocessing_params.get("negative_one_one_scaling", True))
            _target_neg11 = preprocessing_params.get("target_negative_one_one_scaling",
                                                     preprocessing_params.get("negative_one_one_scaling", True))

            processor = BasicProcessing(
                n_features=self.max_number_features,
                max_n_features=self.max_number_features,
                n_train_samples=number_train_samples,
                max_n_train_samples=self.max_number_train_samples,
                n_test_samples=number_test_samples,
                max_n_test_samples=self.max_number_test_samples,
                dropout_prob=preprocessing_params["dropout_prob"],
                target_feature=preprocessing_params["target_feature"],
                intervened_feature=intervention_node,
                random_seed=preprocessing_params["random_seed"],
                test_feature_mask_fraction=preprocessing_params.get("test_feature_mask_fraction", 0.0),
                # Legacy flags retained, but split flags take precedence internally
                negative_one_one_scaling=preprocessing_params.get("negative_one_one_scaling", True),
                standardize=preprocessing_params.get("standardize", True),
                # New split flags
                feature_standardize=_feature_standardize,
                feature_negative_one_one_scaling=_feature_neg11,
                target_negative_one_one_scaling=_target_neg11,
                yeo_johnson=preprocessing_params["yeo_johnson"],
                remove_outliers=preprocessing_params["remove_outliers"],
                outlier_quantile=preprocessing_params["outlier_quantile"],
                shuffle_samples=preprocessing_params["shuffle_data"],
                shuffle_features=True,  # Default
                y_clip_quantile=preprocessing_params.get("y_clip_quantile"),
                eps=preprocessing_params.get("eps", 1e-8),
                device=None,  # Default
                dtype=None,  # Default
            )
            
            # Process observational (train) and interventional (test) splits separately
            processed = processor.process_from_splits(
                train_dataset=obs0,
                test_dataset=interv1,
                mode = "fast"
            )
            
            # Unpack results
            if len(processed) == 4:
                X_obs, Y_obs, X_intv, Y_intv = processed
                result = (X_obs, Y_obs, X_intv, Y_intv)
                has_treatment = False
            else:
                X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv = processed

                if _feature_standardize:
                    T_obs = (T_obs - torch.mean(T_obs)) / (torch.std(T_obs) + preprocessing_params.get("eps", 1e-8))  # standardize treatment
                    T_intv = (T_intv - torch.mean(T_intv)) / (torch.std(T_intv) + preprocessing_params.get("eps", 1e-8))  # standardize treatment
                
                #T_obs = T_obs/ torch.std(T_obs)  # rescale treatment to have unit stddev
                #T_intv = T_intv/ torch.std(T_intv)  # rescale treatment to have unit stddev
                result = (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv)
                has_treatment = True

            
            # Optionally add adjacency or ancestor matrix with proper node ordering
            if self.return_adjacency_matrix or self.return_ancestor_matrix:
                # Construct adjacency matrix with specific ordering to match model's data layout:
                # Position 0: Treatment variable (intervention_node)
                # Position 1: Outcome variable (processor.selected_target_feature)
                # Positions 2 to L+1: Feature variables that were KEPT after dropout (in sorted order)
                #
                # This ordering matches the model's internal representation: [T, Y, X_0, ..., X_{L-1}]
                #
                # This aligns with the REAL (non-padded) columns in the returned tensors:
                # - T_obs/T_intv correspond to position 0 in adjacency matrix
                # - Y_obs/Y_intv correspond to position 1 in adjacency matrix
                # - X_obs/X_intv[:, i] corresponds to position i+2 in adjacency matrix for i < len(kept_features)
                #
                # CRITICAL: The Preprocessor may pad X to max_n_features with zero columns.
                # The adjacency matrix size is (len(kept_features) + 2) x (len(kept_features) + 2),
                # which may be SMALLER than X.shape[1] + 2 if padding was applied.
                #
                # Example: If SCM has 32 nodes, treatment='X5', outcome='X10', and 25 features kept:
                #   - Adjacency matrix is 27x27 (1 treatment + 1 outcome + 25 features)
                #   - X might be padded to shape (N, 50)
                #   - Adjacency position 0 = treatment T
                #   - Adjacency position 1 = outcome Y
                #   - X[:, :25] correspond to real SCM nodes (positions 2-26 in adjacency)
                #   - X[:, 25:] are padded zeros with no correspondence to adjacency matrix
                #
                # The adjacency matrix entry [i,j] = 1.0 indicates a directed edge from 
                # variable at position i to variable at position j in this ordering.
                #
                # The ancestor matrix has the same ordering and shape as adjacency matrix,
                # but entry [i,j] = 1.0 indicates that i is an ancestor of j (transitive closure).
                
                if has_treatment:
                    # Get the target feature and kept features from BasicProcessing
                    target_node = processor.selected_target_feature
                    kept_features = processor.kept_feature_indices  # Node names after dropout AND shuffling
                    
                    # Build ordered list to match model's feature ordering: [treatment, outcome, features]
                    # Model has: [T, Y, X_0, X_1, ..., X_{L-1}]
                    # So adjacency matrix must use the same ordering
                    # 
                    # IMPORTANT: kept_features now correctly reflects the POST-SHUFFLE column order!
                    # If shuffle_features=True was used in BasicProcessing, the Preprocessor
                    # tracks the permutation and BasicProcessing updates kept_feature_indices
                    # to match the actual column order in X. This ensures perfect alignment:
                    # - X[:, i] contains data from node kept_features[i]
                    # - adjacency[i+2, j+2] describes the edge between kept_features[i] and kept_features[j]
                    #
                    # Do NOT sort kept_features - use the exact order from the processor!
                    ordered_nodes = []
                    
                    # Add treatment and outcome first
                    ordered_nodes.append(intervention_node)
                    ordered_nodes.append(target_node)
                    
                    # Then add kept features in the SAME order as they appear in X
                    # (kept_features already reflects any shuffling that was applied)
                    ordered_nodes.extend(kept_features)
                    
                    # Get adjacency matrix with this specific ordering: [T, Y, X_0, ..., X_{L-1}]
                    # where X_i corresponds to X[:, i] in the returned tensor
                    # This will be size (len(kept_features) + 2) x (len(kept_features) + 2)
                    adj_matrix_unpadded = scm.get_adjacency_matrix(node_order=ordered_nodes)
                    
                    # Convert to ancestor matrix if requested
                    if self.return_ancestor_matrix:
                        #print("Converting to ancestor matrix...")
                        graph_matrix_unpadded = adjacency_to_ancestor_matrix(adj_matrix_unpadded)
                    else:
                        graph_matrix_unpadded = adj_matrix_unpadded
            
                    
                    # Pad matrix to match X dimensions
                    # X has been padded to max_n_features, so matrix needs to be (max_n_features + 2) x (max_n_features + 2)
                    # Note: The +2 accounts for treatment T and outcome Y which come AFTER the features in the ordering
                    num_real_features = len(kept_features)  # Number of real (non-padded) features
                    target_size = X_obs.shape[1] + 2  # number of features in X + 2 (treatment + outcome)
                    
                    if graph_matrix_unpadded.shape[0] < target_size:
                        # Need to pad the matrix
                        # Padding maintains the ordering: [X_0, ..., X_{L-1}, T, Y, padding...]
                        padded_matrix = torch.zeros((target_size, target_size), 
                                                dtype=graph_matrix_unpadded.dtype,
                                                device=graph_matrix_unpadded.device)
                        # Copy the real matrix values into the top-left corner
                        padded_matrix[:graph_matrix_unpadded.shape[0], :graph_matrix_unpadded.shape[1]] = graph_matrix_unpadded
                        graph_matrix = padded_matrix
                    else:
                        graph_matrix = graph_matrix_unpadded
                else:
                    # No treatment variable case - just use default topological ordering
                    adj_matrix = scm.get_adjacency_matrix()
                    
                    # Convert to ancestor matrix if requested
                    if self.return_ancestor_matrix:
                        graph_matrix = adjacency_to_ancestor_matrix(adj_matrix)
                    else:
                        graph_matrix = adj_matrix
                    
                    # Pad if needed
                    target_size = X_obs.shape[1]
                    if graph_matrix.shape[0] < target_size:
                        padded_matrix = torch.zeros((target_size, target_size),
                                                dtype=graph_matrix.dtype, 
                                                device=graph_matrix.device)
                        padded_matrix[:graph_matrix.shape[0], :graph_matrix.shape[1]] = graph_matrix
                        graph_matrix = padded_matrix
                
                result = result + (graph_matrix,)
            
            # Optionally add SCM for debugging
            if self.return_scm:
                # Also return processor and intervention_node for detailed debugging
                result = result + (org_scm, processor, intervention_node)
            
            # Save latest result so we can return even if rejection keeps failing
            last_result = result
            
            # If no thresholds or path constraints provided, accept immediately
            no_constraints = (
                self.min_target_variance is None and 
                self.min_unique_target_fraction is None and
                not self.ensure_treatment_outcome_path and
                not self.ensure_outcome_treatment_path and
                not self.ensure_no_connection_treatment_outcome
            )
            if no_constraints:
                break
            
            # --- Check 1: Variance threshold ---
            var_threshold_ok = True
            if self.min_target_variance is not None:
                # Compute variances on the target for observational (train) and interventional (test)
                # Y_* may have shape (N, 1); use flatten for variance computation
                var_obs = torch.var(Y_obs.reshape(-1))
                var_intv = torch.var(Y_intv.reshape(-1))
                
                # Normalize threshold to float if provided as str in YAML
                threshold = self.min_target_variance
                if isinstance(threshold, str):
                    try:
                        threshold = float(threshold)
                    except ValueError:
                        # If parsing fails, disable variance check
                        threshold = None
                
                if threshold is not None:
                    var_threshold_ok = (var_obs.item() >= threshold) and (var_intv.item() >= threshold)
            
            # --- Check 2: Unique values threshold ---
            unique_threshold_ok = True
            if self.min_unique_target_fraction is not None:
                # Normalize threshold to float if provided as str in YAML
                unique_fraction = self.min_unique_target_fraction
                if isinstance(unique_fraction, str):
                    try:
                        unique_fraction = float(unique_fraction)
                    except ValueError:
                        # If parsing fails, disable unique check
                        unique_fraction = None
                
                if unique_fraction is not None:
                    # Count unique values in observational and interventional targets
                    Y_obs_flat = Y_obs.reshape(-1)
                    Y_intv_flat = Y_intv.reshape(-1)
                    
                    n_unique_obs = torch.unique(Y_obs_flat).numel()
                    n_unique_intv = torch.unique(Y_intv_flat).numel()
                    
                    n_obs = Y_obs_flat.numel()
                    n_intv = Y_intv_flat.numel()
                    
                    # Check if fraction of unique values meets threshold
                    obs_unique_ok = (n_unique_obs / n_obs) >= unique_fraction if n_obs > 0 else True
                    intv_unique_ok = (n_unique_intv / n_intv) >= unique_fraction if n_intv > 0 else True
                    
                    unique_threshold_ok = obs_unique_ok and intv_unique_ok
            
            # Note: Path constraints are checked BEFORE intervention (earlier in the loop)
            # This is intentional - we want to check the original causal structure
            
            # Accept if all checks pass (path_constraint_ok was checked earlier)
            if var_threshold_ok and unique_threshold_ok and path_constraint_ok:
                break  # Accept
            
            retry_attempt += 1
            if retry_attempt >= self.max_resample_attempts:
                # Give up and return the last sampled data to avoid infinite loop
                failed_checks = []
                if not var_threshold_ok:
                    failed_checks.append("variance threshold")
                if not unique_threshold_ok:
                    failed_checks.append("unique values threshold")
                if not path_constraint_ok:
                    failed_checks.append("path constraints")
                
                print(f"[InterventionalDataset] Warning: Max resample attempts ({self.max_resample_attempts}) "
                      f"reached. Failed checks: {', '.join(failed_checks)}. Using last sample.")
                break
        
        #breakpoint()
        return last_result