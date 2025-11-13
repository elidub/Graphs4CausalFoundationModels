from __future__ import annotations
from typing import Dict, Any, Optional, Union
from torch.utils.data import Dataset
import torch
import torch.distributions as dist
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from priors.causal_prior.scm.SCMSampler import SCMSampler
from priordata_processing.BasicProcessing import BasicProcessing
from utils import FixedSampler, TorchDistributionSampler, CategoricalSampler, DiscreteUniformSampler


class ObservationalDataset(Dataset):
    """
    Dataset for purely observational causal data (no interventions).
    
    This class directly accepts SCM, preprocessing, and dataset configurations,
    internally creating SCMSampler and sampling hyperparameters on-the-fly.
    
    Parameters
    ----------
    scm_config : Dict[str, Any]
        Configuration for SCM hyperparameters (same format as SCMSampler)
    preprocessing_config : Dict[str, Any]
        Configuration for data preprocessing hyperparameters
    dataset_config : Dict[str, Any]
        Configuration for dataset parameters (size, max samples, etc.)
    seed : Optional[int], default None
        Random seed for reproducibility
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
    }
    
    # Expected dataset configuration parameters
    EXPECTED_DATASET_HYPERPARAMETERS = {
        "dataset_size": int,
        "max_number_features": int,
        "max_number_train_samples": int,
        "max_number_test_samples": int,
        "number_train_samples_per_dataset": (torch.distributions.Distribution, int),
        "number_test_samples_per_dataset": (torch.distributions.Distribution, int),
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
                 seed: Optional[int] = None):
        """
        Initialize the dataset with configuration dictionaries.
        
        Args:
            scm_config: SCM hyperparameter configuration
            preprocessing_config: Preprocessing hyperparameter configuration
            dataset_config: Dataset configuration (size, max samples, etc.)
            seed: Random seed for reproducibility
        """
        self.scm_config = scm_config
        self.preprocessing_config = preprocessing_config
        self.dataset_config = dataset_config
        self.seed = seed
        
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
        
        # Sample dataset parameters once (these are fixed for the whole dataset)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        
        dataset_params = self._sample_parameters(
            self.dataset_samplers,
            self.EXPECTED_DATASET_HYPERPARAMETERS,
            "dataset",
            generator
        )
        
        # Extract dataset parameters
        self.size = dataset_params["dataset_size"]
        self.max_number_features = dataset_params["max_number_features"]
        self.max_number_train_samples = dataset_params["max_number_train_samples"]
        self.max_number_test_samples = dataset_params["max_number_test_samples"]
        
        # Store sample count distributions
        train_dist = dataset_params["number_train_samples_per_dataset"]
        test_dist = dataset_params["number_test_samples_per_dataset"]
        
        # Convert fixed integers to distributions if needed
        if isinstance(train_dist, int):
            self.number_train_samples_per_dataset_distribution = FixedSampler(train_dist)
        elif isinstance(train_dist, torch.distributions.Distribution):
            self.number_train_samples_per_dataset_distribution = train_dist
        else:
            self.number_train_samples_per_dataset_distribution = train_dist
            
        if isinstance(test_dist, int):
            self.number_test_samples_per_dataset_distribution = FixedSampler(test_dist)
        elif isinstance(test_dist, torch.distributions.Distribution):
            self.number_test_samples_per_dataset_distribution = test_dist
        else:
            self.number_test_samples_per_dataset_distribution = test_dist
        
        # Sample preprocessing parameters once (these are fixed for the whole dataset)
        self.preprocessing_params = self._sample_parameters(
            self.preprocessing_samplers,
            self.EXPECTED_PREPROCESSING_HYPERPARAMETERS,
            "preprocessing",
            generator
        )
    
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
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
        
        seed = self.seed + idx if self.seed is not None else idx
        torch.manual_seed(seed)
        
        # Sample an SCM
        scm = self.scm_sampler.sample(seed=seed)
        
        # Sample train and test sample counts
        if isinstance(self.number_train_samples_per_dataset_distribution, FixedSampler):
            number_train_samples = self.number_train_samples_per_dataset_distribution.sample(torch.Generator().manual_seed(seed))
        elif isinstance(self.number_train_samples_per_dataset_distribution, torch.distributions.Distribution):
            number_train_samples = self.number_train_samples_per_dataset_distribution.sample()
        else:
            number_train_samples = self.number_train_samples_per_dataset_distribution.sample(torch.Generator().manual_seed(seed))
        
        if hasattr(number_train_samples, 'item'):
            number_train_samples = int(number_train_samples.item())
        else:
            number_train_samples = int(number_train_samples)
        
        if isinstance(self.number_test_samples_per_dataset_distribution, FixedSampler):
            number_test_samples = self.number_test_samples_per_dataset_distribution.sample(torch.Generator().manual_seed(seed + 1))
        elif isinstance(self.number_test_samples_per_dataset_distribution, torch.distributions.Distribution):
            number_test_samples = self.number_test_samples_per_dataset_distribution.sample()
        else:
            number_test_samples = self.number_test_samples_per_dataset_distribution.sample(torch.Generator().manual_seed(seed + 1))
        
        if hasattr(number_test_samples, 'item'):
            number_test_samples = int(number_test_samples.item())
        else:
            number_test_samples = int(number_test_samples)
        
        # Total samples needed
        total_samples = number_train_samples + number_test_samples
        
        # Generate data from SCM
        scm.sample_exogenous(num_samples=total_samples)
        scm.sample_endogenous(num_samples=total_samples)
        scm_data = scm.propagate(num_samples=total_samples)
        
        # Convert to format expected by BasicProcessing
        # SCM returns {node_id: tensor with shape [num_samples, *node_shape]}
        # BasicProcessing expects {feature_id: tensor with shape [num_samples, 1]}
        dataset = {}
        for key, value in scm_data.items():
            # Reshape to [num_samples, 1] regardless of original node_shape
            dataset[key] = value.reshape(total_samples, -1)
        
        # Create BasicProcessing instance with sampled preprocessing parameters
        processor = BasicProcessing(
            n_features=self.max_number_features,
            max_n_features=self.max_number_features,
            n_train_samples=number_train_samples,
            max_n_train_samples=self.max_number_train_samples,
            n_test_samples=number_test_samples,
            max_n_test_samples=self.max_number_test_samples,
            dropout_prob=self.preprocessing_params["dropout_prob"],
            target_feature=self.preprocessing_params["target_feature"],
            random_seed=self.preprocessing_params["random_seed"],
            negative_one_one_scaling=self.preprocessing_params["negative_one_one_scaling"],
            standardize=self.preprocessing_params["standardize"],
            yeo_johnson=self.preprocessing_params["yeo_johnson"],
            remove_outliers=self.preprocessing_params["remove_outliers"],
            outlier_quantile=self.preprocessing_params["outlier_quantile"],
            shuffle_samples=self.preprocessing_params["shuffle_data"],
            shuffle_features=True,  # Default
            y_clip_quantile=self.preprocessing_params.get("y_clip_quantile"),
            eps=self.preprocessing_params.get("eps", 1e-8),
            device=None,  # Default
            dtype=None,  # Default
        )
        
        # Process the data
        X_train, Y_train, X_test, Y_test = processor.process(dataset)
        
        return X_train, Y_train, X_test, Y_test