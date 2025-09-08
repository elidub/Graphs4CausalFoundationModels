from __future__ import annotations
from typing import Dict, Any
import torch
import torch.distributions as dist
import sys
import os

from priors.causal_prior.scm.SCMSampler import SCMSampler
from priordata_processing.BasicProcessing import BasicProcessing
from priordata_processing.Datasets.PurelyObservationalDataset import PurelyObservationalDataset

# Import from the main utils module (not training.utils)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import FixedSampler, TorchDistributionSampler, CategoricalSampler, DiscreteUniformSampler


class MakePurelyObservationalDataset:
    """
    Factory class for creating PurelyObservationalDataset instances with rigorous hyperparameter validation.
    
    This class takes three hyperparameter configs:
    1. SCM hyperparameter config (for SCMSampler)
    2. Preprocessing hyperparameter config (for BasicProcessing)
    3. Dataset configuration (for PurelyObservationalDataset creation parameters)
    
    All configs follow the same format as in Basic_Configs.py and SCMHyperparameterSampler, where parameters can be:
    - Fixed values: {"value": some_value}
    - Distributions: {"distribution": "uniform", "distribution_parameters": {...}}
    
    No defaults are used - all parameters must be explicitly specified in configurations.
    """
    
    # Define expected hyperparameters for preprocessing and their types
    EXPECTED_PREPROCESSING_HYPERPARAMETERS = {
        "train_fraction": float,
        "dropout_prob": float,
        "shuffle_data": bool,  # maps to shuffle_samples
        "target_feature": (int, type(None)),
        "random_seed": (int, type(None)),
        "negative_one_one_scaling": bool,
        "remove_outliers": bool,
        "outlier_quantile": float,
        "yeo_johnson": bool,
        "standardize": bool,
    }
    
    # Define expected dataset configuration parameters and their types
    EXPECTED_DATASET_HYPERPARAMETERS = {
        "dataset_size": int,
        "max_number_samples": int,
        "max_number_features": int,
        "number_samples_per_dataset": torch.distributions.Distribution,
        "seed": (int, type(None)),
    }
    
    # Distribution factory mapping (same as SCMHyperparameterSampler)
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
    
    # Direct distribution factories for creating PyTorch distributions
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
                 dataset_config: Dict[str, Any]):
        """
        Initialize the factory with hyperparameter configurations.
        
        Args:
            scm_config: Configuration dict for SCMSampler (same format as Basic_Configs.py)
            preprocessing_config: Configuration dict for BasicProcessing hyperparameters
            dataset_config: Configuration dict for dataset creation parameters
        """
        self.scm_config = scm_config
        self.preprocessing_config = preprocessing_config
        self.dataset_config = dataset_config
        
        # Validate and build samplers
        self.preprocessing_samplers = self._build_samplers(
            self.preprocessing_config, self.EXPECTED_PREPROCESSING_HYPERPARAMETERS, "preprocessing"
        )
        self.dataset_samplers = self._build_samplers(
            self.dataset_config, self.EXPECTED_DATASET_HYPERPARAMETERS, "dataset"
        )
        
        # SCM config validation will be handled by SCMSampler itself
    
    def _build_samplers(self, config: Dict[str, Any], expected_params: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Build sampler objects from the configuration, following SCMHyperparameterSampler pattern."""
        samplers = {}
        
        for param_name, param_config in config.items():
            # Check if parameter is known
            if param_name not in expected_params:
                raise ValueError(f"Unknown {config_name} hyperparameter: {param_name}")
            
            # Handle shorthand fixed value notation
            if "value" in param_config and "distribution" not in param_config:
                if param_name == "number_samples_per_dataset":
                    # Special case: if it's a fixed distribution, create it directly
                    sampler = param_config["value"]
                else:
                    sampler = FixedSampler(param_config["value"])
            elif "distribution" in param_config:
                dist_type = param_config["distribution"]
                
                if param_name == "number_samples_per_dataset":
                    # Special case: create PyTorch distribution directly
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
                    
                    # Get distribution parameters
                    dist_params = param_config.get("distribution_parameters", {})
                    if dist_type == "fixed":
                        if "value" not in param_config:
                            raise ValueError(f"Fixed distribution for {param_name} requires 'value' key")
                        dist_params = {"value": param_config["value"]}
                    
                    # Create sampler
                    try:
                        sampler = self.DISTRIBUTION_FACTORIES[dist_type](dist_params)
                    except Exception as e:
                        raise ValueError(f"Error creating sampler for {config_name}.{param_name}: {e}")
            else:
                raise ValueError(f"Configuration for {config_name}.{param_name} must specify 'distribution' or 'value'")
            
            samplers[param_name] = sampler
        
        # Check that all required parameters are specified
        required_params = set(expected_params.keys())
        provided_params = set(config.keys())
        missing_params = required_params - provided_params
        if missing_params:
            raise ValueError(f"Missing required {config_name} parameters: {missing_params}")
        
        return samplers
    
    def _sample_parameters(self, samplers: Dict[str, Any], expected_types: Dict[str, Any], 
                          config_name: str, generator: torch.Generator) -> Dict[str, Any]:
        """Sample parameters from samplers with type validation."""
        sampled_params = {}
        
        for param_name, sampler in samplers.items():
            if param_name == "number_samples_per_dataset":
                # Special case: this is already a PyTorch distribution, not a sampler
                value = sampler
            else:
                value = sampler.sample(generator)
            
            # Type validation (following SCMHyperparameterSampler pattern)
            expected_type = expected_types[param_name]
            if not isinstance(value, expected_type):
                # Special handling for distribution types
                if param_name == "number_samples_per_dataset" and isinstance(value, torch.distributions.Distribution):
                    # This is correct - it's a PyTorch distribution
                    pass
                # Try to convert if possible
                elif isinstance(expected_type, type) and expected_type is int and isinstance(value, float):
                    value = int(value)
                elif isinstance(expected_type, type) and expected_type is float and isinstance(value, int):
                    value = float(value)
                elif isinstance(expected_type, type) and expected_type is tuple and isinstance(value, list):
                    value = tuple(value)
                elif isinstance(expected_type, tuple) and type(None) in expected_type:
                    # Optional parameter - check if value is one of the allowed types
                    allowed_types = [t for t in expected_type if t is not type(None)]
                    if value is not None and not isinstance(value, tuple(allowed_types)):
                        raise ValueError(f"Parameter {config_name}.{param_name} has invalid type. Expected one of {expected_type}, got {type(value)}")
                else:
                    raise ValueError(f"Parameter {config_name}.{param_name} has invalid type. Expected {expected_type}, got {type(value)}")
            
            sampled_params[param_name] = value
        
        return sampled_params
    
    def create_dataset(self, seed: int = None) -> PurelyObservationalDataset:
        """
        Create a PurelyObservationalDataset instance with sampled hyperparameters.
        
        Args:
            seed: Random seed for reproducible dataset creation. If None, no seeding is done.
            
        Returns:
            PurelyObservationalDataset instance
        """
        # Set up generator
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        
        # Sample dataset configuration parameters
        dataset_params = self._sample_parameters(
            self.dataset_samplers, self.EXPECTED_DATASET_HYPERPARAMETERS, "dataset", generator
        )
        
        # Sample preprocessing hyperparameters  
        preprocessing_params = self._sample_parameters(
            self.preprocessing_samplers, self.EXPECTED_PREPROCESSING_HYPERPARAMETERS, "preprocessing", generator
        )
        
        # Create SCMSampler with a derived seed
        scm_seed = None
        if seed is not None:
            scm_seed = (seed * 31 + 17) % (2**32)  # Derive a different seed for SCM
        scm_sampler = SCMSampler(self.scm_config, seed=scm_seed)
        
        # Calculate train/test sample counts from train_fraction
        max_samples = dataset_params["max_number_samples"]
        train_fraction = preprocessing_params["train_fraction"]
        n_train_samples = int(max_samples * train_fraction)
        n_test_samples = max_samples - n_train_samples
        
        # For the dataset, we'll use the sampled number from the distribution
        # but ensure it doesn't exceed our max counts
        max_features = dataset_params["max_number_features"]
        
        # Create BasicProcessing with new parameter structure
        processor = BasicProcessing(
            n_features=max_features,  # We'll use all available features initially
            max_n_features=max_features,
            n_train_samples=n_train_samples,
            max_n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
            max_n_test_samples=n_test_samples,
            dropout_prob=preprocessing_params["dropout_prob"],
            target_feature=preprocessing_params["target_feature"],
            random_seed=preprocessing_params["random_seed"],
            negative_one_one_scaling=preprocessing_params["negative_one_one_scaling"],
            standardize=preprocessing_params["standardize"],
            yeo_johnson=preprocessing_params["yeo_johnson"],
            remove_outliers=preprocessing_params["remove_outliers"],
            outlier_quantile=preprocessing_params["outlier_quantile"],
            shuffle_samples=preprocessing_params["shuffle_data"],
            shuffle_features=True,  # Default from new BasicProcessing
        )
        
        # Get the number of samples per dataset distribution from config
        number_samples_per_dataset_distribution = dataset_params["number_samples_per_dataset"]
        
        # Create and return the dataset
        if seed is not None:
            dataset_seed = (seed * 37 + 23) % (2**32)  # Derive another different seed for dataset
        else:
            dataset_seed = torch.randint(0, 2**32, (1,)).item()  # Generate random seed if none provided
            
        dataset = PurelyObservationalDataset(
            scm_sampler=scm_sampler,
            priordata_processor=processor,
            number_samples_per_dataset_distribution=number_samples_per_dataset_distribution,
            size=dataset_params["dataset_size"],
            max_number_samples=dataset_params["max_number_samples"],
            max_number_features=dataset_params["max_number_features"],
            seed=dataset_seed
        )
        
        return dataset
    
    def _sample_from_config(self, config: Dict[str, Any], seed: int = None) -> Dict[str, Any]:
        """
        Sample hyperparameters from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with distribution specifications
            seed: Random seed for reproducible sampling
            
        Returns:
            Dictionary with sampled parameter values
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        sampled_params = {}
        
        for param_name, param_config in config.items():
            if "value" in param_config:
                # Fixed value
                sampled_params[param_name] = param_config["value"]
            elif "distribution" in param_config:
                # Sample from distribution
                dist_name = param_config["distribution"]
                dist_params = param_config["distribution_parameters"]
                
                if dist_name == "uniform":
                    distribution = torch.distributions.Uniform(
                        low=dist_params["low"], 
                        high=dist_params["high"]
                    )
                    sampled_params[param_name] = distribution.sample().item()
                
                elif dist_name == "discrete_uniform":
                    distribution = torch.distributions.Uniform(
                        low=dist_params["low"], 
                        high=dist_params["high"] + 1
                    )
                    sampled_params[param_name] = int(distribution.sample().item())
                
                elif dist_name == "normal":
                    distribution = torch.distributions.Normal(
                        loc=dist_params["mean"], 
                        scale=dist_params["std"]
                    )
                    sampled_params[param_name] = distribution.sample().item()
                
                elif dist_name == "lognormal":
                    distribution = torch.distributions.LogNormal(
                        loc=dist_params["mean"], 
                        scale=dist_params["std"]
                    )
                    sampled_params[param_name] = distribution.sample().item()
                
                elif dist_name == "beta":
                    distribution = torch.distributions.Beta(
                        concentration1=dist_params["alpha"], 
                        concentration0=dist_params["beta"]
                    )
                    sampled_params[param_name] = distribution.sample().item()
                
                else:
                    raise ValueError(f"Unsupported distribution: {dist_name}")
            
            else:
                raise ValueError(f"Invalid parameter config for {param_name}: {param_config}")
        
        return sampled_params
    
    def get_sampled_params(self, seed: int = None) -> Dict[str, Dict[str, Any]]:
        """
        Get a preview of what parameters would be sampled without creating the dataset.
        
        Args:
            seed: Random seed for reproducible sampling
            
        Returns:
            Dictionary with 'dataset_params', 'preprocessing_params' keys
        """
        # Set up generator
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        
        # Sample parameters
        dataset_params = self._sample_parameters(
            self.dataset_samplers, self.EXPECTED_DATASET_HYPERPARAMETERS, "dataset", generator
        )
        preprocessing_params = self._sample_parameters(
            self.preprocessing_samplers, self.EXPECTED_PREPROCESSING_HYPERPARAMETERS, "preprocessing", generator
        )
        
        # Convert distribution to string representation for readability
        if "number_samples_per_dataset" in dataset_params:
            dist_obj = dataset_params["number_samples_per_dataset"]
            dataset_params["number_samples_per_dataset"] = f"{type(dist_obj).__name__}({dist_obj})"
        
        return {
            "dataset_params": dataset_params,
            "preprocessing_params": preprocessing_params
        }
    
    def get_parameter_summary(self) -> str:
        """
        Get a summary of the configured parameter distributions.
        
        Returns:
            Human-readable summary of all configurations.
        """
        lines = [
            "PurelyObservationalDataset Factory Configuration:",
            "=" * 60,
            ""
        ]
        
        # Dataset configuration
        lines.extend([
            "Dataset Configuration:",
            "-" * 25,
        ])
        for param_name, config in self.dataset_config.items():
            if "value" in config and "distribution" not in config:
                lines.append(f"  {param_name}: Fixed = {config['value']}")
            else:
                dist_type = config["distribution"]
                dist_params = config.get("distribution_parameters", {})
                if dist_type == "fixed":
                    lines.append(f"  {param_name}: Fixed = {config['value']}")
                else:
                    param_str = ", ".join(f"{k}={v}" for k, v in dist_params.items())
                    lines.append(f"  {param_name}: {dist_type}({param_str})")
        
        lines.append("")
        
        # Preprocessing configuration
        lines.extend([
            "Preprocessing Configuration:",
            "-" * 30,
        ])
        for param_name, config in self.preprocessing_config.items():
            if "value" in config and "distribution" not in config:
                lines.append(f"  {param_name}: Fixed = {config['value']}")
            else:
                dist_type = config["distribution"]
                dist_params = config.get("distribution_parameters", {})
                if dist_type == "fixed":
                    lines.append(f"  {param_name}: Fixed = {config['value']}")
                else:
                    param_str = ", ".join(f"{k}={v}" for k, v in dist_params.items())
                    lines.append(f"  {param_name}: {dist_type}({param_str})")
        
        lines.extend([
            "",
            "SCM Configuration: Managed by SCMSampler",
            f"Total dataset parameters configured: {len(self.dataset_samplers)}",
            f"Total preprocessing parameters configured: {len(self.preprocessing_samplers)}",
        ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    """
    Example usage of MakePurelyObservationalDataset with rigorous validation
    """
    print("Testing MakePurelyObservationalDataset with rigorous configuration...")
    
    try:
        # Import the default configs
        from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config
        from priordata_processing.Datasets.ExampleConfigs.BasicConfigs import default_dataset_config, default_preprocessing_config
        
        # Create the dataset factory
        factory = MakePurelyObservationalDataset(
            scm_config=default_sampling_config,
            preprocessing_config=default_preprocessing_config,
            dataset_config=default_dataset_config
        )
        
        # Show configuration summary
        print("\n" + factory.get_parameter_summary())
        
        # Preview what parameters would be sampled
        print("\n1. Previewing sampled parameters...")
        sampled_params = factory.get_sampled_params(seed=42)
        
        print("Dataset Parameters:")
        for key, value in sampled_params["dataset_params"].items():
            print(f"  {key}: {value}")
        
        print("\nPreprocessing Parameters:")
        for key, value in sampled_params["preprocessing_params"].items():
            print(f"  {key}: {value}")
        
        # Create a dataset for testing
        print("\n2. Creating dataset...")
        dataset = factory.create_dataset(seed=42)
        
        print(f"✓ Created dataset with {len(dataset)} items")
        
        # Test accessing an item
        print("\n3. Testing dataset access...")
        item = dataset[0]
        data_dict, metadata = item
        
        print("✓ Successfully retrieved item 0")
        print(f"✓ Data keys: {list(data_dict.keys())}")
        print(f"✓ X_train shape: {data_dict['X_train'].shape}")
        print(f"✓ Y_train shape: {data_dict['Y_train'].shape}")
        print(f"✓ X_test shape: {data_dict['X_test'].shape}")
        print(f"✓ Y_test shape: {data_dict['Y_test'].shape}")
        print(f"✓ Metadata keys: {len(metadata)} keys")
        
        print("\n" + "="*60)
        print("Rigorous MakePurelyObservationalDataset works correctly!")
        print("="*60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
