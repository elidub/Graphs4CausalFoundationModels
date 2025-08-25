from __future__ import annotations
from typing import Dict, Any, Optional, List
import torch
import torch.distributions as dist
import sys
import os

from priors.causal_prior.scm.SCMBuilder import SCMBuilder

# Import from the main utils module (not training.utils)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils import FixedSampler, TorchDistributionSampler, CategoricalSampler, DiscreteUniformSampler, DistributionSampler


class SCMHyperparameterSampler:
    """
    Hyperparameter sampler for SCMBuilder.
    
    This class allows sampling all hyperparameters needed for SCMBuilder from
    specified distributions. Each hyperparameter can either be fixed or sampled
    from a distribution.
    
    Parameters
    ----------
    hyperparameter_config : Dict[str, Dict[str, Any]]
        Configuration dictionary where keys are hyperparameter names and values
        specify the distribution and its parameters.
        
        Format:
        {
            "parameter_name": {
                "distribution": "distribution_type",
                "distribution_parameters": {...}
            },
            "fixed_parameter": {
                "value": fixed_value
            }
        }
        
        Supported distributions:
        - "fixed": Fixed value (alternative: just specify "value" key)
        - "uniform": Uniform distribution (parameters: low, high)
        - "normal": Normal distribution (parameters: mean, std)
        - "lognormal": Log-normal distribution (parameters: mean, std)
        - "exponential": Exponential distribution (parameters: lambd)
        - "gamma": Gamma distribution (parameters: alpha, beta)
        - "beta": Beta distribution (parameters: alpha, beta)
        - "categorical": Categorical choice (parameters: choices, probabilities?)
        - "discrete_uniform": Discrete uniform (parameters: low, high)
    
    seed : Optional[int], default None
        Random seed for reproducibility.
    
    Examples
    --------
    >>> config = {
    ...     "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 3, "high": 10}},
    ...     "graph_edge_prob": {"distribution": "beta", "distribution_parameters": {"alpha": 2, "beta": 5}},
    ...     "graph_seed": {"value": 42},  # Fixed value
    ...     "xgboost_prob": {"distribution": "uniform", "distribution_parameters": {"low": 0.0, "high": 0.5}},
    ...     "mlp_nonlins": {"distribution": "categorical", "distribution_parameters": {
    ...         "choices": ["tanh", "sin", "tabicl"], "probabilities": [0.3, 0.2, 0.5]
    ...     }},
    ...     "mlp_num_hidden_layers": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 0, "high": 3}},
    ...     "mlp_hidden_dim": {"distribution": "categorical", "distribution_parameters": {"choices": [16, 32, 64]}},
    ...     "random_additive_std": {"distribution": "categorical", "distribution_parameters": {"choices": [True, False]}},
    ...     "exo_std": {"distribution": "lognormal", "distribution_parameters": {"mean": 0.0, "std": 0.5}},
    ...     "endo_std": {"distribution": "lognormal", "distribution_parameters": {"mean": -1.0, "std": 0.3}}
    ... }
    >>> 
    >>> sampler = SCMHyperparameterSampler(config, seed=123)
    >>> hyperparams = sampler.sample()
    >>> builder = SCMBuilder(**hyperparams)
    >>> scm = builder.build()
    
    >>> # Sample multiple configurations
    >>> configs = sampler.sample_batch(10)
    >>> for i, config in enumerate(configs):
    ...     builder = SCMBuilder(**config)
    ...     print(f"Configuration {i}: {builder.get_config_summary()}")
    """
    
    # Define SCMBuilder's expected hyperparameters and their types for validation
    EXPECTED_HYPERPARAMETERS = {
        # Required parameters
        "num_nodes": int,
        "graph_edge_prob": float,
        "graph_seed": int,
        
        # Optional parameters with defaults
        "xgboost_prob": float,
        "mechanism_seed": int,
        "mlp_nonlins": str,
        "mlp_num_hidden_layers": int,
        "mlp_hidden_dim": int,
        "mlp_activation_mode": str,
        "mlp_node_shape": tuple,
        "xgb_num_hidden_layers": int,
        "xgb_hidden_dim": int,
        "xgb_activation_mode": str,
        "xgb_node_shape": tuple,
        "xgb_n_training_samples": int,
        "xgb_add_noise": bool,
        "random_additive_std": bool,
        "exo_std_distribution": str,  # New: "gamma" or "pareto"
        "endo_std_distribution": str,  # New: "gamma" or "pareto"
        "exo_std": (float, type(None)),
        "endo_std": (float, type(None)),
        "exo_std_mean": (float, type(None)),
        "exo_std_std": (float, type(None)),
        "endo_std_mean": (float, type(None)),
        "endo_std_std": (float, type(None)),
        "scm_fast": bool,
        "use_exogenous_mechanisms": bool,
        "mechanism_generator_seed": int,
    }
    
    # Distribution factory mapping
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
    
    def __init__(
        self,
        hyperparameter_config: Dict[str, Dict[str, Any]],
        seed: Optional[int] = None
    ):
        self.hyperparameter_config = hyperparameter_config
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        
        # Parse and validate configuration
        self.samplers = self._build_samplers()
    
    def _build_samplers(self) -> Dict[str, DistributionSampler]:
        """Build sampler objects from the configuration."""
        samplers = {}
        
        for param_name, config in self.hyperparameter_config.items():
            # Check if parameter is known
            if param_name not in self.EXPECTED_HYPERPARAMETERS:
                raise ValueError(f"Unknown hyperparameter: {param_name}")
            
            # Handle shorthand fixed value notation
            if "value" in config and "distribution" not in config:
                sampler = FixedSampler(config["value"])
            elif "distribution" in config:
                dist_type = config["distribution"]
                if dist_type not in self.DISTRIBUTION_FACTORIES:
                    raise ValueError(f"Unknown distribution type: {dist_type}")
                
                # Get distribution parameters
                dist_params = config.get("distribution_parameters", {})
                if dist_type == "fixed":
                    if "value" not in config:
                        raise ValueError(f"Fixed distribution for {param_name} requires 'value' key")
                    dist_params = {"value": config["value"]}
                
                # Create sampler
                try:
                    sampler = self.DISTRIBUTION_FACTORIES[dist_type](dist_params)
                except Exception as e:
                    raise ValueError(f"Error creating sampler for {param_name}: {e}")
            else:
                raise ValueError(f"Configuration for {param_name} must specify 'distribution' or 'value'")
            
            samplers[param_name] = sampler
        
        return samplers
    
    def sample(self) -> Dict[str, Any]:
        """
        Sample a single set of hyperparameters.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of hyperparameters ready to be passed to SCMBuilder.
        """
        sampled_params = {}
        
        for param_name, sampler in self.samplers.items():
            value = sampler.sample(self.generator)
            
            # Type validation
            expected_type = self.EXPECTED_HYPERPARAMETERS[param_name]
            if not isinstance(value, expected_type):
                # Try to convert if possible
                if isinstance(expected_type, type) and expected_type is int and isinstance(value, float):
                    value = int(value)
                elif isinstance(expected_type, type) and expected_type is float and isinstance(value, int):
                    value = float(value)
                elif isinstance(expected_type, type) and expected_type is tuple and isinstance(value, list):
                    value = tuple(value)
                elif isinstance(expected_type, tuple) and type(None) in expected_type:
                    # Optional parameter - check if value is one of the allowed types
                    allowed_types = [t for t in expected_type if t is not type(None)]
                    if value is not None and not isinstance(value, tuple(allowed_types)):
                        raise ValueError(f"Parameter {param_name} has invalid type. Expected one of {expected_type}, got {type(value)}")
                else:
                    raise ValueError(f"Parameter {param_name} has invalid type. Expected {expected_type}, got {type(value)}")
            
            sampled_params[param_name] = value
        
        return sampled_params
    
    def sample_batch(self, n: int) -> List[Dict[str, Any]]:
        """
        Sample multiple sets of hyperparameters.
        
        Parameters
        ----------
        n : int
            Number of hyperparameter sets to sample.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of hyperparameter dictionaries.
        """
        return [self.sample() for _ in range(n)]
    
    def sample_scm_builders(self, n: int) -> List[SCMBuilder]:
        """
        Sample multiple SCMBuilder instances.
        
        Parameters
        ----------
        n : int
            Number of SCMBuilder instances to create.
            
        Returns
        -------
        List[SCMBuilder]
            List of configured SCMBuilder instances.
        """
        configs = self.sample_batch(n)
        return [SCMBuilder(**config) for config in configs]
    
    def get_parameter_summary(self) -> str:
        """
        Get a summary of the configured parameter distributions.
        
        Returns
        -------
        str
            Human-readable summary of the sampling configuration.
        """
        lines = [
            "SCM Hyperparameter Sampling Configuration:",
            "=" * 50,
            ""
        ]
        
        for param_name, config in self.hyperparameter_config.items():
            if "value" in config and "distribution" not in config:
                lines.append(f"{param_name}: Fixed = {config['value']}")
            else:
                dist_type = config["distribution"]
                dist_params = config.get("distribution_parameters", {})
                if dist_type == "fixed":
                    lines.append(f"{param_name}: Fixed = {config['value']}")
                else:
                    param_str = ", ".join(f"{k}={v}" for k, v in dist_params.items())
                    lines.append(f"{param_name}: {dist_type}({param_str})")
        
        lines.extend([
            "",
            f"Random seed: {self.seed}",
            f"Total parameters configured: {len(self.samplers)}",
        ])
        
        return "\n".join(lines)



if __name__ == "__main__":
    # Example usage
    print("SCM Hyperparameter Sampler Example")
    print("=" * 40)
    
    # Create a simple configuration
    config = {
        "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 3, "high": 6}},
        "graph_edge_prob": {"distribution": "uniform", "distribution_parameters": {"low": 0.2, "high": 0.6}},
        "graph_seed": {"value": 42},  # Fixed value
        "xgboost_prob": {"distribution": "uniform", "distribution_parameters": {"low": 0.0, "high": 0.3}},
        "mlp_nonlins": {"distribution": "categorical", "distribution_parameters": {"choices": ["tanh", "tabicl"]}},
    }
    
    # Create sampler
    sampler = SCMHyperparameterSampler(config, seed=123)
    print(sampler.get_parameter_summary())
    print()
    
    # Sample some configurations
    print("Sample configurations:")
    for i in range(3):
        params = sampler.sample()
        print(f"Config {i+1}: {params}")
    
    print("\nTesting with SCMBuilder...")
    try:
        # Test that sampled parameters work with SCMBuilder
        sampled_params = sampler.sample()
        builder = SCMBuilder(**sampled_params)
        scm = builder.build()
        samples = builder.build_and_sample(10)
        print(f"✓ Successfully created SCM with {len(samples)} nodes and sampled 10 data points")
    except Exception as e:
        print(f"✗ Error: {e}")
