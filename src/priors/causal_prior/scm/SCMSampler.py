"""
SCMSampler - A high-level interface for sampling Structural Causal Models (SCMs).

This class provides a convenient wrapper around the SCM sampling pipeline,
taking a hyperparameter configuration and producing a fully built SCM ready for data generation.
"""

from typing import Dict, Any, Optional
import warnings

from priors.causal_prior.scm.SCMHyperparameterSampler import SCMHyperparameterSampler
from priors.causal_prior.scm.SCMBuilder import SCMBuilder
from priors.causal_prior.scm.SCM import SCM


class SCMSampler:
    """
    A high-level sampler for Structural Causal Models (SCMs).
    
    This class encapsulates the complete SCM sampling pipeline:
    1. Takes a hyperparameter configuration
    2. Samples specific hyperparameters according to the configuration
    3. Builds and returns a complete SCM ready for data generation
    
    The resulting SCM can be used to generate datasets by calling:
    - scm.sample_exogenous(num_samples)
    - scm.sample_endogenous_noise(num_samples) 
    - data = scm.propagate(num_samples)
    
    Args:
        hyperparameter_config (Dict[str, Any]): Configuration dictionary specifying
                                               distributions over SCM hyperparameters
        seed (int, optional): Random seed for reproducible SCM sampling. Defaults to None
        verbose (bool): Whether to print detailed information during sampling. Defaults to False
        
    Example:
        >>> from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config
        >>> sampler = SCMSampler(default_sampling_config, seed=42)
        >>> scm = sampler.sample()
        >>> 
        >>> # Generate data from the sampled SCM
        >>> scm.sample_exogenous(num_samples=100)
        >>> scm.sample_endogenous_noise(num_samples=100)
        >>> data = scm.propagate(num_samples=100)
        >>> print(f"Generated data with {len(data)} features")
        
    Attributes:
        hyperparameter_config (Dict[str, Any]): The configuration used for sampling
        seed (int, optional): Random seed used for sampling
        verbose (bool): Verbosity flag
        last_sampled_params (Dict[str, Any], optional): Parameters from the most recent sampling
    """
    
    def __init__(
        self, 
        hyperparameter_config: Dict[str, Any],
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize the SCMSampler with the given configuration."""
        self.hyperparameter_config = hyperparameter_config
        self.seed = seed
        self.verbose = verbose
        self.last_sampled_params: Optional[Dict[str, Any]] = None
        
        # Validate the configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate that the hyperparameter configuration is properly formatted."""
        if not isinstance(self.hyperparameter_config, dict):
            raise TypeError("hyperparameter_config must be a dictionary")
        
        if not self.hyperparameter_config:
            raise ValueError("hyperparameter_config cannot be empty")
        
        # Check for required keys (basic validation)
        required_keys = ["num_nodes"]  # At minimum, we need to specify number of nodes
        missing_keys = [key for key in required_keys if key not in self.hyperparameter_config]
        if missing_keys:
            warnings.warn(f"Configuration missing recommended keys: {missing_keys}")
    
    def sample(self, seed: Optional[int] = None) -> SCM:
        """
        Sample a complete SCM according to the hyperparameter configuration.
        
        Args:
            seed (int, optional): Random seed for this sampling. If None, uses the instance seed.
                                If instance seed is also None, sampling will be non-deterministic.
                                
        Returns:
            SCM: A fully constructed Structural Causal Model ready for data generation
            
        Raises:
            RuntimeError: If SCM construction fails due to invalid parameters
            ValueError: If the sampled parameters are inconsistent
        """
        # Determine the seed to use
        sampling_seed = seed if seed is not None else self.seed
        
        if self.verbose:
            print("Sampling SCM with configuration...")
            if sampling_seed is not None:
                print(f"Using seed: {sampling_seed}")
            else:
                print("Using non-deterministic sampling (no seed specified)")
        
        try:
            # Step 1: Sample hyperparameters according to the configuration
            hyperparameter_sampler = SCMHyperparameterSampler(
                self.hyperparameter_config, 
                seed=sampling_seed
            )
            
            if self.verbose:
                print("Hyperparameter sampler created successfully")
                print("Parameter summary:")
                try:
                    summary = hyperparameter_sampler.get_parameter_summary()
                    print(summary)
                except Exception as e:
                    print(f"Could not get parameter summary: {e}")
            
            # Sample the specific parameters
            sampled_params = hyperparameter_sampler.sample()
            self.last_sampled_params = sampled_params.copy()  # Store for inspection
            
            if self.verbose:
                print(f"Sampled parameters for {sampled_params.get('num_nodes', '?')} nodes")
                print("Key parameters:")
                for key, value in sampled_params.items():
                    if key in ['num_nodes', 'graph_edge_prob', 'xgboost_prob']:
                        print(f"  {key}: {value}")
            
            # Step 2: Build the SCM using the sampled parameters
            builder = SCMBuilder(**sampled_params)
            scm = builder.build()
            
            if self.verbose:
                print("SCM built successfully!")
                # Try to get some basic info about the SCM
                try:
                    num_nodes = len(scm.dag.nodes())
                    num_edges = len(scm.dag.edges())
                    print(f"SCM has {num_nodes} nodes and {num_edges} edges")
                except Exception as e:
                    print(f"Could not get SCM structure info: {e}")
            
            return scm
            
        except Exception as e:
            error_msg = f"Failed to sample SCM: {str(e)}"
            if self.verbose:
                print(f"Error: {error_msg}")
                import traceback
                traceback.print_exc()
            raise RuntimeError(error_msg) from e
    
    def sample_multiple(
        self, 
        count: int, 
        base_seed: Optional[int] = None
    ) -> list[SCM]:
        """
        Sample multiple SCMs with different random seeds.
        
        Args:
            count (int): Number of SCMs to sample
            base_seed (int, optional): Base seed for deterministic sampling. 
                                     Each SCM will use base_seed + i as its seed.
                                     If None, uses instance seed as base.
                                     
        Returns:
            List[SCM]: List of sampled SCMs
            
        Raises:
            ValueError: If count is not positive
        """
        if count <= 0:
            raise ValueError("Count must be positive")
        
        # Determine base seed
        if base_seed is None:
            base_seed = self.seed
        
        if self.verbose:
            print(f"Sampling {count} SCMs...")
            if base_seed is not None:
                print(f"Using base seed: {base_seed}")
        
        scms = []
        for i in range(count):
            if self.verbose:
                print(f"\nSampling SCM {i+1}/{count}...")
            
            # Use deterministic seed if base_seed is available
            scm_seed = base_seed + i if base_seed is not None else None
            scm = self.sample(seed=scm_seed)
            scms.append(scm)
        
        if self.verbose:
            print(f"\nSuccessfully sampled {len(scms)} SCMs")
        
        return scms
    
    def get_last_sampled_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the parameters from the most recent call to sample().
        
        Returns:
            Dict[str, Any] or None: The sampled parameters, or None if sample() hasn't been called
        """
        return self.last_sampled_params.copy() if self.last_sampled_params else None
    
    def get_config_summary(self) -> str:
        """
        Get a summary of the hyperparameter configuration.
        
        Returns:
            str: Human-readable summary of the configuration
        """
        summary_lines = ["SCMSampler Configuration Summary:"]
        summary_lines.append(f"  Total parameters: {len(self.hyperparameter_config)}")
        summary_lines.append(f"  Seed: {self.seed}")
        summary_lines.append(f"  Verbose: {self.verbose}")
        summary_lines.append("")
        summary_lines.append("Key configuration parameters:")
        
        # Show important parameters
        important_keys = [
            'num_nodes', 'graph_edge_prob', 'xgboost_prob', 
            'mlp_num_hidden_layers', 'mlp_hidden_dim',
            'random_additive_std', 'scm_fast'
        ]
        
        for key in important_keys:
            if key in self.hyperparameter_config:
                value = self.hyperparameter_config[key]
                summary_lines.append(f"  {key}: {value}")
        
        # Count distribution vs value parameters
        dist_params = sum(1 for v in self.hyperparameter_config.values() 
                         if isinstance(v, dict) and 'distribution' in v)
        value_params = sum(1 for v in self.hyperparameter_config.values() 
                          if isinstance(v, dict) and 'value' in v)
        
        summary_lines.append("")
        summary_lines.append("Parameter types:")
        summary_lines.append(f"  Distribution parameters: {dist_params}")
        summary_lines.append(f"  Fixed value parameters: {value_params}")
        
        return "\n".join(summary_lines)


def create_scm_sampler_from_config(
    config_name: str = "default",
    seed: Optional[int] = None,
    verbose: bool = False
) -> SCMSampler:
    """
    Convenience function to create an SCMSampler from a predefined configuration.
    
    Args:
        config_name (str): Name of the configuration to use. Currently supports:
                          - "default": Uses default_sampling_config
        seed (int, optional): Random seed for the sampler
        verbose (bool): Whether to enable verbose output
        
    Returns:
        SCMSampler: Configured SCMSampler instance
        
    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name == "default":
        from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config
        config = default_sampling_config
    else:
        raise ValueError(f"Unknown config name: {config_name}. Supported: 'default'")
    
    return SCMSampler(config, seed=seed, verbose=verbose)


# Example usage demonstration
def _example_usage():
    """Example of how to use SCMSampler."""
    print("=== SCMSampler Usage Example ===\n")
    
    # Method 1: Using predefined config
    print("1. Creating SCMSampler with default configuration...")
    sampler = create_scm_sampler_from_config("default", seed=42, verbose=True)
    print(sampler.get_config_summary())
    
    print("\n2. Sampling a single SCM...")
    scm = sampler.sample()
    
    print("\n3. Generating data from the SCM...")
    N_SAMPLES = 100
    scm.sample_exogenous(num_samples=N_SAMPLES)
    scm.sample_endogenous_noise(num_samples=N_SAMPLES)
    data = scm.propagate(num_samples=N_SAMPLES)
    
    print(f"Generated dataset with {len(data)} features and {N_SAMPLES} samples")
    for i, (feature_name, feature_data) in enumerate(data.items()):
        if i < 3:  # Show first 3 features
            print(f"  Feature '{feature_name}': shape {feature_data.shape}, "
                  f"mean={feature_data.mean():.3f}, std={feature_data.std():.3f}")
    
    print("\n4. Sampling multiple SCMs...")
    scms = sampler.sample_multiple(3, base_seed=100)
    print(f"Sampled {len(scms)} different SCMs")
    
    print("\n=== Example completed! ===")


if __name__ == "__main__":
    _example_usage()
