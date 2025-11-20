from __future__ import annotations
from typing import Dict, Any, Callable, Optional
import math

import torch
import torch.distributions as dist

from priordata_processing.Datasets.ObservationalDataset import ObservationalDataset
from priordata_processing.BasicProcessing import BasicProcessing
from priors.causal_prior.scm.SCMSampler import SCMSampler
from utils import (
    FixedSampler,
    TorchDistributionSampler,
    CategoricalSampler,
    DiscreteUniformSampler,
)


# Top-level helpers (picklable) to avoid local/closure functions
def _top_level_sample_pair(sampler_t0: Any, sampler_t1: Any, generator: torch.Generator):
    v0 = sampler_t0.sample(generator)
    v1 = sampler_t1.sample(generator)
    if torch.is_tensor(v0) and v0.numel() == 1:
        v0 = v0.item()
    if torch.is_tensor(v1) and v1.numel() == 1:
        v1 = v1.item()
    return v0, v1

def _choose_by_alpha(v0: Any, v1: Any, a: float) -> Any:
    if torch.rand(()) < a:
        return v1
    return v0


class InterpolatedObservationalDataset(ObservationalDataset):
    """
    Dataset that interpolates between two configurations (t0 and t1) based on curriculum time.
    
    Inherits from ObservationalDataset and adds interpolation logic:
    - Each dataset item is assigned a curriculum time t = idx / (size - 1)
    - An interpolation function alpha(t) determines the probability of choosing config t1
    - For each hyperparameter, we sample from both configs and choose based on alpha(t)
    
    Parameters
    ----------
    scm_config_t0, scm_config_t1 : Dict[str, Any]
        SCM configurations for initial and final states.
        Supports all SCMSampler parameters including endo_p_zero for controlling
        the probability of zero endogenous noise.
    preprocessing_config_t0, preprocessing_config_t1 : Dict[str, Any]
        Preprocessing configurations for initial and final states
    dataset_config_t0, dataset_config_t1 : Dict[str, Any]
        Dataset configurations for initial and final states
    interpolation_function : str, default "sigmoid"
        Interpolation function: "linear", "sigmoid", "step", "constant", "immediate"
        - "immediate" or "immediate_jump": α(t<0.001) = 0, α(t>=0.001) = 1
    seed : Optional[int], default None
        Random seed for reproducibility
        
    Examples
    --------
    >>> # Curriculum from dense noise (p_zero=0.0) to sparse noise (p_zero=0.5)
    >>> scm_config_t0 = {"endo_p_zero": {"value": 0.0}, ...}
    >>> scm_config_t1 = {"endo_p_zero": {"value": 0.5}, ...}
    >>> dataset = InterpolatedObservationalDataset(
    ...     scm_config_t0, scm_config_t1,
    ...     preprocessing_config_t0, preprocessing_config_t1,
    ...     dataset_config_t0, dataset_config_t1
    ... )
    """
    
    def __init__(
        self,
        scm_config_t0: Dict[str, Any],
        scm_config_t1: Dict[str, Any],
        preprocessing_config_t0: Dict[str, Any],
        preprocessing_config_t1: Dict[str, Any],
        dataset_config_t0: Dict[str, Any],
        dataset_config_t1: Dict[str, Any],
        interpolation_function: str = "sigmoid",
        seed: Optional[int] = None,
    ):
        # Validate matching keys
        self._validate_matching_keys(scm_config_t0, scm_config_t1, "scm")
        self._validate_matching_keys(preprocessing_config_t0, preprocessing_config_t1, "preprocessing")
        self._validate_matching_keys(dataset_config_t0, dataset_config_t1, "dataset")
        
        # Validate dataset_size matches
        ds_size0 = dataset_config_t0.get("dataset_size", {}).get("value")
        ds_size1 = dataset_config_t1.get("dataset_size", {}).get("value")
        if ds_size0 != ds_size1:
            raise ValueError("dataset_size must match between t0 and t1 configs")
        
        # Initialize base class with t0 configuration
        super().__init__(
            scm_config=scm_config_t0,
            preprocessing_config=preprocessing_config_t0,
            dataset_config=dataset_config_t0,
            seed=seed
        )
        
        # Store t1 configurations
        self.scm_config_t1 = scm_config_t1
        self.preprocessing_config_t1 = preprocessing_config_t1
        self.dataset_config_t1 = dataset_config_t1
        self.interpolation_function = interpolation_function
        
        # Rename base class samplers to be explicit about t0
        self.preprocessing_samplers_t0 = self.preprocessing_samplers
        self.dataset_samplers_t0 = self.dataset_samplers
        self.scm_sampler_t0 = self.scm_sampler
        
        # Build samplers for t1 configurations
        self.preprocessing_samplers_t1 = self._build_samplers(
            self.preprocessing_config_t1,
            self.EXPECTED_PREPROCESSING_HYPERPARAMETERS,
            "preprocessing"
        )
        
        dataset_config_t1_filtered = {k: v for k, v in self.dataset_config_t1.items() if k != 'seed'}
        self.dataset_samplers_t1 = self._build_samplers(
            dataset_config_t1_filtered,
            self.EXPECTED_DATASET_HYPERPARAMETERS,
            "dataset"
        )
        
        # Create second SCMSampler for t1
        scm_seed_t1 = None
        if seed is not None:
            scm_seed_t1 = (seed * 37 + 23) % (2**32)  # Different seed for t1
        self.scm_sampler_t1 = SCMSampler(scm_config_t1, seed=scm_seed_t1)
    
    @staticmethod
    def _validate_matching_keys(c0: Dict[str, Any], c1: Dict[str, Any], name: str) -> None:
        """Ensure both configs have the same keys."""
        k0 = set(c0.keys())
        k1 = set(c1.keys())
        if k0 != k1:
            missing0 = k1 - k0
            missing1 = k0 - k1
            raise ValueError(
                f"Config key mismatch in {name}: missing_in_t0={missing0}, missing_in_t1={missing1}"
            )
    
    def _time_for_index(self, idx: int) -> float:
        """Calculate curriculum time t ∈ [0, 1] for a given index."""
        if self.size <= 1:
            return 1.0
        return float(idx) / float(self.size - 1)
    
    def _alpha(self, t: float) -> float:
        """
        Calculate interpolation probability α(t) ∈ [0, 1].
        
        α(t) determines the probability of choosing config t1 over t0.
        """
        s = (self.interpolation_function or "sigmoid").strip().lower()
        if s.endswith('.'):
            s = s[:-1]
        
        if s == "linear":
            return float(max(0.0, min(1.0, t)))
        
        if s.startswith("step"):
            threshold = 0.5
            if '_' in s:
                try:
                    threshold = float(s.split('_', 1)[1])
                except ValueError:
                    threshold = 0.5
            return 0.0 if t < threshold else 1.0
        
        if s == "immediate" or s == "immediate_jump":
            # Jump to 1 only after t >= 0.001
            return 0.0 if t < 0.001 else 1.0
        
        if s.startswith("constant"):
            p = 1.0
            if '_' in s:
                try:
                    p = float(s.split('_', 1)[1])
                except ValueError:
                    p = 1.0
            return float(max(0.0, min(1.0, p)))
        
        if s.startswith("sigmoid") or True:  # Default to sigmoid
            # Sigmoid with clamped first/last 10%
            if t <= 0.1:
                return 0.0
            if t >= 0.9:
                return 1.0
            # Normalize t to [0,1] within (0.1, 0.9)
            x = (t - 0.1) / 0.8
            # Sigmoid centered at 0.5 with moderate slope
            k = 8.0
            z = k * (x - 0.5)
            raw = 1.0 / (1.0 + math.exp(-z))
            low = 1.0 / (1.0 + math.exp(k * 0.5))
            high = 1.0 / (1.0 + math.exp(-k * 0.5))
            denom = (high - low) if (high - low) != 0.0 else 1.0
            a = (raw - low) / denom
            return float(max(0.0, min(1.0, a)))
    
    def _interpolate_sample(self, sampler_t0: Any, sampler_t1: Any, 
                          alpha: float, generator: torch.Generator) -> Any:
        """
        Sample from both samplers and choose based on alpha.
        
        With probability α, choose the sample from t1, else choose from t0.
        """
        # Sample from both - handle different sampler types
        if isinstance(sampler_t0, torch.distributions.Distribution):
            # PyTorch distributions use global random state, not generator
            v0 = sampler_t0.sample()
        elif hasattr(sampler_t0, 'sample'):
            # Custom samplers that accept generator
            v0 = sampler_t0.sample(generator)
        else:
            # Fixed value
            v0 = sampler_t0
        
        if isinstance(sampler_t1, torch.distributions.Distribution):
            # PyTorch distributions use global random state, not generator
            v1 = sampler_t1.sample()
        elif hasattr(sampler_t1, 'sample'):
            # Custom samplers that accept generator
            v1 = sampler_t1.sample(generator)
        else:
            # Fixed value
            v1 = sampler_t1
        
        # Unwrap tensor values
        if torch.is_tensor(v0) and v0.numel() == 1:
            v0 = v0.item()
        if torch.is_tensor(v1) and v1.numel() == 1:
            v1 = v1.item()
        
        # Choose based on alpha
        if torch.rand((), generator=generator) < alpha:
            return v1
        return v0
    
    def __getitem__(self, idx):
        """
        Sample a dataset item with curriculum-based interpolation.
        
        Returns the same format as ObservationalDataset plus curriculum info:
        (X_train, Y_train, X_test, Y_test, t, alpha)
        """
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
        
        # Calculate curriculum time and interpolation probability
        t = self._time_for_index(idx)
        alpha = self._alpha(t)
        
        seed = self.seed + idx if self.seed is not None else idx
        torch.manual_seed(seed)
        
        # Create generator for this item
        item_generator = torch.Generator()
        item_generator.manual_seed(seed)
        
        # Sample preprocessing parameters (interpolated)
        preprocessing_params = {}
        for param_name in self.preprocessing_samplers_t0.keys():
            preprocessing_params[param_name] = self._interpolate_sample(
                self.preprocessing_samplers_t0[param_name],
                self.preprocessing_samplers_t1[param_name],
                alpha,
                item_generator
            )
        
        # Validate types
        for param_name, value in preprocessing_params.items():
            expected_type = self.EXPECTED_PREPROCESSING_HYPERPARAMETERS[param_name]
            if isinstance(expected_type, type) and expected_type is int and isinstance(value, float):
                preprocessing_params[param_name] = int(value)
            elif isinstance(expected_type, type) and expected_type is float and isinstance(value, int):
                preprocessing_params[param_name] = float(value)
            elif isinstance(expected_type, type) and expected_type is tuple and isinstance(value, list):
                preprocessing_params[param_name] = tuple(value)
        
        # Sample dataset parameters (interpolated)
        dataset_params = {}
        for param_name in self.dataset_samplers_t0.keys():
            dataset_params[param_name] = self._interpolate_sample(
                self.dataset_samplers_t0[param_name],
                self.dataset_samplers_t1[param_name],
                alpha,
                item_generator
            )
        
        # Extract sample counts
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
        
        # Sample SCM (interpolated between t0 and t1)
        # Choose which SCM sampler to use based on alpha
        if torch.rand((), generator=item_generator) < alpha:
            scm = self.scm_sampler_t1.sample(seed=seed)
        else:
            scm = self.scm_sampler_t0.sample(seed=seed)
        
        # Total samples needed
        total_samples = number_train_samples + number_test_samples
        
        # Generate data from SCM
        scm.sample_exogenous(num_samples=total_samples)
        scm.sample_endogenous(num_samples=total_samples)
        scm_data = scm.propagate(num_samples=total_samples)
        
        # Convert to format expected by BasicProcessing
        dataset = {}
        for key, value in scm_data.items():
            dataset[key] = value.reshape(total_samples, -1)
        
        # Create BasicProcessing with sampled preprocessing parameters
        processor = BasicProcessing(
            n_features=self.max_number_features,
            max_n_features=self.max_number_features,
            n_train_samples=number_train_samples,
            max_n_train_samples=self.max_number_train_samples,
            n_test_samples=number_test_samples,
            max_n_test_samples=self.max_number_test_samples,
            dropout_prob=preprocessing_params["dropout_prob"],
            target_feature=preprocessing_params["target_feature"],
            random_seed=preprocessing_params["random_seed"],
            negative_one_one_scaling=preprocessing_params["negative_one_one_scaling"],
            standardize=preprocessing_params["standardize"],
            yeo_johnson=preprocessing_params["yeo_johnson"],
            remove_outliers=preprocessing_params["remove_outliers"],
            outlier_quantile=preprocessing_params["outlier_quantile"],
            shuffle_samples=preprocessing_params["shuffle_data"],
            shuffle_features=True,
            y_clip_quantile=preprocessing_params.get("y_clip_quantile"),
            eps=preprocessing_params.get("eps", 1e-8),
            device=None,
            dtype=None,
        )
        
        # Process the data
        X_train, Y_train, X_test, Y_test = processor.process(dataset)
        
        # Return with curriculum information
        return (
            X_train, 
            Y_train, 
            X_test, 
            Y_test,
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(alpha, dtype=torch.float32),
        )

