from __future__ import annotations
from typing import Dict, Any, Optional, List
import torch
import torch.distributions as dist
from abc import ABC, abstractmethod


class DistributionSampler(ABC):
    """Abstract base class for distribution samplers."""
    
    @abstractmethod
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        """Sample a value from this distribution."""
        pass


class FixedSampler(DistributionSampler):
    """Sampler that always returns a fixed value."""
    
    def __init__(self, value: Any):
        self.value = value
    
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        return self.value


class TorchDistributionSampler(DistributionSampler):
    """Wrapper for torch.distributions samplers."""
    
    def __init__(self, distribution: dist.Distribution):
        self.distribution = distribution
    
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        if generator is not None:
            # Use the generator for sampling
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                value = self.distribution.sample()
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            value = self.distribution.sample()
        
        # Convert to appropriate Python type
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            else:
                return value.tolist()
        return value


class CategoricalSampler(DistributionSampler):
    """Categorical (choice) sampler using torch.distributions."""
    
    def __init__(self, choices: List[Any], probabilities: Optional[List[float]] = None):
        self.choices = choices
        if probabilities is not None:
            if len(probabilities) != len(choices):
                raise ValueError("Length of probabilities must match length of choices")
            self.categorical = dist.Categorical(torch.tensor(probabilities))
        else:
            # Uniform probabilities
            uniform_probs = torch.ones(len(choices)) / len(choices)
            self.categorical = dist.Categorical(uniform_probs)
    
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        if generator is not None:
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                idx = self.categorical.sample()
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            idx = self.categorical.sample()
        
        return self.choices[idx.item()]


class DiscreteUniformSampler(DistributionSampler):
    """Discrete uniform distribution sampler (integers) using torch."""
    
    def __init__(self, low: int, high: int):
        self.low = low
        self.high = high
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")
    
    def sample(self, generator: Optional[torch.Generator] = None) -> int:
        if generator is not None:
            old_generator = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                value = torch.randint(self.low, self.high + 1, (1,))
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_generator)
        else:
            value = torch.randint(self.low, self.high + 1, (1,))
        
        return int(value.item())


class DiscreteLogUniformSampler(CategoricalSampler):
    """Discrete log-uniform sampler over integers in [low, high].
    
    P(k) ∝ 1/k, so each integer is weighted by its log-bin width.
    Optional cutoffs further restrict the support to [cutoff_low, cutoff_high].

    If `normalize_over_full_range` is False (default), the distribution is
    normalized over the cutoff window only — i.e. each cutoff produces a
    standalone, properly normalized PMF over [cutoff_low, cutoff_high].

    If True, the distribution is normalized over the full [low, high] range
    with zero mass outside the cutoff window. This makes multiple cutoffs
    of the same underlying distribution comparable on the same scale (useful
    for plotting / "masked" interpretations).
    """

    def __init__(
        self,
        low: int,
        high: int,
        cutoff_low: Optional[int] = None,
        cutoff_high: Optional[int] = None,
        normalize_over_full_range: bool = False,
    ):
        if low <= 0:
            raise ValueError(f"low ({low}) must be > 0")
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")

        lo = cutoff_low if cutoff_low is not None else low
        hi = cutoff_high if cutoff_high is not None else high

        self.custom = True if (cutoff_low is not None or cutoff_high is not None or normalize_over_full_range) else False

        if lo < low or hi > high or hi < lo:
            raise ValueError(
                f"Invalid cutoff range [{lo}, {hi}] for support [{low}, {high}]"
            )

        if normalize_over_full_range:
            choices = list(range(low, high + 1))
            
            # 1. SAMPLING PROBABILITIES
            # To sample correctly, PyTorch needs probabilities that sum to 1 over the cutoff.
            sample_weights = [(1.0 / k) if (lo <= k <= hi) else 0.0 for k in choices]
            sample_total = sum(sample_weights)
            sample_probs = [w / sample_total for w in sample_weights]
            
            # Initialize the base class (this creates self.categorical)
            super().__init__(choices, sample_probs)
            
            # 2. PLOTTING PROBABILITIES
            # For the plot, we calculate probabilities normalized over the FULL [low, high] range.
            full_total = sum(1.0 / k for k in choices)
            plot_probs = torch.tensor([
                (1.0 / k) / full_total if (lo <= k <= hi) else 0.0 
                for k in choices
            ])
            
            # 3. THE TRICK
            # We wrap `self.categorical` to expose `plot_probs` to your plotting code 
            # while delegating `sample()` to the real PyTorch distribution.
            class _PlottingCategoricalWrapper:
                def __init__(self, base_cat, p):
                    self._base = base_cat
                    self.probs = p
                    
                def sample(self, *args, **kwargs):
                    return self._base.sample(*args, **kwargs)
                    
                def __getattr__(self, name):
                    return getattr(self._base, name)

            self.categorical = _PlottingCategoricalWrapper(self.categorical, plot_probs)
            
        else:
            # Standard behavior (normalized strictly over cutoff)
            choices = list(range(lo, hi + 1))
            weights = [1.0 / k for k in choices]
            total = sum(weights)
            probs = [w / total for w in weights]
            
            super().__init__(choices, probs)

    def quantile(self, quantiles: List[float]) -> List[int]:
        if self.custom:
            raise NotImplementedError("Quantiles are not implemented for custom cutoff distributions.")
        # 1. Calculate the Cumulative Distribution Function (CDF)
        cdf = torch.cumsum(self.categorical.probs, dim=0)
        
        # 2. Find the indices where the CDF crosses the target quantiles
        q_tensor = torch.tensor(quantiles, dtype=torch.float32)
        indices = torch.searchsorted(cdf, q_tensor)
        
        # 3. Map indices back to the actual integer choices
        return [self.choices[i] for i in indices]
