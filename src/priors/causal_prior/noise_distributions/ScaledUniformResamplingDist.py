"""Scaled uniform resampling distribution.

Samples uniformly from the range [mean - a, mean + a] where:
- mean = mean(data)
- a = scale_factor * std(data)

This creates a uniform distribution centered at the observational mean with 
width proportional to the observational standard deviation. For example, with
scale_factor=3, the distribution is U(mean - 3*std, mean + 3*std).

Mathematically: P(T_int) = U(-a, a) + b where:
- a = scale_factor * std(P(T_obs))
- b = mean(P(T_obs))

Implements the ``Distribution`` interface (see ``DistributionInterface.py``).

Typical usage:
    dist = ScaledUniformResamplingDist(data_tensor, scale_factor=3.0)
    x = dist.sample_one()     # scalar () uniformly from [mean-3*std, mean+3*std]
    xs = dist.sample_n(10)    # shape (10,) uniformly from [mean-3*std, mean+3*std]

Edge cases handled:
  - Empty data raises ``ValueError``.
  - If std is zero (all values identical), uses a small epsilon for stability.
"""

from __future__ import annotations

from typing import Tuple
import torch
from torch import Tensor

from .DistributionInterface import Distribution


class ScaledUniformResamplingDist(Distribution):
    """Sample uniformly from [mean - a, mean + a] where a = scale_factor * std.

    Parameters
    ----------
    data : Tensor | list | tuple
        Source values. Will be converted to a 1D tensor (flattened if necessary).
        The distribution will sample uniformly from [mean - a, mean + a] where
        mean and std are computed from this data.
    scale_factor : float, default 3.0
        Multiplier for the standard deviation to determine the width of the uniform
        distribution. For example, scale_factor=3.0 means the distribution spans
        from mean - 3*std to mean + 3*std.
    device, dtype, generator : see ``Distribution`` base class.
    """

    def __init__(
        self,
        data: Tensor | list | tuple,
        scale_factor: float = 3.0,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, generator=generator)

        data_tensor = torch.as_tensor(data, dtype=dtype, device=self.device)
        if data_tensor.numel() == 0:
            raise ValueError("ScaledUniformResamplingDist requires a non-empty data tensor.")
        
        # Flatten to 1D if necessary
        if data_tensor.dim() != 1:
            data_tensor = data_tensor.flatten()
        
        # Compute mean and std from the data
        self.mean_val = data_tensor.mean()
        self.std_val = data_tensor.std()
        self.scale_factor = float(scale_factor)
        
        # Handle edge case: std is zero (all values are identical)
        if self.std_val == 0 or torch.isnan(self.std_val):
            # Use a small epsilon to avoid degenerate distribution
            epsilon = torch.finfo(dtype).eps * 100
            self.std_val = torch.tensor(epsilon, dtype=dtype, device=self.device)
        
        # Compute bounds: [mean - a, mean + a] where a = scale_factor * std
        self.a = self.scale_factor * self.std_val
        self.min_val = self.mean_val - self.a
        self.max_val = self.mean_val + self.a

    # ------------------------------------------------------------------
    # Distribution interface implementations
    # ------------------------------------------------------------------
    def sample_one(self) -> Tensor:
        """Return a scalar tensor () sampled uniformly from [mean - a, mean + a]."""
        return torch.empty([], dtype=self.dtype, device=self.device).uniform_(
            self.min_val.item(), self.max_val.item(), generator=self.generator
        )

    def sample_n(self, n: int) -> Tensor:
        """Return shape (n,) of i.i.d. uniform samples from [mean - a, mean + a]."""
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty(0, dtype=self.dtype, device=self.device)
        
        return torch.empty(n, dtype=self.dtype, device=self.device).uniform_(
            self.min_val.item(), self.max_val.item(), generator=self.generator
        )

    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """Return shape `shape` of i.i.d. uniform samples from [mean - a, mean + a]."""
        if len(shape) == 0:
            return self.sample_one()
        
        total = 1
        for s in shape:
            total *= int(s)
        
        if total == 0:
            return torch.empty(*shape, dtype=self.dtype, device=self.device)
        
        return torch.empty(*shape, dtype=self.dtype, device=self.device).uniform_(
            self.min_val.item(), self.max_val.item(), generator=self.generator
        )

    def parameters(self) -> dict:
        """Return distribution parameters as a dictionary."""
        return {
            "mean": self.mean_val.item(),
            "std": self.std_val.item(),
            "scale_factor": self.scale_factor,
            "a": self.a.item(),
            "min": self.min_val.item(),
            "max": self.max_val.item(),
            "type": "scaled_uniform_resampling"
        }

    # ------------------------------------------------------------------
    # Convenience / representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ScaledUniformResamplingDist("
            f"mean={self.mean_val.item():.4f}, "
            f"std={self.std_val.item():.4f}, "
            f"scale_factor={self.scale_factor:.2f}, "
            f"range=[{self.min_val.item():.4f}, {self.max_val.item():.4f}], "
            f"device={self.device}, dtype={self.dtype})"
        )
