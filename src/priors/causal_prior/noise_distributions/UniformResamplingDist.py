"""Uniform resampling distribution.

Samples uniformly from the range [min(data), max(data)] of the provided data.
Unlike ResamplingDist which samples discrete values from the data without replacement,
this distribution samples continuously from the empirical range.

Implements the ``Distribution`` interface (see ``DistributionInterface.py``).

Typical usage:
    dist = UniformResamplingDist(data_tensor)  # data_tensor shape (N,)
    x = dist.sample_one()                      # scalar () uniformly from [min, max]
    xs = dist.sample_n(10)                     # shape (10,) uniformly from [min, max]

Edge cases handled:
  - Empty data raises ``ValueError``.
  - If all values are identical (min == max), adds small epsilon to max for stability.
"""

from __future__ import annotations

from typing import Tuple
import torch
from torch import Tensor

from .DistributionInterface import Distribution


class UniformResamplingDist(Distribution):
    """Sample uniformly from the range [min(data), max(data)].

    Parameters
    ----------
    data : Tensor | list | tuple
        Source values. Will be converted to a 1D tensor (flattened if necessary).
        The distribution will sample uniformly from [min(data), max(data)].
    device, dtype, generator : see ``Distribution`` base class.
    """

    def __init__(
        self,
        data: Tensor | list | tuple,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, generator=generator)

        data_tensor = torch.as_tensor(data, dtype=dtype, device=self.device)
        if data_tensor.numel() == 0:
            raise ValueError("UniformResamplingDist requires a non-empty data tensor.")
        
        # Flatten to 1D if necessary
        if data_tensor.dim() != 1:
            data_tensor = data_tensor.flatten()
        
        # Compute min and max from the data
        self.min_val = data_tensor.min()
        self.max_val = data_tensor.max()
        
        # Handle edge case: all values are identical
        if self.min_val == self.max_val:
            # Add a small epsilon to avoid degenerate uniform distribution
            epsilon = torch.finfo(dtype).eps * 10
            self.max_val = self.min_val + epsilon

    # ------------------------------------------------------------------
    # Distribution interface implementations
    # ------------------------------------------------------------------
    def sample_one(self) -> Tensor:
        """Return a scalar tensor () sampled uniformly from [min_val, max_val]."""
        return torch.empty([], dtype=self.dtype, device=self.device).uniform_(
            self.min_val.item(), self.max_val.item(), generator=self.generator
        )

    def sample_n(self, n: int) -> Tensor:
        """Return shape (n,) of i.i.d. uniform samples from [min_val, max_val]."""
        if n < 0:
            raise ValueError("n must be non-negative.")
        if n == 0:
            return torch.empty(0, dtype=self.dtype, device=self.device)
        
        return torch.empty(n, dtype=self.dtype, device=self.device).uniform_(
            self.min_val.item(), self.max_val.item(), generator=self.generator
        )

    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """Return shape `shape` of i.i.d. uniform samples from [min_val, max_val]."""
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
            "min": self.min_val.item(),
            "max": self.max_val.item(),
            "type": "uniform_resampling"
        }

    # ------------------------------------------------------------------
    # Convenience / representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"UniformResamplingDist(min={self.min_val.item():.4f}, "
            f"max={self.max_val.item():.4f}, device={self.device}, dtype={self.dtype})"
        )
