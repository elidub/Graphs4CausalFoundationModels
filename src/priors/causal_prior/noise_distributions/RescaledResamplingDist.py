"""Rescaled resampling distribution.

Extends ResamplingDist to apply a multiplicative scale factor to sampled values.
Useful for interventional distributions where the treatment variable's scale 
should be increased relative to its observational marginal.

The rescaling factor is applied as: sampled_value * (1 + rescale_factor)
For example, if rescale_factor = 0.1, sampled values are multiplied by 1.1.

Typical usage:
    # Increase treatment scale by 10%
    dist = RescaledResamplingDist(data_tensor, rescale_factor=0.1)
    x = dist.sample_one()  # Returns value from data_tensor * 1.1
    
    # No rescaling (equivalent to ResamplingDist)
    dist = RescaledResamplingDist(data_tensor, rescale_factor=0.0)
"""

from __future__ import annotations

from typing import Tuple
import torch
from torch import Tensor

from .ResamplingDist import ResamplingDist


class RescaledResamplingDist(ResamplingDist):
    """Resample from a fixed set of scalar values with multiplicative rescaling.
    
    Inherits all functionality from ResamplingDist but applies a scale factor
    to all sampled values: sampled_value * (1 + rescale_factor)
    
    Parameters
    ----------
    data : Tensor | list | tuple
        Source values. Will be converted to a 1D tensor (flattened if necessary).
    rescale_factor : float, default 0.0
        Multiplicative scale increase factor. Final scale is (1 + rescale_factor).
        Examples:
        - 0.0: no scaling (default)
        - 0.1: increase by 10% (multiply by 1.1)
        - 0.5: increase by 50% (multiply by 1.5)
        - -0.1: decrease by 10% (multiply by 0.9)
    device, dtype, generator : see ``Distribution`` base class.
    shuffle_each_epoch : bool, default True
        If True, after all items have been drawn once, a new random permutation
        is generated and sampling continues. If False, further sampling after
        exhaustion raises ``StopIteration``.
    """
    
    def __init__(
        self,
        data: Tensor | list | tuple,
        *,
        rescale_factor: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        shuffle_each_epoch: bool = True,
    ) -> None:
        # Initialize parent ResamplingDist
        super().__init__(
            data=data,
            device=device,
            dtype=dtype,
            generator=generator,
            shuffle_each_epoch=shuffle_each_epoch,
        )
        
        # Store rescale factor and compute scale multiplier
        self.rescale_factor = float(rescale_factor)
        self.scale_multiplier = 1.0 + self.rescale_factor
    
    # ------------------------------------------------------------------
    # Override sampling methods to apply rescaling
    # ------------------------------------------------------------------
    def sample_one(self) -> Tensor:
        """Sample a single value with rescaling applied."""
        value = super().sample_one()
        return value * self.scale_multiplier
    
    def sample_n(self, n: int) -> Tensor:
        """Sample n values with rescaling applied."""
        values = super().sample_n(n)
        return values * self.scale_multiplier
    
    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """Sample values with given shape, rescaling applied."""
        values = super().sample_shape(shape)
        return values * self.scale_multiplier
    
    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover (simple)
        return (
            f"RescaledResamplingDist(N={self._N}, rescale_factor={self.rescale_factor}, "
            f"scale_multiplier={self.scale_multiplier:.4f}, "
            f"shuffle_each_epoch={self.shuffle_each_epoch}, "
            f"device={self.device}, dtype={self.dtype})"
        )
