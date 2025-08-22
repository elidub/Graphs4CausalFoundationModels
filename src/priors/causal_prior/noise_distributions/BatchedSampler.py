from abc import ABC, abstractmethod
from typing import Optional, Tuple
import math
import torch
from torch import Tensor

from priors.causal_prior.noise_distributions.DistributionInterface import Distribution

class BatchedSampler:
    """
    Lift a ScalarDistribution to i.i.d. sampling over an arbitrary shape.
    Tries the distribution's vectorized methods first; otherwise falls back to
    a chunked Python loop to avoid massive per-sample overhead.
    """

    def __init__(self, dist: Distribution, chunk_size: int = 65536) -> None:
        """
        Parameters
        ----------
        dist : ScalarDistribution
            Your scalar distribution instance.
        chunk_size : int
            Fallback loop will draw in chunks of this size for speed/memory.
        """
        self.dist = dist
        self.chunk_size = int(chunk_size)

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...]) -> Tensor:
        """
        Sample i.i.d. from `dist` with overall output shape `shape`.
        """
        # Fast path 1: vectorized by shape
        try:
            return self.dist.sample_shape(shape)  # user override
        except NotImplementedError:
            pass

        # Fast path 2: vectorized by flat count
        N = int(math.prod(shape)) if shape else 1
        try:
            out = self.dist.sample_n(N)  # user override
            return out.reshape(shape)
        except NotImplementedError:
            pass

        # Fallback: chunked loop (still efficient)
        if N == 1:
            return self.dist.sample_one().reshape(shape).to(self.dist.device, self.dist.dtype)

        chunks = []
        remaining = N
        while remaining > 0:
            k = min(self.chunk_size, remaining)
            # draw k scalars
            vals = [self.dist.sample_one() for _ in range(k)]
            chunk = torch.stack(vals)  # (k,)
            chunks.append(chunk)
            remaining -= k
        flat = torch.cat(chunks, dim=0)  # (N,)
        return flat.reshape(shape).to(self.dist.device, self.dist.dtype)
