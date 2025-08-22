from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Type
import math
import torch
from torch import Tensor
import torch.distributions as dist

from priors.causal_prior.noise_distributions.DistributionInterface import Distribution
from priors.causal_prior.noise_distributions.MixedDist import MixedDist  # your existing class
from priors.causal_prior.noise_distributions.Sample_STD import GammaMeanStd  # your existing class



# ---------- Mixed distribution with *random* std per draw ----------

class MixedDistRandomStd(MixedDist):
    """
    Like MixedDist, but the *standard deviation* is re-sampled from a distribution D
    on every draw (independently for each scalar). This makes the mixture *heteroscedastic*.

    The base MixedDist caches components for a fixed std; here we override the sampling
    so components are parameterized by freshly sampled std values.
    """

    def __init__(
        self,
        std_dist: Distribution = GammaMeanStd(mean=1.0, std=0.1),
        distributions: Sequence[Type[dist.Distribution]] = (dist.Normal, dist.Laplace, dist.StudentT, dist.Gumbel),
        mixture_proportions: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        std2scale: Optional[dict[Type[dist.Distribution], Callable[[Tensor], Tensor]]] = None,
        student_t_df: float = 3.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Parameters
        ----------
        std_dist : Distribution
            A positive-valued distribution D that samples the *std* per scalar draw.
        distributions, mixture_proportions, student_t_df, std2scale, device, dtype, generator
            As in MixedDist, except std2scale should accept a Tensor and return a Tensor.
        """
        # Initialize MixedDist with a dummy std (not used) just to reuse metadata
        super().__init__(
            std=1.0,  # placeholder; ignored in overridden sampling
            distributions=distributions,
            mixture_proportions=mixture_proportions,
            std2scale=None,  # we'll set vectorized converters below
            student_t_df=student_t_df,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        self.std_dist = std_dist
        self.student_t_df = float(student_t_df)

        # Vectorized std->scale converters; accept Tensor std and return Tensor scale
        if std2scale is None:
            # Note: we only store functions; Gumbel needs loc later
            std2scale = {
                dist.Normal:  lambda s: s,
                dist.Laplace: lambda s: s / math.sqrt(2.0),
                dist.StudentT: lambda s, df=self.student_t_df: s * math.sqrt((df - 2.0) / df),
                dist.Gumbel:  lambda s: s * math.sqrt(6.0) / math.pi,
            }
        self.std2scale = std2scale

        # Categorical over components (reuse MixedDist weights/categorical)
        # self.weights and self._cat already initialized by parent

    # ---- Helpers to build per-component distributions from per-sample scales ----
    def _build_component_from_scales(
        self, cls: Type[dist.Distribution], scales: Tensor
    ) -> dist.Distribution:
        """
        Given a component class and a 1D tensor of scales (len = count),
        return a batch-parameterized torch.distribution of that class.
        """
        device, dtype = self.device, self.dtype
        if cls is dist.StudentT:
            df = torch.full_like(scales, fill_value=self.student_t_df)
            loc = torch.zeros_like(scales, dtype=dtype, device=device)
            return dist.StudentT(df=df, loc=loc, scale=scales)
        elif cls is dist.Gumbel:
            euler_gamma = 0.5772156649015329
            loc = -euler_gamma * scales  # zero-mean
            return dist.Gumbel(loc=loc, scale=scales)
        elif cls is dist.Normal:
            loc = torch.zeros_like(scales, dtype=dtype, device=device)
            return dist.Normal(loc=loc, scale=scales)
        elif cls is dist.Laplace:
            loc = torch.zeros_like(scales, dtype=dtype, device=device)
            return dist.Laplace(loc=loc, scale=scales)
        else:
            raise ValueError(f"Unsupported distribution class: {cls.__name__}")

    # ---- Distribution API overrides ----
    @torch.no_grad()
    def sample_one(self) -> Tensor:
        # draw std, choose component, build that component with the implied scale, sample one
        s = self.std_dist.sample_one().to(self.device, self.dtype)  # ()
        k = int(self._cat.sample(generator=self.generator).item())

        scale = self.std2scale[self.distributions[k]](s)  # ()
        comp = self._build_component_from_scales(self.distributions[k], scale.reshape(1))
        x = comp.sample(generator=self.generator)  # shape (1,)
        return x.squeeze(0)

    @torch.no_grad()
    def sample_n(self, n: int) -> Tensor:
        if n <= 0:
            return torch.empty((0,), device=self.device, dtype=self.dtype)

        # 1) draw std per scalar
        stds = self.std_dist.sample_n(n).to(self.device, self.dtype)  # (n,)
        # 2) draw component index per scalar
        idx = self._cat.sample((n,))  # (n,)
        out = torch.empty((n,), device=self.device, dtype=self.dtype)

        # 3) for each component, convert std->scale (vectorized) and sample
        for k, cls in enumerate(self.distributions):
            mask = (idx == k)
            count = int(mask.sum().item())
            if count == 0:
                continue
            s_k = stds[mask]                              # (count,)
            scales_k = self.std2scale[cls](s_k)           # (count,)
            comp_k = self._build_component_from_scales(cls, scales_k)
            out[mask] = comp_k.sample()  # (count,)

        return out

    @torch.no_grad()
    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        if not shape:
            return self.sample_one()
        N = int(math.prod(shape))
        return self.sample_n(N).reshape(shape)

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> MixedDistRandomStd:
        # move self + inner std_dist
        super().to(device, dtype)
        self.std_dist = self.std_dist  # ensure interface present
        # try to move std_dist if it has .to
        if hasattr(self.std_dist, "to"):
            self.std_dist = self.std_dist.to(self.device, self.dtype)
        # re-create categorical on new device/dtype
        self.weights = self.weights.to(self.device, self.dtype)
        self._cat = dist.Categorical(probs=self.weights)
        return self
