from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Type, Union
import math
import torch
from torch import Tensor
import torch.distributions as dist

from priors.causal_prior.noise_distributions.DistributionInterface import Distribution
from priors.causal_prior.noise_distributions.MixedDist import MixedDist  # your existing class
from priors.causal_prior.noise_distributions.Sample_STD import GammaMeanStd  # your existing class

# Try to import TabICLResamplingDist
try:
    from priors.causal_prior.noise_distributions.TabICL_prior_resampling_dist import TabICLResamplingDist
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False
    TabICLResamplingDist = None



# ---------- Mixed distribution with *random* std per draw ----------

class MixedDistRandomStd(MixedDist):
    """
    Like MixedDist, but the *standard deviation* is re-sampled from a distribution D
    on every draw (independently for each scalar). This makes the mixture *heteroscedastic*.

    The base MixedDist caches components for a fixed std; here we override the sampling
    so components are parameterized by freshly sampled std values.
    
    Now supports mixing both torch.distributions and custom Distribution objects
    (e.g., TabICLResamplingDist) as components.
    """

    def __init__(
        self,
        std_dist: Distribution = GammaMeanStd(mean=1.0, std=0.1),
        distributions: Sequence[Union[Type[dist.Distribution], Distribution]] = (dist.Normal, dist.Laplace, dist.StudentT, dist.Gumbel),
        mixture_proportions: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        std2scale: Optional[dict[Union[Type[dist.Distribution], str], Callable[[Tensor], Tensor]]] = None,
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
        distributions : Sequence[Union[Type[dist.Distribution], Distribution]]
            Component distributions. Can be torch.distributions classes OR
            Distribution instances (e.g., TabICLResamplingDist).
        mixture_proportions : Sequence[float]
            Mixing weights for components
        std2scale : dict, optional
            Mapping from distribution types/names to scale conversion functions.
            For custom distributions, use string keys like "TabICLResamplingDist".
        student_t_df : float
            Degrees of freedom for StudentT
        device, dtype, generator : standard torch parameters
        """
        # Separate torch distributions from custom Distribution instances
        self.custom_dists = []  # Store Distribution instances
        torch_dist_types = []   # Store torch.distributions types
        
        for d in distributions:
            if isinstance(d, Distribution):
                # It's a Distribution instance (e.g., TabICLResamplingDist instance)
                self.custom_dists.append(d)
                # Use Normal as placeholder for parent class
                torch_dist_types.append(dist.Normal)
            elif isinstance(d, type) and issubclass(d, dist.Distribution):
                # It's a torch.distributions class
                self.custom_dists.append(None)
                torch_dist_types.append(d)
            else:
                raise TypeError(f"Unsupported distribution type: {type(d)}")
        
        # Initialize MixedDist parent with torch.distributions classes
        # For custom Distribution instances, we use Normal as a placeholder
        # since the parent init needs valid torch.distributions classes
        super().__init__(
            std=1.0,  # placeholder; ignored in overridden sampling
            distributions=tuple(torch_dist_types),
            mixture_proportions=mixture_proportions,
            std2scale=None,  # we'll set vectorized converters below
            student_t_df=student_t_df,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        
        # Override with full distributions list
        self.distributions = distributions
        self.std_dist = std_dist
        self.student_t_df = float(student_t_df)

        # Vectorized std->scale converters; accept Tensor std and return Tensor scale
        if std2scale is None:
            # Default converters for standard torch distributions
            std2scale = {
                dist.Normal:  lambda s: s,
                dist.Laplace: lambda s: s / math.sqrt(2.0),
                dist.StudentT: lambda s, df=self.student_t_df: s * math.sqrt((df - 2.0) / df),
                dist.Gumbel:  lambda s: s * math.sqrt(6.0) / math.pi,
            }
            # For custom distributions, they handle their own scaling
            # so we use identity function
            if TABICL_AVAILABLE:
                std2scale['TabICLResamplingDist'] = lambda s: s
        self.std2scale = std2scale

        # Categorical over components (reuse MixedDist weights/categorical)
        # self.weights and self._cat already initialized by parent

    # ---- Helpers to build per-component distributions from per-sample scales ----
    def _build_component_from_scales(
        self, dist_or_cls: Union[Type[dist.Distribution], Distribution], scales: Tensor
    ) -> Union[dist.Distribution, Distribution]:
        """
        Given a component class or Distribution instance and a 1D tensor of scales (len = count),
        return a batch-parameterized distribution.
        
        For torch.distributions classes: build with scales as parameters
        For Distribution instances: return as-is (they handle their own scaling internally)
        """
        # Check if it's already a Distribution instance
        if isinstance(dist_or_cls, Distribution):
            # Custom distributions handle their own scaling
            return dist_or_cls
            
        # Otherwise, it's a torch.distributions class
        cls = dist_or_cls
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
        k = int(self._cat.sample().item())

        dist_or_cls = self.distributions[k]
        
        # For custom Distribution instances, sample directly
        if isinstance(dist_or_cls, Distribution):
            return dist_or_cls.sample_one().to(self.device, self.dtype)
        
        # For torch.distributions classes, convert std->scale and build
        scale = self.std2scale[dist_or_cls](s)  # ()
        comp = self._build_component_from_scales(dist_or_cls, scale.reshape(1))
        x = comp.sample()  # shape (1,)
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
        for k, dist_or_cls in enumerate(self.distributions):
            mask = (idx == k)
            count = int(mask.sum().item())
            if count == 0:
                continue
            
            # Handle custom Distribution instances
            if isinstance(dist_or_cls, Distribution):
                # Sample directly from the custom distribution
                out[mask] = dist_or_cls.sample_n(count).to(self.device, self.dtype)
            else:
                # Handle torch.distributions classes
                s_k = stds[mask]                                    # (count,)
                scales_k = self.std2scale[dist_or_cls](s_k)        # (count,)
                comp_k = self._build_component_from_scales(dist_or_cls, scales_k)
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


# For usage examples with TabICL integration, see:
# - test_mixeddist_main.py: Simple main-method style tests
# - test_mixeddist_with_tabicl.py: Comprehensive test suite

if __name__ == "__main__":
    """Quick tests to verify TabICL integration works."""
    import sys
    from pathlib import Path
    
    # Add src to path for imports when run directly
    src_path = Path(__file__).parent.parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print("="*60)
    print("Testing MixedDistRandomStd with TabICL integration")
    print("="*60)
    
    if not TABICL_AVAILABLE:
        print("\nTabICLResamplingDist not available - skipping tests")
        exit(1)
    
    # Check if TabICL samples exist
    sample_dir = "/Users/arikreuter/CausalPriorFitting/data_cache/TabICL_samples"
    if not Path(sample_dir).exists() or not list(Path(sample_dir).glob("*.pt")):
        print(f"\n No TabICL samples found in {sample_dir}")
        print("Run TabICL_prior_resampling_dist.py first to generate samples")
        exit(1)
    
    # Test (a): Only TabICL resampling distribution
    print("\n" + "-"*60)
    print("Test (a): Pure TabICL resampling distribution")
    print("-"*60)
    
    tabicl_dist = TabICLResamplingDist(
        samples_dir=sample_dir,
        scale=1.0,
        device='cpu'
    )
    
    print("✓ Created TabICLResamplingDist")
    
    # Sample and verify
    sample_one = tabicl_dist.sample_one()
    print(f"  sample_one(): {sample_one.item():.4f}")
    
    samples = tabicl_dist.sample_n(100)
    print(f"  sample_n(100): mean={samples.mean().item():.4f}, std={samples.std().item():.4f}")
    
    # Test (b): Mixture with TabICL as component
    print("\n" + "-"*60)
    print("Test (b): Mixture with TabICL as component")
    print("-"*60)
    
    mixed = MixedDistRandomStd(
        distributions=[dist.Normal, dist.Laplace, tabicl_dist],
        mixture_proportions=[0.33, 0.34, 0.33],
        std_dist=GammaMeanStd(mean=1.0, std=0.2),
        device='cpu',
        dtype=torch.float32
    )
    
    print("✓ Created MixedDistRandomStd with TabICL component")
    
    # Sample and verify
    sample_one = mixed.sample_one()
    print(f"  sample_one(): {sample_one.item():.4f}")
    
    samples = mixed.sample_n(1000)
    print(f"  sample_n(1000): mean={samples.mean().item():.4f}, std={samples.std().item():.4f}")
    
    shape_samples = mixed.sample_shape((10, 5))
    print(f"  sample_shape((10,5)): shape={shape_samples.shape}, mean={shape_samples.mean().item():.4f}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
