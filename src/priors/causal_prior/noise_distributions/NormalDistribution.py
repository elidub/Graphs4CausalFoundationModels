from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.distributions as dist

from priors.causal_prior.noise_distributions.DistributionInterface import Distribution


# ---------- Normal (Gaussian) distribution with specified mean and std ----------

class NormalDistribution(Distribution):
    """
    Normal (Gaussian) distribution with specified mean and standard deviation.
    
    Args:
        mean: Mean of the distribution (can be any real number)
        std: Standard deviation of the distribution (must be > 0)
        device: Device to use for sampling ('cpu' or 'cuda')
        dtype: Data type for samples
        generator: Optional random number generator for reproducibility
        
    Example:
        >>> noise_dist = NormalDistribution(mean=0.0, std=1.0)
        >>> sample = noise_dist.sample_one()  # Single sample
        >>> samples = noise_dist.sample_n(100)  # 100 samples
        >>> samples_2d = noise_dist.sample_shape((10, 5))  # 10x5 samples
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, generator=generator)
        
        if std <= 0:
            raise ValueError(f"NormalDistribution requires std > 0, got std={std}")
        
        self.mean = mean
        self.std = std
        
        # Create PyTorch Normal distribution
        self._normal = dist.Normal(
            loc=torch.tensor(mean, device=self.device, dtype=self.dtype),
            scale=torch.tensor(std, device=self.device, dtype=self.dtype),
        )

    @torch.no_grad()
    def sample_one(self) -> Tensor:
        """Sample a single scalar value from the distribution."""
        return self._normal.sample().reshape(()).to(self.device, self.dtype)

    @torch.no_grad()
    def sample_n(self, n: int) -> Tensor:
        """Sample n i.i.d. values from the distribution."""
        if n <= 0:
            return torch.empty((0,), device=self.device, dtype=self.dtype)
        return self._normal.sample((n,)).reshape((-1,)).to(self.device, self.dtype)

    @torch.no_grad()
    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """Sample a tensor of given shape from the distribution."""
        return self._normal.sample(shape).reshape(shape).to(self.device, self.dtype)
    
    def __repr__(self) -> str:
        return f"NormalDistribution(mean={self.mean}, std={self.std})"


if __name__ == "__main__":
    # Test the normal distribution
    print("Testing NormalDistribution")
    print("=" * 60)
    
    # Test 1: Standard normal (mean=0, std=1)
    print("\n1. Standard Normal (mean=0, std=1)")
    std_normal = NormalDistribution(mean=0.0, std=1.0)
    print(f"   Distribution: {std_normal}")
    
    sample = std_normal.sample_one()
    print(f"   Single sample: {sample.item():.4f}")
    
    samples_100 = std_normal.sample_n(100)
    print(f"   100 samples - mean: {samples_100.mean().item():.4f}, std: {samples_100.std().item():.4f}")
    
    samples_2d = std_normal.sample_shape((10, 5))
    print(f"   Shape (10, 5) samples - shape: {samples_2d.shape}, mean: {samples_2d.mean().item():.4f}")
    
    # Test 2: Non-standard normal (mean=5, std=2)
    print("\n2. Non-Standard Normal (mean=5, std=2)")
    custom_normal = NormalDistribution(mean=5.0, std=2.0)
    print(f"   Distribution: {custom_normal}")
    
    samples_1000 = custom_normal.sample_n(1000)
    print(f"   1000 samples - mean: {samples_1000.mean().item():.4f}, std: {samples_1000.std().item():.4f}")
    print(f"   Expected: mean ≈ 5.0, std ≈ 2.0")
    
    # Test 3: Small std
    print("\n3. Small Standard Deviation (mean=0, std=0.1)")
    small_std_normal = NormalDistribution(mean=0.0, std=0.1)
    print(f"   Distribution: {small_std_normal}")
    
    samples_500 = small_std_normal.sample_n(500)
    print(f"   500 samples - mean: {samples_500.mean().item():.4f}, std: {samples_500.std().item():.4f}")
    print(f"   Min: {samples_500.min().item():.4f}, Max: {samples_500.max().item():.4f}")
    
    # Test 4: Large std
    print("\n4. Large Standard Deviation (mean=0, std=10)")
    large_std_normal = NormalDistribution(mean=0.0, std=10.0)
    print(f"   Distribution: {large_std_normal}")
    
    samples_500 = large_std_normal.sample_n(500)
    print(f"   500 samples - mean: {samples_500.mean().item():.4f}, std: {samples_500.std().item():.4f}")
    print(f"   Min: {samples_500.min().item():.4f}, Max: {samples_500.max().item():.4f}")
    
    # Test 5: Error handling
    print("\n5. Error Handling")
    try:
        bad_normal = NormalDistribution(mean=0.0, std=-1.0)
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    try:
        bad_normal = NormalDistribution(mean=0.0, std=0.0)
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    # Test 6: CUDA compatibility (if available)
    if torch.cuda.is_available():
        print("\n6. CUDA Test")
        cuda_normal = NormalDistribution(mean=0.0, std=1.0, device="cuda")
        cuda_samples = cuda_normal.sample_n(100)
        print(f"   Device: {cuda_samples.device}")
        print(f"   Mean: {cuda_samples.mean().item():.4f}, Std: {cuda_samples.std().item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
