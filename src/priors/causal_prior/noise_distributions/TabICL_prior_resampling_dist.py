import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import random

import torch
from torch import Tensor

# Add tabicl to path
tabicl_path = Path(__file__).parent.parent.parent.parent / "tabicl" / "src"
sys.path.insert(0, str(tabicl_path))

from tabicl.prior.dataset import PriorDataset

# Try relative import first, fall back to absolute
try:
    from .DistributionInterface import Distribution
except ImportError:
    try:
        from DistributionInterface import Distribution
    except ImportError:
        from src.priors.causal_prior.noise_distributions.DistributionInterface import Distribution


class TabICLPriorSampler:
    """
    Samples datasets from the TabICL prior and saves them to disk.
    
    Each sample from the prior is saved as an individual .pt file containing
    only the active features (up to the d dimension for each sample).
    
    Parameters
    ----------
    save_dir : str, default="/Users/arikreuter/CausalPriorFitting/data_cache/TabICL_samples"
        Directory where samples will be saved
    prior_config : dict, optional
        Configuration parameters to pass to PriorDataset
    """
    
    def __init__(
        self, 
        save_dir: str = "/Users/arikreuter/CausalPriorFitting/data_cache/TabICL_samples",
        prior_config: Optional[Dict[str, Any]] = None
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the prior dataset with provided config or defaults
        self.prior_config = prior_config or {}
        self.dataset = PriorDataset(**self.prior_config)
        
        print(f"TabICLPriorSampler initialized")
        print(f"Save directory: {self.save_dir}")
        print(f"Prior config: {self.dataset}")
    
    def sample_and_save(self, num_samples: int, batch_size: Optional[int] = None, start_idx: int = 0):
        """
        Sample datasets from the prior and save each one individually.
        
        Parameters
        ----------
        num_samples : int
            Total number of individual samples to generate and save
        batch_size : int, optional
            Number of samples to generate per batch. If None, uses the prior's default
        start_idx : int, default=0
            Starting index for file naming
        """
        if batch_size is None:
            batch_size = self.dataset.batch_size
        
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        total_saved = 0
        
        print(f"\nGenerating {num_samples} samples in {num_batches} batches...")
        
        for batch_idx in range(num_batches):
            # Determine how many samples to generate in this batch
            samples_in_batch = min(batch_size, num_samples - total_saved)
            
            # Generate a batch
            print(f"Batch {batch_idx + 1}/{num_batches}: generating {samples_in_batch} samples...", end=" ")
            X, y, d, seq_lens, train_sizes = self.dataset.get_batch(batch_size=samples_in_batch)
            
            # Vectorized processing: prepare all samples at once
            is_nested = hasattr(X, 'is_nested') and X.is_nested
            
            # Build list of samples to save
            samples_to_save = []
            for i in range(samples_in_batch):
                sample_idx = start_idx + total_saved + i
                
                # Extract active features only (up to d[i])
                if is_nested:
                    X_i = X[i][:, :d[i]].cpu()  # (seq_len, d[i])
                    y_i = y[i].cpu()  # (seq_len,)
                else:
                    X_i = X[i, :, :d[i]].cpu()  # (seq_len, d[i])
                    y_i = y[i].cpu()  # (seq_len,)
                
                # Create sample dictionary with all relevant info
                sample = {
                    'X': X_i,
                    'y': y_i,
                    'num_features': d[i].item(),
                    'seq_len': seq_lens[i].item(),
                    'train_size': train_sizes[i].item(),
                }
                
                samples_to_save.append((sample_idx, sample))
            
            # Vectorized save: save all samples in batch
            for sample_idx, sample in samples_to_save:
                filename = self.save_dir / f"sample_{sample_idx:06d}.pt"
                torch.save(sample, filename)
            
            total_saved += samples_in_batch
            print(f"saved {samples_in_batch} samples (total: {total_saved}/{num_samples})")
        
        print(f"\nCompleted! Saved {total_saved} samples to {self.save_dir}")
    
    def load_sample(self, sample_idx: int) -> Dict[str, Any]:
        """
        Load a previously saved sample.
        
        Parameters
        ----------
        sample_idx : int
            Index of the sample to load
            
        Returns
        -------
        dict
            Dictionary containing 'X', 'y', 'num_features', 'seq_len', 'train_size'
        """
        filename = self.save_dir / f"sample_{sample_idx:06d}.pt"
        if not filename.exists():
            raise FileNotFoundError(f"Sample {sample_idx} not found at {filename}")
        return torch.load(filename)
    
    def get_num_saved_samples(self) -> int:
        """
        Count the number of samples currently saved.
        
        Returns
        -------
        int
            Number of .pt files in the save directory
        """
        return len(list(self.save_dir.glob("sample_*.pt")))


class TabICLResamplingDist(Distribution):
    """
    A distribution that resamples from pre-computed TabICL prior samples.
    
    This distribution loads TabICL samples from disk and uses individual features
    as reservoirs for sampling. Each feature column is normalized (if non-constant)
    and then rescaled to the specified scale/shape parameters.
    
    The distribution samples without replacement from a reservoir until it's empty,
    then throws an error. This ensures we don't reuse the same data points.
    
    Parameters
    ----------
    samples_dir : str, default="/Users/arikreuter/CausalPriorFitting/data_cache/TabICL_samples"
        Directory containing the saved TabICL samples
    num_samples : int, default=100
        Number of sample files available (sample_000000.pt to sample_{num_samples-1}.pt)
    scale : float, default=1.0
        Scale parameter to rescale the normalized features
    shape : float, default=1.0
        Shape parameter (for compatibility with other distributions, currently unused)
    device : torch.device | str, default="cpu"
        Device to load tensors onto
    dtype : torch.dtype, default=torch.float32
        Data type for tensors
    """
    
    def __init__(
        self,
        samples_dir: str = "/Users/arikreuter/CausalPriorFitting/data_cache/TabICL_samples",
        num_samples: int = 100,
        scale: float = 1.0,
        shape: float = 1.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.samples_dir = Path(samples_dir)
        self.num_samples = num_samples
        self.scale = scale
        self.shape = shape
        self.device = device
        self.dtype = dtype
        
        # Reservoir state
        self._reservoir: Optional[Tensor] = None
        self._reservoir_idx: int = 0
        
        # Verify samples directory exists
        if not self.samples_dir.exists():
            raise FileNotFoundError(f"Samples directory not found: {self.samples_dir}")
    
    def _load_new_reservoir(self) -> None:
        """
        Load a new feature column from a randomly selected TabICL sample.
        
        This method:
        1. Randomly picks a sample file
        2. Loads it
        3. Randomly picks a feature column
        4. Normalizes it (if variance > 0) - VECTORIZED
        5. Rescales it by self.scale - VECTORIZED
        6. Takes absolute value - VECTORIZED
        """
        # Randomly select a sample file
        sample_idx = random.randint(0, self.num_samples - 1)
        sample_path = self.samples_dir / f"sample_{sample_idx:06d}.pt"
        
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_path}")
        
        # Load the sample
        sample = torch.load(sample_path, map_location='cpu')
        X = sample['X']  # (seq_len, num_features)
        num_features = sample['num_features']
        
        # Randomly select a feature column
        feature_idx = random.randint(0, num_features - 1)
        feature_col = X[:, feature_idx].to(self.dtype)  # (seq_len,)
        
        # Vectorized normalization: compute mean and std in one pass
        mean = feature_col.mean()
        std = feature_col.std(unbiased=False)
        
        # Vectorized conditional normalization
        if std > 1e-8:
            # Vectorized: normalize entire tensor at once
            feature_col = (feature_col - mean) / std
        
        # Vectorized: scale and take absolute value in one operation
        feature_col = torch.abs(feature_col * self.scale)
        
        # Move to device (vectorized operation)
        self._reservoir = feature_col.to(self.device)
        self._reservoir_idx = 0
    
    def _ensure_reservoir(self) -> None:
        """Ensure reservoir is loaded and has data available."""
        if self._reservoir is None or self._reservoir_idx >= len(self._reservoir):
            self._load_new_reservoir()
    
    @torch.no_grad()
    def sample_one(self) -> Tensor:
        """
        Sample a single scalar value from the reservoir.
        
        Returns
        -------
        Tensor
            A scalar tensor (already absolute value from reservoir)
        """
        self._ensure_reservoir()
        
        if self._reservoir_idx >= len(self._reservoir):
            raise RuntimeError(
                "Reservoir exhausted! All values have been sampled without replacement. "
                "This should not happen if _ensure_reservoir is working correctly."
            )
        
        # Reservoir already contains absolute values, no need to call abs() again
        value = self._reservoir[self._reservoir_idx]
        self._reservoir_idx += 1
        return value
    
    @torch.no_grad()
    def sample_n(self, n: int) -> Tensor:
        """
        Sample n values from the reservoir (potentially across multiple reservoirs).
        
        Uses vectorized slicing for efficient batch sampling.
        
        Parameters
        ----------
        n : int
            Number of samples to draw
        
        Returns
        -------
        Tensor
            Shape (n,) containing sampled values (absolute values to ensure positivity)
        """
        if n <= 0:
            return torch.empty((0,), device=self.device, dtype=self.dtype)
        
        # Pre-allocate output tensor for efficiency
        output = torch.empty(n, device=self.device, dtype=self.dtype)
        output_idx = 0
        
        while output_idx < n:
            self._ensure_reservoir()
            
            # Vectorized: calculate how many we can take from current reservoir
            available = len(self._reservoir) - self._reservoir_idx
            take = min(n - output_idx, available)
            
            # Vectorized: slice and copy entire chunk at once (no loops)
            output[output_idx:output_idx + take] = self._reservoir[self._reservoir_idx:self._reservoir_idx + take]
            
            # Update indices
            self._reservoir_idx += take
            output_idx += take
        
        return output
    
    @torch.no_grad()
    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """
        Sample values with the given shape.
        
        Parameters
        ----------
        shape : tuple of int
            Desired output shape
        
        Returns
        -------
        Tensor
            Sampled values with the specified shape
        """
        if not shape:
            return self.sample_one()
        
        import math
        n = int(math.prod(shape))
        return self.sample_n(n).reshape(shape)
    
    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> "TabICLResamplingDist":
        """
        Move distribution to a different device/dtype.
        
        Parameters
        ----------
        device : torch.device | str
            Target device
        dtype : torch.dtype, optional
            Target dtype
        
        Returns
        -------
        TabICLResamplingDist
            Returns self for chaining
        """
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        
        # Move current reservoir if it exists
        if self._reservoir is not None:
            self._reservoir = self._reservoir.to(device=self.device, dtype=self.dtype)
        
        return self


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate and save TabICL prior samples to disk"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation (default: 32)"
    )
    parser.add_argument(
        "--min_features",
        type=int,
        default=2,
        help="Minimum number of features (default: 2)"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=100,
        help="Maximum number of features (default: 100)"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=10_000,
        help="Maximum sequence length (default: 10,000)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/Users/arikreuter/CausalPriorFitting/data_cache/TabICL_samples",
        help="Directory to save samples (default: data_cache/TabICL_samples)"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for file naming (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Create sampler with configuration
    print("="*70)
    print("TabICL Prior Sample Generation")
    print("="*70)
    
    sampler = TabICLPriorSampler(
        save_dir=args.save_dir,
        prior_config={
            'batch_size': args.batch_size,
            'min_features': args.min_features,
            'max_features': args.max_features,
            'max_seq_len': args.max_seq_len,
        }
    )
    
    # Generate and save samples
    sampler.sample_and_save(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        start_idx=args.start_idx
    )
    
    # Verify by loading and inspecting the first sample
    print("\n" + "="*70)
    print("Verification: Inspecting first generated sample")
    print("="*70)
    try:
        sample = sampler.load_sample(args.start_idx)
        print(f"Sample {args.start_idx}:")
        print(f"  X shape: {sample['X'].shape}")
        print(f"  y shape: {sample['y'].shape}")
        print(f"  num_features: {sample['num_features']}")
        print(f"  seq_len: {sample['seq_len']}")
        print(f"  train_size: {sample['train_size']}")
        print(f"\n✓ Sample generation completed successfully!")
    except FileNotFoundError as e:
        print(f"Could not load sample for verification: {e}")
    
    print("\n" + "="*70)