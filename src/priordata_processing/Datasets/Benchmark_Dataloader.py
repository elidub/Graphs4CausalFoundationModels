"""
Benchmark class for measuring dataloader performance.

This module provides a simple class to benchmark how quickly a dataloader
can load batches of specific sizes, measuring average loading times.
"""

import time
import statistics
from typing import Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm

from MakePurelyObservationalDataset import MakePurelyObservationalDataset
from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config as prior_config
from priordata_processing.ExampleConfigs.BasicConfigs import default_dataset_config as dataset_config
from priordata_processing.ExampleConfigs.BasicConfigs import default_preprocessing_config as preprocessing_config


class DataloaderBenchmark:
    """
    Simple class to benchmark dataloader performance by measuring batch loading times.
    
    This class can measure:
    - Average time per batch
    - Total time for a specified number of batches
    - Statistics including min, max, median, standard deviation and 95% bootstrap CIs
    """
    
    def __init__(self, dataloader: DataLoader, verbose: bool = True):
        """
        Initialize the benchmark with a pre-configured dataloader.
        
        Args:
            dataloader: Pre-configured PyTorch DataLoader to benchmark
            verbose: Whether to print progress information
        """
        self.dataloader = dataloader
        self.verbose = verbose
        
    def benchmark_batch_loading(self, num_batches: int = 10) -> Dict[str, float]:
        """
        Benchmark batch loading performance using the pre-configured dataloader.
        
        Args:
            num_batches: Number of batches to load for benchmarking
            
        Returns:
            Dictionary containing timing statistics:
            - 'average_time': Average time per batch (seconds)
            - 'total_time': Total time for all batches (seconds)
            - 'min_time': Minimum time for a single batch (seconds)
            - 'max_time': Maximum time for a single batch (seconds)
            - 'median_time': Median time per batch (seconds)
            - 'std_dev': Standard deviation of batch times (seconds)
            - 'batches_per_second': Average number of batches per second
            - 'ci_lower': Lower bound of 95% bootstrap CI for average time
            - 'ci_upper': Upper bound of 95% bootstrap CI for average time
        """
        if self.verbose:
            print(f"Benchmarking dataloader with num_batches={num_batches}")
        
        # Measure loading times
        batch_times = []
        
        start_total = time.time()
        
        # Create iterator from dataloader
        dataloader_iter = iter(self.dataloader)
        
        # Create progress bar
        progress_bar = tqdm(
            range(num_batches),
            desc="Benchmarking batches",
            unit="batch",
            disable=not self.verbose
        )
        
        for i in progress_bar:
            try:
                start_batch = time.time()
                # This is where the actual loading happens
                batch = next(dataloader_iter)
                # Ensure the batch is actually accessed to force full loading
                if isinstance(batch, dict):
                    for value in batch.values():
                        if hasattr(value, 'shape'):
                            _ = value.shape
                elif hasattr(batch, 'shape'):
                    _ = batch.shape
                end_batch = time.time()
                
                batch_time = end_batch - start_batch
                batch_times.append(batch_time)
                
                # Update progress bar with current batch time
                progress_bar.set_postfix({
                    'batch_time': f'{batch_time*1000:.2f}ms',
                    'avg_time': f'{statistics.mean(batch_times)*1000:.2f}ms'
                })
                
            except StopIteration:
                # No more batches available
                break
        
        progress_bar.close()
        end_total = time.time()
        total_time = end_total - start_total
        
        # Calculate statistics
        if batch_times:
            average_time = statistics.mean(batch_times)
            min_time = min(batch_times)
            max_time = max(batch_times)
            median_time = statistics.median(batch_times)
            std_dev = statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0
            batches_per_second = len(batch_times) / total_time if total_time > 0 else 0.0
            
            # Calculate 95% bootstrap confidence interval
            ci_lower, ci_upper = self._bootstrap_ci(batch_times, confidence=0.95, n_bootstrap=1000)
        else:
            average_time = min_time = max_time = median_time = std_dev = batches_per_second = 0.0
            ci_lower = ci_upper = 0.0
        
        results = {
            'average_time': average_time,
            'total_time': total_time,
            'min_time': min_time,
            'max_time': max_time,
            'median_time': median_time,
            'std_dev': std_dev,
            'batches_per_second': batches_per_second,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'num_batches': len(batch_times),
        }
        
        if self.verbose:
            self._print_results(results)
            
        return results
    
    def _bootstrap_ci(self, data: List[float], confidence: float = 0.95, n_bootstrap: int = 1000) -> tuple:
        """
        Calculate bootstrap confidence interval for the mean.
        
        Args:
            data: List of observed values
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound) for the confidence interval
        """
        if len(data) < 2:
            return (0.0, 0.0)
        
        bootstrap_means = []
        n = len(data)
        
        # Use random module with manual implementation to avoid numpy dependency
        import random
        random.seed(42)  # For reproducibility
        
        for _ in range(n_bootstrap):
            # Bootstrap sample: sample with replacement
            bootstrap_sample = [random.choice(data) for _ in range(n)]
            bootstrap_means.append(statistics.mean(bootstrap_sample))
        
        # Calculate percentiles for confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        bootstrap_means.sort()
        lower_idx = int(lower_percentile / 100 * len(bootstrap_means))
        upper_idx = int(upper_percentile / 100 * len(bootstrap_means))
        
        # Ensure indices are within bounds
        lower_idx = max(0, min(lower_idx, len(bootstrap_means) - 1))
        upper_idx = max(0, min(upper_idx, len(bootstrap_means) - 1))
        
        return (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
    
    def _print_results(self, results: Dict[str, float]) -> None:
        """Print benchmark results in a formatted way."""
        print("Benchmark Results:")
        print(f"  Average time per batch: {results['average_time']*1000:.2f}ms")
        print(f"  95% CI: [{results['ci_lower']*1000:.2f}ms, {results['ci_upper']*1000:.2f}ms]")
        print(f"  Total time: {results['total_time']:.4f}s")
        print(f"  Min/Max time: {results['min_time']*1000:.2f}ms / {results['max_time']*1000:.2f}ms")
        print(f"  Median time: {results['median_time']*1000:.2f}ms")
        print(f"  Standard deviation: {results['std_dev']*1000:.2f}ms")
        print(f"  Batches per second: {results['batches_per_second']:.2f}")
        print(f"  Batches loaded: {results['num_batches']}")


if __name__ == "__main__":
    """
    Example usage of the DataloaderBenchmark class.
    Define the dataloader in main and then benchmark it.
    """
    # Configuration parameters
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 8
    SHUFFLE = True
    
    print("Creating example dataset for benchmarking...")
    
    # Create dataset using the factory
    dataset_factory = MakePurelyObservationalDataset(
        scm_config=prior_config,
        preprocessing_config=preprocessing_config,
        dataset_config=dataset_config,
    )
    dataset = dataset_factory.create_dataset(seed=42)
    print(f"Dataset created with {len(dataset)} items")

    # Define the dataloader with specific configuration
    print("\nCreating DataLoader with:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Num workers: {NUM_WORKERS}")
    print(f"  Prefetch factor: {PREFETCH_FACTOR}")
    print(f"  Shuffle: {SHUFFLE}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE, 
        num_workers=NUM_WORKERS, 
        prefetch_factor=PREFETCH_FACTOR
    )

    # Create benchmark instance with the pre-configured dataloader
    benchmark = DataloaderBenchmark(dataloader, verbose=True)
    
    print("\n" + "="*60)
    print("DATALOADER BENCHMARK WITH 95% BOOTSTRAP CIs")
    print("="*60)
    
    # Benchmark the dataloader
    results = benchmark.benchmark_batch_loading(num_batches=50)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)
