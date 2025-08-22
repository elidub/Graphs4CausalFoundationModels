"""
Stress testing configuration sampling with timing and error analysis.

This script samples many datasets according to a passed configuration, records timing
with bootstrap confidence intervals, and tracks any errors that occur during sampling.
"""

import time
import traceback
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback for when tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable

from priors.causal_prior.scm.SCMHyperparameterSampler import SCMHyperparameterSampler
from priors.causal_prior.scm.SCMBuilder import SCMBuilder
from priors.causal_prior.scm.Basic_Configs import default_sampling_config


@dataclass
class SamplingResult:
    """Container for individual sampling attempt results."""
    success: bool
    duration: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    num_nodes: Optional[int] = None
    num_samples: Optional[int] = None
    scm_params: Optional[Dict[str, Any]] = None
    has_nans: Optional[bool] = None
    nan_nodes: Optional[List[str]] = None
    total_nan_count: Optional[int] = None


@dataclass
class StressTestResults:
    """Container for overall stress test results."""
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    success_rate: float
    durations: List[float]
    mean_duration: float
    median_duration: float
    duration_std: float
    duration_ci_lower: float
    duration_ci_upper: float
    error_counts: Dict[str, int]
    error_examples: Dict[str, List[str]]
    detailed_results: List[SamplingResult]
    nan_attempts: int
    nan_rate: float
    nan_node_counts: Dict[str, int]


class ConfigStressTester:
    """Stress tester for SCM configuration sampling."""
    
    def __init__(self, config: Dict[str, Any], base_seed: int = 42):
        """
        Initialize stress tester.
        
        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary for SCMHyperparameterSampler
        base_seed : int, default=42
            Base seed for reproducibility
        """
        self.config = config
        self.base_seed = base_seed
    
    def _check_for_nans(self, data: Dict[str, Any]) -> Tuple[bool, List[str], int]:
        """
        Check for NaN values in the sampled data.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary of node names to tensor data
            
        Returns
        -------
        Tuple[bool, List[str], int]
            (has_nans, nodes_with_nans, total_nan_count)
        """
        has_nans = False
        nan_nodes = []
        total_nan_count = 0
        
        for node_name, tensor_data in data.items():
            if isinstance(tensor_data, torch.Tensor):
                node_nan_mask = torch.isnan(tensor_data)
                if node_nan_mask.any():
                    has_nans = True
                    nan_nodes.append(node_name)
                    total_nan_count += node_nan_mask.sum().item()
            elif isinstance(tensor_data, (list, tuple, np.ndarray)):
                # Convert to numpy for checking
                arr = np.asarray(tensor_data)
                if np.isnan(arr).any():
                    has_nans = True
                    nan_nodes.append(node_name)
                    total_nan_count += np.isnan(arr).sum()
                    
        return has_nans, nan_nodes, total_nan_count
    
    def _print_error_with_config(self, result: SamplingResult, attempt_num: int):
        """Print error details along with the configuration that caused it."""
        print(f"\n{'='*60}")
        print(f"ERROR IN ATTEMPT {attempt_num + 1}")
        print(f"{'='*60}")
        
        if result.error_type:
            print(f"Error Type: {result.error_type}")
        if result.error_message:
            print(f"Error Message: {result.error_message}")
        
        if result.has_nans:
            print(f"NaN Detection: {result.total_nan_count} NaNs found in nodes: {result.nan_nodes}")
        
        if result.scm_params:
            print("\nConfiguration parameters that led to this error:")
            for key, value in result.scm_params.items():
                print(f"  {key}: {value}")
        
        print(f"{'='*60}\n")
        
    def sample_single_dataset(self, 
                             seed: int, 
                             num_samples: int = 256, 
                             timeout: float = 30.0) -> SamplingResult:
        """
        Sample a single dataset and record timing/errors.
        
        Parameters
        ----------
        seed : int
            Random seed for this attempt
        num_samples : int, default=256
            Number of samples to generate in the dataset
        timeout : float, default=30.0
            Maximum time to allow for sampling (seconds)
            
        Returns
        -------
        SamplingResult
            Results of the sampling attempt
        """
        start_time = time.time()
        
        try:
            # Create sampler with unique seed
            sampler = SCMHyperparameterSampler(self.config, seed=seed)
            
            # Sample parameters
            sampled_params = sampler.sample()
            
            # Build SCM
            builder = SCMBuilder(**sampled_params)
            scm = builder.build()
            
            # Sample noise and propagate
            scm.sample_exogenous(num_samples=num_samples)
            scm.sample_endogenous_noise(num_samples=num_samples)
            
            # Generate dataset
            data = scm.propagate(num_samples=num_samples)
            
            # Check for NaNs in the data
            has_nans, nan_nodes, total_nan_count = self._check_for_nans(data)
            
            duration = time.time() - start_time
            
            # Check for timeout
            if duration > timeout:
                return SamplingResult(
                    success=False,
                    duration=duration,
                    error_type="TimeoutError",
                    error_message=f"Sampling took {duration:.2f}s, exceeding timeout of {timeout}s",
                    num_samples=num_samples,
                    scm_params=sampled_params,
                    has_nans=has_nans,
                    nan_nodes=nan_nodes,
                    total_nan_count=total_nan_count
                )
            
            # Check if we have NaNs - treat as a special error case
            if has_nans:
                return SamplingResult(
                    success=False,
                    duration=duration,
                    error_type="NaNError",
                    error_message=f"Generated data contains {total_nan_count} NaN values in nodes: {nan_nodes}",
                    num_nodes=sampled_params.get('num_nodes'),
                    num_samples=num_samples,
                    scm_params=sampled_params,
                    has_nans=has_nans,
                    nan_nodes=nan_nodes,
                    total_nan_count=total_nan_count
                )
            
            return SamplingResult(
                success=True,
                duration=duration,
                num_nodes=sampled_params.get('num_nodes'),
                num_samples=num_samples,
                scm_params=sampled_params,
                has_nans=has_nans,
                nan_nodes=nan_nodes,
                total_nan_count=total_nan_count
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_type = type(e).__name__
            error_message = str(e)
            error_traceback = traceback.format_exc()
            
            return SamplingResult(
                success=False,
                duration=duration,
                error_type=error_type,
                error_message=error_message,
                error_traceback=error_traceback,
                num_samples=num_samples
            )
    
    def run_stress_test(self, 
                       num_attempts: int = 1000,
                       num_samples: int = 256,
                       timeout: float = 30.0,
                       bootstrap_samples: int = 10000,
                       confidence_level: float = 0.95,
                       verbose: bool = True) -> StressTestResults:
        """
        Run stress test with many sampling attempts.
        
        Parameters
        ----------
        num_attempts : int, default=1000
            Number of sampling attempts to perform
        num_samples : int, default=256
            Number of samples per dataset
        timeout : float, default=30.0
            Timeout for individual sampling attempts
        bootstrap_samples : int, default=10000
            Number of bootstrap samples for confidence intervals
        confidence_level : float, default=0.95
            Confidence level for bootstrap CIs
        verbose : bool, default=True
            Whether to print progress information
            
        Returns
        -------
        StressTestResults
            Comprehensive results of the stress test
        """
        if verbose:
            print(f"Starting stress test with {num_attempts} attempts...")
            print(f"Dataset size: {num_samples} samples")
            print(f"Timeout per attempt: {timeout}s")
            print(f"Bootstrap samples for CIs: {bootstrap_samples}")
            print("-" * 60)
        
        results = []
        error_counts = defaultdict(int)
        error_examples = defaultdict(list)
        nan_node_counts = defaultdict(int)
        
        # Create progress bar
        if verbose and HAS_TQDM:
            pbar = tqdm(range(num_attempts), desc="Sampling datasets", unit="attempt")
        else:
            pbar = range(num_attempts)
        
        for i in pbar:
            seed = self.base_seed + i
            result = self.sample_single_dataset(seed, num_samples, timeout)
            results.append(result)
            
            if not result.success:
                error_type = result.error_type or "UnknownError"
                error_counts[error_type] += 1
                
                # Store a few examples of each error type
                if len(error_examples[error_type]) < 3:
                    error_examples[error_type].append(result.error_message or "No message")
                
                # Print error details immediately
                if verbose:
                    self._print_error_with_config(result, i)
            
            # Track NaN occurrences (even for successful ones, in case we want separate tracking)
            if result.has_nans and result.nan_nodes:
                for node in result.nan_nodes:
                    nan_node_counts[node] += 1
                    
            # Update progress bar description if using tqdm
            if verbose and HAS_TQDM and isinstance(pbar, tqdm):
                success_so_far = len([r for r in results if r.success])
                pbar.set_postfix({"Success": f"{success_so_far}/{i+1}", "Errors": f"{len(results) - success_so_far}"})
        
        if verbose and HAS_TQDM and isinstance(pbar, tqdm):
            pbar.close()
        
        # Calculate basic statistics
        successful_results = [r for r in results if r.success]
        durations = [r.duration for r in successful_results]
        
        # Calculate NaN statistics (including both successful and failed attempts)
        nan_results = [r for r in results if r.has_nans]
        nan_attempts = len(nan_results)
        nan_rate = nan_attempts / num_attempts
        
        if durations:
            mean_duration = np.mean(durations)
            median_duration = np.median(durations)
            duration_std = np.std(durations)
            
            # Bootstrap confidence intervals for mean duration
            ci_lower, ci_upper = self._bootstrap_ci(
                durations, bootstrap_samples, confidence_level
            )
        else:
            mean_duration = median_duration = duration_std = 0.0
            ci_lower = ci_upper = 0.0
        
        success_rate = len(successful_results) / num_attempts
        
        if verbose:
            print("\nStress test completed!")
            print(f"Success rate: {success_rate:.2%}")
            print(f"NaN rate: {nan_rate:.2%} ({nan_attempts}/{num_attempts} attempts had NaNs)")
            if durations:
                print(f"Mean duration: {mean_duration:.3f}s ± {duration_std:.3f}s")
                print(f"Duration 95% CI: [{ci_lower:.3f}s, {ci_upper:.3f}s]")
        
        return StressTestResults(
            total_attempts=num_attempts,
            successful_attempts=len(successful_results),
            failed_attempts=num_attempts - len(successful_results),
            success_rate=success_rate,
            durations=durations,
            mean_duration=mean_duration,
            median_duration=median_duration,
            duration_std=duration_std,
            duration_ci_lower=ci_lower,
            duration_ci_upper=ci_upper,
            error_counts=dict(error_counts),
            error_examples=dict(error_examples),
            detailed_results=results,
            nan_attempts=nan_attempts,
            nan_rate=nan_rate,
            nan_node_counts=dict(nan_node_counts)
        )
    
    def _bootstrap_ci(self, 
                     data: List[float], 
                     n_bootstrap: int, 
                     confidence_level: float) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for the mean."""
        if not data:
            return 0.0, 0.0
            
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ci_lower, ci_upper


def print_detailed_results(results: StressTestResults, show_errors: bool = True):
    """Print detailed analysis of stress test results."""
    print("\n" + "=" * 80)
    print("STRESS TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print("\nOverall Performance:")
    print(f"  Total attempts: {results.total_attempts}")
    print(f"  Successful: {results.successful_attempts} ({results.success_rate:.2%})")
    print(f"  Failed: {results.failed_attempts} ({1-results.success_rate:.2%})")
    
    print("\nNaN Analysis:")
    print(f"  Attempts with NaNs: {results.nan_attempts} ({results.nan_rate:.2%})")
    if results.nan_node_counts:
        print("  Nodes with NaNs (frequency):")
        for node, count in sorted(results.nan_node_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / results.total_attempts) * 100
            print(f"    {node}: {count} times ({percentage:.1f}%)")
    else:
        print("  No NaN values detected in any datasets!")
    
    if results.durations:
        print("\nTiming Analysis (successful attempts only):")
        print(f"  Mean duration: {results.mean_duration:.3f}s")
        print(f"  Median duration: {results.median_duration:.3f}s")
        print(f"  Standard deviation: {results.duration_std:.3f}s")
        print(f"  95% Confidence Interval: [{results.duration_ci_lower:.3f}s, {results.duration_ci_upper:.3f}s]")
        print(f"  Min duration: {min(results.durations):.3f}s")
        print(f"  Max duration: {max(results.durations):.3f}s")
    
    if results.error_counts and show_errors:
        print("\nError Analysis:")
        for error_type, count in sorted(results.error_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / results.total_attempts) * 100
            print(f"  {error_type}: {count} occurrences ({percentage:.1f}%)")
            
            if error_type in results.error_examples:
                print("    Example messages:")
                for i, example in enumerate(results.error_examples[error_type][:2], 1):
                    truncated_msg = (example[:100] + "...") if len(example) > 100 else example
                    print(f"      {i}. {truncated_msg}")


def main():
    """Main function for running stress tests."""
    # ===== CONFIGURATION PARAMETERS - MODIFY AS NEEDED =====
    NUM_ATTEMPTS = 1_000        # Number of sampling attempts
    NUM_SAMPLES = 1_000        # Number of samples per dataset
    TIMEOUT = 30.0           # Timeout per attempt (seconds)
    BASE_SEED = 42           # Base random seed
    # =========================================================
    
    # Use default config
    config = default_sampling_config
    
    print("Configuration Stress Testing")
    print("=" * 40)
    print("Using default sampling configuration")
    print(f"Attempts: {NUM_ATTEMPTS}")
    print(f"Samples per dataset: {NUM_SAMPLES}")
    print(f"Timeout: {TIMEOUT}s")
    print(f"Base seed: {BASE_SEED}")
    
    if not HAS_TQDM:
        print("Note: Install 'tqdm' for better progress tracking: pip install tqdm")
    
    # Run stress test
    tester = ConfigStressTester(config, base_seed=BASE_SEED)
    results = tester.run_stress_test(
        num_attempts=NUM_ATTEMPTS,
        num_samples=NUM_SAMPLES,
        timeout=TIMEOUT,
        verbose=True
    )
    
    # Display results
    print_detailed_results(results)


if __name__ == "__main__":
    main()
