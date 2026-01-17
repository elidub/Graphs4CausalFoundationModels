"""
ComplexMech2 Benchmark - Simple version that generates data from config.yaml.

This benchmark:
1. Loads config.yaml to define the SCM and dataset parameters
2. Creates an InterventionalDataset
3. Samples data for evaluation
"""

import os
import sys
import yaml
import pickle
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

from sklearn.metrics import r2_score, mean_squared_error

# Add repo root and src to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
src_path = os.path.join(repo_root, 'src')

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset
from priordata_processing.Datasets.Collator import BatchSplitCollator
from models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn


class ComplexMechBenchmark:
    """
    Simple benchmark class that generates data from a config.yaml file.
    
    Usage:
        benchmark = ComplexMechBenchmark()
        benchmark.generate_data(num_samples=100)
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the benchmark.
        
        Args:
            config_path: Path to config.yaml file (default: ./config/config.yaml)
            cache_dir: Directory to store cached data (default: ./data_cache)
            verbose: Whether to print progress messages
        """
        # Get the directory where this file is located
        self.benchmark_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        
        # Set config path
        if config_path is None:
            self.config_path = self.benchmark_dir / "config" / "config.yaml"
        else:
            self.config_path = Path(config_path)
        
        # Set cache directory
        if cache_dir is None:
            self.cache_dir = self.benchmark_dir / "data_cache"
        else:
            self.cache_dir = Path(cache_dir)
        
        self.verbose = verbose
        self.config = None
        self.dataset = None
        self.collator = None  # BatchSplitCollator for varying dataset sizes
        self.model = None  # GraphConditionedInterventionalPFNSklearn model
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"ComplexMech2 Benchmark initialized")
            print(f"  Benchmark dir: {self.benchmark_dir}")
            print(f"  Config path: {self.config_path}")
            print(f"  Cache dir: {self.cache_dir}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load the configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if self.verbose:
            print(f"Loaded config: {self.config_path}")
            print(f"  Experiment name: {self.config.get('experiment_name', 'N/A')}")
            print(f"  Mode: {self.config.get('mode', 'N/A')}")
        
        return self.config
    
    def create_collator(self) -> BatchSplitCollator:
        """
        Create a BatchSplitCollator from the dataset config.
        This handles varying train/test splits just like in training.
        
        Returns:
            BatchSplitCollator instance
        """
        if self.config is None:
            self.load_config()
        
        dataset_config = self.config.get('dataset_config', {})
        
        # Import torch.distributions for creating distribution objects
        import torch.distributions as dist
        
        # Helper to extract values from config (handles both plain values and dict with 'value' key)
        def _get_value(cfg_entry):
            """Extract value from config entry that may be plain value or dict."""
            if isinstance(cfg_entry, dict):
                if "value" in cfg_entry:
                    return cfg_entry["value"]
                # If it's a distribution config, return it as-is
                return cfg_entry
            return cfg_entry
        
        # Get max sizes from config
        max_total = int(_get_value(dataset_config.get("max_number_samples_per_dataset", 1000)))
        max_train_cap = int(_get_value(dataset_config.get("max_number_train_samples_per_dataset", max_total)))
        max_test_cap = int(_get_value(dataset_config.get("max_number_test_samples_per_dataset", max_total)))
        
        # Build distribution for n_test per dataset (must be torch.distributions.Distribution or int)
        n_test_cfg = dataset_config.get("n_test_samples_per_dataset")
        n_test_val = _get_value(n_test_cfg)
        
        if isinstance(n_test_val, dict) and n_test_val.get("distribution"):
            # Distribution-based test size - create actual torch.distributions object
            dist_type = n_test_val.get("distribution")
            params = n_test_val.get("distribution_parameters", {})
            
            if dist_type == "discrete_uniform":
                low = int(params.get("low", 0))
                high = int(params.get("high", max_total))
                # Use Uniform and cast to int inside collator
                n_test_dist = dist.Uniform(low=low, high=high)
            elif dist_type == "uniform":
                n_test_dist = dist.Uniform(low=float(params.get("low", 0)), high=float(params.get("high", max_total)))
            elif dist_type == "normal":
                n_test_dist = dist.Normal(loc=float(params.get("mean", max_total // 2)), scale=float(params.get("std", max(1, max_total/10))))
            else:
                # Fallback: fixed half split
                n_test_dist = max_total // 2
        elif isinstance(n_test_val, int):
            # Fixed test size
            n_test_dist = n_test_val
        else:
            # Default to half split
            n_test_dist = max_total // 2
        
        if self.verbose:
            print(f"Creating BatchSplitCollator:")
            print(f"  max_number_samples_per_dataset: {max_total}")
            print(f"  max_number_train_samples_per_dataset: {max_train_cap}")
            print(f"  max_number_test_samples_per_dataset: {max_test_cap}")
            print(f"  n_test_samples_distribution: {n_test_dist}")
        
        self.collator = BatchSplitCollator(
            max_number_samples_per_dataset=max_total,
            max_number_train_samples_per_dataset=max_train_cap,
            max_number_test_samples_per_dataset=max_test_cap,
            n_test_samples_distribution=n_test_dist,
        )
        
        return self.collator
    
    def create_dataset(self, seed: Optional[int] = None) -> InterventionalDataset:
        """
        Create an InterventionalDataset from the loaded config.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            InterventionalDataset instance
        """
        if self.config is None:
            self.load_config()
        
        # Extract config sections
        scm_config = self.config.get('scm_config', {})
        dataset_config = self.config.get('dataset_config', {})
        preprocessing_config = self.config.get('preprocessing_config', {})
        
        if self.verbose:
            print("\nCreating InterventionalDataset...")
            print(f"  SCM config keys: {list(scm_config.keys())}")
            print(f"  Dataset config keys: {list(dataset_config.keys())}")
            print(f"  Preprocessing config keys: {list(preprocessing_config.keys())}")
        
        # Create the dataset
        self.dataset = InterventionalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config,
            seed=seed,
        )
        
        if self.verbose:
            print(f"  Dataset created with {len(self.dataset)} potential samples")
        
        return self.dataset
    
    def generate_data(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        save_to_cache: bool = True,
        overwrite: bool = False,
        keep_in_memory: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate data samples from the dataset and save each to individual .pkl files.
        Uses BatchSplitCollator to vary train/test split sizes like in training.
        
        Args:
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            save_to_cache: Whether to save each sample to disk as individual .pkl file
            overwrite: Whether to overwrite existing cached files
            keep_in_memory: Whether to keep samples in memory (if False, only saves to disk)
            
        Returns:
            List of sampled data dictionaries (empty if keep_in_memory=False)
        """
        # Create dataset if not already created
        if self.dataset is None:
            self.create_dataset(seed=seed)
        
        # Create collator if not already created
        if self.collator is None:
            self.create_collator()
        
        # Create samples subdirectory
        samples_dir = self.cache_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\nGenerating {num_samples} data samples...")
            print(f"  Saving individual samples to: {samples_dir}")
            print(f"  Using BatchSplitCollator for varying train/test splits")
        
        # Sample data
        sampled_data = [] if keep_in_memory else None
        iterator = tqdm(range(num_samples), desc="Sampling") if self.verbose else range(num_samples)

        for idx in iterator:
            # Check if this sample already exists
            sample_path = samples_dir / f"sample_{idx:06d}.pkl"
            
            if sample_path.exists() and not overwrite:
                # Load existing sample
                with open(sample_path, 'rb') as f:
                    sample_dict = pickle.load(f)
                if keep_in_memory:
                    sampled_data.append(sample_dict)
                continue
            
            try:
                # Get sample from dataset
                raw_sample = self.dataset[idx]
                
                # Parse the raw sample tuple
                # Expected format from InterventionalDataset with ancestor matrix:
                # (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv) - 6 elements
                # (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adj_matrix) - 7 elements (adjacency or ancestor)
                
                if len(raw_sample) == 6:
                    X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv = raw_sample
                    ancestor_matrix = None
                elif len(raw_sample) == 7:
                    X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, ancestor_matrix = raw_sample
                else:
                    raise ValueError(f"Unexpected sample length: {len(raw_sample)}")
                
                # Apply collator to get varying train/test splits
                # Collator expects a batch (list of samples), so wrap in list
                batch = [raw_sample]
                collated_batch = self.collator(batch)
                
                # Extract the collated sample (first and only element in batch)
                if len(collated_batch) == 6:
                    X_obs_c, T_obs_c, Y_obs_c, X_intv_c, T_intv_c, Y_intv_c = collated_batch
                    ancestor_matrix_c = ancestor_matrix  # Use original if not in collated output
                elif len(collated_batch) == 7:
                    X_obs_c, T_obs_c, Y_obs_c, X_intv_c, T_intv_c, Y_intv_c, ancestor_matrix_c = collated_batch
                else:
                    raise ValueError(f"Unexpected collated batch length: {len(collated_batch)}")
                
                # Store sample as dictionary (using collated versions)
                sample_dict = {
                    'X_obs': X_obs_c.numpy() if torch.is_tensor(X_obs_c) else X_obs_c,
                    'T_obs': T_obs_c.numpy() if torch.is_tensor(T_obs_c) else T_obs_c,
                    'Y_obs': Y_obs_c.numpy() if torch.is_tensor(Y_obs_c) else Y_obs_c,
                    'X_intv': X_intv_c.numpy() if torch.is_tensor(X_intv_c) else X_intv_c,
                    'T_intv': T_intv_c.numpy() if torch.is_tensor(T_intv_c) else T_intv_c,
                    'Y_intv': Y_intv_c.numpy() if torch.is_tensor(Y_intv_c) else Y_intv_c,
                    'sample_idx': idx,
                }
                
                if ancestor_matrix_c is not None:
                    sample_dict['ancestor_matrix'] = ancestor_matrix_c.numpy() if torch.is_tensor(ancestor_matrix_c) else ancestor_matrix_c
                
                # Optionally keep in memory
                if keep_in_memory:
                    sampled_data.append(sample_dict)

                # Save individual sample to disk
                if save_to_cache:
                    with open(sample_path, 'wb') as f:
                        pickle.dump(sample_dict, f)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to sample index {idx}: {e}")
                continue
        
        if self.verbose:
            if keep_in_memory:
                print(f"  Successfully processed {len(sampled_data)} datasets")
            else:
                # Count files present
                cached_files = len(list(samples_dir.glob("sample_*.pkl")))
                print(f"  Successfully processed (files saved) {cached_files} datasets")
            if save_to_cache:
                print(f"  Samples saved to: {samples_dir}")

        return sampled_data if keep_in_memory else []
    
    def load_sample(self, idx: int) -> Dict[str, Any]:
        """
        Load a single cached sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary
        """
        sample_path = self.cache_dir / "samples" / f"sample_{idx:06d}.pkl"
        
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample not found: {sample_path}")
        
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)
        
        return sample
    
    def load_all_samples(self) -> List[Dict[str, Any]]:
        """
        Load all cached samples from disk.
        
        Returns:
            List of sample dictionaries
        """
        samples_dir = self.cache_dir / "samples"
        
        if not samples_dir.exists():
            raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
        
        # Find all sample files
        sample_files = sorted(samples_dir.glob("sample_*.pkl"))
        
        if self.verbose:
            print(f"Loading {len(sample_files)} samples from: {samples_dir}")
        
        samples = []
        for sample_path in sample_files:
            with open(sample_path, 'rb') as f:
                samples.append(pickle.load(f))
        
        return samples
    
    def load_model(
        self,
        config_path: str,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> GraphConditionedInterventionalPFNSklearn:
        """
        Load a GraphConditionedInterventionalPFNSklearn model.
        
        Args:
            config_path: Path to the model config YAML file
            checkpoint_path: Path to the model checkpoint (.pt file)
            device: Device for inference ('cpu', 'cuda', etc.). Auto-detects if None.
            
        Returns:
            Loaded model
        """
        if self.verbose:
            print(f"\nLoading model...")
            print(f"  Config: {config_path}")
            print(f"  Checkpoint: {checkpoint_path}")
        
        self.model = GraphConditionedInterventionalPFNSklearn(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            verbose=self.verbose,
        )
        self.model.load()
        
        # Store config path for later use
        self._model_config_path = config_path
        self._model_checkpoint_path = checkpoint_path
        
        if self.verbose:
            print(f"  Model loaded successfully")
        
        return self.model
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        model: Optional[GraphConditionedInterventionalPFNSklearn] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a single sample and return metrics.
        
        Args:
            sample: Sample dictionary with X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, ancestor_matrix
            model: Model to evaluate (default: use self.model)
            
        Returns:
            Dictionary with 'mse', 'r2', 'nll' metrics
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Extract data
        X_obs = sample['X_obs']
        T_obs = sample['T_obs']
        Y_obs = sample['Y_obs']
        X_intv = sample['X_intv']
        T_intv = sample['T_intv']
        Y_intv = sample['Y_intv']
        ancestor_matrix = sample.get('ancestor_matrix', None)
        
        if ancestor_matrix is None:
            raise ValueError("Sample missing ancestor_matrix")
        
        # Ensure correct shapes
        if T_obs.ndim == 1:
            T_obs = T_obs.reshape(-1, 1)
        if T_intv.ndim == 1:
            T_intv = T_intv.reshape(-1, 1)
        if Y_obs.ndim == 2:
            Y_obs = Y_obs.flatten()
        if Y_intv.ndim == 2:
            Y_intv = Y_intv.flatten()
        
        # Get predictions
        pred = model.predict(
            X_obs=X_obs,
            T_obs=T_obs,
            Y_obs=Y_obs,
            X_intv=X_intv,
            T_intv=T_intv,
            adjacency_matrix=ancestor_matrix,
            prediction_type="mean",
            batched=False,
        )
        
        # Compute MSE
        mse = float(mean_squared_error(Y_intv, pred))
        
        # Compute R2
        r2 = float(r2_score(Y_intv, pred))
        
        # Compute NLL (negative log-likelihood)
        try:
            log_likelihood = model.log_likelihood(
                X_obs=X_obs,
                T_obs=T_obs,
                Y_obs=Y_obs,
                X_intv=X_intv,
                T_intv=T_intv,
                Y_intv=Y_intv,
                adjacency_matrix=ancestor_matrix,
                batched=False,
            )
            nll = float(-np.mean(log_likelihood))
        except Exception as e:
            if self.verbose:
                print(f"  Warning: NLL computation failed: {e}")
            nll = float('nan')
        
        return {
            'mse': mse,
            'r2': r2,
            'nll': nll,
        }
    
    def run_evaluation(
        self,
        model: Optional[GraphConditionedInterventionalPFNSklearn] = None,
        max_samples: Optional[int] = None,
        save_results: bool = True,
        results_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation on all cached samples.
        
        Args:
            model: Model to evaluate (default: use self.model)
            max_samples: Maximum number of samples to evaluate (default: all)
            save_results: Whether to save results to disk
            results_name: Name for results file (default: auto-generated)
            
        Returns:
            List of result dictionaries, one per sample
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Load all samples
        samples = self.load_all_samples()
        
        if max_samples is not None and len(samples) > max_samples:
            samples = samples[:max_samples]
            if self.verbose:
                print(f"Limiting to {max_samples} samples")
        
        if self.verbose:
            print(f"\nRunning evaluation on {len(samples)} samples...")
        
        results = []
        iterator = tqdm(samples, desc="Evaluating") if self.verbose else samples
        
        for sample in iterator:
            try:
                metrics = self.evaluate_sample(sample, model=model)
                result = {
                    'sample_idx': sample['sample_idx'],
                    'mse': metrics['mse'],
                    'r2': metrics['r2'],
                    'nll': metrics['nll'],
                    'n_obs': sample['X_obs'].shape[0],
                    'n_intv': sample['X_intv'].shape[0],
                    'n_features': sample['X_obs'].shape[1],
                }
                results.append(result)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to evaluate sample {sample.get('sample_idx', '?')}: {e}")
                results.append({
                    'sample_idx': sample.get('sample_idx', -1),
                    'mse': float('nan'),
                    'r2': float('nan'),
                    'nll': float('nan'),
                    'error': str(e),
                })
        
        if self.verbose:
            # Print summary statistics
            valid_results = [r for r in results if not np.isnan(r.get('mse', float('nan')))]
            if valid_results:
                mse_values = [r['mse'] for r in valid_results]
                r2_values = [r['r2'] for r in valid_results]
                nll_values = [r['nll'] for r in valid_results if not np.isnan(r['nll'])]
                
                print(f"\nEvaluation Summary ({len(valid_results)}/{len(results)} samples):")
                print(f"  MSE:  mean={np.mean(mse_values):.4f}, std={np.std(mse_values):.4f}")
                print(f"  R2:   mean={np.mean(r2_values):.4f}, std={np.std(r2_values):.4f}")
                if nll_values:
                    print(f"  NLL:  mean={np.mean(nll_values):.4f}, std={np.std(nll_values):.4f}")
        
        # Save results
        if save_results:
            results_dir = self.cache_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            if results_name is None:
                # Generate name from checkpoint path
                if hasattr(self, '_model_checkpoint_path'):
                    ckpt_name = Path(self._model_checkpoint_path).stem
                    results_name = f"eval_{ckpt_name}"
                else:
                    results_name = "eval_results"
            
            # Save individual results as JSON
            results_path = results_dir / f"{results_name}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if self.verbose:
                print(f"\nResults saved to: {results_path}")
        
        return results
    
    def get_sample_info(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about a sample.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Dictionary with sample information
        """
        info = {
            'n_obs_samples': sample['X_obs'].shape[0] if sample['X_obs'] is not None else 0,
            'n_intv_samples': sample['X_intv'].shape[0] if sample['X_intv'] is not None else 0,
            'n_features': sample['X_obs'].shape[1] if sample['X_obs'] is not None and len(sample['X_obs'].shape) > 1 else 0,
            'has_ancestor_matrix': 'ancestor_matrix' in sample and sample['ancestor_matrix'] is not None,
        }
        
        if info['has_ancestor_matrix']:
            info['ancestor_matrix_shape'] = sample['ancestor_matrix'].shape
        
        return info


# Simple test / example usage
if __name__ == "__main__":
    print("=" * 60)
    print("ComplexMech2 Benchmark - Data Generation Test")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = ComplexMechBenchmark(verbose=True)
    
    # Load config
    config = benchmark.load_config()
    
    # Create dataset
    dataset = benchmark.create_dataset(seed=42)
    
    # Create collator (will vary train/test sizes)
    collator = benchmark.create_collator()
    
    # Generate a few samples (saved as individual .pkl files)
    # Each sample will have DIFFERENT train/test splits due to collator
    samples = benchmark.generate_data(
        num_samples=5,
        seed=42,
        save_to_cache=True,
        overwrite=True
    )
    
    # Print info about generated samples
    print("\n" + "=" * 60)
    print("Sample Information:")
    print("=" * 60)
    for i, sample in enumerate(samples):
        info = benchmark.get_sample_info(sample)
        print(f"\nSample {i}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Test loading individual sample
    print("\n" + "=" * 60)
    print("Testing individual sample loading:")
    print("=" * 60)
    loaded_sample = benchmark.load_sample(0)
    print(f"Loaded sample 0: keys = {list(loaded_sample.keys())}")
    
    # Test loading all samples
    print("\n" + "=" * 60)
    print("Testing loading all samples:")
    print("=" * 60)
    all_samples = benchmark.load_all_samples()
    print(f"Loaded {len(all_samples)} samples total")
    
    # Test model evaluation (uncomment and set path to test)
    # print("\n" + "=" * 60)
    # print("Testing Model Evaluation:")
    # print("=" * 60)
    # 
    # model_checkpoint = "/path/to/model.ckpt"
    # model_config = {
    #     "model_path": "/path/to/model/directory",  # Optional, directory with config.yaml
    # }
    # 
    # # Load model
    # model = benchmark.load_model(
    #     model_checkpoint=model_checkpoint,
    #     model_config=model_config
    # )
    # 
    # # Evaluate all samples
    # results = benchmark.evaluate_all_samples(
    #     model_checkpoint=model_checkpoint,
    #     model_config=model_config,
    #     save_results=True
    # )
    # 
    # # Print aggregate results
    # print(f"\nAggregate Results (over {len(results)} samples):")
    # print(f"  Mean MSE: {np.mean([r['mse'] for r in results]):.4f}")
    # print(f"  Mean R2:  {np.mean([r['r2'] for r in results]):.4f}")
    # print(f"  Mean NLL: {np.mean([r['nll'] for r in results]):.4f}")
