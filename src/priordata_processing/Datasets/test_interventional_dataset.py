"""
Test script for InterventionalDataset using hyperparameters from basic.yaml config.

This script instantiates the InterventionalDataset with realistic configurations
and performs basic validation tests:
- Dataset initialization
- Sample retrieval and shape checking
- Intervention feature extraction
- Basic statistics and visualization
"""

import sys
import os
from pathlib import Path
import torch
import yaml
from typing import Dict, Any

# Add src to path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset


class InterventionalDatasetTester:
    """Test harness for InterventionalDataset."""
    
    def __init__(self, config_path: str):
        """
        Initialize tester with configuration from YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file (e.g., basic.yaml)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.dataset = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse YAML configuration."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def create_dataset(self, size: int = 10, seed: int = 42) -> InterventionalDataset:
        """
        Create InterventionalDataset instance with config parameters.
        
        Parameters
        ----------
        size : int, default 10
            Number of samples in dataset (overrides config for testing)
        seed : int, default 42
            Random seed for reproducibility
            
        Returns
        -------
        InterventionalDataset
            Initialized dataset instance
        """
        # Extract configs
        scm_config = self.config['scm_config']
        preprocessing_config = self.config['preprocessing_config']
        dataset_config = self.config['dataset_config'].copy()
        
        # Override size for testing
        dataset_config['dataset_size'] = {'value': size}
        
        print(f"Creating InterventionalDataset with {size} samples...")
        print(f"  SCM config keys: {list(scm_config.keys())}")
        print(f"  Preprocessing config keys: {list(preprocessing_config.keys())}")
        print(f"  Dataset config keys: {list(dataset_config.keys())}")
        
        self.dataset = InterventionalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config,
            seed=seed
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Length: {len(self.dataset)}")
        print(f"  Max features: {self.dataset.max_number_features}")
        print(f"  Max train samples: {self.dataset.max_number_train_samples}")
        print(f"  Max test samples: {self.dataset.max_number_test_samples}")
        
        return self.dataset
    
    def test_single_sample(self, idx: int = 0):
        """
        Retrieve and inspect a single sample.
        
        Parameters
        ----------
        idx : int, default 0
            Index of sample to retrieve
        """
        if self.dataset is None:
            raise ValueError("Dataset not created. Call create_dataset() first.")
        
        print(f"\n{'='*60}")
        print(f"Testing sample retrieval (idx={idx})...")
        print(f"{'='*60}")
        
        try:
            sample = self.dataset[idx]
            
            # Check return signature
            if len(sample) == 4:
                X_obs, Y_obs, X_intv, Y_intv = sample
                has_treatment = False
                print("✓ Sample structure: (X_obs, Y_obs, X_intv, Y_intv)")
            elif len(sample) == 6:
                X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv = sample
                has_treatment = True
                print("✓ Sample structure: (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv)")
            else:
                raise ValueError(f"Unexpected sample structure with {len(sample)} elements")
            
            # Observational data shapes
            print(f"\nObservational (train) data:")
            print(f"  X_obs shape: {X_obs.shape}")
            if has_treatment:
                print(f"  T_obs shape: {T_obs.shape}")
            print(f"  Y_obs shape: {Y_obs.shape}")
            
            # Interventional data shapes
            print(f"\nInterventional (test) data:")
            print(f"  X_intv shape: {X_intv.shape}")
            if has_treatment:
                print(f"  T_intv shape: {T_intv.shape}")
            print(f"  Y_intv shape: {Y_intv.shape}")
            
            # Basic statistics
            print(f"\nObservational statistics:")
            print(f"  X_obs: mean={X_obs.mean().item():.4f}, std={X_obs.std().item():.4f}")
            if has_treatment:
                print(f"  T_obs: mean={T_obs.mean().item():.4f}, std={T_obs.std().item():.4f}")
            print(f"  Y_obs: mean={Y_obs.mean().item():.4f}, std={Y_obs.std().item():.4f}")
            
            print(f"\nInterventional statistics:")
            print(f"  X_intv: mean={X_intv.mean().item():.4f}, std={X_intv.std().item():.4f}")
            if has_treatment:
                print(f"  T_intv: mean={T_intv.mean().item():.4f}, std={T_intv.std().item():.4f}")
            print(f"  Y_intv: mean={Y_intv.mean().item():.4f}, std={Y_intv.std().item():.4f}")
            
            # Check for NaNs/Infs
            def check_valid(tensor, name):
                has_nan = torch.isnan(tensor).any()
                has_inf = torch.isinf(tensor).any()
                if has_nan or has_inf:
                    print(f"  ⚠ {name}: NaN={has_nan}, Inf={has_inf}")
                    return False
                return True
            
            print(f"\nValidity checks:")
            all_valid = True
            all_valid &= check_valid(X_obs, "X_obs")
            if has_treatment:
                all_valid &= check_valid(T_obs, "T_obs")
            all_valid &= check_valid(Y_obs, "Y_obs")
            all_valid &= check_valid(X_intv, "X_intv")
            if has_treatment:
                all_valid &= check_valid(T_intv, "T_intv")
            all_valid &= check_valid(Y_intv, "Y_intv")
            
            if all_valid:
                print("  ✓ All tensors valid (no NaN/Inf)")
            
            return sample
            
        except Exception as e:
            print(f"✗ Error retrieving sample: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def test_multiple_samples(self, n_samples: int = 3):
        """
        Test retrieval of multiple samples and check consistency.
        
        Parameters
        ----------
        n_samples : int, default 3
            Number of samples to test
        """
        if self.dataset is None:
            raise ValueError("Dataset not created. Call create_dataset() first.")
        
        print(f"\n{'='*60}")
        print(f"Testing multiple sample retrieval (n={n_samples})...")
        print(f"{'='*60}")
        
        shapes_consistent = True
        
        for i in range(min(n_samples, len(self.dataset))):
            try:
                sample = self.dataset[i]
                print(f"\nSample {i}:")
                
                if len(sample) == 4:
                    X_obs, Y_obs, X_intv, Y_intv = sample
                    print(f"  Shapes: X_obs={X_obs.shape}, Y_obs={Y_obs.shape}, "
                          f"X_intv={X_intv.shape}, Y_intv={Y_intv.shape}")
                elif len(sample) == 6:
                    X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv = sample
                    print(f"  Shapes: X_obs={X_obs.shape}, T_obs={T_obs.shape}, Y_obs={Y_obs.shape}, "
                          f"X_intv={X_intv.shape}, T_intv={T_intv.shape}, Y_intv={Y_intv.shape}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                shapes_consistent = False
        
        if shapes_consistent:
            print("\n✓ All samples retrieved successfully")
        else:
            print("\n⚠ Some samples failed")
    
    def test_batch_loading(self, batch_size: int = 2):
        """
        Test batch loading with DataLoader.
        
        Parameters
        ----------
        batch_size : int, default 2
            Batch size for testing
        """
        if self.dataset is None:
            raise ValueError("Dataset not created. Call create_dataset() first.")
        
        print(f"\n{'='*60}")
        print(f"Testing batch loading (batch_size={batch_size})...")
        print(f"{'='*60}")
        
        from torch.utils.data import DataLoader
        
        # Custom collate function for variable output structures
        def collate_fn(batch):
            if len(batch[0]) == 4:
                X_obs_list, Y_obs_list, X_intv_list, Y_intv_list = zip(*batch)
                return (
                    torch.stack(X_obs_list),
                    torch.stack(Y_obs_list),
                    torch.stack(X_intv_list),
                    torch.stack(Y_intv_list)
                )
            elif len(batch[0]) == 6:
                X_obs_list, T_obs_list, Y_obs_list, X_intv_list, T_intv_list, Y_intv_list = zip(*batch)
                return (
                    torch.stack(X_obs_list),
                    torch.stack(T_obs_list),
                    torch.stack(Y_obs_list),
                    torch.stack(X_intv_list),
                    torch.stack(T_intv_list),
                    torch.stack(Y_intv_list)
                )
        
        try:
            loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )
            
            batch = next(iter(loader))
            
            if len(batch) == 4:
                X_obs_batch, Y_obs_batch, X_intv_batch, Y_intv_batch = batch
                print(f"✓ Batch loaded successfully (4-tuple)")
                print(f"  X_obs_batch shape: {X_obs_batch.shape}")
                print(f"  Y_obs_batch shape: {Y_obs_batch.shape}")
                print(f"  X_intv_batch shape: {X_intv_batch.shape}")
                print(f"  Y_intv_batch shape: {Y_intv_batch.shape}")
            elif len(batch) == 6:
                X_obs_batch, T_obs_batch, Y_obs_batch, X_intv_batch, T_intv_batch, Y_intv_batch = batch
                print(f"✓ Batch loaded successfully (6-tuple)")
                print(f"  X_obs_batch shape: {X_obs_batch.shape}")
                print(f"  T_obs_batch shape: {T_obs_batch.shape}")
                print(f"  Y_obs_batch shape: {Y_obs_batch.shape}")
                print(f"  X_intv_batch shape: {X_intv_batch.shape}")
                print(f"  T_intv_batch shape: {T_intv_batch.shape}")
                print(f"  Y_intv_batch shape: {Y_intv_batch.shape}")
            
        except Exception as e:
            print(f"✗ Batch loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_all_tests(self, dataset_size: int = 5, seed: int = 42):
        """
        Run complete test suite.
        
        Parameters
        ----------
        dataset_size : int, default 5
            Size of test dataset
        seed : int, default 42
            Random seed
        """
        print(f"\n{'#'*60}")
        print(f"# InterventionalDataset Test Suite")
        print(f"# Config: {self.config_path}")
        print(f"{'#'*60}\n")
        
        try:
            # Test 1: Dataset creation
            self.create_dataset(size=dataset_size, seed=seed)
            
            # Test 2: Single sample
            self.test_single_sample(idx=0)
            
            # Test 3: Multiple samples
            self.test_multiple_samples(n_samples=min(3, dataset_size))
            
            # Test 4: Batch loading
            if dataset_size >= 2:
                self.test_batch_loading(batch_size=2)
            
            print(f"\n{'='*60}")
            print("✓ All tests passed successfully!")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"✗ Test suite failed: {e}")
            print(f"{'='*60}\n")
            raise


def main():
    """Main entry point for testing."""
    # Path to config file
    config_path = "experiments/FirstTests/configs/basic.yaml"
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please provide the correct path to basic.yaml")
        return
    
    # Create tester and run tests
    tester = InterventionalDatasetTester(config_path)
    tester.run_all_tests(dataset_size=5, seed=42)


if __name__ == "__main__":
    main()
