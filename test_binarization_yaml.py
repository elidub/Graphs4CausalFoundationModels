#!/usr/bin/env python3
"""
Test script to verify treatment binarization in InterventionalDataset.

Loads the config from the provided YAML file and tests the binarization feature.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import yaml
import torch
from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset


def load_config(yaml_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def test_binarization_from_yaml():
    """Test binarization using the YAML config file."""
    
    # Load the config
    yaml_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FinalModel/configs_early/final_earlytest_binarize.yaml"
    config = load_config(yaml_path)
    
    scm_config = config['scm_config']
    preprocessing_config = config['preprocessing_config']
    dataset_config = config['dataset_config'].copy()
    
    print("=" * 60)
    print("Testing InterventionalDataset Treatment Binarization")
    print(f"Config file: {yaml_path}")
    print("=" * 60)
    
    # Check what binarize_treatment_prob is set to
    binarize_prob = dataset_config.get('binarize_treatment_prob', {}).get('value', 'not set')
    print(f"\nCurrent binarize_treatment_prob: {binarize_prob}")
    
    # =========================================================================
    # TEST 1: Binarization WITH rejection sampling disabled
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Binarization WITH rejection sampling DISABLED")
    print("=" * 60)
    
    dataset_config_no_reject = dataset_config.copy()
    dataset_config_no_reject['binarize_treatment_prob'] = {'value': 1.0}
    dataset_config_no_reject['dataset_size'] = {'value': 100}
    dataset_config_no_reject['min_target_variance'] = {'value': None}
    dataset_config_no_reject['min_unique_target_fraction'] = {'value': None}
    
    print(f"\nmin_target_variance: None (disabled)")
    print(f"min_unique_target_fraction: None (disabled)")
    
    try:
        dataset_no_reject = InterventionalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config_no_reject,
        )
        
        num_test = 5
        binary_count = 0
        
        for i in range(num_test):
            sample = dataset_no_reject[i]
            
            if isinstance(sample, tuple) and len(sample) >= 2:
                T_obs = sample[1]
                unique_vals = torch.unique(T_obs)
                if len(unique_vals) <= 2:
                    binary_count += 1
                print(f"   Sample {i}: {len(unique_vals)} unique treatment values")
        
        print(f"\n   Binary treatment datasets: {binary_count}/{num_test}")
        if binary_count == num_test:
            print(f"   ✓ SUCCESS: All treatments are binary!")
        else:
            print(f"   ⚠ Some treatments are not binary")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================================================================
    # TEST 2: Binarization WITH rejection sampling ENABLED
    # =========================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Binarization WITH rejection sampling ENABLED")
    print("=" * 60)
    
    dataset_config_with_reject = dataset_config.copy()
    dataset_config_with_reject['binarize_treatment_prob'] = {'value': 1.0}
    dataset_config_with_reject['dataset_size'] = {'value': 100}
    # Keep rejection sampling enabled (from original config)
    # min_target_variance: 1e-2
    # min_unique_target_fraction: 0.2
    
    min_var = dataset_config_with_reject.get('min_target_variance', {}).get('value', 'not set')
    min_unique = dataset_config_with_reject.get('min_unique_target_fraction', {}).get('value', 'not set')
    print(f"\nmin_target_variance: {min_var}")
    print(f"min_unique_target_fraction: {min_unique}")
    print(f"max_resample_attempts: {dataset_config_with_reject.get('max_resample_attempts', {}).get('value', 'not set')}")
    
    try:
        dataset_with_reject = InterventionalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config_with_reject,
        )
        
        num_test = 5
        binary_count = 0
        
        for i in range(num_test):
            sample = dataset_with_reject[i]
            
            if isinstance(sample, tuple) and len(sample) >= 2:
                T_obs = sample[1]
                unique_vals = torch.unique(T_obs)
                if len(unique_vals) <= 2:
                    binary_count += 1
                print(f"   Sample {i}: {len(unique_vals)} unique treatment values")
        
        print(f"\n   Binary treatment datasets: {binary_count}/{num_test}")
        if binary_count == num_test:
            print(f"   ✓ SUCCESS: All treatments are binary even with rejection sampling!")
        elif binary_count > 0:
            print(f"   ⚠ PARTIAL: Some treatments are binary ({binary_count}/{num_test})")
            print(f"   Note: Rejection sampling may be rejecting some binarized samples")
        else:
            print(f"   ✗ FAIL: No binary treatments - rejection sampling is likely rejecting them all")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_binarization_from_yaml()
    sys.exit(0 if success else 1)
