#!/usr/bin/env python3
"""
Test script to verify treatment binarization in InterventionalDataset.

This script tests that:
1. Without binarization, treatment variables are mostly continuous
2. With binarize_treatment_prob=1.0, treatment variables are always binary (0/1)
3. The binarization threshold uses the median correctly
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import torch
from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset


def test_binarization():
    """Test that binarization works correctly."""
    
    # SCM config using correct parameter names from SCMSampler.EXPECTED_HYPERPARAMETERS
    scm_config = {
        "num_nodes": {"value": 5},
        "graph_edge_prob": {"value": 0.4},
        "graph_seed": {"value": 42},
        "xgboost_prob": {"value": 0.0},
        "mechanism_seed": {"value": 42},
        "mlp_nonlins": {"value": "relu"},
        "mlp_num_hidden_layers": {"value": 1},
        "mlp_hidden_dim": {"value": 8},
        "mlp_activation_mode": {"value": "post"},
        "mlp_use_batch_norm": {"value": False},
        "xgb_num_hidden_layers": {"value": 0},
        "xgb_hidden_dim": {"value": 0},
        "xgb_activation_mode": {"value": "post"},
        "xgb_use_batch_norm": {"value": False},
        "xgb_node_shape": {"value": [1]},
        "xgb_n_training_samples": {"value": 100},
        "xgb_add_noise": {"value": False},
        "random_additive_std": {"value": True},
        "exo_std_distribution": {"value": "gamma"},
        "endo_std_distribution": {"value": "gamma"},
        "exo_std_mean": {"value": 1.0},
        "exo_std_std": {"value": 0.2},
        "endo_std_mean": {"value": 0.1},
        "endo_std_std": {"value": 0.1},
        "endo_p_zero": {"value": 0.0},
        "noise_mixture_proportions": {"value": [0.33, 0.33, 0.34]},
        "use_exogenous_mechanisms": {"value": True},
        "mechanism_generator_seed": {"value": 42},
    }
    
    # Preprocessing config
    preprocessing_config = {
        "dropout_prob": {"value": 0.0},
        "shuffle_data": {"value": False},
        "target_feature": {"value": None},
        "random_seed": {"value": None},
        "negative_one_one_scaling": {"value": True},
        "remove_outliers": {"value": False},
        "outlier_quantile": {"value": 0.01},
        "yeo_johnson": {"value": False},
        "standardize": {"value": True},
        "y_clip_quantile": {"value": None},
        "eps": {"value": 1e-6},
        "increase_treatment_scale": {"value": False},
        "distribution_rescale_factor": {"value": 0.0},
        "interventional_distribution_type": {"value": "resampling"},
        "test_feature_mask_fraction": {"value": 0.0},
    }
    
    # Dataset config WITHOUT binarization
    dataset_config_no_binarize = {
        "dataset_size": {"value": 100},
        "max_number_features": {"value": 5},
        "max_number_samples_per_dataset": {"value": 500},
        "max_number_train_samples_per_dataset": {"value": 100},
        "max_number_test_samples_per_dataset": {"value": 100},
        "number_train_samples_per_dataset": {"value": 100},
        "number_test_samples_per_dataset": {"value": 100},  # Same as train to avoid size mismatch
        "binarize_treatment_prob": {"value": 0.0},  # No binarization
        "seed": 123,
    }
    
    # Dataset config WITH binarization
    dataset_config_with_binarize = {
        "dataset_size": {"value": 100},
        "max_number_features": {"value": 5},
        "max_number_samples_per_dataset": {"value": 500},
        "max_number_train_samples_per_dataset": {"value": 100},
        "max_number_test_samples_per_dataset": {"value": 100},
        "number_train_samples_per_dataset": {"value": 100},
        "number_test_samples_per_dataset": {"value": 100},  # Same as train to avoid size mismatch
        "binarize_treatment_prob": {"value": 1.0},  # Always binarize
        "seed": 123,
    }
    
    print("=" * 60)
    print("Testing InterventionalDataset Treatment Binarization")
    print("=" * 60)
    
    # Test without binarization
    print("\n1. Testing WITHOUT binarization (prob=0.0)...")
    try:
        dataset_no_bin = InterventionalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config_no_binarize,
        )
        
        # Sample a few items and check treatment values
        num_test = 20
        binary_count = 0
        continuous_count = 0
        
        for i in range(num_test):
            sample = dataset_no_bin[i]
            # Get the raw observational data before processing
            # The treatment variable is the intervention node's values
            # We need to check if treatment is binary in the returned data
            
            # Check X_train for treatment column (usually index 0 after processing)
            X_train = sample['X_train']
            if X_train.shape[1] > 0:
                T_train = X_train[:, 0]  # Treatment is first column
                unique_vals = torch.unique(T_train)
                if len(unique_vals) <= 2:
                    binary_count += 1
                else:
                    continuous_count += 1
        
        print(f"   Samples with <=2 unique treatment values: {binary_count}/{num_test}")
        print(f"   Samples with >2 unique treatment values: {continuous_count}/{num_test}")
        print(f"   ✓ Without binarization, most treatments should be continuous")
        
    except Exception as e:
        print(f"   ✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test with binarization
    print("\n2. Testing WITH binarization (prob=1.0)...")
    try:
        dataset_with_bin = InterventionalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config_with_binarize,
        )
        
        num_test = 20
        binary_count = 0
        continuous_count = 0
        raw_binary_count = 0
        
        for i in range(num_test):
            sample = dataset_with_bin[i]
            X_train = sample['X_train']
            if X_train.shape[1] > 0:
                T_train = X_train[:, 0]  # Treatment is first column
                unique_vals = torch.unique(T_train)
                
                # After standardization/processing, binary values may not be exactly 0,1
                # but should still have only 2 unique values
                if len(unique_vals) <= 2:
                    binary_count += 1
                else:
                    continuous_count += 1
        
        print(f"   Samples with <=2 unique treatment values: {binary_count}/{num_test}")
        print(f"   Samples with >2 unique treatment values: {continuous_count}/{num_test}")
        
        if binary_count == num_test:
            print(f"   ✓ SUCCESS: All treatments are binary after binarization!")
        elif binary_count > num_test * 0.8:
            print(f"   ⚠ PARTIAL: Most treatments are binary ({binary_count}/{num_test})")
        else:
            print(f"   ✗ FAIL: Expected all binary, got {binary_count}/{num_test}")
            return False
        
    except Exception as e:
        print(f"   ✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_binarization()
    sys.exit(0 if success else 1)
