#!/usr/bin/env python3
"""
Quick test script for SimplePFN training to catch shape and API errors early.
"""

import sys
import torch
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.SimplePFN import SimplePFNRegressor


def test_simplepfn_shapes():
    """Test SimplePFN with various input shapes to catch dimension errors."""
    print("=== Testing SimplePFN Shapes ===")
    
    # Create a small SimplePFN model
    model = SimplePFNRegressor(
        num_features=5,
        d_model=8,
        depth=1,
        heads_feat=2,
        heads_samp=2,
        dropout=0.1
    )
    
    print(f"Model created: {model}")
    
    # Test case 1: Normal case
    print("\n--- Test 1: Normal case ---")
    try:
        X_train = torch.randn(1, 4, 5)  # (batch=1, N=4, features=5)
        y_train = torch.randn(1, 4, 1)  # (batch=1, N=4, targets=1)
        X_test = torch.randn(1, 3, 5)   # (batch=1, M=3, features=5)
        
        print(f"X_train.shape: {X_train.shape}")
        print(f"y_train.shape: {y_train.shape}")
        print(f"X_test.shape: {X_test.shape}")
        
        output = model(X_train, y_train, X_test)
        print(f"Output type: {type(output)}")
        if isinstance(output, dict):
            print(f"Output keys: {list(output.keys())}")
            for k, v in output.items():
                print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
        else:
            print(f"Output shape: {output.shape}")
        print("✅ Test 1 PASSED")
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Test case 2: Wrong number of dimensions
    print("\n--- Test 2: Wrong dimensions (should fail) ---")
    try:
        X_train_bad = torch.randn(4, 5)  # Missing batch dimension
        y_train = torch.randn(1, 4, 1)
        X_test = torch.randn(1, 3, 5)
        
        print(f"X_train_bad.shape: {X_train_bad.shape} (should be 3D)")
        output = model(X_train_bad, y_train, X_test)
        print("❌ Test 2 should have failed but didn't!")
        
    except Exception as e:
        print(f"✅ Test 2 correctly failed: {e}")
    
    # Test case 3: Feature mismatch
    print("\n--- Test 3: Feature count mismatch (should fail) ---")
    try:
        X_train = torch.randn(1, 4, 7)  # Wrong feature count (7 instead of 5)
        y_train = torch.randn(1, 4, 1)
        X_test = torch.randn(1, 3, 7)
        
        print(f"X_train.shape: {X_train.shape} (features=7, expected=5)")
        output = model(X_train, y_train, X_test)
        print("❌ Test 3 should have failed but didn't!")
        
    except Exception as e:
        print(f"✅ Test 3 correctly failed: {e}")
    
    print("\n=== SimplePFN Shape Tests Complete ===")


def test_dataloader_shapes():
    """Test what shapes we get from the actual dataloader."""
    print("\n=== Testing Dataloader Shapes ===")
    
    try:
        # Import dataloader components
        from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
        from priordata_processing.Datasets.ExampleConfigs.BasicConfigs import (
            default_dataset_config, 
            default_preprocessing_config
        )
        from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config
        from torch.utils.data import DataLoader
        
        print("Creating small test dataset...")
        
        # Create minimal dataset config for quick testing
        test_dataset_config = default_dataset_config.copy()
        test_dataset_config['dataset_size'] = {'value': 100}  # Very small
        test_dataset_config['max_number_samples'] = {'value': 20}
        
        dataset_maker = MakePurelyObservationalDataset(
            scm_config=default_sampling_config,
            preprocessing_config=default_preprocessing_config,
            dataset_config=test_dataset_config
        )
        
        dataset = dataset_maker.create_dataset(seed=42)
        print(f"Dataset created with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        # Test first batch
        print("Testing first batch...")
        first_batch = next(iter(dataloader))
        
        print(f"Batch type: {type(first_batch)}")
        if isinstance(first_batch, (list, tuple)):
            print(f"Batch length: {len(first_batch)}")
            for i, item in enumerate(first_batch):
                print(f"  Item {i}: type={type(item)}, shape={item.shape if hasattr(item, 'shape') else 'no shape'}")
        else:
            print(f"Batch shape: {first_batch.shape if hasattr(first_batch, 'shape') else 'no shape'}")
        
        print("✅ Dataloader test completed")
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("SimplePFN Quick Test Script")
    print("=" * 50)
    
    test_simplepfn_shapes()
    test_dataloader_shapes()
    
    print("\n" + "=" * 50)
    print("Test script completed!")


if __name__ == "__main__":
    main()
