import torch
import numpy as np
from priordata_processing.BasicProcessing import BasicProcessing

def test_padding_consistency():
    """Test that BasicProcessing always returns consistent shapes."""
    
    # Set parameters
    max_samples = 100
    max_features = 20
    
    processor = BasicProcessing(
        max_num_samples=max_samples,
        max_num_features=max_features,
        train_fraction=0.6,
        random_seed=42
    )
    
    # Test datasets of different sizes
    test_cases = [
        # Small dataset
        {i: torch.randn(10, 1) for i in range(3)},
        # Medium dataset
        {i: torch.randn(50, 1) for i in range(8)},
        # Large dataset (should be truncated)
        {i: torch.randn(200, 1) for i in range(25)},
        # Very small dataset
        {i: torch.randn(5, 1) for i in range(2)},
    ]
    
    expected_train_samples = int(max_samples * 0.6)
    expected_test_samples = max_samples - expected_train_samples
    
    print(f"Testing padding consistency:")
    print(f"Expected shapes:")
    print(f"  X_train: ({expected_train_samples}, {max_features})")
    print(f"  X_test: ({expected_test_samples}, {max_features})")
    print(f"  Y_train: ({expected_train_samples}, 1)")
    print(f"  Y_test: ({expected_test_samples}, 1)")
    print()
    
    for i, dataset in enumerate(test_cases):
        print(f"Test case {i+1}: {len(dataset)} features, {list(dataset.values())[0].shape[0]} samples")
        
        try:
            result, metadata = processor.process(dataset, mode='safe')
            
            # Check shapes
            X_train_shape = result['X_train'].shape
            X_test_shape = result['X_test'].shape
            Y_train_shape = result['Y_train'].shape
            Y_test_shape = result['Y_test'].shape
            
            print(f"  Actual shapes:")
            print(f"    X_train: {X_train_shape}")
            print(f"    X_test: {X_test_shape}")
            print(f"    Y_train: {Y_train_shape}")
            print(f"    Y_test: {Y_test_shape}")
            
            # Verify shapes match expectations
            shape_correct = (
                X_train_shape == (expected_train_samples, max_features) and
                X_test_shape == (expected_test_samples, max_features) and
                Y_train_shape == (expected_train_samples, 1) and
                Y_test_shape == (expected_test_samples, 1)
            )
            
            if shape_correct:
                print(f"  ✓ Shapes are correct!")
            else:
                print(f"  ❌ Shapes are incorrect!")
                
            # Check for NaN or inf values
            all_finite = (
                torch.isfinite(result['X_train']).all() and
                torch.isfinite(result['X_test']).all() and
                torch.isfinite(result['Y_train']).all() and
                torch.isfinite(result['Y_test']).all()
            )
            
            if all_finite:
                print(f"  ✓ All values are finite")
            else:
                print(f"  ❌ Contains NaN or inf values")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            
        print()

if __name__ == "__main__":
    test_padding_consistency()
