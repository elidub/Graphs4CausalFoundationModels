#!/usr/bin/env python3
"""
Test file for PurelyObservationalDataset class
"""

import sys
import os

# Add the src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from priordata_processing.Datasets.PurelyObservationalDataset import PurelyObservationalDataset
from priors.causal_prior.scm.SCMSampler import SCMSampler
from priordata_processing.BasicProcessing import BasicProcessing


def test_purely_observational_dataset():
    """
    Comprehensive tests for PurelyObservationalDataset class
    """
    print("Testing PurelyObservationalDataset...")
    
    try:
        # Test 1: Import required modules
        print("1. Testing imports...")
        from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config
        print("✓ Successfully imported required modules")
        
        # Test 2: Create SCMSampler and BasicProcessing instances
        print("\n2. Testing component initialization...")
        
        # Create SCMSampler
        scm_sampler = SCMSampler(default_sampling_config, seed=42)
        print("✓ Created SCMSampler")
        
        # Create BasicProcessing
        processor = BasicProcessing(
            max_num_samples=50,
            max_num_features=20,
            train_fraction=0.7,
            random_seed=123
        )
        print("✓ Created BasicProcessing")
        
        # Create distribution for number of samples
        samples_distribution = torch.distributions.Uniform(low=10, high=30)
        print("✓ Created samples distribution")
        
        # Test the distribution sampling to ensure it works correctly
        sample_test = samples_distribution.sample()
        sample_int = int(sample_test.item())
        print(f"✓ Distribution sampling works: {sample_test} -> {sample_int}")
        
        # Test 3: Initialize PurelyObservationalDataset
        print("\n3. Testing dataset initialization...")
        dataset = PurelyObservationalDataset(
            scm_sampler=scm_sampler,
            priordata_processor=processor,
            number_samples_per_dataset_distribution=samples_distribution,
            size=5,  # Small size for testing
            max_number_samples=50,
            max_number_features=20,
            seed=456
        )
        print("✓ Successfully created PurelyObservationalDataset")
        
        # Test 4: Check dataset length
        print("\n4. Testing dataset length...")
        length = len(dataset)
        print(f"✓ Dataset length: {length} (expected: 5)")
        assert length == 5, f"Expected length 5, got {length}"
        
        # Test 5: Test __getitem__ method
        print("\n5. Testing dataset item retrieval...")
        item = dataset[0]
        print("✓ Retrieved item 0")
        
        # Check that item is a tuple with two elements (data_dict, metadata)
        assert isinstance(item, tuple) and len(item) == 2, f"Expected tuple of length 2, got {type(item)} of length {len(item) if hasattr(item, '__len__') else 'unknown'}"
        
        data_dict, metadata = item
        print("✓ Item structure correct: data_dict and metadata")
        
        # Check data_dict structure
        expected_keys = {'X_train', 'Y_train', 'X_test', 'Y_test'}
        assert isinstance(data_dict, dict), f"Expected dict for data_dict, got {type(data_dict)}"
        assert set(data_dict.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(data_dict.keys())}"
        print(f"✓ Data dictionary has correct keys: {list(data_dict.keys())}")
        
        # Check tensor shapes
        for key, tensor in data_dict.items():
            assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor for {key}, got {type(tensor)}"
            print(f"✓ {key} shape: {tensor.shape}")
        
        # Check metadata
        assert isinstance(metadata, dict), f"Expected dict for metadata, got {type(metadata)}"
        print(f"✓ Metadata keys: {list(metadata.keys())}")
        
        # Test 6: Test multiple items (reproducibility check)
        print("\n6. Testing reproducibility...")
        item1_a = dataset[1]
        item1_b = dataset[1]  # Should be identical due to deterministic seeding
        
        # Check that the same index returns the same result
        data1_a, meta1_a = item1_a
        data1_b, meta1_b = item1_b
        
        for key in data1_a.keys():
            assert torch.allclose(data1_a[key], data1_b[key]), f"Reproducibility failed for {key}"
        print("✓ Same index returns identical results")
        
        # Test 7: Test different items are different
        print("\n7. Testing that different indices give different results...")
        item2 = dataset[2]
        data2, meta2 = item2
        
        # At least one tensor should be different
        any_different = False
        for key in data1_a.keys():
            if not torch.allclose(data1_a[key], data2[key]):
                any_different = True
                break
        
        assert any_different, "Different indices should produce different datasets"
        print("✓ Different indices produce different datasets")
        
        # Test 8: Test edge cases
        print("\n8. Testing edge cases...")
        
        # Test last index
        last_item = dataset[length - 1]
        print(f"✓ Successfully retrieved last item (index {length - 1})")
        
        # Test that out-of-bounds raises appropriate error
        try:
            dataset[length]
            assert False, "Should have raised IndexError for out-of-bounds access"
        except IndexError:
            print("✓ IndexError correctly raised for out-of-bounds access")
        
        # Test negative index out-of-bounds
        try:
            dataset[-1]
            assert False, "Should have raised IndexError for negative index access"
        except IndexError:
            print("✓ IndexError correctly raised for negative index access")
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("PurelyObservationalDataset is working correctly.")
        print("="*50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("\nDebugging info:")
        print(f"Current working directory: {os.getcwd()}")
        return False


def test_basic_functionality():
    """
    Quick test for basic functionality
    """
    print("\n" + "="*50)
    print("BASIC FUNCTIONALITY TEST")
    print("="*50)
    
    try:
        from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config
        
        # Create minimal setup
        scm_sampler = SCMSampler(default_sampling_config, seed=42)
        processor = BasicProcessing(max_num_samples=20, max_num_features=10, random_seed=123)
        distribution = torch.distributions.Uniform(low=5, high=15)
        
        # Create small dataset
        dataset = PurelyObservationalDataset(
            scm_sampler=scm_sampler,
            priordata_processor=processor,
            number_samples_per_dataset_distribution=distribution,
            size=2,
            seed=789
        )
        
        print(f"✓ Created dataset with {len(dataset)} items")
        
        # Test item access
        item = dataset[0]
        data_dict, metadata = item
        
        print(f"✓ Retrieved item successfully")
        print(f"✓ Data keys: {list(data_dict.keys())}")
        print(f"✓ Train set shape: X={data_dict['X_train'].shape}, Y={data_dict['Y_train'].shape}")
        print(f"✓ Test set shape: X={data_dict['X_test'].shape}, Y={data_dict['Y_test'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


if __name__ == "__main__":
    """
    Run all tests when script is executed directly
    """
    print("Running PurelyObservationalDataset tests...")
    print("="*60)
    
    # Run comprehensive tests
    success1 = test_purely_observational_dataset()
    
    # Run basic tests
    success2 = test_basic_functionality()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("PurelyObservationalDataset is working correctly.")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    print("="*60)
