#!/usr/bin/env python3
"""
Debug script to understand dataset shapes and configuration.
"""

import sys
import torch
import yaml
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset


def test_yaml_config():
    """Test loading and interpreting the YAML config."""
    print("=== Testing YAML Config ===")
    
    config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/configs/early_test2.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Dataset config: {config['dataset_config']}")
    print(f"Preprocessing config: {config['preprocessing_config']}")
    print(f"Model config: {config['model_config']}")
    
    # Key parameters
    max_feat = config['dataset_config']['max_number_features']['value']
    expected_input_feat = config['model_config']['num_features']
    
    print(f"\nMax features in dataset: {max_feat}")
    print(f"Expected input features for model: {expected_input_feat}")
    print(f"Expected relationship: {max_feat} - 1 = {expected_input_feat}")
    
    return config


def test_dataset_creation_single_sample():
    """Test creating a single sample to debug shapes."""
    print("\n=== Testing Single Dataset Sample ===")
    
    try:
        config = test_yaml_config()
        
        # Import dataset components
        from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config
        
        # Use the YAML configs
        scm_config = config['scm_config']
        dataset_config = config['dataset_config']
        preprocessing_config = config['preprocessing_config']
        
        print(f"\nCreating dataset maker with max_features={dataset_config['max_number_features']['value']}...")
        
        dataset_maker = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config
        )
        
        print("Creating minimal dataset...")
        
        # Override size for quick test
        test_dataset_config = dataset_config.copy()
        test_dataset_config['dataset_size'] = {'value': 10}
        
        dataset_maker_small = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=test_dataset_config
        )
        
        dataset = dataset_maker_small.create_dataset(seed=42)
        print(f"Dataset created with {len(dataset)} samples")
        
        # Get first sample
        print("Getting first sample...")
        sample = dataset[0]
        
        print(f"Sample type: {type(sample)}")
        if isinstance(sample, (list, tuple)):
            print(f"Sample length: {len(sample)}")
            for i, item in enumerate(sample):
                if hasattr(item, 'shape'):
                    print(f"  Item {i}: shape={item.shape}, dtype={item.dtype}")
                else:
                    print(f"  Item {i}: type={type(item)}")
        else:
            print(f"Sample shape: {sample.shape if hasattr(sample, 'shape') else 'no shape'}")
        
        return sample
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run dataset debugging."""
    print("Dataset Shape Debug Script")
    print("=" * 50)
    
    sample = test_dataset_creation_single_sample()
    
    print("\n" + "=" * 50)
    print("Debug completed!")


if __name__ == "__main__":
    main()
