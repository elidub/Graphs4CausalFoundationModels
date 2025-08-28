#!/usr/bin/env python3

# -------------------------------------------

"""
Dataloader Test - Only test dataloading functionality.
"""

import sys
import yaml
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# AFTER the preamble, do project imports
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
from priordata_processing.Datasets.ExampleConfigs.BasicConfigs import (
    default_dataset_config, 
    default_preprocessing_config
)
from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config


def main():
    """Main entry point for dataloader testing."""
    parser = argparse.ArgumentParser(description='Dataloader Test Runner')
    parser.add_argument('--config', help='Optional path to YAML configuration file. If not provided, uses default configs from BasicConfigs.py')
    
    args = parser.parse_args()
    
    try:
        print("=== Testing Dataloader Only ===")
        
        if args.config:
            # Load config from YAML file
            print(f"Loading config from YAML file: {args.config}")
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Extract data configs
            scm_config = config.get('scm_config', {})
            dataset_config = config.get('dataset_config', {})
            preprocessing_config = config.get('preprocessing_config', {})
            
            print(f"Loaded config from: {args.config}")
        else:
            # Use default configs from BasicConfigs.py
            print("Using default configs from BasicConfigs.py")
            scm_config = default_sampling_config
            dataset_config = default_dataset_config
            preprocessing_config = default_preprocessing_config
        
        print(f"SCM config keys: {list(scm_config.keys())}")
        print(f"Dataset config keys: {list(dataset_config.keys())}")
        print(f"Preprocessing config keys: {list(preprocessing_config.keys())}")
        
        print("Creating dataset maker...")
        dataset_maker = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config
        )
        
        print("Creating dataset...")
        dataset = dataset_maker.create_dataset(seed=42)
        print(f"Dataset created with {len(dataset)} samples")
        
        # Test DataLoader
        print("Testing DataLoader...")
        #t = torch.zeros(10)  # trying if using torch before messes it up
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4
        )
        
        print("Testing first few batches...")
        for i, batch in enumerate(dataloader):
            if i >= 20:  # Only test first 20 batches
                break
            print(f"Batch {i}: shape = {batch[0].shape if isinstance(batch, (list, tuple)) else batch.shape}")
        
        print("=== Dataloader Test Complete ===")
        print("Dataloader test completed successfully!")
        return dataset
        
    except Exception as e:
        print(f"Dataloader test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
