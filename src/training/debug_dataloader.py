#!/usr/bin/env python3
"""
Debug script to see exact dataloader output format.
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
from torch.utils.data import DataLoader


def test_dataloader_output():
    """See exactly what the dataloader produces."""
    print("=== Testing DataLoader Output ===")
    
    config_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/configs/early_test2.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use configs
    scm_config = config['scm_config']
    dataset_config = config['dataset_config']
    preprocessing_config = config['preprocessing_config']
    
    # Create small dataset for testing
    test_dataset_config = dataset_config.copy()
    test_dataset_config['dataset_size'] = {'value': 20}
    
    print("Creating dataset...")
    dataset_maker = MakePurelyObservationalDataset(
        scm_config=scm_config,
        preprocessing_config=preprocessing_config,
        dataset_config=test_dataset_config
    )
    
    dataset = dataset_maker.create_dataset(seed=42)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test single sample
    print("\n--- Single Sample ---")
    sample = dataset[0]
    print(f"Sample type: {type(sample)}")
    print(f"Sample structure:")
    if isinstance(sample, (tuple, list)):
        for i, item in enumerate(sample):
            print(f"  Item {i}: {type(item)}, shape={item.shape if hasattr(item, 'shape') else 'no shape'}")
    
    # Test dataloader with different batch sizes
    for batch_size in [1, 2]:
        print(f"\n--- DataLoader with batch_size={batch_size} ---")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        batch = next(iter(dataloader))
        print(f"Batch type: {type(batch)}")
        
        if isinstance(batch, (tuple, list)):
            print(f"Batch length: {len(batch)}")
            for i, item in enumerate(batch):
                print(f"  Batch item {i}: {type(item)}, shape={item.shape if hasattr(item, 'shape') else 'no shape'}")
        else:
            print(f"Batch shape: {batch.shape if hasattr(batch, 'shape') else 'no shape'}")


if __name__ == "__main__":
    test_dataloader_output()
