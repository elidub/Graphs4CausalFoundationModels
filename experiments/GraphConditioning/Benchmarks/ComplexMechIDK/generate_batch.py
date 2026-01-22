#!/usr/bin/env python
"""
Generate a single batch of samples. Called by generate_batched.py.
Designed to be run in subprocess isolation.
"""

import argparse
import sys
from pathlib import Path
import pickle
import gc


def main():
    parser = argparse.ArgumentParser(description='Generate a batch of samples')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, required=True, help='Path to output pickle file')
    parser.add_argument('--start-idx', type=int, required=True, help='Starting sample index')
    parser.add_argument('--num-samples', type=int, required=True, help='Number of samples in batch')
    args = parser.parse_args()
    
    # Add src to path
    repo_root = Path(__file__).resolve().parents[4]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset
    import yaml
    
    config_path = Path(args.config)
    output_path = Path(args.output)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = InterventionalDataset(
        dataset_config=config.get('dataset_config', {}),
        scm_config=config.get('scm_config', {}),
        preprocessing_config=config.get('preprocessing_config', {}),
    )
    
    # Sample data
    samples = []
    for i in range(args.start_idx, args.start_idx + args.num_samples):
        sample = dataset[i]
        samples.append(sample)
        
        # Explicit cleanup every 10 samples
        if (i - args.start_idx) % 10 == 0:
            gc.collect()
    
    # Final cleanup before save
    gc.collect()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)
    
    print(f"Saved {len(samples)} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
