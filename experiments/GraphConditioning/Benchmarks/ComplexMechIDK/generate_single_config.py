"""
Generate data for a SINGLE config file.
Run this in a separate process to isolate memory corruption issues.
"""

import argparse
import sys
from pathlib import Path
import pickle

# Add src to path
repo_root = Path(__file__).resolve().parents[4]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset
from tqdm import tqdm
import yaml


def main():
    parser = argparse.ArgumentParser(description='Generate data for a single config')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, required=True, help='Path to output pickle file')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    config_path = Path(args.config)
    output_path = Path(args.output)
    
    # Skip if output already exists
    if output_path.exists():
        print(f"Output already exists: {output_path}")
        return 0
    
    print(f"Loading config: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset using the specific config keys
    dataset = InterventionalDataset(
        dataset_config=config.get('dataset_config', {}),
        scm_config=config.get('scm_config', {}),
        preprocessing_config=config.get('preprocessing_config', {}),
    )
    
    # Sample data
    print(f"Sampling {args.num_samples} samples...")
    samples = []
    for i in tqdm(range(args.num_samples)):
        sample = dataset[i]
        samples.append(sample)
    
    # Save
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)
    
    print(f"Done! Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
