"""
Generate data for a SINGLE config file with multiprocessing isolation.
Uses multiprocessing.spawn to create a truly fresh Python process.
"""

import argparse
import sys
from pathlib import Path
import pickle
import multiprocessing as mp
from multiprocessing import get_context


def generate_samples_in_subprocess(config_path: str, output_path: str, num_samples: int, seed: int):
    """This function runs in a completely fresh subprocess."""
    # Import inside subprocess to get fresh state
    import sys
    from pathlib import Path
    
    # Add src to path
    repo_root = Path(__file__).resolve().parents[4]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset
    from tqdm import tqdm
    import yaml
    
    config_path = Path(config_path)
    output_path = Path(output_path)
    
    print(f"[subprocess] Loading config: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = InterventionalDataset(
        dataset_config=config.get('dataset_config', {}),
        scm_config=config.get('scm_config', {}),
        preprocessing_config=config.get('preprocessing_config', {}),
    )
    
    # Sample data with smaller batches and explicit garbage collection
    import gc
    
    print(f"[subprocess] Sampling {num_samples} samples...")
    samples = []
    batch_size = 100  # Process in smaller batches
    
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_samples = []
        
        for i in tqdm(range(batch_start, batch_end), desc=f"  Batch {batch_start//batch_size + 1}"):
            try:
                sample = dataset[i]
                batch_samples.append(sample)
            except Exception as e:
                print(f"\n  Warning: Failed to sample item {i}: {e}")
                continue
        
        samples.extend(batch_samples)
        
        # Force garbage collection between batches
        gc.collect()
    
    # Save
    print(f"[subprocess] Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)
    
    print(f"[subprocess] Done! Saved {len(samples)} samples, Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    return True


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
    
    print(f"Starting generation with spawn context...")
    
    # Use spawn context to create truly fresh process
    ctx = get_context('spawn')
    
    # Run in subprocess
    p = ctx.Process(
        target=generate_samples_in_subprocess,
        args=(str(config_path), str(output_path), args.num_samples, args.seed)
    )
    p.start()
    p.join()
    
    if p.exitcode == 0:
        print("Success!")
        return 0
    else:
        print(f"Process failed with exit code: {p.exitcode}")
        return 1


if __name__ == "__main__":
    # Needed for spawn on macOS
    mp.freeze_support()
    sys.exit(main())
