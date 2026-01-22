#!/usr/bin/env python
"""
Ultra-safe batch generator: generates data in small batches with subprocess isolation.
Each batch is generated in a fresh subprocess to avoid memory corruption accumulation.
"""

import argparse
import sys
import os
from pathlib import Path
import pickle
import subprocess
import json
import tempfile


def main():
    parser = argparse.ArgumentParser(description='Generate data safely in batches')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, required=True, help='Path to output pickle file')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--batch-size', type=int, default=50, help='Samples per batch (subprocess)')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries per batch')
    args = parser.parse_args()
    
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve()
    
    # Skip if output already exists
    if output_path.exists():
        print(f"Output already exists: {output_path}")
        return 0
    
    # Create temp directory for batch outputs
    temp_dir = Path(tempfile.mkdtemp(prefix='batch_gen_'))
    print(f"Temp directory: {temp_dir}")
    
    all_samples = []
    sample_idx = 0
    batch_num = 0
    
    while sample_idx < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - sample_idx)
        batch_output = temp_dir / f"batch_{batch_num:04d}.pkl"
        
        success = False
        for retry in range(args.max_retries):
            print(f"\nBatch {batch_num+1}: samples {sample_idx}-{sample_idx+batch_size-1} (attempt {retry+1}/{args.max_retries})")
            
            # Run batch in subprocess
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / 'generate_batch.py'),
                    '--config', str(config_path),
                    '--output', str(batch_output),
                    '--start-idx', str(sample_idx),
                    '--num-samples', str(batch_size),
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout per batch
            )
            
            if result.returncode == 0 and batch_output.exists():
                success = True
                print(f"  Batch {batch_num+1} completed successfully")
                break
            else:
                print(f"  Batch failed: {result.stderr[-500:] if result.stderr else 'unknown error'}")
        
        if not success:
            print(f"ERROR: Batch {batch_num+1} failed after {args.max_retries} retries")
            # Continue anyway, just skip these samples
            sample_idx += batch_size
            batch_num += 1
            continue
        
        # Load batch and append to all_samples
        try:
            with open(batch_output, 'rb') as f:
                batch_samples = pickle.load(f)
            all_samples.extend(batch_samples)
            print(f"  Loaded {len(batch_samples)} samples (total: {len(all_samples)})")
        except Exception as e:
            print(f"  Warning: Failed to load batch: {e}")
        
        # Clean up batch file
        try:
            batch_output.unlink()
        except:
            pass
        
        sample_idx += batch_size
        batch_num += 1
    
    # Save all samples
    print(f"\nSaving {len(all_samples)} total samples to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(all_samples, f)
    
    print(f"Done! Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Clean up temp directory
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
