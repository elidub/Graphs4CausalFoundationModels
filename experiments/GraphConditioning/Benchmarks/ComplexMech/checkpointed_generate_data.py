#!/usr/bin/env python3
"""
Ultra-hacky checkpointed auto-retry wrapper for ComplexMech data generation.

This version:
1. Generates configs one at a time instead of all at once
2. Retries each config up to 25 times if it fails
3. Skips configs that already have output files
4. Keeps going even if some configs fail
"""

import subprocess
import sys
import time
import os
import argparse
from pathlib import Path


def parse_original_args():
    """Parse the original script arguments to understand what needs to be generated."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--node-counts", type=int, nargs="+", default=[2, 5, 20, 50])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--variants", type=str, nargs="+", default=["base", "path_TY", "path_YT", "path_independent_TY"])
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[500, 700, 800, 900, 950])
    
    return parser.parse_args()


def get_output_filename(node_count, variant, sample_size, num_samples, seed):
    """Generate the expected output filename."""
    if variant == "base":
        return f"complexmech_{node_count}nodes_base_{num_samples}samples_seed{seed}.pkl"
    else:
        return f"complexmech_{node_count}nodes_{variant}_ntest{sample_size}_{num_samples}samples_seed{seed}.pkl"


def file_exists_and_complete(filepath):
    """Check if file exists and seems complete (> 1MB)."""
    if not filepath.exists():
        return False
    
    # Basic size check - should be at least 1MB for real data
    size_bytes = filepath.stat().st_size
    return size_bytes > 1024 * 1024  # 1MB


def run_single_config(node_count, variant, sample_size, args, max_retries=25):
    """Run data generation for a single config with retries."""
    
    cache_dir = Path("data_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Generate expected output filename
    output_filename = get_output_filename(node_count, variant, sample_size, args.num_samples, args.base_seed)
    output_path = cache_dir / output_filename
    
    # Check if already exists and complete
    if file_exists_and_complete(output_path) and not args.overwrite:
        print(f"✓ SKIP: {output_filename} already exists ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    
    # Build command for this specific config
    cmd_args = [
        "--num-samples", str(args.num_samples),
        "--base-seed", str(args.base_seed),
        "--node-counts", str(node_count),
        "--variants", variant,
    ]
    
    if variant != "base":
        cmd_args.extend(["--sample-sizes", str(sample_size)])
    
    if args.overwrite:
        cmd_args.append("--overwrite")
    
    print(f"\n{'='*80}")
    print(f"GENERATING: {node_count} nodes, {variant}" + (f" (ntest={sample_size})" if variant != "base" else ""))
    print(f"Output: {output_filename}")
    print(f"Command args: {' '.join(cmd_args)}")
    print("=" * 80)
    
    # Retry loop for this specific config
    python_executable = "python"
    script_name = "generate_all_variants_data.py"
    
    for retry in range(1, max_retries + 1):
        print(f"\n--- Retry {retry}/{max_retries} for {variant} ---")
        
        try:
            cmd = [python_executable, script_name] + cmd_args
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout per config
            )
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                # Check if output file was actually created and is reasonable size
                if file_exists_and_complete(output_path):
                    size_mb = output_path.stat().st_size / 1024 / 1024
                    print(f"✓ SUCCESS: Generated {output_filename} ({size_mb:.1f} MB) in {elapsed:.1f}s")
                    return True
                else:
                    print(f"✗ FAILED: Script returned 0 but output file missing or too small")
                    print("STDOUT:", result.stdout[-500:] if result.stdout else "None")
                    print("STDERR:", result.stderr[-500:] if result.stderr else "None")
            else:
                print(f"✗ FAILED: Exit code {result.returncode} after {elapsed:.1f}s")
                
                # Show error details
                if result.stderr:
                    print("STDERR (last 500 chars):", result.stderr[-500:])
                
                # Analyze error type
                if result.returncode == 134:
                    error_type = "SIGABRT (C extension refcount error)"
                elif result.returncode == 139:
                    error_type = "SIGSEGV (segmentation fault)" 
                elif result.returncode == 6:
                    error_type = "SIGABRT (abort trap)"
                elif result.returncode == 130:
                    print("  User interrupted. Stopping retries for this config.")
                    return False
                else:
                    error_type = f"Unknown error"
                
                print(f"  Error type: {error_type}")
        
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT: Config exceeded 30 minute limit on retry {retry}")
            
        except KeyboardInterrupt:
            print(f"\n✗ INTERRUPTED: User stopped execution on retry {retry}")
            return False
            
        except Exception as e:
            print(f"✗ UNEXPECTED ERROR on retry {retry}: {e}")
        
        # Wait before retrying (with backoff)
        if retry < max_retries:
            wait_time = min(10 + retry * 5, 60)  # 10s to 60s
            print(f"  Waiting {wait_time}s before retry {retry + 1}...")
            time.sleep(wait_time)
    
    print(f"✗ FAILED: {variant} failed after {max_retries} retries")
    return False


def main():
    """Main execution with individual config retry."""
    
    args = parse_original_args()
    
    print("=" * 80)
    print("Checkpointed Auto-Retry ComplexMech Data Generation")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Samples per config: {args.num_samples}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Node counts: {args.node_counts}")
    print(f"  Variants: {args.variants}")
    print(f"  Sample sizes: {args.sample_sizes}")
    print(f"  Overwrite existing: {args.overwrite}")
    print(f"  Max retries per config: 25")
    print("=" * 80)
    
    # Build list of all configs to process
    configs_to_process = []
    
    for node_count in args.node_counts:
        for variant in args.variants:
            if variant == "base":
                configs_to_process.append((node_count, variant, None))
            else:
                for sample_size in args.sample_sizes:
                    configs_to_process.append((node_count, variant, sample_size))
    
    print(f"\nTotal configs to process: {len(configs_to_process)}")
    
    # Process each config individually
    successful = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    for i, (node_count, variant, sample_size) in enumerate(configs_to_process, 1):
        print(f"\n{'#'*80}")
        print(f"CONFIG {i}/{len(configs_to_process)}")
        print("#" * 80)
        
        result = run_single_config(node_count, variant, sample_size, args)
        
        if result is True:
            successful += 1
        elif result is False:
            failed += 1
        else:
            skipped += 1  # This shouldn't happen with current logic
    
    # Final summary
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total configs processed: {len(configs_to_process)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"⊘ Skipped: {skipped}")
    print(f"Total time: {elapsed/3600:.2f} hours")
    
    # List generated files
    cache_dir = Path("data_cache")
    if cache_dir.exists():
        pkl_files = list(cache_dir.glob("*.pkl"))
        total_size_mb = sum(f.stat().st_size for f in pkl_files) / 1024 / 1024
        print(f"\nGenerated files: {len(pkl_files)} files, {total_size_mb:.1f} MB total")
    
    if failed == 0:
        print(f"\n🎉 ALL CONFIGS SUCCESSFUL!")
        return 0
    else:
        print(f"\n⚠️  {failed} configs failed - but {successful} succeeded!")
        return 1 if successful == 0 else 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n✗ User interrupted execution")
        sys.exit(1)