"""
Generate benchmark datasets for ComplexMech benchmark.

This script generates synthetic datasets using the benchmark configs for different
node counts (2, 5, 10, 20, 35, 50 nodes). The datasets are saved to the data_cache
directory for later use in benchmarking.

Usage:
    python generate_benchmark_data.py [--num-samples N] [--overwrite] [--node-counts 2 5 10]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[4]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from experiments.GraphConditioning.Benchmarks.ComplexMechIDK.ComplexMechBenchmarkIDK import ComplexMechBenchmarkIDK


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark datasets for LinGaus benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate per node configuration",
    )
    
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed (each config gets seed + offset*1000)",
    )
    
    parser.add_argument(
        "--node-counts",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20, 35, 50],
        help="Node counts to generate data for",
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data files",
    )
    
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Path to benchmark directory (default: script directory)",
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Path to cache directory (default: benchmark_dir/data_cache)",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    
    return parser.parse_args()


def main():
    """Main function to generate benchmark data."""
    args = parse_args()
    
    # Setup benchmark directory
    if args.benchmark_dir is None:
        benchmark_dir = Path(__file__).parent
    else:
        benchmark_dir = Path(args.benchmark_dir)
    
    print("="*80)
    print("ComplexMech Benchmark Data Generation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Benchmark directory: {benchmark_dir}")
    print(f"  Number of samples per config: {args.num_samples}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Node counts: {args.node_counts}")
    print(f"  Overwrite existing: {args.overwrite}")
    print("="*80)
    
    # Initialize benchmark
    benchmark = ComplexMechBenchmarkIDK(
        benchmark_dir=str(benchmark_dir),
        cache_dir=args.cache_dir,
        verbose=not args.quiet,
    )
    
    # Generate data for all specified configs
    saved_paths = benchmark.sample_all_configs(
        num_samples_per_config=args.num_samples,
        dataset_seed=args.base_seed,
        node_counts=args.node_counts,
        overwrite=args.overwrite,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("Data Generation Summary")
    print("="*80)
    print(f"\nGenerated {len(saved_paths)}/{len(args.node_counts)} datasets:")
    
    total_size_mb = 0
    for node_count, path in sorted(saved_paths.items()):
        path_obj = Path(path)
        size_mb = path_obj.stat().st_size / 1024 / 1024
        total_size_mb += size_mb
        print(f"  {node_count:2d} nodes: {path_obj.name} ({size_mb:.2f} MB)")
    
    print(f"\nTotal size: {total_size_mb:.2f} MB")
    print(f"Cache directory: {benchmark.cache_dir}")
    print("="*80)
    print("\nData generation complete! ✓")
    print("\nYou can now run benchmarks using these datasets with:")
    print("  python ComplexMechBenchmarkIDK.py")
    print("  # or use the ComplexMechBenchmarkIDK class programmatically")
    print("="*80)


if __name__ == "__main__":
    main()
