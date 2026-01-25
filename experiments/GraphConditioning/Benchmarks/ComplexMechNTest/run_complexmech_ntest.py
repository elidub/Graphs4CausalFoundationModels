#!/usr/bin/env python3
"""
Standalone ComplexMechNTest Benchmark Runner.

This script runs the ComplexMechNTest benchmark on a trained model checkpoint.
It can be used independently or as part of a cluster job submission.

The ComplexMechNTest benchmark tests partial graph knowledge with varying numbers
of test samples to evaluate model performance under different data regimes.

Usage:
    python run_complexmech_ntest.py --config /path/to/config.yaml --checkpoint /path/to/checkpoint.pt --fidelity high

Arguments:
    --config: Path to the model's YAML configuration file
    --checkpoint: Path to the model checkpoint (.pt file)
    --fidelity: Benchmark fidelity level (minimal, low, high)
    --output_dir: Directory to save results (default: same as checkpoint dir)
    --use_ancestor_matrix: If set, use ancestor matrix instead of adjacency matrix
"""

import os
import sys
import argparse
import json
import traceback
from pathlib import Path
from datetime import datetime

# Add paths for imports
script_dir = Path(__file__).parent
repo_root = script_dir.parent.parent.parent.parent  # CausalPriorFitting/
src_path = repo_root / "src"

# Add repo root and src to path
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import the benchmark
from ComplexMechBenchmarkNTest import ComplexMechBenchmarkNTest


def main():
    parser = argparse.ArgumentParser(
        description="Run ComplexMechNTest benchmark on a trained model checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to model YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--fidelity",
        type=str,
        default="high",
        choices=["minimal", "low", "high"],
        help="Benchmark fidelity level: minimal (3 samples), low (100 samples), high (1000 samples)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (default: checkpoint directory/complexmech_ntest_<fidelity>)"
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=None,
        help="Directory containing ComplexMechNTest configs and data_cache (default: script directory)"
    )
    parser.add_argument(
        "--use_ancestor_matrix",
        action="store_true",
        help="If set, use ancestor matrix instead of adjacency matrix"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override max samples (ignores fidelity)"
    )
    parser.add_argument(
        "--node_counts",
        type=int,
        nargs="+",
        default=None,
        help="Specific node counts to evaluate (default: all [2, 5, 10, 20, 35, 50])"
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help="Specific variants to evaluate (default: all [base, path_TY, path_YT, path_independent_TY])"
    )
    parser.add_argument(
        "--hide_fractions",
        type=float,
        nargs="+",
        default=None,
        help="Specific hide fractions for path variants (default: all [0.0, 0.25, 0.5, 0.75, 1.0])"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ComplexMechNTest Benchmark Runner")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Validate inputs
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    # Determine benchmark directory
    if args.benchmark_dir:
        benchmark_dir = Path(args.benchmark_dir)
    else:
        # Check environment variable first (for cluster jobs)
        env_benchmark_dir = os.environ.get("COMPLEXMECH_NTEST_BENCHMARK_DIR")
        if env_benchmark_dir:
            benchmark_dir = Path(env_benchmark_dir)
            print(f"Using COMPLEXMECH_NTEST_BENCHMARK_DIR from environment: {benchmark_dir}")
        else:
            # Default to script directory
            benchmark_dir = script_dir
    
    if not benchmark_dir.exists():
        print(f"ERROR: Benchmark directory not found: {benchmark_dir}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to checkpoint directory / complexmech_ntest_<fidelity>
        output_dir = checkpoint_path.parent / f"complexmech_ntest_{args.fidelity}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Config file: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Fidelity: {args.fidelity}")
    print(f"  Benchmark dir: {benchmark_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Use ancestor matrix: {args.use_ancestor_matrix}")
    if args.max_samples:
        print(f"  Max samples override: {args.max_samples}")
    if args.node_counts:
        print(f"  Node counts: {args.node_counts}")
    if args.variants:
        print(f"  Variants: {args.variants}")
    if args.hide_fractions:
        print(f"  Hide fractions: {args.hide_fractions}")
    print()
    
    try:
        # Create benchmark instance
        print("Initializing ComplexMechNTest benchmark...")
        benchmark = ComplexMechBenchmarkNTest(
            benchmark_dir=str(benchmark_dir),
            cache_dir=None,  # Uses default: benchmark_dir/data_cache
            verbose=True,
            max_samples=args.max_samples,  # Will be overridden by fidelity if None
            use_ancestor_matrix=args.use_ancestor_matrix,
        )
        
        print(f"  Data cache: {benchmark.cache_dir}")
        print(f"  Available node counts: {benchmark.NODE_COUNTS}")
        print(f"  Available variants: {benchmark.VARIANTS}")
        print(f"  Available hide fractions: {benchmark.HIDE_FRACTIONS}")
        print()
        
        # Check if data files exist
        print("Checking data files...")
        data_cache = benchmark.cache_dir
        if not data_cache.exists():
            print(f"ERROR: Data cache directory not found: {data_cache}")
            print(f"Please run generate_benchmark_data.py first to generate benchmark data.")
            sys.exit(1)
        
        # Count available data files
        pkl_files = list(data_cache.glob("*.pkl"))
        print(f"  Found {len(pkl_files)} data files in cache")
        if len(pkl_files) == 0:
            print(f"ERROR: No .pkl data files found in {data_cache}")
            print(f"Please run generate_benchmark_data.py first to generate benchmark data.")
            sys.exit(1)
        print()
        
        # Run the benchmark
        print("Running benchmark...")
        print("-" * 80)
        
        results = benchmark.run(
            fidelity=args.fidelity,
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            output_dir=str(output_dir),
            node_counts=args.node_counts,
            variants=args.variants,
            hide_fractions=args.hide_fractions,
        )
        
        print("-" * 80)
        print()
        
        # Save combined results
        print("Saving combined results...")
        
        # Convert tuple keys to strings for JSON
        results_serializable = {}
        for key, value in results.items():
            if isinstance(key, tuple):
                if len(key) == 3:
                    node_count, variant, hide_fraction = key
                    if hide_fraction is None:
                        str_key = f"{node_count}node_{variant}"
                    else:
                        str_key = f"{node_count}node_{variant}_hide{hide_fraction}"
                else:
                    str_key = "_".join(str(k) for k in key)
                results_serializable[str_key] = value
            else:
                results_serializable[str(key)] = value
        
        # Save JSON
        results_json_path = output_dir / f"complexmech_ntest_benchmark_{args.fidelity}.json"
        with open(results_json_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"  Saved JSON results: {results_json_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Benchmark completed successfully!")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"  JSON: {results_json_path.name}")
        
        # Print summary statistics
        if results:
            print("\nSummary Statistics:")
            print("-" * 80)
            
            # Group by node count
            node_results = {}
            for key, metrics in results.items():
                if isinstance(key, str) and 'node' in key:
                    node_count = key.split('node')[0]
                    if node_count not in node_results:
                        node_results[node_count] = []
                    if isinstance(metrics, dict) and 'mean_mise' in metrics:
                        node_results[node_count].append(metrics['mean_mise'])
            
            for node_count in sorted(node_results.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                mises = node_results[node_count]
                if mises:
                    avg_mise = sum(mises) / len(mises)
                    print(f"  {node_count} nodes: avg MISE = {avg_mise:.4f} ({len(mises)} configs)")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print("\nERROR: Benchmark execution failed!")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
