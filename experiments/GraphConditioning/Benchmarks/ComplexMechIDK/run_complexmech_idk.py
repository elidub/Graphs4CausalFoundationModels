#!/usr/bin/env python3
"""
Standalone ComplexMechIDK Benchmark Runner.

This script runs the ComplexMechIDK benchmark on a trained model checkpoint.
It can be used independently or as part of a cluster job submission.

The ComplexMechIDK benchmark tests partial graph knowledge with three-state
adjacency matrices {-1, 0, 1} where 0 represents unknown edges (I Don't Know).

Usage:
    python run_complexmech_idk.py --config /path/to/config.yaml --checkpoint /path/to/checkpoint.pt --fidelity high

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
from ComplexMechBenchmarkIDK import ComplexMechBenchmarkIDK


def main():
    parser = argparse.ArgumentParser(
        description="Run ComplexMechIDK benchmark on a trained model checkpoint.",
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
        help="Directory to save results (default: checkpoint directory/complexmech_idk_<fidelity>)"
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=None,
        help="Directory containing ComplexMechIDK configs and data_cache (default: script directory)"
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
    print("ComplexMechIDK Benchmark Runner")
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
        env_benchmark_dir = os.environ.get("COMPLEXMECH_IDK_BENCHMARK_DIR")
        if env_benchmark_dir:
            benchmark_dir = Path(env_benchmark_dir)
            print(f"Using COMPLEXMECH_IDK_BENCHMARK_DIR from environment: {benchmark_dir}")
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
        # Default to checkpoint directory / complexmech_idk_<fidelity>
        output_dir = checkpoint_path.parent / f"complexmech_idk_{args.fidelity}"
    
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
        print("Initializing ComplexMechIDK benchmark...")
        benchmark = ComplexMechBenchmarkIDK(
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
        results_json_path = output_dir / f"complexmech_idk_benchmark_{args.fidelity}.json"
        with open(results_json_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        print(f"  Saved JSON: {results_json_path}")
        
        # Save CSV
        try:
            import pandas as pd
            
            rows = []
            for config_key, node_results in results.items():
                if isinstance(config_key, tuple) and len(config_key) == 3:
                    node_count, variant, hide_fraction = config_key
                    row = {
                        'node_count': node_count,
                        'variant': variant,
                        'hide_fraction': hide_fraction if hide_fraction is not None else 'base'
                    }
                else:
                    row = {'node_count': config_key, 'variant': 'base', 'hide_fraction': 'N/A'}
                
                for metric_name in ['mse', 'r2', 'nll']:
                    if metric_name in node_results:
                        for stat_name, stat_value in node_results[metric_name].items():
                            row[f'{metric_name}_{stat_name}'] = stat_value
                rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                results_csv_path = output_dir / f"complexmech_idk_benchmark_{args.fidelity}.csv"
                df.to_csv(results_csv_path, index=False)
                print(f"  Saved CSV: {results_csv_path}")
        except ImportError:
            print("  Warning: pandas not available, skipping CSV output")
        
        # Print summary
        print()
        print("=" * 80)
        print("Benchmark Summary")
        print("=" * 80)
        print()
        
        # Group results by variant for cleaner output
        variants_summary = {}
        for config_key, node_results in results.items():
            if isinstance(config_key, tuple) and len(config_key) == 3:
                node_count, variant, hide_fraction = config_key
            else:
                node_count = config_key
                variant = 'base'
                hide_fraction = None
            
            if variant not in variants_summary:
                variants_summary[variant] = []
            
            mse_mean = node_results.get('mse', {}).get('mean', float('nan'))
            r2_mean = node_results.get('r2', {}).get('mean', float('nan'))
            
            variants_summary[variant].append({
                'node_count': node_count,
                'hide_fraction': hide_fraction,
                'mse_mean': mse_mean,
                'r2_mean': r2_mean,
            })
        
        for variant, results_list in sorted(variants_summary.items()):
            print(f"\nVariant: {variant}")
            print("-" * 40)
            for r in sorted(results_list, key=lambda x: (x['node_count'], x['hide_fraction'] or 0)):
                hide_str = f"hide={r['hide_fraction']}" if r['hide_fraction'] is not None else "uniform"
                print(f"  {r['node_count']:2d} nodes, {hide_str:12s}: MSE={r['mse_mean']:.6f}, R²={r['r2_mean']:.4f}")
        
        print()
        print("=" * 80)
        print("Benchmark completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print()
        print("=" * 80)
        print("BENCHMARK FAILED!")
        print("=" * 80)
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
