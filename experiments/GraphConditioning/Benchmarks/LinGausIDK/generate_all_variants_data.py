"""
Generate benchmark datasets for all config variants with partial graph knowledge.

This script generates synthetic datasets for:
- Base configs (uniform hide_fraction distribution)
- Path variants with fixed hide fractions:
  * Treatment→Outcome path (path_TY)
  * Outcome→Treatment path (path_YT)  
  * Independent T⊥Y (path_independent_TY)
  * Hide fractions: 0.0, 0.25, 0.5, 0.75, 1.0

For node counts: 2, 5, 10, 20, 35, 50

Usage:
    python generate_all_variants_data.py [--num-samples N] [--overwrite] [--node-counts 2 5 10]
    python generate_all_variants_data.py --variants base --hide-fractions 0.5  # Only base or specific fractions
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add src to path
repo_root = Path(__file__).resolve().parents[4]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add benchmark dir to path for LingausBenchmark import
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset
import pickle
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark datasets for all config variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate per config variant",
    )
    
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed",
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
        "--variants",
        type=str,
        nargs="+",
        default=["base", "path_TY", "path_YT", "path_independent_TY"],
        help="Config variants to generate (base, path_TY, path_YT, path_independent_TY)",
    )
    
    parser.add_argument(
        "--hide-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Hide fractions to generate for path variants (ignored for base)",
    )
    
    return parser.parse_args()


def load_config(config_path: Path):
    """Load a YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def sample_config(config_path: Path, num_samples: int, seed: int, verbose: bool = True):
    """
    Sample data from a config file.
    
    Args:
        config_path: Path to the YAML config file
        num_samples: Number of samples to generate
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Dictionary with sampled data and metadata
    """
    if verbose:
        print(f"  Loading config from {config_path.name}...")
    
    config = load_config(config_path)
    
    # Create dataset
    dataset = InterventionalDataset(
        dataset_config=config['dataset_config'],
        scm_config=config['scm_config'],
        preprocessing_config=config['preprocessing_config'],
    )
    
    # Sample data
    all_data = []
    if verbose:
        pbar = tqdm(total=num_samples, desc="  Sampling")
    
    for i in range(num_samples):
        try:
            sample = dataset[i]
            all_data.append(sample)
            if verbose:
                pbar.update(1)
        except Exception as e:
            if verbose:
                print(f"\n  Warning: Failed to sample item {i}: {e}")
            continue
    
    if verbose:
        pbar.close()
    
    # Package results
    result = {
        'data': all_data,
        'config': config,
        'metadata': {
            'num_samples': len(all_data),
            'seed': seed,
            'config_file': str(config_path),
            'node_count': config['scm_config']['num_nodes']['value'],
        }
    }
    
    return result


def main():
    """Main function to generate all variant data."""
    args = parse_args()
    
    benchmark_dir = Path(__file__).parent
    configs_dir = benchmark_dir / "configs"
    cache_dir = benchmark_dir / "data_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LinGausIDK Benchmark - All Variants Data Generation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Benchmark directory: {benchmark_dir}")
    print(f"  Configs directory: {configs_dir}")
    print(f"  Cache directory: {cache_dir}")
    print(f"  Number of samples per config: {args.num_samples}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Node counts: {args.node_counts}")
    print(f"  Variants: {args.variants}")
    print(f"  Hide fractions: {args.hide_fractions}")
    print(f"  Overwrite existing: {args.overwrite}")
    print("="*80)
    
    # Build list of all config files to process
    configs_to_process = []
    
    for node_count in args.node_counts:
        for variant in args.variants:
            if variant == "base":
                # Base config: configs/{node_count}node/base.yaml
                config_path = configs_dir / f"{node_count}node" / "base.yaml"
                
                if not config_path.exists():
                    print(f"\nWarning: Config file not found: {config_path}")
                    continue
                
                # Output filename for base variant
                output_filename = f"lingaus_{node_count}nodes_base_{args.num_samples}samples_seed{args.base_seed}.pkl"
                output_path = cache_dir / output_filename
                
                # Check if already exists
                if output_path.exists() and not args.overwrite:
                    print(f"\nSkipping base config for {node_count} nodes: output file already exists")
                    print(f"  Use --overwrite to regenerate")
                    continue
                
                configs_to_process.append({
                    'node_count': node_count,
                    'variant': variant,
                    'hide_fraction': None,
                    'config_path': config_path,
                    'output_path': output_path,
                })
            else:
                # Path variant configs: configs/{node_count}node/{variant}/hide_{fraction}.yaml
                for hide_fraction in args.hide_fractions:
                    config_path = configs_dir / f"{node_count}node" / variant / f"hide_{hide_fraction}.yaml"
                    
                    if not config_path.exists():
                        print(f"\nWarning: Config file not found: {config_path}")
                        continue
                    
                    # Output filename for path variant with hide fraction
                    output_filename = f"lingaus_{node_count}nodes_{variant}_hide{hide_fraction}_{args.num_samples}samples_seed{args.base_seed}.pkl"
                    output_path = cache_dir / output_filename
                    
                    # Check if already exists
                    if output_path.exists() and not args.overwrite:
                        print(f"\nSkipping {variant} (hide={hide_fraction}) for {node_count} nodes: output file already exists")
                        print(f"  Use --overwrite to regenerate")
                        continue
                    
                    configs_to_process.append({
                        'node_count': node_count,
                        'variant': variant,
                        'hide_fraction': hide_fraction,
                        'config_path': config_path,
                        'output_path': output_path,
                    })
    
    if not configs_to_process:
        print("\nNo configs to process!")
        return
    
    print(f"\nProcessing {len(configs_to_process)} config variants...")
    print("="*80)
    
    # Process each config
    saved_paths = []
    
    for i, config_info in enumerate(configs_to_process):
        node_count = config_info['node_count']
        variant = config_info['variant']
        hide_fraction = config_info['hide_fraction']
        config_path = config_info['config_path']
        output_path = config_info['output_path']
        
        if variant == "base":
            print(f"\n[{i+1}/{len(configs_to_process)}] Processing: {node_count} nodes, variant: base")
        else:
            print(f"\n[{i+1}/{len(configs_to_process)}] Processing: {node_count} nodes, variant: {variant}, hide_fraction: {hide_fraction}")
        print(f"  Config: {config_path.relative_to(benchmark_dir)}")
        print(f"  Output: {output_path.name}")
        
        try:
            # Sample data
            result = sample_config(
                config_path=config_path,
                num_samples=args.num_samples,
                seed=args.base_seed + i * 1000,  # Different seed per config
                verbose=True,
            )
            
            # Save to disk
            print(f"  Saving to {output_path.name}...")
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            
            saved_paths.append({
                'node_count': node_count,
                'variant': variant,
                'hide_fraction': hide_fraction,
                'path': output_path,
                'size_mb': output_path.stat().st_size / 1024 / 1024,
            })
            
            print(f"  ✓ Saved successfully ({saved_paths[-1]['size_mb']:.2f} MB)")
            
        except Exception as e:
            print(f"  ✗ ERROR: Failed to process config: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("Data Generation Summary")
    print("="*80)
    print(f"\nSuccessfully generated {len(saved_paths)}/{len(configs_to_process)} datasets:\n")
    
    # Group by node count
    by_node_count = {}
    for item in saved_paths:
        nc = item['node_count']
        if nc not in by_node_count:
            by_node_count[nc] = []
        by_node_count[nc].append(item)
    
    total_size_mb = 0
    for node_count in sorted(by_node_count.keys()):
        print(f"{node_count} nodes:")
        for item in by_node_count[node_count]:
            if item['variant'] == "base":
                variant_label = "base"
            else:
                variant_label = f"{item['variant']} (hide={item['hide_fraction']})"
            print(f"  - {variant_label:35s}: {item['path'].name} ({item['size_mb']:.2f} MB)")
            total_size_mb += item['size_mb']
        print()
    
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Cache directory: {cache_dir}")
    print("="*80)
    print("\nData generation complete! ✓")
    print("="*80)


if __name__ == "__main__":
    main()
