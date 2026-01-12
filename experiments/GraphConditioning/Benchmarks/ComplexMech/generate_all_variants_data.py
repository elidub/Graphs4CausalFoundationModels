"""
Generate benchmark datasets for all ComplexMech config variants.

This script generates synthetic datasets for:
- Base configs 
- Path variants with different sample sizes:
  * Treatment→Outcome path (path_TY)
  * Outcome→Treatment path (path_YT)  
  * Independent T⊥Y (path_independent_TY)
  * Sample sizes: 500, 700, 800, 900, 950

For node counts: 2, 5, 20, 50

Usage:
    python generate_all_variants_data.py [--num-samples N] [--overwrite] [--node-counts 2 5 20]
    python generate_all_variants_data.py --variants base --sample-sizes 500  # Only base or specific sizes
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

# Add benchmark dir to path for ComplexMechBenchmark import
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset
import pickle
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark datasets for all ComplexMech config variants",
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
        default=[2, 5, 20, 50],
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
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[500, 700, 800, 900, 950],
        help="Sample sizes to generate for path variants (ignored for base)",
    )
    
    return parser.parse_args()


def load_config(config_path: Path):
    """Load a YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def sample_config(config_path: Path, num_samples: int, seed: int, verbose: bool = True):
    """
    Sample data from a config file using the same approach as LinGausIDK benchmark.
    
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
    
    # Extract the three main configs (following LinGausIDK approach)
    scm_config = config['scm_config']
    dataset_config = config['dataset_config']
    preprocessing_config = config['preprocessing_config']
    
    # Ensure adjacency matrix is returned (we always need it)
    if 'return_adjacency_matrix' not in dataset_config:
        dataset_config['return_adjacency_matrix'] = {'value': True}
    else:
        dataset_config['return_adjacency_matrix']['value'] = True
    
    # Ensure ancestor matrix is not directly requested from dataset
    if 'return_ancestor_matrix' in dataset_config:
        dataset_config['return_ancestor_matrix']['value'] = False
    
    # Create dataset (following LinGausIDK approach)
    dataset = InterventionalDataset(
        scm_config=scm_config,
        preprocessing_config=preprocessing_config,
        dataset_config=dataset_config,
        seed=seed,
        return_scm=True,  # Include SCM for debugging/analysis
    )
    
    if verbose:
        print(f"  Dataset created with size {len(dataset)}")
        print(f"  Sampling {num_samples} elements...")
    
    # Sample data (following LinGausIDK approach)
    sampled_data = []
    idx = 0
    failed_attempts = 0
    max_failed_attempts = num_samples * 10  # Allow up to 10x failed attempts before giving up
    
    if verbose:
        pbar = tqdm(total=num_samples, desc="  Sampling")
    
    while len(sampled_data) < num_samples and failed_attempts < max_failed_attempts:
        try:
            # Get data tuple: (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adj, scm, processor, intervention_node)
            data_item = dataset[idx]
            
            # Debug first sample structure
            if verbose and len(sampled_data) == 0:
                print(f"\n  Debug: Dataset returns {len(data_item)} elements")
            
            # Get adjacency matrix (position 6 based on LinGausIDK)
            adjacency_matrix = data_item[6]
            
            # Store as dict format (following LinGausIDK structure)
            sample_dict = {
                'X_obs': data_item[0],
                'T_obs': data_item[1],
                'Y_obs': data_item[2],
                'X_intv': data_item[3],
                'T_intv': data_item[4],
                'Y_intv': data_item[5],
                'adjacency_matrix': adjacency_matrix,
                'intervention_node': data_item[9] if len(data_item) > 9 else None,
                'metadata': {
                    'intervened_feature': data_item[8].intervened_feature if len(data_item) > 8 else None,
                    'selected_target_feature': data_item[8].selected_target_feature if len(data_item) > 8 else None,
                    'kept_feature_indices': data_item[8].kept_feature_indices if len(data_item) > 8 else None,
                }
            }
            
            sampled_data.append(sample_dict)
            if verbose:
                pbar.update(1)
            
        except Exception as e:
            failed_attempts += 1
            if verbose and "Inconsistent PAM" in str(e):
                # Just show a brief message for PAM inconsistencies, not full traceback
                print(f"\n  Warning: Skipping item {idx} due to PAM inconsistency")
            elif verbose:
                print(f"\n  Warning: Failed to sample item {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        idx += 1
    
    if verbose:
        pbar.close()
        if failed_attempts >= max_failed_attempts:
            print(f"\n  Warning: Reached maximum failed attempts ({max_failed_attempts}), collected {len(sampled_data)}/{num_samples} samples")
        elif failed_attempts > 0:
            print(f"\n  Successfully collected {len(sampled_data)} samples after {failed_attempts} failed attempts")
    
    # Create metadata (following LinGausIDK structure)
    metadata = {
        'num_samples': len(sampled_data),
        'seed': seed,
        'config_file': str(config_path),
        'node_count': scm_config['num_nodes']['value'],
        'config': config,
        'scm_config': scm_config,
        'dataset_config': dataset_config,
        'preprocessing_config': preprocessing_config,
    }
    
    # Package results
    result = {
        'data': sampled_data,
        'metadata': metadata,
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
    print("ComplexMech Benchmark - All Variants Data Generation")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Benchmark directory: {benchmark_dir}")
    print(f"  Configs directory: {configs_dir}")
    print(f"  Cache directory: {cache_dir}")
    print(f"  Number of samples per config: {args.num_samples}")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Node counts: {args.node_counts}")
    print(f"  Variants: {args.variants}")
    print(f"  Sample sizes: {args.sample_sizes}")
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
                output_filename = f"complexmech_{node_count}nodes_base_{args.num_samples}samples_seed{args.base_seed}.pkl"
                output_path = cache_dir / output_filename
                
                # Check if already exists
                if output_path.exists() and not args.overwrite:
                    print(f"\nSkipping base config for {node_count} nodes: output file already exists")
                    print(f"  Use --overwrite to regenerate")
                    continue
                
                configs_to_process.append({
                    'node_count': node_count,
                    'variant': variant,
                    'sample_size': None,
                    'config_path': config_path,
                    'output_path': output_path,
                })
            else:
                # Path variant configs: configs/{node_count}node/{variant}/ntest_{sample_size}.yaml
                for sample_size in args.sample_sizes:
                    config_path = configs_dir / f"{node_count}node" / variant / f"ntest_{sample_size}.yaml"
                    
                    if not config_path.exists():
                        print(f"\nWarning: Config file not found: {config_path}")
                        continue
                    
                    # Output filename for path variant with sample size
                    output_filename = f"complexmech_{node_count}nodes_{variant}_ntest{sample_size}_{args.num_samples}samples_seed{args.base_seed}.pkl"
                    output_path = cache_dir / output_filename
                    
                    # Check if already exists
                    if output_path.exists() and not args.overwrite:
                        print(f"\nSkipping {variant} (ntest={sample_size}) for {node_count} nodes: output file already exists")
                        print(f"  Use --overwrite to regenerate")
                        continue
                    
                    configs_to_process.append({
                        'node_count': node_count,
                        'variant': variant,
                        'sample_size': sample_size,
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
        sample_size = config_info['sample_size']
        config_path = config_info['config_path']
        output_path = config_info['output_path']
        
        if variant == "base":
            print(f"\n[{i+1}/{len(configs_to_process)}] Processing: {node_count} nodes, variant: base")
        else:
            print(f"\n[{i+1}/{len(configs_to_process)}] Processing: {node_count} nodes, variant: {variant}, sample_size: {sample_size}")
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
            
            # Add variant and sample_size info to metadata
            result['metadata']['variant'] = variant
            result['metadata']['sample_size'] = sample_size
            
            # Save to disk
            print(f"  Saving to {output_path.name}...")
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            
            saved_paths.append({
                'node_count': node_count,
                'variant': variant,
                'sample_size': sample_size,
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
                variant_label = f"{item['variant']} (ntest={item['sample_size']})"
            print(f"  - {variant_label:40s}: {item['path'].name} ({item['size_mb']:.2f} MB)")
            total_size_mb += item['size_mb']
        print()
    
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"Cache directory: {cache_dir}")
    print("="*80)
    print("\nData generation complete! ✓")
    
    # Print usage instructions
    print("\nGenerated datasets can be used with ComplexMechBenchmark:")
    print("="*60)
    print("from ComplexMechBenchmark import ComplexMechBenchmark")
    print("")
    print("benchmark = ComplexMechBenchmark(")
    print(f"    benchmark_dir='{benchmark_dir}',")
    print("    max_samples=100")
    print(")")
    print("")
    print("# Load and evaluate a dataset")
    print("data, metadata = benchmark.load_data('complexmech_5nodes_base_1000samples_seed42.pkl')")
    print("benchmark.load_model(config_path, checkpoint_path)")
    print("results = benchmark.evaluate_dataset(data)")
    print("="*60)
    print("="*80)


if __name__ == "__main__":
    main()