#!/usr/bin/env python3
"""
Checkpointed data generation for 50 node ComplexMech configurations
Processes each configuration individually and can resume from where it left off
"""
import os
import sys
import yaml
import pickle
import argparse
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parents[4]
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Add benchmark dir to path 
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset

def load_config(config_path):
    """Load a YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_checkpoint_if_exists(output_path):
    """Load existing checkpoint data if available"""
    checkpoint_files = []
    batch_idx = 1
    
    # Find all checkpoint files for this output
    while True:
        checkpoint_path = str(output_path) + f".checkpoint_{batch_idx}"
        if os.path.exists(checkpoint_path):
            checkpoint_files.append((batch_idx, checkpoint_path))
            batch_idx += 1
        else:
            break
    
    if not checkpoint_files:
        return None, 0
    
    # Load the latest checkpoint
    latest_batch, latest_path = checkpoint_files[-1]
    try:
        with open(latest_path, 'rb') as f:
            sampled_data = pickle.load(f)
        
        print(f"  Found checkpoint from batch {latest_batch} with {len(sampled_data)} samples")
        return sampled_data, latest_batch
    except Exception as e:
        print(f"  Warning: Could not load checkpoint {latest_path}: {e}")
        return None, 0

def generate_single_config(config_path, output_path, num_samples, seed):
    """Generate data for a single configuration with progress tracking"""
    print(f"\n{'='*80}")
    print(f"Processing: {config_path}")
    print(f"Output: {output_path}")
    print(f"{'='*80}")
    
    # Check if output already exists
    if os.path.exists(output_path):
        print(f"✓ Output file already exists, skipping: {output_path}")
        return True
    
    # Check for existing checkpoints
    existing_data, last_batch = load_checkpoint_if_exists(output_path)
    if existing_data:
        print(f"Resuming from checkpoint at batch {last_batch}...")
        sampled_data = existing_data
        start_batch = last_batch
    else:
        sampled_data = []
        start_batch = 0
        
    try:
        # Load config
        print(f"Loading config from {config_path}...")
        config = load_config(config_path)
        
        # Debug config matrix settings  
        scm_config = config.get('scm_config', {})
        preprocessing_config = config.get('preprocessing_config', {})
        dataset_config = config.get('dataset_config', {})
        
        print(f"Config matrix settings:")
        print(f"  return_adjacency_matrix: {scm_config.get('return_adjacency_matrix', 'not specified')}")
        print(f"  return_ancestor_matrix: {scm_config.get('return_ancestor_matrix', 'not specified')}")
        print(f"  use_partial_graph_format: {scm_config.get('use_partial_graph_format', 'not specified')}")
        
        # Create dataset (following training approach - use config as-is)
        dataset = InterventionalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config,
            seed=seed,
            return_scm=True,  # Include SCM for debugging/analysis
        )
        print(f"Dataset created with size {len(dataset)}")
        
        # Sample data with progress tracking and checkpoint resume
        print(f"Sampling {num_samples} elements...")
        
        # Sample in batches to show progress and avoid memory issues
        batch_size = min(100, num_samples)
        batches_needed = (num_samples + batch_size - 1) // batch_size
        
        # Start from checkpoint if available
        if start_batch > 0:
            print(f"Resuming from batch {start_batch + 1}/{batches_needed} with {len(sampled_data)} existing samples")
        
        for batch_idx in range(start_batch, batches_needed):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_samples = end_idx - start_idx
            
            print(f"  Batch {batch_idx + 1}/{batches_needed}: sampling {batch_samples} elements...")
            
            # Sample this batch
            for i in range(batch_samples):
                try:
                    # Use a deterministic seed for reproducibility
                    sample_seed = seed + start_idx + i
                    data_item = dataset[sample_seed % len(dataset)]
                    
                    # Extract only the essential tensor/array data, skip complex objects with lambdas
                    # data_item structure: (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, graph_matrix, scm, processor, intervention_node)
                    essential_sample = {
                        'X_obs': data_item[0],
                        'T_obs': data_item[1], 
                        'Y_obs': data_item[2],
                        'X_intv': data_item[3],
                        'T_intv': data_item[4],
                        'Y_intv': data_item[5],
                        'graph_matrix': data_item[6],
                        'intervention_node': data_item[9] if len(data_item) > 9 else None
                    }
                    # Skip the SCM (data_item[7]) and processor (data_item[8]) as they contain lambdas
                    
                    sampled_data.append(essential_sample)
                    
                    # Progress indicator
                    if (start_idx + i + 1) % 50 == 0:
                        print(f"    Progress: {start_idx + i + 1}/{num_samples} samples collected...")
                        
                except Exception as e:
                    print(f"    Error sampling element {start_idx + i}: {e}")
                    # Continue with next sample
                    continue
            
            # Checkpoint after each batch - save just the essential data
            if len(sampled_data) > 0:
                checkpoint_path = str(output_path) + f".checkpoint_{batch_idx + 1}"
                print(f"  Saving checkpoint with {len(sampled_data)} samples to {checkpoint_path}")
                try:
                    # Just save the raw data, not the complex objects
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(sampled_data, f)
                    print(f"    Checkpoint saved successfully")
                except Exception as e:
                    print(f"  Warning: Could not save checkpoint: {e}")
                    # Continue without checkpointing
        
        print(f"\nCollected {len(sampled_data)} samples total")
        
        if len(sampled_data) == 0:
            print("❌ No samples collected - dataset sampling failed")
            return False
            
        # Save final data
        print(f"Saving final data to {output_path}...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(sampled_data, f)
            
        # Clean up checkpoint files
        print("Cleaning up checkpoint files...")
        for batch_idx in range(batches_needed):
            checkpoint_path = str(output_path) + f".checkpoint_{batch_idx + 1}"
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                    print(f"  Removed {checkpoint_path}")
                except:
                    pass  # Ignore cleanup errors
        
        print(f"✓ Successfully generated and saved {len(sampled_data)} samples")
        return True
        
    except Exception as e:
        print(f"❌ Error processing config: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate ComplexMech 50 node benchmark data with checkpointing')
    parser.add_argument('--config', help='Specific config file to process (optional)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples per config')
    parser.add_argument('--seed', type=int, default=42, help='Base seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup paths
    benchmark_dir = Path(__file__).parent
    configs_dir = benchmark_dir / "configs" / "50node"
    data_cache_dir = benchmark_dir / "data_cache"
    
    print("=" * 80)
    print("ComplexMech 50 Node Checkpointed Data Generation")
    print("=" * 80)
    print(f"Configs directory: {configs_dir}")
    print(f"Cache directory: {data_cache_dir}")
    print(f"Number of samples per config: {args.num_samples}")
    print(f"Base seed: {args.seed}")
    
    # Ensure output directory exists
    os.makedirs(data_cache_dir, exist_ok=True)
    
    # Define all 50 node configurations to process
    node_count = 50
    variants = ['base', 'path_TY', 'path_YT', 'path_independent_TY']
    sample_sizes = [500, 700, 800, 900, 950]  # ntest values for path variants
    
    configs_to_process = []
    
    if args.config:
        # Process specific config
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = configs_dir / args.config
        
        # Determine output filename from config path
        if 'base' in str(config_path):
            output_name = f"complexmech_{node_count}nodes_base_{args.num_samples}samples_seed{args.seed}.pkl"
        else:
            # Extract variant and ntest from path
            parts = config_path.parts
            variant = None
            ntest = None
            for part in parts:
                if part in ['path_TY', 'path_YT', 'path_independent_TY']:
                    variant = part
                elif part.startswith('ntest_'):
                    ntest = part.split('_')[1]
            
            if variant and ntest:
                output_name = f"complexmech_{node_count}nodes_{variant}_ntest{ntest}_{args.num_samples}samples_seed{args.seed}.pkl"
            else:
                output_name = f"complexmech_{node_count}nodes_custom_{args.num_samples}samples_seed{args.seed}.pkl"
        
        output_path = data_cache_dir / output_name
        configs_to_process = [(config_path, output_path)]
    else:
        # Process all 50 node configs
        # Base config
        base_config = configs_dir / "base.yaml"
        base_output = data_cache_dir / f"complexmech_{node_count}nodes_base_{args.num_samples}samples_seed{args.seed}.pkl"
        configs_to_process.append((base_config, base_output))
        
        # Variant configs
        for variant in ['path_TY', 'path_YT', 'path_independent_TY']:
            for ntest in sample_sizes:
                config_path = configs_dir / variant / f"ntest_{ntest}.yaml"
                output_name = f"complexmech_{node_count}nodes_{variant}_ntest{ntest}_{args.num_samples}samples_seed{args.seed}.pkl"
                output_path = data_cache_dir / output_name
                configs_to_process.append((config_path, output_path))
    
    print(f"\nFound {len(configs_to_process)} configurations to process:")
    for i, (config_path, output_path) in enumerate(configs_to_process):
        status = "✓ EXISTS" if os.path.exists(output_path) else "⚠ MISSING"
        print(f"  {i+1:2d}. {config_path.name:25s} → {output_path.name} [{status}]")
    
    # Process each configuration
    successful = 0
    failed = 0
    skipped = 0
    
    for i, (config_path, output_path) in enumerate(configs_to_process):
        print(f"\n{'='*20} CONFIG {i+1}/{len(configs_to_process)} {'='*20}")
        print(f"Config: {config_path}")
        print(f"Output: {output_path.name}")
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            failed += 1
            continue
        
        if os.path.exists(output_path):
            print(f"✓ Output already exists, skipping")
            skipped += 1
            continue
        
        # Generate data for this config
        success = generate_single_config(config_path, output_path, args.num_samples, args.seed)
        
        if success:
            successful += 1
            print(f"✓ Config {i+1}/{len(configs_to_process)} completed successfully")
        else:
            failed += 1
            print(f"❌ Config {i+1}/{len(configs_to_process)} failed")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total configurations: {len(configs_to_process)}")
    print(f"✓ Successful: {successful}")
    print(f"⚠ Skipped (already exist): {skipped}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print(f"\n⚠ {failed} configurations failed to generate")
        sys.exit(1)
    else:
        print(f"\n✓ All configurations processed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()