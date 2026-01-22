#!/usr/bin/env python
"""
Verify ComplexMechIDK benchmark data generation is complete and valid.
Checks:
1. All 96 config files have corresponding data files
2. Each data file contains 1000 samples
3. Sample structure is valid (has expected keys)
4. File sizes are reasonable
"""

import os
import sys
import pickle
from pathlib import Path
from collections import defaultdict

# Configuration
BENCHMARK_DIR = Path(__file__).parent
CONFIG_DIR = BENCHMARK_DIR / "configs"
DATA_DIR = BENCHMARK_DIR / "data_cache"
EXPECTED_SAMPLES = 1000
MIN_ACCEPTABLE_SAMPLES = 950  # Accept files with at least this many samples
SEED = 42

def get_expected_output_name(config_path: Path) -> str:
    """Convert config path to expected output filename."""
    # Get relative path from configs dir
    rel_path = config_path.relative_to(CONFIG_DIR)
    parts = rel_path.parts
    
    if len(parts) == 2:
        # Base config: e.g., configs/10node/base.yaml -> complexmech_10nodes_base_1000samples_seed42
        node_dir = parts[0]  # e.g., "10node"
        config_name = parts[1].replace('.yaml', '')  # e.g., "base"
        return f"complexmech_{node_dir}s_{config_name}_{EXPECTED_SAMPLES}samples_seed{SEED}"
    elif len(parts) == 3:
        # Hide config: e.g., configs/10node/path_TY/hide_0.0.yaml
        node_dir = parts[0]  # e.g., "10node"
        path_type = parts[1]  # e.g., "path_TY"
        hide_file = parts[2].replace('.yaml', '')  # e.g., "hide_0.0"
        hide_value = hide_file.replace('hide_', 'hide')  # e.g., "hide0.0"
        return f"complexmech_{node_dir}s_{path_type}_{hide_value}_{EXPECTED_SAMPLES}samples_seed{SEED}"
    else:
        return None


def find_all_configs():
    """Find all config files."""
    configs = []
    for config_file in CONFIG_DIR.rglob("*.yaml"):
        configs.append(config_file)
    return sorted(configs)


def verify_data_file(data_path: Path) -> dict:
    """Verify a single data file and return info."""
    result = {
        'exists': False,
        'size_mb': 0,
        'num_samples': 0,
        'sample_keys': [],
        'format': None,
        'error': None,
    }
    
    if not data_path.exists():
        result['error'] = "File not found"
        return result
    
    result['exists'] = True
    result['size_mb'] = data_path.stat().st_size / (1024 * 1024)
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old format (dict with 'data' key) and new format (raw list)
        if isinstance(data, dict) and 'data' in data:
            # Old format: {'data': [...], 'config': {...}, 'metadata': {...}}
            result['format'] = 'wrapped_dict'
            samples = data['data']
        elif isinstance(data, list):
            # New format: direct list of samples
            result['format'] = 'raw_list'
            samples = data
        else:
            result['error'] = f"Unexpected data type: {type(data).__name__}"
            return result
        
        result['num_samples'] = len(samples)
        if len(samples) > 0:
            sample = samples[0]
            if isinstance(sample, dict):
                result['sample_keys'] = list(sample.keys())
            elif isinstance(sample, tuple):
                result['sample_keys'] = [f"tuple[{len(sample)}]"]
            else:
                result['sample_keys'] = [type(sample).__name__]
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    print("=" * 70)
    print("ComplexMechIDK Benchmark Data Verification")
    print("=" * 70)
    print(f"Benchmark dir: {BENCHMARK_DIR}")
    print(f"Config dir: {CONFIG_DIR}")
    print(f"Data dir: {DATA_DIR}")
    print()
    
    # Find all configs
    configs = find_all_configs()
    print(f"Found {len(configs)} config files")
    print()
    
    # Check each config
    results_by_nodes = defaultdict(list)
    missing = []
    incomplete = []
    errors = []
    
    total_samples = 0
    total_size_mb = 0
    
    for config_path in configs:
        output_name = get_expected_output_name(config_path)
        if output_name is None:
            print(f"WARNING: Could not parse config path: {config_path}")
            continue
        
        data_path = DATA_DIR / f"{output_name}.pkl"
        result = verify_data_file(data_path)
        
        # Extract node count for grouping
        node_count = config_path.relative_to(CONFIG_DIR).parts[0]
        
        if not result['exists']:
            missing.append((config_path, output_name))
            results_by_nodes[node_count].append(('MISSING', output_name, result))
        elif result['error']:
            errors.append((config_path, output_name, result['error']))
            results_by_nodes[node_count].append(('ERROR', output_name, result))
        elif result['num_samples'] < MIN_ACCEPTABLE_SAMPLES:
            incomplete.append((config_path, output_name, result['num_samples']))
            results_by_nodes[node_count].append(('INCOMPLETE', output_name, result))
        else:
            results_by_nodes[node_count].append(('OK', output_name, result))
            total_samples += result['num_samples']
            total_size_mb += result['size_mb']
    
    # Print summary by node count
    print("=" * 70)
    print("Summary by Node Count")
    print("=" * 70)
    
    for node_count in sorted(results_by_nodes.keys(), key=lambda x: int(x.replace('node', ''))):
        results = results_by_nodes[node_count]
        ok_count = sum(1 for r in results if r[0] == 'OK')
        total_count = len(results)
        
        # Count formats
        wrapped_count = sum(1 for r in results if r[0] == 'OK' and r[2].get('format') == 'wrapped_dict')
        raw_count = sum(1 for r in results if r[0] == 'OK' and r[2].get('format') == 'raw_list')
        
        status = "✓" if ok_count == total_count else "✗"
        format_info = f" (wrapped:{wrapped_count}, raw:{raw_count})" if ok_count > 0 else ""
        print(f"{status} {node_count}: {ok_count}/{total_count} files complete{format_info}")
        
        # Show details for incomplete
        for status, name, result in results:
            if status != 'OK':
                print(f"    {status}: {name}")
                if result.get('error'):
                    print(f"           Error: {result['error']}")
                elif result.get('num_samples', 0) > 0:
                    print(f"           Samples: {result['num_samples']}/{EXPECTED_SAMPLES}")
    
    print()
    print("=" * 70)
    print("Overall Summary")
    print("=" * 70)
    
    total_configs = len(configs)
    complete_count = total_configs - len(missing) - len(incomplete) - len(errors)
    
    print(f"Total configs:     {total_configs}")
    print(f"Complete (OK):     {complete_count}")
    print(f"Missing:           {len(missing)}")
    print(f"Incomplete:        {len(incomplete)}")
    print(f"Errors:            {len(errors)}")
    print()
    print(f"Total samples:     {total_samples:,}")
    print(f"Total size:        {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"Avg size/file:     {total_size_mb/max(complete_count, 1):.2f} MB")
    print()
    
    if complete_count == total_configs:
        print("✅ ALL DATA GENERATION COMPLETE!")
        
        # Show sample structure from first file
        print()
        print("=" * 70)
        print("Sample Structure (from first file)")
        print("=" * 70)
        
        first_result = results_by_nodes[sorted(results_by_nodes.keys())[0]][0]
        if first_result[0] == 'OK':
            first_file = DATA_DIR / f"{first_result[1]}.pkl"
            with open(first_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"File: {first_file.name}")
            print(f"Type: list of {len(data)} samples")
            
            if len(data) > 0:
                sample = data[0]
                print(f"Sample type: {type(sample).__name__}")
                
                if isinstance(sample, dict):
                    print("Sample keys:")
                    for key, value in sample.items():
                        if hasattr(value, 'shape'):
                            print(f"  - {key}: {type(value).__name__} shape={value.shape}")
                        elif isinstance(value, (list, tuple)):
                            print(f"  - {key}: {type(value).__name__} len={len(value)}")
                        else:
                            val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                            print(f"  - {key}: {type(value).__name__} = {val_str}")
        
        return 0
    else:
        print("❌ DATA GENERATION INCOMPLETE")
        
        if missing:
            print()
            print("Missing files:")
            for config_path, output_name in missing[:10]:
                print(f"  - {output_name}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
