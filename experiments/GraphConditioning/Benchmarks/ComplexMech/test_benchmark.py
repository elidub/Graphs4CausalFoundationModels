#!/usr/bin/env python3
"""
Test script for ComplexMechBenchmark using local paths.
"""

import os
import sys
from pathlib import Path

# Get the current directory (ComplexMech folder)
current_dir = Path(__file__).parent.absolute()

# Import the benchmark with local path
sys.path.insert(0, str(current_dir))
from ComplexMechBenchmark import ComplexMechBenchmark

def main():
    print("ComplexMechBenchmark - Local Test")
    print("="*80)
    
    MAX_SAMPLES = 5  # Very small for quick testing
    
    # Initialize with current directory
    benchmark = ComplexMechBenchmark(
        benchmark_dir=str(current_dir),
        verbose=True, 
        max_samples=MAX_SAMPLES,
        use_ancestor_matrix=False,
    )
    
    # Test loading a config
    print("\nTesting config loading...")
    try:
        config = benchmark.load_config(2, "base")
        print(f"  ✓ Loaded base config keys: {list(config.keys())}")
        print(f"  ✓ Num nodes: {config['scm_config']['num_nodes']['value']}")
        
        config_path = benchmark.load_config(2, "path_TY", sample_size=500)
        print(f"  ✓ Loaded path_TY config with ntest_500")
        
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ComplexMechBenchmark test completed successfully!")
    print(f"✓ Benchmark directory: {current_dir}")
    print(f"✓ Cache directory: {current_dir / 'data_cache'}")
    print("\nThe benchmark is ready to use with:")
    print("  benchmark = ComplexMechBenchmark(")
    print(f"      benchmark_dir='{current_dir}',")
    print("      max_samples=100")
    print("  )")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)