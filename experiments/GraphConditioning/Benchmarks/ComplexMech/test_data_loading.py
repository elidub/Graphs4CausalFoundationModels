#!/usr/bin/env python3
"""
Test ComplexMechBenchmark with generated data.
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
    print("ComplexMechBenchmark - Data Loading Test")
    print("="*80)
    
    # Initialize with current directory
    benchmark = ComplexMechBenchmark(
        benchmark_dir=str(current_dir),
        verbose=True, 
        max_samples=5,
        use_ancestor_matrix=False,
    )
    
    # Test loading generated data
    print("\nTesting data loading...")
    try:
        data_filename = "complexmech_2nodes_base_2samples_seed42.pkl"
        data, metadata = benchmark.load_data(data_filename)
        
        print(f"  ✓ Loaded {len(data)} samples")
        print(f"  ✓ Node count: {metadata.get('node_count')}")
        print(f"  ✓ Variant: {metadata.get('variant')}")
        print(f"  ✓ Sample structure: {list(data[0].keys()) if data else 'No data'}")
        
        if data:
            first_sample = data[0]
            print(f"  ✓ X_obs shape: {first_sample['X_obs'].shape}")
            print(f"  ✓ Adjacency matrix shape: {first_sample['adjacency_matrix'].shape}")
        
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ComplexMech data generation and loading test completed successfully!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)