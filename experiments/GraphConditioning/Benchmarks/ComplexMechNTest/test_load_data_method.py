#!/usr/bin/env python3
"""Test the fixed load_data method."""

import sys
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting')

from experiments.GraphConditioning.Benchmarks.ComplexMechNTest.ComplexMechBenchmarkNTest import ComplexMechBenchmarkNTest

# Initialize benchmark
benchmark = ComplexMechBenchmarkNTest(
    benchmark_dir="/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/GraphConditioning/Benchmarks/ComplexMechNTest",
    cache_dir="/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/GraphConditioning/Benchmarks/ComplexMechNTest/data_cache/data_cache",
    verbose=True
)

print("Testing load_data with ntest100 (new format)...")
data1, metadata1 = benchmark.load_data("complexmech_2nodes_path_TY_ntest100_1000samples_seed42.pkl")
print(f"Success! Loaded {len(data1)} samples")
print(f"First sample keys: {data1[0].keys()}")
print()

print("Testing load_data with ntest500 (old format)...")
data2, metadata2 = benchmark.load_data("complexmech_2nodes_path_TY_ntest500_1000samples_seed42.pkl")
print(f"Success! Loaded {len(data2)} samples")
print(f"First sample keys: {data2[0].keys()}")
print()

print("Testing data access...")
try:
    X_obs = data2[0]['X_obs']
    print(f"Successfully accessed data2[0]['X_obs']: shape {X_obs.shape}")
    print("\nAll tests passed!")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
