"""Simple script to load and inspect a single sample pickle file."""

import pickle
import numpy as np
import sys
from pathlib import Path

# Get sample index from command line, default to 0
sample_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Get the script directory and construct the path
script_dir = Path(__file__).parent
filepath = script_dir / "data_cache" / "samples" / f"sample_{sample_idx:06d}.pkl"
print(f"Loading: {filepath}")
print("="*80)

with open(filepath, 'rb') as f:
    data = pickle.load(f)

print("\nKeys in data:")
print(data.keys())

print("\n" + "="*80)
print("DATA SHAPES:")
print("="*80)
for key, value in data.items():
    if hasattr(value, 'shape'):
        print(f"{key:20s}: shape = {value.shape}, dtype = {value.dtype}")
    else:
        print(f"{key:20s}: {type(value).__name__} = {value}")

print("\n" + "="*80)
print("SAMPLE DATA:")
print("="*80)

print("\nX_obs (first 3 rows, first 10 columns):")
print(data['X_obs'][:3, :10])

print("\nX_obs (last 3 rows, last 10 columns):")
print(data['X_obs'][-3:, -10:])

print("\nY_obs (first 10 values):")
print(data['Y_obs'][:10].flatten())

print("\nT_obs (first 10 values):")
print(data['T_obs'][:10].flatten())

print("\n" + "="*80)
print("STATISTICS:")
print("="*80)
print(f"X_obs non-zero counts per row (first 10): {[np.count_nonzero(data['X_obs'][i]) for i in range(10)]}")
print(f"X_obs mean: {data['X_obs'].mean():.4f}, std: {data['X_obs'].std():.4f}")
print(f"Y_obs mean: {data['Y_obs'].mean():.4f}, std: {data['Y_obs'].std():.4f}")

if 'ancestor_matrix' in data:
    print(f"\nAncestor matrix shape: {data['ancestor_matrix'].shape}")
    print(f"Ancestor matrix unique values: {np.unique(data['ancestor_matrix'])}")
    print(f"Ancestor matrix:\n{data['ancestor_matrix']}")
