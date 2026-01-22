#!/usr/bin/env python3
"""
Debug script to trace graph matrix processing from dataset to model.

This will show exactly what happens to the adjacency matrix at each step.
"""

import sys
import os
import numpy as np
import dill
import torch

# Add paths - CausalPriorFitting must be first for src imports to work
sys.path.insert(0, '/fast/arikreuter/DoPFN_v2/CausalPriorFitting')

# Change working directory so relative imports work
os.chdir('/fast/arikreuter/DoPFN_v2/CausalPriorFitting')

from src.models.PreprocessingGraphConditionedPFN import PreprocessingGraphConditionedPFN

# Import graph_utils directly to avoid utils.py / utils/ conflict
import importlib.util
spec = importlib.util.spec_from_file_location("graph_utils", "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/src/utils/graph_utils.py")
graph_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_utils)
adjacency_to_ancestor_matrix = graph_utils.adjacency_to_ancestor_matrix

# Load dataset - use absolute path
dataset_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval2/PlantsEval/plant_data/normal_with_graph.pkl"
print(f"Loading dataset from: {dataset_path}")
with open(dataset_path, 'rb') as f:
    benchmark = dill.load(f)

# Check format
print(f"Type: {type(benchmark)}")
if isinstance(benchmark, dict):
    print(f"Keys: {benchmark.keys()}")
    if 'realizations' in benchmark:
        realizations = benchmark['realizations']
        print(f"Realizations type: {type(realizations)}")
        if isinstance(realizations, list):
            realization = realizations[0]
        elif isinstance(realizations, dict):
            first_key = list(realizations.keys())[0]
            realization = realizations[first_key]
        else:
            realization = realizations
    else:
        first_key = list(benchmark.keys())[0]
        realization = benchmark[first_key]
elif hasattr(benchmark, 'realizations'):
    realization = benchmark.realizations[0]
else:
    print(f"Unknown format. Attributes: {dir(benchmark)}")
    sys.exit(1)

# Check realization format
print(f"Realization type: {type(realization)}")
if isinstance(realization, dict):
    print(f"Realization keys: {realization.keys()}")
    if 'adjacency_matrix' in realization:
        dataset_adjacency = realization['adjacency_matrix']
    else:
        print("No adjacency_matrix in realization!")
        sys.exit(1)
elif hasattr(realization, 'adjacency_matrix'):
    dataset_adjacency = realization.adjacency_matrix
else:
    print(f"Realization attributes: {dir(realization)}")
    sys.exit(1)
    
print(f"\n{'='*80}")
print("STEP 1: Raw dataset adjacency matrix")
print('='*80)

# dataset_adjacency already set above from the dict/object
print(f"Shape: {dataset_adjacency.shape}")
print(f"Unique values: {np.unique(dataset_adjacency)}")
print(f"Number of 1s (edges): {(dataset_adjacency == 1).sum()}")
print(f"Number of 0s (unknown): {(dataset_adjacency == 0).sum()}")
print(f"Number of -1s (no edge): {(dataset_adjacency == -1).sum()}")

# Variable order in dataset
print("\nDataset variable order (8 variables):")
print("  0-5: Features (X, Y, Bed_Position, Red/Blue/Far_Red Light)")
print("  6: Treatment (Light_Intensity_umols)")
print("  7: Outcome (Biomass_g)")

print(f"\nDataset adjacency (treatment row idx=6):")
print(f"  From treatment to features [0:6]: {dataset_adjacency[6, 0:6]}")
print(f"  From treatment to treatment [6]: {dataset_adjacency[6, 6]}")
print(f"  From treatment to outcome [7]: {dataset_adjacency[6, 7]}")

print(f"\nDataset adjacency (outcome row idx=7):")
print(f"  From outcome to features [0:6]: {dataset_adjacency[7, 0:6]}")
print(f"  From outcome to treatment [6]: {dataset_adjacency[7, 6]}")
print(f"  From outcome to outcome [7]: {dataset_adjacency[7, 7]}")

# Also show incoming edges to outcome
print(f"\nIncoming edges to outcome (column 7):")
for i in range(8):
    if dataset_adjacency[i, 7] == 1:
        var_names = ["X", "Y", "Bed_Pos", "Red", "Blue", "FarRed", "Treatment", "Outcome"]
        print(f"  {var_names[i]} -> Outcome")

print(f"\n{'='*80}")
print("STEP 2: Reorder to model format (run_dofm2.py logic)")
print('='*80)

# Same logic as run_dofm2.py
n_dataset_vars = 8
model_n_features = 50  # Model expects 50 features
n_real_features = 6

# Create mapping: dataset idx -> model idx
# Dataset: [features(0-5), treatment(6), outcome(7)]
# Model: [treatment(0), outcome(1), features(2-7)]
dataset_to_model = np.zeros(n_dataset_vars, dtype=int)
dataset_to_model[6] = 0  # treatment: dataset idx 6 -> model idx 0
dataset_to_model[7] = 1  # outcome: dataset idx 7 -> model idx 1
for i in range(6):  # features: dataset idx 0-5 -> model idx 2-7
    dataset_to_model[i] = 2 + i

print(f"Index mapping (dataset -> model):")
for i in range(8):
    var_names = ["X", "Y", "Bed_Pos", "Red", "Blue", "FarRed", "Treatment", "Outcome"]
    print(f"  {var_names[i]}: dataset[{i}] -> model[{dataset_to_model[i]}]")

# Create model adjacency matrix
model_adjacency = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)

# Copy edges with reordering
for i in range(n_dataset_vars):
    for j in range(n_dataset_vars):
        model_i = dataset_to_model[i]
        model_j = dataset_to_model[j]
        model_adjacency[model_i, model_j] = dataset_adjacency[i, j]

# Pad remaining with 0 (unknown) - UPDATED to match run_dofm2.py
for i in range(n_real_features, model_n_features):
    feat_idx_pad = 2 + i
    model_adjacency[feat_idx_pad, :] = 0.0  # Unknown instead of -1
    model_adjacency[:, feat_idx_pad] = 0.0  # Unknown instead of -1

print(f"\nModel adjacency shape: {model_adjacency.shape}")
print(f"Unique values: {np.unique(model_adjacency)}")
print(f"Number of 1s (edges): {(model_adjacency == 1).sum()}")
print(f"Number of 0s (unknown): {(model_adjacency == 0).sum()}")
print(f"Number of -1s (no edge): {(model_adjacency == -1).sum()}")

print(f"\nModel adjacency (treatment row idx=0):")
print(f"  From treatment to treatment [0]: {model_adjacency[0, 0]}")
print(f"  From treatment to outcome [1]: {model_adjacency[0, 1]}")
print(f"  From treatment to features [2:8]: {model_adjacency[0, 2:8]}")

print(f"\nModel adjacency (outcome row idx=1):")
print(f"  From outcome to treatment [0]: {model_adjacency[1, 0]}")
print(f"  From outcome to outcome [1]: {model_adjacency[1, 1]}")
print(f"  From outcome to features [2:8]: {model_adjacency[1, 2:8]}")

# Show incoming edges to outcome
print(f"\nIncoming edges to outcome (column 1) in model format:")
for i in range(10):  # Check first 10 rows
    if model_adjacency[i, 1] == 1:
        if i == 0:
            print(f"  Treatment -> Outcome")
        elif i == 1:
            print(f"  Outcome -> Outcome (should not exist!)")
        else:
            print(f"  Feature[{i-2}] -> Outcome")

print(f"\n{'='*80}")
print("STEP 3: Convert adjacency to ancestor matrix (PreprocessingGraphConditionedPFN logic)")
print('='*80)

# Same logic as _convert_adjacency_to_ancestor_if_needed
adj_torch = torch.from_numpy(model_adjacency)

# Handle three-state format
unknown_mask = (adj_torch == 0.0)
no_edge_mask = (adj_torch == -1.0)

print(f"Unknown entries (==0): {unknown_mask.sum().item()}")
print(f"No-edge entries (==-1): {no_edge_mask.sum().item()}")
print(f"Edge entries (==1): {(adj_torch == 1.0).sum().item()}")

# Convert to binary
adj_binary = (adj_torch == 1.0).float()
print(f"\nBinary adjacency edges: {adj_binary.sum().item()}")

# Compute ancestor matrix
ancestor_torch = adjacency_to_ancestor_matrix(adj_binary, assume_dag=True, remove_diagonal=True)
print(f"Ancestor edges (after transitive closure): {ancestor_torch.sum().item()}")

# Convert back to three-state
ancestor_three_state = 2.0 * ancestor_torch - 1.0
print(f"\nAfter three-state conversion:")
print(f"  Values == 1: {(ancestor_three_state == 1.0).sum().item()}")
print(f"  Values == -1: {(ancestor_three_state == -1.0).sum().item()}")
print(f"  Values == 0: {(ancestor_three_state == 0.0).sum().item()}")

# Restore unknowns
ancestor_three_state = torch.where(unknown_mask, torch.zeros_like(ancestor_three_state), ancestor_three_state)
print(f"\nAfter restoring unknowns:")
print(f"  Values == 1: {(ancestor_three_state == 1.0).sum().item()}")
print(f"  Values == -1: {(ancestor_three_state == -1.0).sum().item()}")
print(f"  Values == 0: {(ancestor_three_state == 0.0).sum().item()}")

final_ancestor = ancestor_three_state.numpy().astype(np.float32)

print(f"\n{'='*80}")
print("STEP 4: Compare with ALL_UNKNOWN mode")
print('='*80)

# All unknown = all zeros
all_unknown_adj = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
adj_torch_unknown = torch.from_numpy(all_unknown_adj)

unknown_mask_u = (adj_torch_unknown == 0.0)
adj_binary_u = (adj_torch_unknown == 1.0).float()
ancestor_torch_u = adjacency_to_ancestor_matrix(adj_binary_u, assume_dag=True, remove_diagonal=True)
ancestor_three_state_u = 2.0 * ancestor_torch_u - 1.0
ancestor_three_state_u = torch.where(unknown_mask_u, torch.zeros_like(ancestor_three_state_u), ancestor_three_state_u)

print(f"All-unknown mode final ancestor matrix:")
print(f"  Values == 1: {(ancestor_three_state_u == 1.0).sum().item()}")
print(f"  Values == -1: {(ancestor_three_state_u == -1.0).sum().item()}")
print(f"  Values == 0: {(ancestor_three_state_u == 0.0).sum().item()}")

print(f"\n{'='*80}")
print("COMPARISON: What model sees")
print('='*80)
print(f"\nFull Graph mode:")
print(f"  Real structure with {(final_ancestor == 1.0).sum()} ancestor relationships")
print(f"  {(final_ancestor == -1.0).sum()} definitely-no-ancestor entries")
print(f"  {(final_ancestor == 0.0).sum()} unknown entries")

print(f"\nAll Unknown mode:")
print(f"  {(ancestor_three_state_u == 1.0).sum().item()} ancestor relationships")
print(f"  {(ancestor_three_state_u == -1.0).sum().item()} definitely-no-ancestor entries") 
print(f"  {(ancestor_three_state_u == 0.0).sum().item()} unknown entries")

print(f"\n{'='*80}")
print("HYPOTHESIS CHECK: Training distribution vs inference")
print('='*80)
print("""
During training with hide_fraction_matrix ~ Uniform(0, 1):
- hide_frac = 0.0 (0% of time): Model sees full graph with NO unknowns (0s)
- hide_frac = 0.5 (median): Model sees ~50% unknowns
- hide_frac = 1.0 (0% of time, edge case): Model sees ALL unknowns

Key insight: The hide_fraction is UNIFORMLY distributed!
- Most training samples have MIXED known/unknown entries
- Fully known graphs (hide_frac=0) are rare (measure zero on continuous distribution)
- Fully unknown graphs (hide_frac=1) are also rare

At inference time:
- FULL_GRAPH mode: 0 unknown entries (rare in training!)
- ALL_UNKNOWN mode: All unknown entries (also rare, but model may have learned to be robust)

This explains why ALL_UNKNOWN performs BETTER:
- Model learned to be robust to unknowns
- Fully known graphs may over-constrain the model
""")
