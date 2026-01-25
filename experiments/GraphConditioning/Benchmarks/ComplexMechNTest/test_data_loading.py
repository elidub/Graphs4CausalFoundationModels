#!/usr/bin/env python3
"""Test script to debug ComplexMechNTest data loading issue."""

import pickle
import sys
from pathlib import Path

# Test loading one of the data files that FAILED (ntest500)
data_file = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/GraphConditioning/Benchmarks/ComplexMechNTest/data_cache/data_cache/complexmech_2nodes_path_TY_ntest500_1000samples_seed42.pkl"

print(f"Loading: {data_file}")
print("=" * 80)

with open(data_file, 'rb') as f:
    save_data = pickle.load(f)

print(f"Type of save_data: {type(save_data)}")

if isinstance(save_data, dict):
    print(f"save_data is a dict with keys: {save_data.keys()}")
    print()
    
    data = save_data['data']
    metadata = save_data.get('metadata', {})
elif isinstance(save_data, list):
    print(f"save_data is a list (old format) with length: {len(save_data)}")
    print()
    
    data = save_data
    metadata = {}
else:
    print(f"ERROR: Unexpected save_data type: {type(save_data)}")
    sys.exit(1)

print(f"Type of data: {type(data)}")
print(f"Length of data: {len(data)}")
print()

if len(data) > 0:
    print(f"Type of data[0]: {type(data[0])}")
    
    if isinstance(data[0], (tuple, list)):
        print(f"Length of data[0]: {len(data[0])}")
        print("Data[0] is a tuple/list - showing first item types:")
        for i, item in enumerate(data[0]):
            print(f"  data[0][{i}]: {type(item)} - shape {getattr(item, 'shape', 'N/A')}")
    elif isinstance(data[0], dict):
        print(f"Data[0] is a dict with keys: {data[0].keys()}")
        for key in data[0].keys():
            item = data[0][key]
            print(f"  data[0]['{key}']: {type(item)} - shape {getattr(item, 'shape', 'N/A')}")
    else:
        print(f"Data[0] is an unexpected type: {type(data[0])}")
        print(f"Data[0] content: {data[0]}")

print()
print("Metadata:")
for key, value in metadata.items():
    if key in ['config', 'scm_config', 'dataset_config', 'preprocessing_config']:
        print(f"  {key}: <dict with {len(value)} keys>")
    else:
        print(f"  {key}: {value}")

print()
print("=" * 80)
print("Testing conversion logic...")

# Test the conversion
if len(data) > 0 and isinstance(data[0], (tuple, list)):
    print("Data is in tuple/list format - attempting conversion...")
    try:
        converted_item = {
            'X_obs': data[0][0],
            'T_obs': data[0][1],
            'Y_obs': data[0][2],
            'X_intv': data[0][3],
            'T_intv': data[0][4],
            'Y_intv': data[0][5],
            'adjacency_matrix': data[0][6] if len(data[0]) >= 7 else None,
        }
        print("Conversion successful!")
        print(f"Converted dict keys: {converted_item.keys()}")
        
        # Try accessing with string index
        try:
            X_obs = converted_item['X_obs']
            print(f"Successfully accessed converted_item['X_obs']: shape {X_obs.shape}")
        except Exception as e:
            print(f"ERROR accessing converted_item['X_obs']: {e}")
            
    except Exception as e:
        print(f"ERROR during conversion: {e}")
        import traceback
        traceback.print_exc()
elif len(data) > 0 and isinstance(data[0], dict):
    print("Data is already in dict format")
    try:
        X_obs = data[0]['X_obs']
        print(f"Successfully accessed data[0]['X_obs']: shape {X_obs.shape}")
    except Exception as e:
        print(f"ERROR accessing data[0]['X_obs']: {e}")
        import traceback
        traceback.print_exc()
