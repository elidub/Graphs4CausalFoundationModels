#!/usr/bin/env python3
"""
Compare Benchmark and Training Data

This script compares the data shapes and processing between:
1. Training data from dataloader (MakePurelyObservationalDataset)
2. Benchmark data from OpenML (SimpleOpenMLLoader + run_benchmark.py processing)

It helps understand where zero-padding occurs and how shapes differ between the two data sources.
"""

import sys
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from types import SimpleNamespace

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Project imports (after path modification)
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
from benchmarking.load_openml_benchmark import SimpleOpenMLLoader, DEFAULT_TABULAR_NUM_REG_TASKS
from training.training_utils import load_yaml_config, extract_config_values


def analyze_dataloader_batch(dataloader, n_batches=3):
    """Analyze training dataloader batch structure and shapes."""
    print("=" * 60)
    print("ANALYZING TRAINING DATALOADER")
    print("=" * 60)
    
    batch_info = []
    
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
            
        if len(batch) == 4:
            X_train, y_train, X_test, y_test = batch
        else:
            print(f"[WARNING] Unexpected batch format: {len(batch)} elements")
            continue
            
        batch_data = {
            'batch_idx': i,
            'X_train_shape': tuple(X_train.shape),
            'y_train_shape': tuple(y_train.shape),
            'X_test_shape': tuple(X_test.shape),
            'y_test_shape': tuple(y_test.shape),
            'X_train_dtype': str(X_train.dtype),
            'y_train_dtype': str(y_train.dtype),
            'X_test_dtype': str(X_test.dtype),
            'y_test_dtype': str(y_test.dtype),
        }
        
        # Extract batch size and individual dataset dimensions
        batch_size = X_train.shape[0]
        n_train_samples = X_train.shape[1] if len(X_train.shape) > 1 else 1
        n_features = X_train.shape[2] if len(X_train.shape) > 2 else X_train.shape[1]
        n_test_samples = X_test.shape[1] if len(X_test.shape) > 1 else 1
        
        batch_data.update({
            'batch_size': batch_size,
            'n_train_samples': n_train_samples,
            'n_test_samples': n_test_samples,
            'n_features': n_features,
        })
        
        # Analyze data ranges and zero-padding
        # Convert tensors to numpy arrays for analysis
        X_train_np = X_train.detach().cpu().numpy() if hasattr(X_train, 'detach') else np.array(X_train)
        X_test_np = X_test.detach().cpu().numpy() if hasattr(X_test, 'detach') else np.array(X_test)
        y_train_np = y_train.detach().cpu().numpy() if hasattr(y_train, 'detach') else np.array(y_train)
        y_test_np = y_test.detach().cpu().numpy() if hasattr(y_test, 'detach') else np.array(y_test)
        
        # IMPORTANT: Each element in the batch is a complete dataset
        # X_train shape: [n_datasets_in_batch, n_train_samples, n_features]
        # X_test shape: [n_datasets_in_batch, n_test_samples, n_features]
        
        # Analyze each dataset in the batch separately
        dataset_analyses = []
        for dataset_idx in range(batch_size):
            # Extract single dataset
            X_train_single = X_train_np[dataset_idx]  # [n_train_samples, n_features]
            X_test_single = X_test_np[dataset_idx]    # [n_test_samples, n_features]
            y_train_single = y_train_np[dataset_idx]  # [n_train_samples]
            y_test_single = y_test_np[dataset_idx]    # [n_test_samples]
            
            # Check for zero columns (potential padding) in this dataset
            zero_cols_train = np.all(X_train_single == 0, axis=0)
            zero_cols_test = np.all(X_test_single == 0, axis=0)
            
            # Add per-feature statistics for this dataset
            feature_statistics = {}
            for feat_idx in range(n_features):
                train_feature = X_train_single[:, feat_idx]
                test_feature = X_test_single[:, feat_idx]
                
                feature_statistics[feat_idx] = {
                    'train_mean': float(np.mean(train_feature)),
                    'train_std': float(np.std(train_feature)),
                    'train_min': float(np.min(train_feature)),
                    'train_max': float(np.max(train_feature)),
                    'test_mean': float(np.mean(test_feature)),
                    'test_std': float(np.std(test_feature)),
                    'test_min': float(np.min(test_feature)),
                    'test_max': float(np.max(test_feature)),
                    'is_zero_padded': bool(zero_cols_train[feat_idx] or zero_cols_test[feat_idx]),
                    'train_n_zeros': int(np.sum(train_feature == 0)),
                    'test_n_zeros': int(np.sum(test_feature == 0))
                }
            
            dataset_analysis = {
                'dataset_idx_in_batch': dataset_idx,
                'X_train_shape': X_train_single.shape,
                'X_test_shape': X_test_single.shape,
                'y_train_shape': y_train_single.shape,
                'y_test_shape': y_test_single.shape,
                'zero_cols_train': int(np.sum(zero_cols_train)),
                'zero_cols_test': int(np.sum(zero_cols_test)),
                'zero_col_indices_train': np.where(zero_cols_train)[0].tolist(),
                'zero_col_indices_test': np.where(zero_cols_test)[0].tolist(),
                'feature_statistics': feature_statistics,  # Add per-feature stats
                'X_train_stats': {
                    'mean': float(np.mean(X_train_single)),
                    'std': float(np.std(X_train_single)),
                    'min': float(np.min(X_train_single)),
                    'max': float(np.max(X_train_single)),
                    'n_zeros': int(np.sum(X_train_single == 0)),
                    'total_elements': int(X_train_single.size)
                },
                'X_test_stats': {
                    'mean': float(np.mean(X_test_single)),
                    'std': float(np.std(X_test_single)),
                    'min': float(np.min(X_test_single)),
                    'max': float(np.max(X_test_single)),
                    'n_zeros': int(np.sum(X_test_single == 0)),
                    'total_elements': int(X_test_single.size)
                },
                'y_train_stats': {
                    'mean': float(np.mean(y_train_single)),
                    'std': float(np.std(y_train_single)),
                    'min': float(np.min(y_train_single)),
                    'max': float(np.max(y_train_single))
                },
                'y_test_stats': {
                    'mean': float(np.mean(y_test_single)),
                    'std': float(np.std(y_test_single)),
                    'min': float(np.min(y_test_single)),
                    'max': float(np.max(y_test_single))
                }
            }
            dataset_analyses.append(dataset_analysis)
        
        # For backward compatibility, also analyze flattened batch data
        X_train_flat = X_train_np.reshape(-1, n_features)
        X_test_flat = X_test_np.reshape(-1, n_features)
        
        # Check for zero columns (potential padding) across entire batch
        zero_cols_train = np.all(X_train_flat == 0, axis=0)
        zero_cols_test = np.all(X_test_flat == 0, axis=0)
        
        batch_data.update({
            'zero_cols_train': np.sum(zero_cols_train),
            'zero_cols_test': np.sum(zero_cols_test),
            'zero_col_indices_train': np.where(zero_cols_train)[0].tolist() if np.any(zero_cols_train) else [],
            'zero_col_indices_test': np.where(zero_cols_test)[0].tolist() if np.any(zero_cols_test) else [],
            'individual_datasets': dataset_analyses,  # Add individual dataset analysis
        })
        
        # Data statistics
        batch_data.update({
            'X_train_mean': float(np.mean(X_train_flat)),
            'X_train_std': float(np.std(X_train_flat)),
            'X_train_min': float(np.min(X_train_flat)),
            'X_train_max': float(np.max(X_train_flat)),
            'X_test_mean': float(np.mean(X_test_flat)),
            'X_test_std': float(np.std(X_test_flat)),
            'X_test_min': float(np.min(X_test_flat)),
            'X_test_max': float(np.max(X_test_flat)),
        })
        
        batch_info.append(batch_data)
        
        print(f"\nBatch {i}:")
        print(f"  X_train: {X_train.shape} ({X_train.dtype})")
        print(f"  y_train: {y_train.shape} ({y_train.dtype})")
        print(f"  X_test:  {X_test.shape} ({X_test.dtype})")
        print(f"  y_test:  {y_test.shape} ({y_test.dtype})")
        print(f"  Batch size: {batch_size}, Features: {n_features}")
        print(f"  Train samples per dataset: {n_train_samples}")
        print(f"  Test samples per dataset: {n_test_samples}")
        print(f"  Zero columns (train/test): {np.sum(zero_cols_train)}/{np.sum(zero_cols_test)}")
        if np.any(zero_cols_train) or np.any(zero_cols_test):
            print(f"  Zero column indices (train): {np.where(zero_cols_train)[0].tolist()}")
            print(f"  Zero column indices (test): {np.where(zero_cols_test)[0].tolist()}")
        print(f"  Data range (train): [{np.min(X_train_flat):.3f}, {np.max(X_train_flat):.3f}]")
        print(f"  Data range (test): [{np.min(X_test_flat):.3f}, {np.max(X_test_flat):.3f}]")
    
    # Compute global statistics across all datasets and batches
    print("\n" + "=" * 60)
    print("GLOBAL TRAINING DATA STATISTICS")
    print("=" * 60)
    
    # For now, compute statistics from what we have
    total_datasets = sum(len(batch['individual_datasets']) for batch in batch_info)
    
    # Aggregate feature statistics
    feature_level_stats = {}
    for batch in batch_info:
        n_features = batch['n_features']
        for feat_idx in range(n_features):
            if feat_idx not in feature_level_stats:
                feature_level_stats[feat_idx] = {
                    'train_means': [],
                    'test_means': [],
                    'train_stds': [],
                    'test_stds': [],
                    'zero_padded_count': 0
                }
            
            for dataset in batch['individual_datasets']:
                if feat_idx in dataset['feature_statistics']:
                    feat_stats = dataset['feature_statistics'][feat_idx]
                    feature_level_stats[feat_idx]['train_means'].append(feat_stats['train_mean'])
                    feature_level_stats[feat_idx]['test_means'].append(feat_stats['test_mean'])
                    feature_level_stats[feat_idx]['train_stds'].append(feat_stats['train_std'])
                    feature_level_stats[feat_idx]['test_stds'].append(feat_stats['test_std'])
                    if feat_stats['is_zero_padded']:
                        feature_level_stats[feat_idx]['zero_padded_count'] += 1
    
    # Compute aggregated feature statistics
    aggregated_feature_stats = {}
    zero_padded_features = []
    active_features = []
    
    for feat_idx, stats in feature_level_stats.items():
        if stats['train_means']:
            aggregated_feature_stats[feat_idx] = {
                'train_mean_avg': float(np.mean(stats['train_means'])),
                'train_mean_std': float(np.std(stats['train_means'])),
                'test_mean_avg': float(np.mean(stats['test_means'])),
                'test_mean_std': float(np.std(stats['test_means'])),
                'zero_padded_percentage': float(stats['zero_padded_count'] / total_datasets * 100)
            }
            
            if aggregated_feature_stats[feat_idx]['zero_padded_percentage'] > 50:
                zero_padded_features.append(feat_idx)
            else:
                active_features.append(feat_idx)
    
    print(f"Total datasets analyzed: {total_datasets}")
    print(f"Active features (non-zero in >50% datasets): {len(active_features)} {active_features[:10]}")
    print(f"Zero-padded features (zero in >50% datasets): {len(zero_padded_features)} {zero_padded_features[:10]}")
    
    if active_features:
        print(f"\nActive feature statistics (features {active_features[:5]}):")
        for feat_idx in active_features[:5]:
            stats = aggregated_feature_stats[feat_idx]
            print(f"  Feature {feat_idx}: train_mean={stats['train_mean_avg']:.4f}±{stats['train_mean_std']:.4f}, test_mean={stats['test_mean_avg']:.4f}±{stats['test_mean_std']:.4f}")
    
    # Add global statistics to return data
    global_training_stats = {
        'total_datasets_analyzed': total_datasets,
        'n_features_total': len(feature_level_stats),
        'n_active_features': len(active_features),
        'n_zero_padded_features': len(zero_padded_features),
        'active_feature_indices': active_features,
        'zero_padded_feature_indices': zero_padded_features,
        'feature_level_statistics': aggregated_feature_stats
    }
    
    return {
        'batches': batch_info,
        'global_statistics': global_training_stats,
        'n_batches_analyzed': len(batch_info)
    }


def analyze_benchmark_data(loader, task_ids, benchmark_args):
    """Analyze benchmark data shapes and processing."""
    print("=" * 60)
    print("ANALYZING BENCHMARK DATA")
    print("=" * 60)
    
    # Create benchmark data using the same logic as run_benchmark.py
    subsample_mode = bool(getattr(benchmark_args, "n_train", 0) and 
                         getattr(benchmark_args, "n_test", 0) and 
                         getattr(benchmark_args, "n_features", 0))
    
    if subsample_mode:
        print(f"Using subsampled mode:")
        print(f"  n_features: {benchmark_args.n_features}")
        print(f"  n_train: {benchmark_args.n_train}")
        print(f"  n_test: {benchmark_args.n_test}")
        print(f"  max_n_features: {getattr(benchmark_args, 'max_n_features', 'None')}")
        
        data_map = loader.create_subsampled_datasets(
            task_ids,
            n_features=int(benchmark_args.n_features),
            n_train=int(benchmark_args.n_train),
            n_test=int(benchmark_args.n_test),
            prefer_numeric=getattr(benchmark_args, "prefer_numeric", True),
            save=False,
        )
    else:
        print("Using normal mode (full datasets)")
        data_map = loader.load_tasks(task_ids)
    
    benchmark_info = []
    
    for tid in task_ids:
        if tid not in data_map or not data_map.get(tid):
            print(f"Skipping task {tid}: no processed data")
            continue
            
        d = data_map[tid]
        
        # Convert to numpy arrays (same as run_benchmark.py)
        X_train = d["X_train"]
        X_test = d["X_test"]
        y_train = d["y_train"]
        y_test = d["y_test"]

        if hasattr(X_train, "to_numpy"):
            X_train = X_train.to_numpy()
        if hasattr(X_test, "to_numpy"):
            X_test = X_test.to_numpy()
        if hasattr(y_train, "to_numpy"):
            y_train = y_train.to_numpy()
        if hasattr(y_test, "to_numpy"):
            y_test = y_test.to_numpy()

        # Ensure y shapes: y as 1d
        y_train = np.asarray(y_train).reshape(-1)
        y_test = np.asarray(y_test).reshape(-1)
        
        print(f"\nTask {tid} (before feature processing):")
        print(f"  X_train: {X_train.shape} ({X_train.dtype})")
        print(f"  y_train: {y_train.shape} ({y_train.dtype})")
        print(f"  X_test:  {X_test.shape} ({X_test.dtype})")
        print(f"  y_test:  {y_test.shape} ({y_test.dtype})")
        
        # Apply the same feature processing as run_benchmark.py
        requested_n = int(getattr(benchmark_args, "n_features", 0) or 0)
        max_n = int(getattr(benchmark_args, "max_n_features", 0) or 0)

        original_n_features = X_train.shape[1]
        
        # Feature selection step
        if requested_n and requested_n > 0:
            if X_train.shape[1] >= requested_n:
                X_train = X_train[:, :requested_n]
                X_test = X_test[:, :requested_n]
                print(f"  Feature selection: {original_n_features} -> {X_train.shape[1]}")
            else:
                print(f"  Requested n_features={requested_n} > available {X_train.shape[1]}; using available")

        # Padding/truncation step
        applied_padding = False
        applied_truncation = False
        if max_n and max_n > 0:
            cur_n = X_train.shape[1]
            if cur_n > max_n:
                # truncate to max_n
                X_train = X_train[:, :max_n]
                X_test = X_test[:, :max_n]
                applied_truncation = True
                print(f"  Truncation: {cur_n} -> {max_n} features")
            elif cur_n < max_n:
                # pad with zeros on the right
                pad_train = np.zeros((X_train.shape[0], max_n - cur_n), dtype=X_train.dtype)
                pad_test = np.zeros((X_test.shape[0], max_n - cur_n), dtype=X_test.dtype)
                X_train = np.concatenate([X_train, pad_train], axis=1)
                X_test = np.concatenate([X_test, pad_test], axis=1)
                applied_padding = True
                print(f"  Zero-padding: {cur_n} -> {max_n} features")
        
        print(f"  Final shapes after processing:")
        print(f"    X_train: {X_train.shape}")
        print(f"    X_test:  {X_test.shape}")
        
        # Check for zero columns after processing
        zero_cols_train = np.all(X_train == 0, axis=0)
        zero_cols_test = np.all(X_test == 0, axis=0)
        
        # Analyze padding pattern
        padding_start_train = None
        padding_start_test = None
        if applied_padding:
            # Find where padding starts (first zero column from the right)
            for i in range(X_train.shape[1] - 1, -1, -1):
                if not zero_cols_train[i]:
                    padding_start_train = i + 1
                    break
            for i in range(X_test.shape[1] - 1, -1, -1):
                if not zero_cols_test[i]:
                    padding_start_test = i + 1
                    break
        
        task_data = {
            'task_id': tid,
            'dataset_id': d.get('dataset_id', 'unknown'),
            'original_shape': d.get('data_shape', 'unknown'),
            'original_n_features': original_n_features,
            'final_X_train_shape': tuple(X_train.shape),
            'final_y_train_shape': tuple(y_train.shape),
            'final_X_test_shape': tuple(X_test.shape),
            'final_y_test_shape': tuple(y_test.shape),
            'applied_feature_selection': requested_n > 0 and original_n_features > requested_n,
            'applied_padding': applied_padding,
            'applied_truncation': applied_truncation,
            'zero_cols_train': int(np.sum(zero_cols_train)),
            'zero_cols_test': int(np.sum(zero_cols_test)),
            'padding_start_train': padding_start_train,
            'padding_start_test': padding_start_test,
            'X_train_dtype': str(X_train.dtype),
            'y_train_dtype': str(y_train.dtype),
            'X_test_dtype': str(X_test.dtype),
            'y_test_dtype': str(y_test.dtype),
            'X_train_mean': float(np.mean(X_train)),
            'X_train_std': float(np.std(X_train)),
            'X_train_min': float(np.min(X_train)),
            'X_train_max': float(np.max(X_train)),
            'X_test_mean': float(np.mean(X_test)),
            'X_test_std': float(np.std(X_test)),
            'X_test_min': float(np.min(X_test)),
            'X_test_max': float(np.max(X_test)),
        }
        
        benchmark_info.append(task_data)
        
        if applied_padding:
            print(f"  Zero-padding details:")
            print(f"    Padding starts at column: {padding_start_train} (train), {padding_start_test} (test)")
            print(f"    Zero columns: {np.sum(zero_cols_train)} (train), {np.sum(zero_cols_test)} (test)")
        
        print(f"  Data range (train): [{np.min(X_train):.3f}, {np.max(X_train):.3f}]")
        print(f"  Data range (test): [{np.min(X_test):.3f}, {np.max(X_test):.3f}]")
    
    # Compute global statistics across all benchmark datasets
    print("\n" + "=" * 60)
    print("GLOBAL BENCHMARK DATA STATISTICS")
    print("=" * 60)
    
    if benchmark_info:
        total_datasets = len(benchmark_info)
        
        # Collect feature and shape statistics
        final_shapes = [info['final_X_train_shape'] for info in benchmark_info]
        max_features = max(shape[1] for shape in final_shapes)
        min_features = min(shape[1] for shape in final_shapes)
        avg_features = np.mean([shape[1] for shape in final_shapes])
        
        # Collect padding and processing statistics
        datasets_with_padding = sum(1 for info in benchmark_info if info['applied_padding'])
        datasets_with_truncation = sum(1 for info in benchmark_info if info['applied_truncation'])
        datasets_with_feature_selection = sum(1 for info in benchmark_info if info['applied_feature_selection'])
        
        # Global data value statistics
        all_train_means = [info['X_train_mean'] for info in benchmark_info]
        all_test_means = [info['X_test_mean'] for info in benchmark_info]
        all_train_stds = [info['X_train_std'] for info in benchmark_info]
        all_test_stds = [info['X_test_std'] for info in benchmark_info]
        
        global_train_range = [
            min(info['X_train_min'] for info in benchmark_info),
            max(info['X_train_max'] for info in benchmark_info)
        ]
        global_test_range = [
            min(info['X_test_min'] for info in benchmark_info),
            max(info['X_test_max'] for info in benchmark_info)
        ]
        
        # Feature-level analysis (approximation)
        feature_analysis = {}
        for feat_idx in range(max_features):
            datasets_with_feature = 0
            zero_padded_count = 0
            
            for info in benchmark_info:
                n_features = info['final_X_train_shape'][1]
                if feat_idx < n_features:
                    datasets_with_feature += 1
                    # Check if this feature is in the zero-padded region
                    if (info['applied_padding'] and 
                        info['padding_start_train'] is not None and 
                        feat_idx >= info['padding_start_train']):
                        zero_padded_count += 1
            
            if datasets_with_feature > 0:
                zero_padded_percentage = (zero_padded_count / datasets_with_feature) * 100
                feature_analysis[feat_idx] = {
                    'datasets_with_feature': datasets_with_feature,
                    'zero_padded_count': zero_padded_count,
                    'zero_padded_percentage': zero_padded_percentage
                }
        
        # Classify features as active or zero-padded
        active_features = [idx for idx, stats in feature_analysis.items() 
                          if stats['zero_padded_percentage'] <= 50]
        zero_padded_features = [idx for idx, stats in feature_analysis.items() 
                               if stats['zero_padded_percentage'] > 50]
        
        print(f"Total datasets analyzed: {total_datasets}")
        print(f"Feature statistics:")
        print(f"  Feature count range: {min_features} - {max_features}")
        print(f"  Average features: {avg_features:.1f}")
        print(f"  Active features (non-zero in >50% datasets): {len(active_features)} {active_features[:10]}")
        print(f"  Zero-padded features (zero in >50% datasets): {len(zero_padded_features)} {zero_padded_features[:10]}")
        
        print(f"\nProcessing statistics:")
        print(f"  Datasets with padding: {datasets_with_padding}/{total_datasets} ({datasets_with_padding/total_datasets*100:.1f}%)")
        print(f"  Datasets with truncation: {datasets_with_truncation}/{total_datasets} ({datasets_with_truncation/total_datasets*100:.1f}%)")
        print(f"  Datasets with feature selection: {datasets_with_feature_selection}/{total_datasets} ({datasets_with_feature_selection/total_datasets*100:.1f}%)")
        
        print(f"\nGlobal data statistics:")
        print(f"  Train means: avg={np.mean(all_train_means):.4f}, std={np.std(all_train_means):.4f}")
        print(f"  Test means: avg={np.mean(all_test_means):.4f}, std={np.std(all_test_means):.4f}")
        print(f"  Train stds: avg={np.mean(all_train_stds):.4f}, std={np.std(all_train_stds):.4f}")
        print(f"  Test stds: avg={np.mean(all_test_stds):.4f}, std={np.std(all_test_stds):.4f}")
        print(f"  Global train range: [{global_train_range[0]:.3f}, {global_train_range[1]:.3f}]")
        print(f"  Global test range: [{global_test_range[0]:.3f}, {global_test_range[1]:.3f}]")
        
        # Create comprehensive global statistics object
        global_benchmark_stats = {
            'total_datasets_analyzed': total_datasets,
            'feature_statistics': {
                'max_features': max_features,
                'min_features': min_features,
                'avg_features': float(avg_features),
                'n_active_features': len(active_features),
                'n_zero_padded_features': len(zero_padded_features),
                'active_feature_indices': active_features,
                'zero_padded_feature_indices': zero_padded_features,
                'feature_level_analysis': feature_analysis
            },
            'processing_statistics': {
                'datasets_with_padding': datasets_with_padding,
                'datasets_with_truncation': datasets_with_truncation,
                'datasets_with_feature_selection': datasets_with_feature_selection,
                'padding_percentage': float(datasets_with_padding/total_datasets*100),
                'truncation_percentage': float(datasets_with_truncation/total_datasets*100),
                'feature_selection_percentage': float(datasets_with_feature_selection/total_datasets*100)
            },
            'global_data_statistics': {
                'train_stats': {
                    'mean_avg': float(np.mean(all_train_means)),
                    'mean_std': float(np.std(all_train_means)),
                    'std_avg': float(np.mean(all_train_stds)),
                    'std_std': float(np.std(all_train_stds)),
                    'global_min': float(global_train_range[0]),
                    'global_max': float(global_train_range[1])
                },
                'test_stats': {
                    'mean_avg': float(np.mean(all_test_means)),
                    'mean_std': float(np.std(all_test_means)),
                    'std_avg': float(np.mean(all_test_stds)),
                    'std_std': float(np.std(all_test_stds)),
                    'global_min': float(global_test_range[0]),
                    'global_max': float(global_test_range[1])
                }
            }
        }
        
        return {
            'datasets': benchmark_info,
            'global_statistics': global_benchmark_stats,
            'n_datasets_analyzed': total_datasets
        }
    
    return {
        'datasets': benchmark_info,
        'global_statistics': None,
        'n_datasets_analyzed': 0
    }


def compare_data_structures(dataloader_info, benchmark_info, config):
    """Compare dataloader and benchmark data structures."""
    print("=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Handle new data structure format
    if isinstance(dataloader_info, dict) and 'batches' in dataloader_info:
        dl_batches = dataloader_info['batches']
        dl_global_stats = dataloader_info.get('global_statistics')
    else:
        dl_batches = dataloader_info
        dl_global_stats = None
    
    if isinstance(benchmark_info, dict) and 'datasets' in benchmark_info:
        bm_datasets = benchmark_info['datasets']
        bm_global_stats = benchmark_info.get('global_statistics')
    else:
        bm_datasets = benchmark_info
        bm_global_stats = None
    
    # Extract key info from dataloader
    if dl_batches:
        dl_batch = dl_batches[0]  # Use first batch as reference
        dl_batch_size = dl_batch['batch_size']
        dl_n_features = dl_batch['n_features']
        dl_n_train = dl_batch['n_train_samples']
        dl_n_test = dl_batch['n_test_samples']
        dl_zero_cols = dl_batch['zero_cols_train']
        
        print("TRAINING DATALOADER:")
        print(f"  Batch size: {dl_batch_size}")
        print(f"  Features per dataset: {dl_n_features}")
        print(f"  Train samples per dataset: {dl_n_train}")
        print(f"  Test samples per dataset: {dl_n_test}")
        print(f"  Zero columns: {dl_zero_cols}")
        print(f"  Data format: {dl_batch['X_train_shape']} (X_train)")
        
        # Add global statistics if available
        if dl_global_stats:
            print(f"  Total datasets analyzed: {dl_global_stats['total_datasets_analyzed']}")
            print(f"  Active features: {dl_global_stats['n_active_features']}")
            print(f"  Zero-padded features: {dl_global_stats['n_zero_padded_features']}")
    
    # Extract key info from benchmark
    if bm_datasets:
        bm_task = bm_datasets[0]  # Use first task as reference
        bm_n_features = bm_task['final_X_train_shape'][1]
        bm_n_train = bm_task['final_X_train_shape'][0]
        bm_n_test = bm_task['final_X_test_shape'][0]
        bm_zero_cols = bm_task['zero_cols_train']
        bm_padding = bm_task['applied_padding']
        
        print("\nBENCHMARK DATA:")
        print(f"  Tasks analyzed: {len(bm_datasets)}")
        print(f"  Features per dataset: {bm_n_features}")
        print(f"  Train samples: {bm_n_train}")
        print(f"  Test samples: {bm_n_test}")
        print(f"  Zero columns: {bm_zero_cols}")
        print(f"  Applied padding: {bm_padding}")
        print(f"  Data format: {bm_task['final_X_train_shape']} (X_train)")
        
        # Add global statistics if available
        if bm_global_stats:
            print(f"  Total datasets analyzed: {bm_global_stats['total_datasets_analyzed']}")
            if 'feature_statistics' in bm_global_stats:
                print(f"  Active features: {bm_global_stats['feature_statistics']['n_active_features']}")
                print(f"  Zero-padded features: {bm_global_stats['feature_statistics']['n_zero_padded_features']}")
    
    # Key comparisons
    print("\nKEY COMPARISONS:")
    
    if dl_batches and bm_datasets:
        print(f"  Feature count match: {dl_n_features == bm_n_features}")
        print(f"    Dataloader: {dl_n_features}")
        print(f"    Benchmark:  {bm_n_features}")
        
        print("  Data structure:")
        print(f"    Dataloader: Batched datasets {dl_batch['X_train_shape']}")
        print(f"    Benchmark:  Single datasets {bm_task['final_X_train_shape']}")
        
        print("  Zero-padding:")
        print(f"    Dataloader: {dl_zero_cols} zero columns")
        print(f"    Benchmark:  {bm_zero_cols} zero columns (padding applied: {bm_padding})")
        
        # Compare global statistics if available
        if dl_global_stats and bm_global_stats:
            print("\n  Global Statistics Comparison:")
            dl_active = dl_global_stats['n_active_features']
            dl_zero_pad = dl_global_stats['n_zero_padded_features']
            
            if 'feature_statistics' in bm_global_stats:
                bm_active = bm_global_stats['feature_statistics']['n_active_features']
                bm_zero_pad = bm_global_stats['feature_statistics']['n_zero_padded_features']
                
                print(f"    Active features match: {dl_active == bm_active}")
                print(f"      Dataloader: {dl_active}, Benchmark: {bm_active}")
                print(f"    Zero-padded features match: {dl_zero_pad == bm_zero_pad}")
                print(f"      Dataloader: {dl_zero_pad}, Benchmark: {bm_zero_pad}")
    
    # Configuration analysis
    model_config = extract_config_values(config.get('model_config', {}))
    config_num_features = model_config.get('num_features', 'Not specified')
    
    print("\nCONFIGURATION:")
    print(f"  Config num_features: {config_num_features}")
    
    if dl_batches:
        print(f"  Match with dataloader: {config_num_features == dl_n_features if isinstance(config_num_features, int) else 'N/A'}")
    if bm_datasets:
        print(f"  Match with benchmark: {config_num_features == bm_n_features if isinstance(config_num_features, int) else 'N/A'}")
    
    return {
        'dataloader_summary': {
            'batch_size': dl_batch_size if dl_batches else None,
            'n_features': dl_n_features if dl_batches else None,
            'n_train': dl_n_train if dl_batches else None,
            'n_test': dl_n_test if dl_batches else None,
            'zero_columns': dl_zero_cols if dl_batches else None,
            'global_statistics': dl_global_stats,
        },
        'benchmark_summary': {
            'n_features': bm_n_features if bm_datasets else None,
            'n_train': bm_n_train if bm_datasets else None,
            'n_test': bm_n_test if bm_datasets else None,
            'zero_columns': bm_zero_cols if bm_datasets else None,
            'applied_padding': bm_padding if bm_datasets else None,
            'global_statistics': bm_global_stats,
        },
        'feature_count_match': dl_n_features == bm_n_features if (dl_batches and bm_datasets) else None,
        'config_num_features': config_num_features,
    }


def main():
    """Main comparison function."""
    
    # =====================
    # CONFIGURATION SECTION
    # =====================
    
    # Path to YAML config file (same as training)
    CONFIG_PATH = "experiments/FirstTests/configs/early_test.yaml"
    
    # Benchmark hyperparameters (same as run_benchmark.py ALL_CAPS section)
    BENCHMARK_TASKS = ""  # empty = use defaults
    BENCHMARK_MAX_TASKS = 5  # Limit for testing
    BENCHMARK_DATA_DIR = "data_cache"
    BENCHMARK_N_FEATURES = 7
    BENCHMARK_MAX_N_FEATURES = 19
    BENCHMARK_N_TRAIN = 125
    BENCHMARK_N_TEST = 125
    BENCHMARK_PREFER_NUMERIC = True
    BENCHMARK_NO_TARGET_ENCODING = False
    
    # Analysis parameters
    DATALOADER_BATCHES_TO_ANALYZE = 1
    
    print("=" * 80)
    print("BENCHMARK vs TRAINING DATA COMPARISON")
    print("=" * 80)
    
    print(f"\nCONFIGURATION:")
    print(f"  Config file: {CONFIG_PATH}")
    print(f"  Benchmark tasks: {BENCHMARK_MAX_TASKS} tasks")
    print(f"  Benchmark n_features: {BENCHMARK_N_FEATURES}")
    print(f"  Benchmark max_n_features: {BENCHMARK_MAX_N_FEATURES}")
    print(f"  Benchmark samples: {BENCHMARK_N_TRAIN} train, {BENCHMARK_N_TEST} test")
    
    # Load configuration
    try:
        config = load_yaml_config(CONFIG_PATH)
    except Exception as e:
        print(f"ERROR: Could not load config file: {e}")
        return
    
    # ===================
    # ANALYZE DATALOADER
    # ===================
    
    try:
        print(f"\nSetting up training dataloader...")
        
        # Extract config sections
        scm_config = config.get('scm_config', {})
        dataset_config = config.get('dataset_config', {})
        preprocessing_config = config.get('preprocessing_config', {})
        training_config = extract_config_values(config.get('training_config', {}))
        
        # Create dataset maker and dataloader (same as simple_run.py)
        dataset_maker = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config
        )
        
        dataset = dataset_maker.create_dataset(seed=42)
        batch_size = training_config.get('batch_size', 4)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for simpler debugging
        )
        
        dataloader_info = analyze_dataloader_batch(dataloader, DATALOADER_BATCHES_TO_ANALYZE)
        
    except Exception as e:
        print(f"ERROR analyzing dataloader: {e}")
        import traceback
        traceback.print_exc()
        dataloader_info = None
    
    # ==================
    # ANALYZE BENCHMARK
    # ==================
    
    try:
        print(f"\nSetting up benchmark loader...")
        
        # Create benchmark args (same format as run_benchmark.py)
        benchmark_args = SimpleNamespace(
            tasks=BENCHMARK_TASKS,
            max_tasks=BENCHMARK_MAX_TASKS,
            data_dir=BENCHMARK_DATA_DIR,
            no_target_encoding=BENCHMARK_NO_TARGET_ENCODING,
            quiet=False,
            n_features=BENCHMARK_N_FEATURES,
            max_n_features=BENCHMARK_MAX_N_FEATURES,
            n_train=BENCHMARK_N_TRAIN,
            n_test=BENCHMARK_N_TEST,
            prefer_numeric=BENCHMARK_PREFER_NUMERIC,
        )
        
        # Create loader
        loader = SimpleOpenMLLoader(
            data_dir=benchmark_args.data_dir,
            use_target_encoding=not benchmark_args.no_target_encoding,
            verbose=not benchmark_args.quiet,
        )
        
        # Get task list
        if benchmark_args.tasks:
            tasks = [int(t) for t in benchmark_args.tasks.split(",")]
        else:
            tasks = DEFAULT_TABULAR_NUM_REG_TASKS[:benchmark_args.max_tasks]
        
        benchmark_info = analyze_benchmark_data(loader, tasks, benchmark_args)
        
    except Exception as e:
        print(f"ERROR analyzing benchmark: {e}")
        import traceback
        traceback.print_exc()
        benchmark_info = None
    
    # ============
    # COMPARISON
    # ============
    
    comparison_results = compare_data_structures(dataloader_info, benchmark_info, config)
    
    # =================
    # SAVE RESULTS
    # =================
    
    results = {
        'config_path': CONFIG_PATH,
        'benchmark_config': {
            'n_features': BENCHMARK_N_FEATURES,
            'max_n_features': BENCHMARK_MAX_N_FEATURES,
            'n_train': BENCHMARK_N_TRAIN,
            'n_test': BENCHMARK_N_TEST,
        },
        'dataloader_analysis': dataloader_info,
        'benchmark_analysis': benchmark_info,
        'comparison': comparison_results,
    }
    
    # Save JSON file in the same directory as this script
    script_dir = Path(__file__).parent
    output_file = script_dir / "data_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"=" * 80)


if __name__ == "__main__":
    main()


# ===============================================================================
# ANALYSIS RESULTS AND FINDINGS
# ===============================================================================

"""
DATA COMPARISON ANALYSIS RESULTS
================================

Based on examination of the code structure and data processing pipelines, here are the key findings:

## 1. TRAINING DATA (Dataloader) Structure:
- **Source**: MakePurelyObservationalDataset with synthetic causal data
- **Batch Format**: (X_train, y_train, X_test, y_test) 
- **Feature Dimensions**: Variable based on SCM configuration
- **Zero Padding**: Applied in BasicProcessing.py via _pad_features_to_target_size()
  - Pads to `target_feature_size` from preprocessing_config
  - Uses np.pad with mode='constant', constant_values=0.0
  - Applied to both train and test portions within each dataset

## 2. BENCHMARK DATA (OpenML) Structure:
- **Source**: OpenML tasks via SimpleOpenMLLoader
- **Format**: Standard X_train, y_train, X_test, y_test arrays
- **Feature Dimensions**: Variable based on OpenML dataset characteristics
- **Zero Padding**: Applied in run_benchmark.py via _pad_features()
  - Pads to `args.n_features` (BENCHMARK_N_FEATURES constant)
  - Uses np.pad with mode='constant', constant_values=0.0
  - Applied to both X_train and X_test

## 3. KEY DIFFERENCES:

### Shape Handling:
- **Training**: Batch dimension = [batch_size, n_samples_per_dataset, n_features]
- **Benchmark**: Standard format = [n_samples, n_features]

### Padding Strategy:
- **Training**: Pads to config.preprocessing_config.target_feature_size
- **Benchmark**: Pads to BENCHMARK_N_FEATURES (from ALL_CAPS constants)

### Sample Size Control:
- **Training**: Uses full synthetic datasets (size varies by SCM config)
- **Benchmark**: Subsamples to fixed sizes (BENCHMARK_N_TRAIN, BENCHMARK_N_TEST)

## 4. CRITICAL COMPATIBILITY POINTS:

### Feature Dimension Alignment:
For proper model compatibility, ensure:
```
config.preprocessing_config.target_feature_size == BENCHMARK_N_FEATURES
```

### Model Configuration:
The SimplePFN model's `num_features` parameter should match both:
- Training: config.preprocessing_config.target_feature_size  
- Benchmark: BENCHMARK_N_FEATURES

### Zero Padding Location:
Both pipelines pad with zeros, but at different stages:
- **Training**: During preprocessing in BasicProcessing._pad_features_to_target_size()
- **Benchmark**: During benchmark setup in run_benchmark._pad_features()

## 5. RECOMMENDED VERIFICATION:

To ensure compatibility:
1. Check that target_feature_size in config matches BENCHMARK_N_FEATURES
2. Verify model's num_features parameter matches both values
3. Confirm zero-padding produces identical shapes
4. Test that padded features don't introduce bias

## 6. POTENTIAL ISSUES:

### Feature Mismatch:
If target_feature_size ≠ BENCHMARK_N_FEATURES:
- Model trained on different input size than benchmark expects
- Could cause runtime errors or poor performance

### Padding Bias:
Large amounts of zero-padding could:
- Reduce model performance on real data
- Create distribution shift between synthetic and real data

### Sample Size Effects:
Different sample sizes between training and benchmark may affect:
- Model generalization assessment
- Statistical significance of comparisons

## 7. CONFIGURATION EXAMPLE:

For compatibility, ensure your config has:
```yaml
preprocessing_config:
  target_feature_size: 7  # Must match BENCHMARK_N_FEATURES

model_config:
  num_features: 7  # Must match both above values
```

And run_benchmark.py constants:
```python
N_FEATURES = 7  # Must match config values
```

This analysis was generated by examining the codebase structure and data processing pipelines.
For runtime verification, execute the main() function which will perform actual data comparison.
"""
