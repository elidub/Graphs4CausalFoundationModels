#!/usr/bin/env python3
"""
Compare Benchmark and Training Data

This script compares the data shapes and processing between:
1. Training data from dataloader (MakePurelyObservationalDataset)
2. Benchmark data from OpenML (SimpleOpenMLLoader + run_benchmark.py processing)

UPDATED: Now uses the new Preprocessor-based architecture for consistent preprocessing
across both training and benchmark data pipelines.

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


def compute_unified_statistics(datasets_info, data_source_name):
    """
    Compute unified statistics for either training or benchmark datasets.
    
    Args:
        datasets_info: List of dataset dictionaries containing X_train, X_test, y_train, y_test, etc.
        data_source_name: String name for the data source (e.g., "Training", "Benchmark")
    
    Returns:
        Dictionary with comprehensive statistics
    """
    if not datasets_info:
        return {}
    
    n_datasets = len(datasets_info)
    
    # Shape analysis
    train_shapes = [d['X_train_shape'] for d in datasets_info]
    test_shapes = [d['X_test_shape'] for d in datasets_info]
    
    max_features = max(shape[-1] for shape in train_shapes)
    min_features = min(shape[-1] for shape in train_shapes)
    avg_features = np.mean([shape[-1] for shape in train_shapes])
    
    # Global feature analysis - check which features are active vs zero-padded
    feature_analysis = {}
    for feat_idx in range(max_features):
        active_count = 0
        zero_padded_count = 0
        
        for d in datasets_info:
            if 'feature_statistics' in d and feat_idx in d['feature_statistics']:
                feat_stats = d['feature_statistics'][feat_idx]
                if feat_stats['is_zero_padded']:
                    zero_padded_count += 1
                else:
                    active_count += 1
            elif feat_idx < d['X_train_shape'][-1]:
                # Fallback: assume not zero-padded if we don't have specific info
                active_count += 1
        
        total_with_feature = active_count + zero_padded_count
        if total_with_feature > 0:
            zero_padded_percentage = (zero_padded_count / total_with_feature) * 100
            feature_analysis[feat_idx] = {
                'active_count': active_count,
                'zero_padded_count': zero_padded_count,
                'zero_padded_percentage': zero_padded_percentage,
                'total_datasets_with_feature': total_with_feature
            }
    
    # Classify features
    active_features = [idx for idx, stats in feature_analysis.items() 
                      if stats['zero_padded_percentage'] <= 50]
    zero_padded_features = [idx for idx, stats in feature_analysis.items() 
                           if stats['zero_padded_percentage'] > 50]
    
    # Global data statistics
    all_train_means = [d.get('X_train_mean', d.get('X_train_stats', {}).get('mean', 0)) for d in datasets_info]
    all_test_means = [d.get('X_test_mean', d.get('X_test_stats', {}).get('mean', 0)) for d in datasets_info]
    all_train_stds = [d.get('X_train_std', d.get('X_train_stats', {}).get('std', 0)) for d in datasets_info]
    all_test_stds = [d.get('X_test_std', d.get('X_test_stats', {}).get('std', 0)) for d in datasets_info]
    
    all_train_mins = [d.get('X_train_min', d.get('X_train_stats', {}).get('min', 0)) for d in datasets_info]
    all_train_maxs = [d.get('X_train_max', d.get('X_train_stats', {}).get('max', 0)) for d in datasets_info]
    all_test_mins = [d.get('X_test_min', d.get('X_test_stats', {}).get('min', 0)) for d in datasets_info]
    all_test_maxs = [d.get('X_test_max', d.get('X_test_stats', {}).get('max', 0)) for d in datasets_info]
    
    # Zero-padding statistics
    zero_cols_train = [d.get('zero_cols_train', 0) for d in datasets_info]
    zero_cols_test = [d.get('zero_cols_test', 0) for d in datasets_info]
    
    # Processing statistics (for benchmark data)
    datasets_with_padding = sum(1 for d in datasets_info if d.get('applied_padding', False))
    datasets_with_truncation = sum(1 for d in datasets_info if d.get('applied_truncation', False))
    datasets_with_feature_selection = sum(1 for d in datasets_info if d.get('applied_feature_selection', False))
    
    # Per-feature detailed statistics (for active features)
    feature_detailed_stats = {}
    for feat_idx in active_features[:10]:  # Show stats for first 10 active features
        train_values = []
        test_values = []
        
        for d in datasets_info:
            if 'feature_statistics' in d and feat_idx in d['feature_statistics']:
                feat_stats = d['feature_statistics'][feat_idx]
                if not feat_stats['is_zero_padded']:
                    train_values.append(feat_stats['train_mean'])
                    test_values.append(feat_stats['test_mean'])
        
        if train_values and test_values:
            feature_detailed_stats[feat_idx] = {
                'train_mean_avg': np.mean(train_values),
                'train_mean_std': np.std(train_values),
                'test_mean_avg': np.mean(test_values),
                'test_mean_std': np.std(test_values),
                'n_datasets': len(train_values)
            }
    
    return {
        'data_source': data_source_name,
        'n_datasets': n_datasets,
        'shape_stats': {
            'min_features': min_features,
            'max_features': max_features,
            'avg_features': avg_features,
            'train_shapes': train_shapes[:5],  # Show first 5 shapes
            'test_shapes': test_shapes[:5]
        },
        'feature_classification': {
            'n_active_features': len(active_features),
            'n_zero_padded_features': len(zero_padded_features),
            'active_features': active_features,
            'zero_padded_features': zero_padded_features,
            'feature_analysis': feature_analysis
        },
        'global_data_stats': {
            'train_means': {
                'avg': np.mean(all_train_means) if all_train_means else 0,
                'std': np.std(all_train_means) if all_train_means else 0,
                'min': np.min(all_train_means) if all_train_means else 0,
                'max': np.max(all_train_means) if all_train_means else 0
            },
            'test_means': {
                'avg': np.mean(all_test_means) if all_test_means else 0,
                'std': np.std(all_test_means) if all_test_means else 0,
                'min': np.min(all_test_means) if all_test_means else 0,
                'max': np.max(all_test_means) if all_test_means else 0
            },
            'train_stds': {
                'avg': np.mean(all_train_stds) if all_train_stds else 0,
                'std': np.std(all_train_stds) if all_train_stds else 0,
                'min': np.min(all_train_stds) if all_train_stds else 0,
                'max': np.max(all_train_stds) if all_train_stds else 0
            },
            'test_stds': {
                'avg': np.mean(all_test_stds) if all_test_stds else 0,
                'std': np.std(all_test_stds) if all_test_stds else 0,
                'min': np.min(all_test_stds) if all_test_stds else 0,
                'max': np.max(all_test_stds) if all_test_stds else 0
            },
            'global_ranges': {
                'train_range': [np.min(all_train_mins) if all_train_mins else 0, 
                               np.max(all_train_maxs) if all_train_maxs else 0],
                'test_range': [np.min(all_test_mins) if all_test_mins else 0, 
                              np.max(all_test_maxs) if all_test_maxs else 0]
            }
        },
        'zero_padding_stats': {
            'zero_cols_train': {
                'avg': np.mean(zero_cols_train) if zero_cols_train else 0,
                'std': np.std(zero_cols_train) if zero_cols_train else 0,
                'min': np.min(zero_cols_train) if zero_cols_train else 0,
                'max': np.max(zero_cols_train) if zero_cols_train else 0
            },
            'zero_cols_test': {
                'avg': np.mean(zero_cols_test) if zero_cols_test else 0,
                'std': np.std(zero_cols_test) if zero_cols_test else 0,
                'min': np.min(zero_cols_test) if zero_cols_test else 0,
                'max': np.max(zero_cols_test) if zero_cols_test else 0
            }
        },
        'processing_stats': {
            'datasets_with_padding': datasets_with_padding,
            'datasets_with_truncation': datasets_with_truncation,
            'datasets_with_feature_selection': datasets_with_feature_selection,
            'padding_percentage': (datasets_with_padding / n_datasets * 100) if n_datasets > 0 else 0,
            'truncation_percentage': (datasets_with_truncation / n_datasets * 100) if n_datasets > 0 else 0,
            'feature_selection_percentage': (datasets_with_feature_selection / n_datasets * 100) if n_datasets > 0 else 0
        },
        'feature_detailed_stats': feature_detailed_stats
    }


def print_unified_statistics(stats):
    """Print unified statistics in a clear format."""
    data_source = stats['data_source']
    
    print("=" * 60)
    print(f"UNIFIED STATISTICS: {data_source.upper()} DATA")
    print("=" * 60)
    
    # Dataset and shape information
    print(f"Dataset Information:")
    print(f"  Total datasets analyzed: {stats['n_datasets']}")
    print(f"  Feature count range: {stats['shape_stats']['min_features']} - {stats['shape_stats']['max_features']}")
    print(f"  Average features per dataset: {stats['shape_stats']['avg_features']:.1f}")
    
    if stats['shape_stats']['train_shapes']:
        print(f"  Example shapes (first 5):")
        for i, (train_shape, test_shape) in enumerate(zip(stats['shape_stats']['train_shapes'], 
                                                           stats['shape_stats']['test_shapes'])):
            print(f"    Dataset {i}: train={train_shape}, test={test_shape}")
    
    # Feature classification
    fc = stats['feature_classification']
    print(f"\nFeature Classification:")
    print(f"  Active features (non-zero in >50% datasets): {fc['n_active_features']} {fc['active_features']}")
    print(f"  Zero-padded features (zero in >50% datasets): {fc['n_zero_padded_features']} {fc['zero_padded_features']}")
    
    # Global data statistics
    gds = stats['global_data_stats']
    print(f"\nGlobal Data Statistics:")
    print(f"  Train data means: avg={gds['train_means']['avg']:.4f}±{gds['train_means']['std']:.4f}, range=[{gds['train_means']['min']:.4f}, {gds['train_means']['max']:.4f}]")
    print(f"  Test data means: avg={gds['test_means']['avg']:.4f}±{gds['test_means']['std']:.4f}, range=[{gds['test_means']['min']:.4f}, {gds['test_means']['max']:.4f}]")
    print(f"  Train data stds: avg={gds['train_stds']['avg']:.4f}±{gds['train_stds']['std']:.4f}, range=[{gds['train_stds']['min']:.4f}, {gds['train_stds']['max']:.4f}]")
    print(f"  Test data stds: avg={gds['test_stds']['avg']:.4f}±{gds['test_stds']['std']:.4f}, range=[{gds['test_stds']['min']:.4f}, {gds['test_stds']['max']:.4f}]")
    print(f"  Global train range: [{gds['global_ranges']['train_range'][0]:.3f}, {gds['global_ranges']['train_range'][1]:.3f}]")
    print(f"  Global test range: [{gds['global_ranges']['test_range'][0]:.3f}, {gds['global_ranges']['test_range'][1]:.3f}]")
    
    # Zero-padding statistics
    zps = stats['zero_padding_stats']
    print(f"\nZero-Padding Statistics:")
    print(f"  Zero columns (train): avg={zps['zero_cols_train']['avg']:.1f}±{zps['zero_cols_train']['std']:.1f}, range=[{zps['zero_cols_train']['min']:.0f}, {zps['zero_cols_train']['max']:.0f}]")
    print(f"  Zero columns (test): avg={zps['zero_cols_test']['avg']:.1f}±{zps['zero_cols_test']['std']:.1f}, range=[{zps['zero_cols_test']['min']:.0f}, {zps['zero_cols_test']['max']:.0f}]")
    
    # Processing statistics (mainly for benchmark data)
    ps = stats['processing_stats']
    if ps['datasets_with_padding'] > 0 or ps['datasets_with_truncation'] > 0 or ps['datasets_with_feature_selection'] > 0:
        print(f"\nProcessing Statistics:")
        print(f"  Datasets with padding: {ps['datasets_with_padding']}/{stats['n_datasets']} ({ps['padding_percentage']:.1f}%)")
        print(f"  Datasets with truncation: {ps['datasets_with_truncation']}/{stats['n_datasets']} ({ps['truncation_percentage']:.1f}%)")
        print(f"  Datasets with feature selection: {ps['datasets_with_feature_selection']}/{stats['n_datasets']} ({ps['feature_selection_percentage']:.1f}%)")
    
    # Per-feature detailed statistics (for active features)
    if stats['feature_detailed_stats']:
        print(f"\nPer-Feature Statistics (active features, first 5):")
        for feat_idx, feat_stats in list(stats['feature_detailed_stats'].items())[:5]:
            print(f"  Feature {feat_idx}: train_mean={feat_stats['train_mean_avg']:.4f}±{feat_stats['train_mean_std']:.4f}, test_mean={feat_stats['test_mean_avg']:.4f}±{feat_stats['test_mean_std']:.4f} (n={feat_stats['n_datasets']})")


def analyze_dataloader_batch(dataloader, n_batches=3):
    """Analyze training dataloader batch structure and shapes."""
    print("=" * 60)
    print("ANALYZING TRAINING DATALOADER")
    print("=" * 60)
    
    batch_info = []
    all_datasets_info = []  # Collect all individual dataset info for unified analysis
    
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
            
            # Create simplified dataset info for unified analysis
            dataset_info = {
                'X_train_shape': X_train_single.shape,
                'X_test_shape': X_test_single.shape,
                'y_train_shape': y_train_single.shape,
                'y_test_shape': y_test_single.shape,
                'zero_cols_train': int(np.sum(zero_cols_train)),
                'zero_cols_test': int(np.sum(zero_cols_test)),
                'feature_statistics': feature_statistics,
                'X_train_mean': float(np.mean(X_train_single)),
                'X_train_std': float(np.std(X_train_single)),
                'X_train_min': float(np.min(X_train_single)),
                'X_train_max': float(np.max(X_train_single)),
                'X_test_mean': float(np.mean(X_test_single)),
                'X_test_std': float(np.std(X_test_single)),
                'X_test_min': float(np.min(X_test_single)),
                'X_test_max': float(np.max(X_test_single)),
                'applied_padding': bool(np.any(zero_cols_train) or np.any(zero_cols_test)),
                'applied_truncation': False,
                'applied_feature_selection': False
            }
            all_datasets_info.append(dataset_info)
        
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
    
    # Compute and print unified statistics
    training_unified_stats = compute_unified_statistics(all_datasets_info, "Training")
    print_unified_statistics(training_unified_stats)
    
    return {
        'batches': batch_info,
        'global_statistics': global_training_stats,
        'n_batches_analyzed': len(batch_info),
        'unified_statistics': training_unified_stats
    }


def analyze_benchmark_data(loader, task_ids, benchmark_args):
    """Analyze benchmark data shapes and processing using new Preprocessor-based approach."""
    print("=" * 60)
    print("ANALYZING BENCHMARK DATA")
    print("=" * 60)
    
    # Create benchmark data using the new Preprocessor-based SimpleOpenMLLoader
    subsample_mode = bool(getattr(benchmark_args, "n_train", 0) and 
                         getattr(benchmark_args, "n_test", 0) and 
                         getattr(benchmark_args, "n_features", 0))
    
    if subsample_mode:
        print(f"Using subsampled mode with new Preprocessor:")
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
        print("Using normal mode (full datasets) with new Preprocessor")
        data_map = loader.load_tasks(task_ids)
    
    benchmark_info = []
    benchmark_datasets_info = []  # For unified analysis
    
    for tid in task_ids:
        if tid not in data_map or not data_map.get(tid):
            print(f"Skipping task {tid}: no processed data")
            continue
            
        d = data_map[tid]
        
        # Data is now already processed through Preprocessor - convert to numpy for analysis
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
        
        print(f"\nTask {tid} (processed by Preprocessor):")
        print(f"  X_train: {X_train.shape} ({X_train.dtype})")
        print(f"  y_train: {y_train.shape} ({y_train.dtype})")
        print(f"  X_test:  {X_test.shape} ({X_test.dtype})")
        print(f"  y_test:  {y_test.shape} ({y_test.dtype})")
        
        # Analyze the preprocessed data
        original_n_features = d.get('original_n_features', X_train.shape[1])
        
        # Check for zero columns (indicates padding)
        zero_cols_train = np.all(X_train == 0, axis=0)
        zero_cols_test = np.all(X_test == 0, axis=0)
        
        # Detect padding pattern
        padding_start_train = None
        padding_start_test = None
        applied_padding = np.any(zero_cols_train)
        
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
        
        # Determine if feature selection or truncation was applied
        requested_n = int(getattr(benchmark_args, "n_features", 0) or 0)
        max_n = int(getattr(benchmark_args, "max_n_features", 0) or 0)
        
        applied_feature_selection = requested_n > 0 and original_n_features > requested_n
        applied_truncation = max_n > 0 and original_n_features > max_n
        
        print(f"  Original features: {original_n_features}")
        print(f"  Final features: {X_train.shape[1]}")
        if applied_feature_selection:
            print(f"  Feature selection applied: {original_n_features} -> {requested_n}")
        if applied_padding:
            print(f"  Zero-padding applied: padding starts at column {padding_start_train}")
        if applied_truncation:
            print(f"  Truncation applied: limited to {max_n} features")
        
        # Add per-feature statistics for unified analysis
        feature_statistics = {}
        n_features = X_train.shape[1]
        for feat_idx in range(n_features):
            train_feature = X_train[:, feat_idx]
            test_feature = X_test[:, feat_idx]
            
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
        
        task_data = {
            'task_id': tid,
            'dataset_id': d.get('dataset_id', 'unknown'),
            'original_shape': d.get('data_shape', 'unknown'),
            'original_n_features': original_n_features,
            'final_X_train_shape': tuple(X_train.shape),
            'final_y_train_shape': tuple(y_train.shape),
            'final_X_test_shape': tuple(X_test.shape),
            'final_y_test_shape': tuple(y_test.shape),
            'applied_feature_selection': applied_feature_selection,
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
            'preprocessing_applied': d.get('preprocessing_applied', True),
            'feature_statistics': feature_statistics,  # Add per-feature stats
        }
        
        benchmark_info.append(task_data)
        
        # Create simplified dataset info for unified analysis
        simplified_data = {
            'X_train_shape': tuple(X_train.shape),
            'X_test_shape': tuple(X_test.shape),
            'y_train_shape': tuple(y_train.shape),
            'y_test_shape': tuple(y_test.shape),
            'zero_cols_train': task_data['zero_cols_train'],
            'zero_cols_test': task_data['zero_cols_test'],
            'feature_statistics': feature_statistics,
            'X_train_mean': task_data['X_train_mean'],
            'X_train_std': task_data['X_train_std'],
            'X_train_min': task_data['X_train_min'],
            'X_train_max': task_data['X_train_max'],
            'X_test_mean': task_data['X_test_mean'],
            'X_test_std': task_data['X_test_std'],
            'X_test_min': task_data['X_test_min'],
            'X_test_max': task_data['X_test_max'],
            'applied_padding': applied_padding,
            'applied_truncation': applied_truncation,
            'applied_feature_selection': applied_feature_selection
        }
        benchmark_datasets_info.append(simplified_data)
        
        if applied_padding:
            print(f"  Zero-padding details:")
            print(f"    Padding starts at column: {padding_start_train} (train), {padding_start_test} (test)")
            print(f"    Zero columns: {np.sum(zero_cols_train)} (train), {np.sum(zero_cols_test)} (test)")
        
        print(f"  Data range (train): [{np.min(X_train):.3f}, {np.max(X_train):.3f}]")
        print(f"  Data range (test): [{np.min(X_test):.3f}, {np.max(X_test):.3f}]")
        print(f"  Preprocessor applied: {task_data['preprocessing_applied']}")
    
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
        
        # Compute and print unified statistics
        benchmark_unified_stats = compute_unified_statistics(benchmark_datasets_info, "Benchmark")
        print_unified_statistics(benchmark_unified_stats)
        
        return {
            'datasets': benchmark_info,
            'global_statistics': global_benchmark_stats,
            'n_datasets_analyzed': total_datasets,
            'unified_statistics': benchmark_unified_stats
        }
    
    return {
        'datasets': benchmark_info,
        'global_statistics': None,
        'n_datasets_analyzed': 0
    }


def compare_unified_statistics(training_stats, benchmark_stats):
    """Compare training and benchmark data using unified statistics."""
    print("=" * 80)
    print("UNIFIED STATISTICS COMPARISON")
    print("=" * 80)
    
    if not training_stats or not benchmark_stats:
        print("Missing statistics for comparison")
        return
    
    print(f"SIDE-BY-SIDE COMPARISON:")
    print(f"{'Metric':<40} {'Training':<25} {'Benchmark':<25}")
    print("=" * 90)
    
    # Dataset counts
    print(f"{'Total datasets':<40} {training_stats['n_datasets']:<25} {benchmark_stats['n_datasets']:<25}")
    
    # Shape statistics
    ts = training_stats['shape_stats']
    bs = benchmark_stats['shape_stats']
    print(f"{'Feature count range':<40} {ts['min_features']}-{ts['max_features']:<25} {bs['min_features']}-{bs['max_features']:<25}")
    print(f"{'Average features':<40} {ts['avg_features']:.1f:<25} {bs['avg_features']:.1f:<25}")
    
    # Feature classification
    tfc = training_stats['feature_classification']
    bfc = benchmark_stats['feature_classification']
    print(f"{'Active features':<40} {tfc['n_active_features']:<25} {bfc['n_active_features']:<25}")
    print(f"{'Zero-padded features':<40} {tfc['n_zero_padded_features']:<25} {bfc['n_zero_padded_features']:<25}")
    
    # Global data statistics
    tgds = training_stats['global_data_stats']
    bgds = benchmark_stats['global_data_stats']
    print(f"\nDATA VALUE STATISTICS:")
    print(f"{'Metric':<40} {'Training':<25} {'Benchmark':<25}")
    print("-" * 90)
    print(f"{'Train means (avg±std)':<40} {tgds['train_means']['avg']:.4f}±{tgds['train_means']['std']:.4f:<20} {bgds['train_means']['avg']:.4f}±{bgds['train_means']['std']:.4f:<20}")
    print(f"{'Test means (avg±std)':<40} {tgds['test_means']['avg']:.4f}±{tgds['test_means']['std']:.4f:<20} {bgds['test_means']['avg']:.4f}±{bgds['test_means']['std']:.4f:<20}")
    print(f"{'Train stds (avg±std)':<40} {tgds['train_stds']['avg']:.4f}±{tgds['train_stds']['std']:.4f:<20} {bgds['train_stds']['avg']:.4f}±{bgds['train_stds']['std']:.4f:<20}")
    print(f"{'Test stds (avg±std)':<40} {tgds['test_stds']['avg']:.4f}±{tgds['test_stds']['std']:.4f:<20} {bgds['test_stds']['avg']:.4f}±{bgds['test_stds']['std']:.4f:<20}")
    
    # Data ranges
    trange = tgds['global_ranges']
    brange = bgds['global_ranges']
    print(f"{'Global train range':<40} [{trange['train_range'][0]:.3f}, {trange['train_range'][1]:.3f}]<20 [{brange['train_range'][0]:.3f}, {brange['train_range'][1]:.3f}]<20")
    print(f"{'Global test range':<40} [{trange['test_range'][0]:.3f}, {trange['test_range'][1]:.3f}]<20 [{brange['test_range'][0]:.3f}, {brange['test_range'][1]:.3f}]<20")
    
    # Zero-padding statistics
    tzps = training_stats['zero_padding_stats']
    bzps = benchmark_stats['zero_padding_stats']
    print(f"\nZERO-PADDING STATISTICS:")
    print(f"{'Metric':<40} {'Training':<25} {'Benchmark':<25}")
    print("-" * 90)
    print(f"{'Zero cols train (avg±std)':<40} {tzps['zero_cols_train']['avg']:.1f}±{tzps['zero_cols_train']['std']:.1f:<20} {bzps['zero_cols_train']['avg']:.1f}±{bzps['zero_cols_train']['std']:.1f:<20}")
    print(f"{'Zero cols test (avg±std)':<40} {tzps['zero_cols_test']['avg']:.1f}±{tzps['zero_cols_test']['std']:.1f:<20} {bzps['zero_cols_test']['avg']:.1f}±{bzps['zero_cols_test']['std']:.1f:<20}")
    
    # Processing statistics
    tps = training_stats['processing_stats']
    bps = benchmark_stats['processing_stats']
    print(f"\nPROCESSING STATISTICS:")
    print(f"{'Metric':<40} {'Training':<25} {'Benchmark':<25}")
    print("-" * 90)
    print(f"{'Datasets with padding (%)':<40} {tps['padding_percentage']:.1f}%<22 {bps['padding_percentage']:.1f}%<22")
    print(f"{'Datasets with truncation (%)':<40} {tps['truncation_percentage']:.1f}%<22 {bps['truncation_percentage']:.1f}%<22")
    print(f"{'Datasets with feat. selection (%)':<40} {tps['feature_selection_percentage']:.1f}%<22 {bps['feature_selection_percentage']:.1f}%<22")
    
    # Key differences analysis
    print(f"\nKEY DIFFERENCES ANALYSIS:")
    print("-" * 50)
    
    # Feature count consistency
    feature_count_match = (ts['min_features'] == bs['min_features'] and 
                          ts['max_features'] == bs['max_features'])
    print(f"Feature count consistency: {'✓ MATCH' if feature_count_match else '✗ MISMATCH'}")
    if not feature_count_match:
        print(f"  Training: {ts['min_features']}-{ts['max_features']} features")
        print(f"  Benchmark: {bs['min_features']}-{bs['max_features']} features")
    
    # Active feature consistency
    active_features_match = tfc['n_active_features'] == bfc['n_active_features']
    print(f"Active features consistency: {'✓ MATCH' if active_features_match else '✗ MISMATCH'}")
    if not active_features_match:
        print(f"  Training: {tfc['n_active_features']} active features")
        print(f"  Benchmark: {bfc['n_active_features']} active features")
    
    # Data range consistency (check if both use similar scaling)
    train_range_similar = (abs(trange['train_range'][0] - brange['train_range'][0]) < 0.1 and 
                          abs(trange['train_range'][1] - brange['train_range'][1]) < 0.1)
    print(f"Data range consistency: {'✓ SIMILAR' if train_range_similar else '✗ DIFFERENT'}")
    if not train_range_similar:
        print(f"  Training range: [{trange['train_range'][0]:.3f}, {trange['train_range'][1]:.3f}]")
        print(f"  Benchmark range: [{brange['train_range'][0]:.3f}, {brange['train_range'][1]:.3f}]")
    
    # Zero-padding consistency
    padding_similar = (abs(tzps['zero_cols_train']['avg'] - bzps['zero_cols_train']['avg']) < 2)
    print(f"Zero-padding consistency: {'✓ SIMILAR' if padding_similar else '✗ DIFFERENT'}")
    if not padding_similar:
        print(f"  Training avg zero cols: {tzps['zero_cols_train']['avg']:.1f}")
        print(f"  Benchmark avg zero cols: {bzps['zero_cols_train']['avg']:.1f}")


# ============
# COMPARISON
# ============

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
    BENCHMARK_LEAVE_ONE_OUT_ENCODING = True
    
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
            leave_one_out_encoding=BENCHMARK_LEAVE_ONE_OUT_ENCODING,
            quiet=False,
            n_features=BENCHMARK_N_FEATURES,
            max_n_features=BENCHMARK_MAX_N_FEATURES,
            n_train=BENCHMARK_N_TRAIN,
            n_test=BENCHMARK_N_TEST,
            prefer_numeric=BENCHMARK_PREFER_NUMERIC,
        )
        
        # Create loader with new Preprocessor parameters
        loader = SimpleOpenMLLoader(
            data_dir=benchmark_args.data_dir,
            verbose=not benchmark_args.quiet,
            max_n_features=benchmark_args.max_n_features,
            max_n_train_samples=benchmark_args.n_train,
            max_n_test_samples=benchmark_args.n_test,
            # Use standardization (matching typical benchmark settings)
            standardize=True,
            yeo_johnson=False,
            negative_one_one_scaling=False,
            remove_outliers=True,
            outlier_quantile=0.95,
            shuffle_samples=True,
            shuffle_features=True,
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
    
    # Unified comparison using standardized statistics
    if (dataloader_info and 'unified_statistics' in dataloader_info and 
        benchmark_info and 'unified_statistics' in benchmark_info):
        compare_unified_statistics(
            dataloader_info['unified_statistics'], 
            benchmark_info['unified_statistics']
        )
    
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
