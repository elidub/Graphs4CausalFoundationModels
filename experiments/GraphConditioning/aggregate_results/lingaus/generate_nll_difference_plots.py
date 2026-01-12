#!/usr/bin/env python3
"""
NLL Difference Analysis Script

This script analyzes benchmark results by computing the difference in NLL between
all approaches and the baseline model, then generates box plots showing the 
improvement (or degradation) relative to baseline.

Usage:
    python generate_nll_difference_plots.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Configuration
checkpoint_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints"
checkpoint_base = Path(checkpoint_path)

# Create timestamped output directory with subfolders for each metric
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output_dir = Path(__file__).parent / "plots" / f"metric_differences_{timestamp}"
base_output_dir.mkdir(parents=True, exist_ok=True)

# Create subfolders for each metric
nll_output_dir = base_output_dir / "nll"
mse_output_dir = base_output_dir / "mse" 
r2_output_dir = base_output_dir / "r2"

nll_output_dir.mkdir(exist_ok=True)
mse_output_dir.mkdir(exist_ok=True)
r2_output_dir.mkdir(exist_ok=True)

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 12)  # More square figure
# Set uniform large font sizes throughout
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['figure.titlesize'] = 24

# Define model order and colors (excluding baseline since it will be the reference)
# Order follows standard color sequence: blue, red, green, yellow, purple, magenta
MODEL_ORDER = [
    'hardatt', 'ancestor_hardatt',           # Blue family (1st)
    'softatt', 'ancestor_softatt',           # Red family (2nd)
    'gcn', 'ancestor_gcn',                   # Green family (3rd)
    'gcn_and_softatt', 'ancestor_gcn_and_softatt',  # Yellow family (4th) - moved up to use yellow
    'gcn_and_hartatt', 'ancestor_gcn_and_hartatt',  # Purple family (5th) - if available
    'gcn_and_softatt_fixed'                  # Magenta (6th)
]

MODEL_COLORS = {
    # Base models - light colors in standard order
    'hardatt': '#96d2eb',                    # light blue (150, 210, 235)
    'softatt': '#ffb3b3',                    # light red (255, 179, 179)
    'gcn': '#a5d7af',                        # light green (165, 215, 175)
    'gcn_and_softatt': '#fff08c',            # light yellow (255, 240, 140) - moved to 4th position
    'gcn_and_hartatt': '#afa5d7',            # light purple (175, 165, 215) - moved to 5th position
    'gcn_and_softatt_fixed': '#eba5c8',      # light magenta (235, 165, 200)
    
    # Ancestor models - much darker, more distinct shades
    'ancestor_hardatt': '#2856a3',           # much darker blue
    'ancestor_softatt': '#cc0000',           # much darker red
    'ancestor_gcn': '#4d7c57',               # much darker green
    'ancestor_gcn_and_softatt': '#b8a000',   # much darker yellow/gold - moved to 4th position
    'ancestor_gcn_and_hartatt': '#4a3f70',   # much darker purple - moved to 5th position
    'baseline': '#d2d2d2'                    # lgrey for baseline if needed
}

# Model display names for plots
MODEL_DISPLAY_NAMES = {
    'hardatt': 'Hard Attention',
    'ancestor_hardatt': 'Hard Attention (Ancestor)',
    'softatt': 'Soft Attention',
    'ancestor_softatt': 'Soft Attention (Ancestor)', 
    'gcn': 'GCN',
    'ancestor_gcn': 'GCN (Ancestor)',
    'gcn_and_hartatt': 'GCN + Hard Attention',
    'ancestor_gcn_and_hartatt': 'GCN + Hard Attention (Ancestor)',
    'gcn_and_softatt': 'GCN + Soft Attention', 
    'ancestor_gcn_and_softatt': 'GCN + Soft Attention (Ancestor)',
    'gcn_and_softatt_fixed': 'GCN + Soft Attention (Fixed)',
    'baseline': 'Baseline'
}


def get_model_display_name(model_key):
    """Get display name for model, fallback to original key if not found."""
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def load_individual_results(checkpoint_base, checkpoint_keys):
    """Load individual sample results for bootstrap analysis."""
    results_dict = {}
    
    for ckpt_key in checkpoint_keys:
        ckpt_dir = checkpoint_base / ckpt_key
        results_dir = ckpt_dir / "lingaus_final"
        
        if not results_dir.exists():
            print(f"Missing results dir: {results_dir}")
            continue
        
        checkpoint_results = {}
        
        # Look for individual result files - optimized file loading
        node_counts = [2, 5, 10, 20, 35, 50]
        variant_suffixes = ['path_YT', 'path_TY', 'path_independent_TY']
        
        for node_count in node_counts:
            # Check for base variant
            base_file = results_dir / f"individual_{node_count}nodes.json"
            if base_file.exists():
                if 'base' not in checkpoint_results:
                    checkpoint_results['base'] = {}
                try:
                    with open(base_file, 'r') as f:
                        data = json.load(f)
                        checkpoint_results['base'][str(node_count)] = data
                except Exception as e:
                    print(f"Error loading {base_file}: {e}")
            
            # Check for path variants - batch file existence checks
            for variant_suffix in variant_suffixes:
                variant_file = results_dir / f"individual_{node_count}nodes_{variant_suffix}.json"
                if variant_file.exists():
                    if variant_suffix not in checkpoint_results:
                        checkpoint_results[variant_suffix] = {}
                    try:
                        with open(variant_file, 'r') as f:
                            data = json.load(f)
                            checkpoint_results[variant_suffix][str(node_count)] = data
                    except Exception as e:
                        print(f"Error loading {variant_file}: {e}")
        
        if checkpoint_results:
            results_dict[ckpt_key] = checkpoint_results
            print(f"Loaded individual results: {ckpt_key} ({len(checkpoint_results)} variants)")
    
    return results_dict


def bootstrap_confidence_interval(data, confidence_level=0.90, n_bootstrap=1000):
    """Compute bootstrap confidence interval for the mean of data."""
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan
    
    # Original mean
    original_mean = np.mean(data)
    
    # Bootstrap resampling - reduced from 10000 to 1000 for speed
    bootstrap_means = []
    np.random.seed(42)  # For reproducibility
    
    # Vectorized bootstrap sampling for speed
    bootstrap_samples = np.random.choice(data, size=(n_bootstrap, n), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return original_mean, ci_lower, ci_upper


def compute_aggregated_nll_differences_with_bootstrap(results_dict, confidence_level=0.90):
    """Compute NLL differences with bootstrap CI for aggregated TY, YT, and independent_TY data.
    
    For each variant (TY, YT, independent_TY):
    1. Compute mean difference (baseline - model) for that variant using baseline from SAME variant
    2. Collect all mean differences across variants
    3. Apply bootstrap to the collection of mean differences
    """
    difference_data = []
    
    # Variants to aggregate
    variants_to_aggregate = ['path_TY', 'path_YT', 'path_independent_TY']
    
    # Collect mean differences for each model across all variants
    model_mean_differences = {}  # {model: {node_count: [mean_diffs_from_each_variant]}}
    
    # First get all baseline data organized by variant
    baseline_data = {}  # {variant: {node_count: [nll_values]}}
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        if model_name == 'baseline':
            for variant, variant_data in ckpt_results.items():
                if variant in variants_to_aggregate:  # Only get baseline for variants we're aggregating
                    baseline_data[variant] = {}
                    for node_count, individual_results in variant_data.items():
                        baseline_data[variant][node_count] = [sample['nll'] for sample in individual_results]
    
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        
        if model_name == 'baseline':
            continue  # Skip baseline
            
        # Process each variant separately
        for variant in variants_to_aggregate:
            if variant not in ckpt_results:
                continue
                
            variant_data = ckpt_results[variant]
            
            # Check if we have baseline data for this specific variant
            if variant not in baseline_data:
                print(f"Warning: No baseline data found for variant {variant}")
                continue
                
            for node_count, individual_results in variant_data.items():
                if node_count not in baseline_data[variant]:
                    continue
                    
                # Get NLL values for this variant and node count
                model_nll_values = [sample['nll'] for sample in individual_results]
                baseline_nll_values = baseline_data[variant][node_count]
                
                if len(model_nll_values) == 0 or len(baseline_nll_values) == 0:
                    continue
                
                # Compute mean NLLs for this variant/node_count combination
                model_mean_nll = np.mean(model_nll_values)
                baseline_mean_nll = np.mean(baseline_nll_values)
                
                # Compute difference for this variant (baseline - model, so positive = improvement)
                variant_mean_diff = baseline_mean_nll - model_mean_nll
                
                # Store this mean difference
                if model_name not in model_mean_differences:
                    model_mean_differences[model_name] = {}
                if node_count not in model_mean_differences[model_name]:
                    model_mean_differences[model_name][node_count] = []
                
                model_mean_differences[model_name][node_count].append(variant_mean_diff)
                print(f"Variant {variant}, Model {model_name}, Node {node_count}: "
                      f"baseline_mean={baseline_mean_nll:.4f}, model_mean={model_mean_nll:.4f}, "
                      f"diff={variant_mean_diff:.4f}")
    
    # Now compute bootstrap CI for the collection of mean differences
    for model_name, model_node_data in model_mean_differences.items():
        for node_count, mean_differences in model_node_data.items():
            if len(mean_differences) == 0:
                continue
                
            print(f"Model {model_name}, Node {node_count}: mean_differences = {mean_differences}")
                
            # Bootstrap confidence interval for the mean of mean differences
            mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                mean_differences, confidence_level=confidence_level
            )
            
            difference_data.append({
                'model': model_name,
                'variant': 'aggregated_TY_YT_indep',
                'node_count': int(node_count),
                'n_samples': len(mean_differences),
                'nll_difference_mean': mean_diff,
                'nll_difference_ci_lower': ci_lower,
                'nll_difference_ci_upper': ci_upper,
                'baseline_nll_mean': np.nan,  # Not meaningful for aggregated
                'model_nll_mean': np.nan      # Not meaningful for aggregated
            })
    
    return pd.DataFrame(difference_data)


def compute_nll_differences_with_bootstrap(results_dict, confidence_level=0.90):
    """Compute NLL differences with bootstrap confidence intervals."""
    difference_data = []
    
    # First, organize data by variant and node count
    baseline_data = {}  # {variant: {node_count: [nll_values]}}
    model_data = {}     # {model: {variant: {node_count: [nll_values]}}}
    
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        
        for variant, variant_data in ckpt_results.items():
            for node_count, individual_results in variant_data.items():
                nll_values = [sample['nll'] for sample in individual_results]
                
                if model_name == 'baseline':
                    if variant not in baseline_data:
                        baseline_data[variant] = {}
                    baseline_data[variant][node_count] = nll_values
                else:
                    if model_name not in model_data:
                        model_data[model_name] = {}
                    if variant not in model_data[model_name]:
                        model_data[model_name][variant] = {}
                    model_data[model_name][variant][node_count] = nll_values
    
    # Compute differences for each model relative to baseline
    for model_name, model_variants in model_data.items():
        for variant, variant_node_data in model_variants.items():
            for node_count, model_nll_values in variant_node_data.items():
                # Get corresponding baseline values
                if variant in baseline_data and node_count in baseline_data[variant]:
                    baseline_nll_values = baseline_data[variant][node_count]
                    
                    # Ensure same number of samples
                    min_samples = min(len(model_nll_values), len(baseline_nll_values))
                    model_nll_subset = model_nll_values[:min_samples]
                    baseline_nll_subset = baseline_nll_values[:min_samples]
                    
                    # Compute differences (baseline - model, so positive = improvement)
                    nll_differences = np.array(baseline_nll_subset) - np.array(model_nll_subset)
                    
                    # Bootstrap confidence interval for the mean difference
                    mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                        nll_differences, confidence_level=confidence_level
                    )
                    
                    difference_data.append({
                        'model': model_name,
                        'variant': variant,
                        'node_count': int(node_count),
                        'nll_difference_mean': float(mean_diff),
                        'nll_difference_ci_lower': float(ci_lower),
                        'nll_difference_ci_upper': float(ci_upper),
                        'n_samples': len(nll_differences),
                        'baseline_nll_mean': float(np.mean(baseline_nll_subset)),
                        'model_nll_mean': float(np.mean(model_nll_subset))
                    })
    
    return pd.DataFrame(difference_data)


def compute_aggregated_mse_differences_with_bootstrap(results_dict, confidence_level=0.90):
    """Compute MSE differences with bootstrap CI for aggregated TY, YT, and independent_TY data."""
    difference_data = []
    
    # Variants to aggregate
    variants_to_aggregate = ['path_TY', 'path_YT', 'path_independent_TY']
    
    # Collect mean differences for each model across all variants
    model_mean_differences = {}  # {model: {node_count: [mean_diffs_from_each_variant]}}
    
    # First get all baseline data organized by variant
    baseline_data = {}  # {variant: {node_count: [mse_values]}}
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        if model_name == 'baseline':
            for variant, variant_data in ckpt_results.items():
                if variant in variants_to_aggregate:  # Only get baseline for variants we're aggregating
                    baseline_data[variant] = {}
                    for node_count, individual_results in variant_data.items():
                        baseline_data[variant][node_count] = [sample['mse'] for sample in individual_results]
    
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        
        if model_name == 'baseline':
            continue  # Skip baseline
            
        # Process each variant separately
        for variant in variants_to_aggregate:
            if variant not in ckpt_results:
                continue
                
            variant_data = ckpt_results[variant]
            
            # Check if we have baseline data for this specific variant
            if variant not in baseline_data:
                continue
                
            for node_count, individual_results in variant_data.items():
                if node_count not in baseline_data[variant]:
                    continue
                    
                # Get MSE values for this variant and node count
                model_mse_values = [sample['mse'] for sample in individual_results]
                baseline_mse_values = baseline_data[variant][node_count]
                
                if len(model_mse_values) == 0 or len(baseline_mse_values) == 0:
                    continue
                
                # Compute mean MSEs for this variant/node_count combination
                model_mean_mse = np.mean(model_mse_values)
                baseline_mean_mse = np.mean(baseline_mse_values)
                
                # Compute difference for this variant (baseline - model, so positive = improvement)
                variant_mean_diff = baseline_mean_mse - model_mean_mse
                
                # Store this mean difference
                if model_name not in model_mean_differences:
                    model_mean_differences[model_name] = {}
                if node_count not in model_mean_differences[model_name]:
                    model_mean_differences[model_name][node_count] = []
                
                model_mean_differences[model_name][node_count].append(variant_mean_diff)
    
    # Now compute bootstrap CI for the collection of mean differences
    for model_name, model_node_data in model_mean_differences.items():
        for node_count, mean_differences in model_node_data.items():
            if len(mean_differences) == 0:
                continue
                
            # Bootstrap confidence interval for the mean of mean differences
            mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                mean_differences, confidence_level=confidence_level
            )
            
            difference_data.append({
                'model': model_name,
                'variant': 'aggregated_TY_YT_indep',
                'node_count': int(node_count),
                'n_samples': len(mean_differences),
                'mse_difference_mean': mean_diff,
                'mse_difference_ci_lower': ci_lower,
                'mse_difference_ci_upper': ci_upper,
                'baseline_mse_mean': np.nan,  # Not meaningful for aggregated
                'model_mse_mean': np.nan      # Not meaningful for aggregated
            })
    
    return pd.DataFrame(difference_data)


def compute_mse_differences_with_bootstrap(results_dict, confidence_level=0.90):
    """Compute MSE differences with bootstrap confidence intervals."""
    difference_data = []
    
    # First, organize data by variant and node count
    baseline_data = {}  # {variant: {node_count: [mse_values]}}
    model_data = {}     # {model: {variant: {node_count: [mse_values]}}}
    
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        
        for variant, variant_data in ckpt_results.items():
            for node_count, individual_results in variant_data.items():
                mse_values = [sample['mse'] for sample in individual_results]
                
                if model_name == 'baseline':
                    if variant not in baseline_data:
                        baseline_data[variant] = {}
                    baseline_data[variant][node_count] = mse_values
                else:
                    if model_name not in model_data:
                        model_data[model_name] = {}
                    if variant not in model_data[model_name]:
                        model_data[model_name][variant] = {}
                    model_data[model_name][variant][node_count] = mse_values
    
    # Compute differences for each model relative to baseline
    for model_name, model_variants in model_data.items():
        for variant, variant_node_data in model_variants.items():
            for node_count, model_mse_values in variant_node_data.items():
                # Get corresponding baseline values
                if variant in baseline_data and node_count in baseline_data[variant]:
                    baseline_mse_values = baseline_data[variant][node_count]
                    
                    # Ensure same number of samples
                    min_samples = min(len(model_mse_values), len(baseline_mse_values))
                    model_mse_subset = model_mse_values[:min_samples]
                    baseline_mse_subset = baseline_mse_values[:min_samples]
                    
                    # Compute differences (baseline - model, so positive = improvement)
                    mse_differences = np.array(baseline_mse_subset) - np.array(model_mse_subset)
                    
                    # Bootstrap confidence interval for the mean difference
                    mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                        mse_differences, confidence_level=confidence_level
                    )
                    
                    difference_data.append({
                        'model': model_name,
                        'variant': variant,
                        'node_count': int(node_count),
                        'n_samples': min_samples,
                        'mse_difference_mean': mean_diff,
                        'mse_difference_ci_lower': ci_lower,
                        'mse_difference_ci_upper': ci_upper,
                        'baseline_mse_mean': np.mean(baseline_mse_subset),
                        'model_mse_mean': np.mean(model_mse_subset)
                    })
    
    return pd.DataFrame(difference_data)


def compute_aggregated_r2_differences_with_bootstrap(results_dict, confidence_level=0.90):
    """Compute R2 differences with bootstrap CI for aggregated TY, YT, and independent_TY data."""
    difference_data = []
    
    # Variants to aggregate
    variants_to_aggregate = ['path_TY', 'path_YT', 'path_independent_TY']
    
    # Collect mean differences for each model across all variants
    model_mean_differences = {}  # {model: {node_count: [mean_diffs_from_each_variant]}}
    
    # First get all baseline data organized by variant
    baseline_data = {}  # {variant: {node_count: [r2_values]}}
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        if model_name == 'baseline':
            for variant, variant_data in ckpt_results.items():
                if variant in variants_to_aggregate:  # Only get baseline for variants we're aggregating
                    baseline_data[variant] = {}
                    for node_count, individual_results in variant_data.items():
                        baseline_data[variant][node_count] = [sample['r2'] for sample in individual_results]
    
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        
        if model_name == 'baseline':
            continue  # Skip baseline
            
        # Process each variant separately
        for variant in variants_to_aggregate:
            if variant not in ckpt_results:
                continue
                
            variant_data = ckpt_results[variant]
            
            # Check if we have baseline data for this specific variant
            if variant not in baseline_data:
                continue
                
            for node_count, individual_results in variant_data.items():
                if node_count not in baseline_data[variant]:
                    continue
                    
                # Get R2 values for this variant and node count
                model_r2_values = [sample['r2'] for sample in individual_results]
                baseline_r2_values = baseline_data[variant][node_count]
                
                if len(model_r2_values) == 0 or len(baseline_r2_values) == 0:
                    continue
                
                # Compute mean R2s for this variant/node_count combination
                model_mean_r2 = np.mean(model_r2_values)
                baseline_mean_r2 = np.mean(baseline_r2_values)
                
                # Compute difference for this variant (model - baseline, so positive = improvement for R2)
                variant_mean_diff = model_mean_r2 - baseline_mean_r2
                
                # Store this mean difference
                if model_name not in model_mean_differences:
                    model_mean_differences[model_name] = {}
                if node_count not in model_mean_differences[model_name]:
                    model_mean_differences[model_name][node_count] = []
                
                model_mean_differences[model_name][node_count].append(variant_mean_diff)
    
    # Now compute bootstrap CI for the collection of mean differences
    for model_name, model_node_data in model_mean_differences.items():
        for node_count, mean_differences in model_node_data.items():
            if len(mean_differences) == 0:
                continue
                
            # Bootstrap confidence interval for the mean of mean differences
            mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                mean_differences, confidence_level=confidence_level
            )
            
            difference_data.append({
                'model': model_name,
                'variant': 'aggregated_TY_YT_indep',
                'node_count': int(node_count),
                'n_samples': len(mean_differences),
                'r2_difference_mean': mean_diff,
                'r2_difference_ci_lower': ci_lower,
                'r2_difference_ci_upper': ci_upper,
                'baseline_r2_mean': np.nan,  # Not meaningful for aggregated
                'model_r2_mean': np.nan      # Not meaningful for aggregated
            })
    
    return pd.DataFrame(difference_data)


def compute_r2_differences_with_bootstrap(results_dict, confidence_level=0.90):
    """Compute R2 differences with bootstrap confidence intervals."""
    difference_data = []
    
    # First, organize data by variant and node count
    baseline_data = {}  # {variant: {node_count: [r2_values]}}
    model_data = {}     # {model: {variant: {node_count: [r2_values]}}}
    
    for ckpt_key, ckpt_results in results_dict.items():
        model_name = extract_model_name(ckpt_key)
        
        for variant, variant_data in ckpt_results.items():
            for node_count, individual_results in variant_data.items():
                r2_values = [sample['r2'] for sample in individual_results]
                
                if model_name == 'baseline':
                    if variant not in baseline_data:
                        baseline_data[variant] = {}
                    baseline_data[variant][node_count] = r2_values
                else:
                    if model_name not in model_data:
                        model_data[model_name] = {}
                    if variant not in model_data[model_name]:
                        model_data[model_name][variant] = {}
                    model_data[model_name][variant][node_count] = r2_values
    
    # Compute differences for each model relative to baseline
    for model_name, model_variants in model_data.items():
        for variant, variant_node_data in model_variants.items():
            for node_count, model_r2_values in variant_node_data.items():
                # Get corresponding baseline values
                if variant in baseline_data and node_count in baseline_data[variant]:
                    baseline_r2_values = baseline_data[variant][node_count]
                    
                    # Ensure same number of samples
                    min_samples = min(len(model_r2_values), len(baseline_r2_values))
                    model_r2_subset = model_r2_values[:min_samples]
                    baseline_r2_subset = baseline_r2_values[:min_samples]
                    
                    # Compute differences (model - baseline, so positive = improvement for R2)
                    r2_differences = np.array(model_r2_subset) - np.array(baseline_r2_subset)
                    
                    # Bootstrap confidence interval for the mean difference
                    mean_diff, ci_lower, ci_upper = bootstrap_confidence_interval(
                        r2_differences, confidence_level=confidence_level
                    )
                    
                    difference_data.append({
                        'model': model_name,
                        'variant': variant,
                        'node_count': int(node_count),
                        'n_samples': min_samples,
                        'r2_difference_mean': mean_diff,
                        'r2_difference_ci_lower': ci_lower,
                        'r2_difference_ci_upper': ci_upper,
                        'baseline_r2_mean': np.mean(baseline_r2_subset),
                        'model_r2_mean': np.mean(model_r2_subset)
                    })
    
    return pd.DataFrame(difference_data)


def extract_model_name(checkpoint_name):
    """Extract model name from checkpoint directory name."""
    parts = checkpoint_name.split('_')
    model_name_parts = []
    is_ancestor = False
    
    if "ancestor" in parts:
        is_ancestor = True
    
    start_collecting = False
    for i, part in enumerate(parts):
        if part == "benchmarked":
            start_collecting = True
            continue
        
        if "node" in part and part.replace("node", "").isdigit():
            start_collecting = True
            continue
        
        if start_collecting and not part.replace('.', '').replace('-', '').isdigit():
            model_name_parts.append(part)
    
    model_name = '_'.join(model_name_parts) if model_name_parts else 'unknown'
    
    if is_ancestor and model_name != 'unknown':
        model_name = f'ancestor_{model_name}'
    
    return model_name


def extract_model_name(checkpoint_name):
    """Extract model name from checkpoint directory name."""
    parts = checkpoint_name.split('_')
    model_name_parts = []
    is_ancestor = False
    
    # Check if this is an ancestor model
    if "ancestor" in parts:
        is_ancestor = True
    
    start_collecting = False
    for i, part in enumerate(parts):
        # Start collecting after "benchmarked" keyword
        if part == "benchmarked":
            start_collecting = True
            continue
        
        # For ancestor pattern: start collecting after "Xnode" (e.g., "5node")
        if "node" in part and part.replace("node", "").isdigit():
            start_collecting = True
            continue
        
        # Collect parts that are not PIDs (numbers with dots)
        if start_collecting and not part.replace('.', '').replace('-', '').isdigit():
            model_name_parts.append(part)
    
    model_name = '_'.join(model_name_parts) if model_name_parts else 'unknown'
    
    # Add "ancestor_" prefix if this is an ancestor model
    if is_ancestor and model_name != 'unknown':
        model_name = f'ancestor_{model_name}'
    
    return model_name


def plot_nll_differences(df, node_counts_to_compare=None, title_suffix="", output_prefix="nll_diff", show_values=False, output_dir=None):
    """Create box plots showing NLL differences relative to baseline using bootstrap CI bounds.
    
    Args:
        df: DataFrame with NLL difference data
        node_counts_to_compare: List of node counts to include in plots
        title_suffix: Additional text for plot title
        output_prefix: Prefix for output filenames
        show_values: Whether to display mean values as text labels on the plot
        output_dir: Directory to save plots to
    """
    if output_dir is None:
        output_dir = nll_output_dir
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    variants = sorted(df['variant'].unique())
    
    variant_display_names = {
        'base': 'Base Causal Structure',
        'path_YT': 'Y → T Path',
        'path_TY': 'T → Y Path', 
        'path_independent_TY': 'T ⊥ Y (Independent)',
        'aggregated_TY_YT_indep': 'Aggregated (TY + YT + Independent)'
    }
    
    # Create separate figure for each variant
    for variant in variants:
        variant_data = df[df['variant'] == variant].copy()
        
        if len(variant_data) == 0:
            print(f"No data for variant: {variant}")
            continue
        
        # Create more square subplots - adjust for number of subplots
        fig, axes = plt.subplots(1, len(node_counts_to_compare), figsize=(12 * len(node_counts_to_compare), 12))
        
        # Handle single column case
        if len(node_counts_to_compare) == 1:
            axes = [axes]
        
        for node_idx, node_count in enumerate(node_counts_to_compare):
            ax = axes[node_idx]
            
            # Filter data for this node count
            node_data = variant_data[variant_data['node_count'] == node_count].copy()
            
            if len(node_data) == 0:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{node_count} Nodes')
                continue
            
            # Get unique models and sort by predefined order
            available_models = node_data['model'].unique()
            models = [m for m in MODEL_ORDER if m in available_models]
            
            if not models:
                ax.text(0.5, 0.5, f'No models', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{node_count} Nodes')
                continue
            
            box_data = []
            positions = []
            colors = []
            mean_values = []
            
            for model_idx, model in enumerate(models):
                model_data = node_data[node_data['model'] == model]
                
                if len(model_data) == 0:
                    continue
                
                # Extract difference statistics
                mean_diff = model_data['nll_difference_mean'].values[0]
                ci_lower = model_data['nll_difference_ci_lower'].values[0]
                ci_upper = model_data['nll_difference_ci_upper'].values[0]
                
                # Create synthetic distribution from bootstrap CI bounds (same as original plots)
                synthetic_data = [
                    ci_lower,
                    ci_lower + (mean_diff - ci_lower) * 0.5,  # Q1
                    mean_diff,  # Median
                    mean_diff + (ci_upper - mean_diff) * 0.5,  # Q3
                    ci_upper
                ]
                
                box_data.append(synthetic_data)
                positions.append(model_idx)
                colors.append(MODEL_COLORS.get(model, '#808080'))
                mean_values.append(mean_diff)
            
            if box_data:
                # Create box plot (same style as original plots)
                bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                               patch_artist=True, showmeans=False,
                               medianprops=dict(color='red', linewidth=2),
                               boxprops=dict(linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
                
                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add mean value labels (optional)
                if show_values:
                    for pos, mean_val in zip(positions, mean_values):
                        ax.text(pos, mean_val, f'{mean_val:.4f}', 
                               ha='center', va='bottom', fontsize=30, rotation=0,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor='none', alpha=0.7))
                
                # Add horizontal line at zero (no difference)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
            
            # Set x-axis labels with display names
            ax.set_xticks(range(len(models)))
            model_display_names = [get_model_display_name(model) for model in models]
            ax.set_xticklabels(model_display_names, rotation=45, ha='right', fontsize=30)
            
            ax.set_title(f'{node_count} Nodes', fontsize=30, fontweight='bold')
            ax.set_ylabel('Absolute improvement', fontsize=30, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Main title - use simple title for aggregated variants
        variant_display = variant_display_names.get(variant, variant)
        if variant == 'aggregated_TY_YT_indep':
            fig.suptitle('Improvement over Baseline (NLL)', fontsize=30, fontweight='bold')
        else:
            if title_suffix:
                fig.suptitle(f'NLL Improvement over Baseline: {variant_display}\n{title_suffix}\n(90% Bootstrap CI)', 
                            fontsize=30, fontweight='bold', y=0.95)
            else:
                fig.suptitle('Improvement over Baseline (NLL)', fontsize=30, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        
        # Save plot with optimized settings
        filename = f"{output_prefix}_{variant}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')  # Reduced DPI from 300 to 200
        print(f"Saved: {filepath}")
        plt.close(fig)


def plot_mse_differences(df, node_counts_to_compare=None, title_suffix="", output_prefix="mse_diff", show_values=False, output_dir=None):
    """Create box plots showing MSE differences relative to baseline using bootstrap CI bounds.
    
    Args:
        df: DataFrame with MSE difference data
        node_counts_to_compare: List of node counts to include in plots
        title_suffix: Additional text for plot title
        output_prefix: Prefix for output filenames
        show_values: Whether to display mean values as text labels on the plot
        output_dir: Directory to save plots to
    """
    if output_dir is None:
        output_dir = mse_output_dir
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    # Get variant display names
    variant_display_names = {
        'base': 'Base',
        'path_TY': 'Path T→Y', 
        'path_YT': 'Path Y→T',
        'path_independent_TY': 'Independent T,Y',
        'aggregated_TY_YT_indep': 'Aggregated (TY, YT, Indep)'
    }
    
    variants = df['variant'].unique()
    
    # Create separate figure for each variant
    for variant in variants:
        variant_data = df[df['variant'] == variant].copy()
        
        if len(variant_data) == 0:
            print(f"No data for variant: {variant}")
            continue
        
        # Create more square subplots - adjust for number of subplots
        fig, axes = plt.subplots(1, len(node_counts_to_compare), figsize=(12 * len(node_counts_to_compare), 12))
        
        # Handle single column case
        if len(node_counts_to_compare) == 1:
            axes = [axes]
        
        for node_idx, node_count in enumerate(node_counts_to_compare):
            ax = axes[node_idx]
            
            # Filter data for this node count
            node_data = variant_data[variant_data['node_count'] == node_count].copy()
            
            if len(node_data) == 0:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{node_count} Nodes')
                continue
            
            # Get unique models and sort by predefined order
            available_models = node_data['model'].unique()
            models = [m for m in MODEL_ORDER if m in available_models]
            
            if not models:
                ax.text(0.5, 0.5, f'No models', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{node_count} Nodes')
                continue
            
            box_data = []
            positions = []
            colors = []
            mean_values = []
            
            for model_idx, model in enumerate(models):
                model_data = node_data[node_data['model'] == model]
                
                if len(model_data) == 0:
                    continue
                
                # Extract difference statistics
                mean_diff = model_data['mse_difference_mean'].values[0]
                ci_lower = model_data['mse_difference_ci_lower'].values[0]
                ci_upper = model_data['mse_difference_ci_upper'].values[0]
                
                # Create synthetic distribution from bootstrap CI bounds
                synthetic_data = [
                    ci_lower,
                    ci_lower + (mean_diff - ci_lower) * 0.5,  # Q1
                    mean_diff,  # Median
                    mean_diff + (ci_upper - mean_diff) * 0.5,  # Q3
                    ci_upper
                ]
                
                box_data.append(synthetic_data)
                positions.append(model_idx)
                colors.append(MODEL_COLORS.get(model, '#808080'))
                mean_values.append(mean_diff)
            
            if box_data:
                # Create box plot
                bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                               patch_artist=True, showmeans=False,
                               medianprops=dict(color='red', linewidth=2),
                               boxprops=dict(linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
                
                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add mean value labels (optional)
                if show_values:
                    for pos, mean_val in zip(positions, mean_values):
                        ax.text(pos, mean_val, f'{mean_val:.4f}', 
                               ha='center', va='bottom', fontsize=30, rotation=0,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor='none', alpha=0.7))
                
                # Add horizontal line at zero (no difference)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
            
            # Set x-axis labels with display names
            ax.set_xticks(range(len(models)))
            model_display_names = [get_model_display_name(model) for model in models]
            ax.set_xticklabels(model_display_names, rotation=45, ha='right', fontsize=30)
            
            ax.set_title(f'{node_count} Nodes', fontsize=30, fontweight='bold')
            ax.set_ylabel('Absolute improvement', fontsize=30, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Main title - simple for aggregated variants
        variant_display = variant_display_names.get(variant, variant)
        if variant == 'aggregated_TY_YT_indep':
            fig.suptitle('Improvement over Baseline (MSE)', fontsize=30, fontweight='bold')
        else:
            fig.suptitle(f'MSE Improvement: {variant_display}{title_suffix}', fontsize=30, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot with white background and optimized DPI
        filename = f"{output_prefix}_{variant}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
        plt.close(fig)


def plot_r2_differences(df, node_counts_to_compare=None, title_suffix="", output_prefix="r2_diff", show_values=False, output_dir=None):
    """Create box plots showing R2 differences relative to baseline using bootstrap CI bounds.
    
    Args:
        df: DataFrame with R2 difference data
        node_counts_to_compare: List of node counts to include in plots
        title_suffix: Additional text for plot title
        output_prefix: Prefix for output filenames
        show_values: Whether to display mean values as text labels on the plot
        output_dir: Directory to save plots to
    """
    if output_dir is None:
        output_dir = r2_output_dir
    if node_counts_to_compare is None:
        node_counts_to_compare = sorted(df['node_count'].unique())
    
    # Get variant display names
    variant_display_names = {
        'base': 'Base',
        'path_TY': 'Path T→Y', 
        'path_YT': 'Path Y→T',
        'path_independent_TY': 'Independent T,Y',
        'aggregated_TY_YT_indep': 'Aggregated (TY, YT, Indep)'
    }
    
    variants = df['variant'].unique()
    
    # Create separate figure for each variant
    for variant in variants:
        variant_data = df[df['variant'] == variant].copy()
        
        if len(variant_data) == 0:
            print(f"No data for variant: {variant}")
            continue
        
        # Create more square subplots - adjust for number of subplots
        fig, axes = plt.subplots(1, len(node_counts_to_compare), figsize=(12 * len(node_counts_to_compare), 12))
        
        # Handle single column case
        if len(node_counts_to_compare) == 1:
            axes = [axes]
        
        for node_idx, node_count in enumerate(node_counts_to_compare):
            ax = axes[node_idx]
            
            # Filter data for this node count
            node_data = variant_data[variant_data['node_count'] == node_count].copy()
            
            if len(node_data) == 0:
                ax.text(0.5, 0.5, f'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{node_count} Nodes')
                continue
            
            # Get unique models and sort by predefined order
            available_models = node_data['model'].unique()
            models = [m for m in MODEL_ORDER if m in available_models]
            
            if not models:
                ax.text(0.5, 0.5, f'No models', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{node_count} Nodes')
                continue
            
            box_data = []
            positions = []
            colors = []
            mean_values = []
            
            for model_idx, model in enumerate(models):
                model_data = node_data[node_data['model'] == model]
                
                if len(model_data) == 0:
                    continue
                
                # Extract difference statistics
                mean_diff = model_data['r2_difference_mean'].values[0]
                ci_lower = model_data['r2_difference_ci_lower'].values[0]
                ci_upper = model_data['r2_difference_ci_upper'].values[0]
                
                # Create synthetic distribution from bootstrap CI bounds
                synthetic_data = [
                    ci_lower,
                    ci_lower + (mean_diff - ci_lower) * 0.5,  # Q1
                    mean_diff,  # Median
                    mean_diff + (ci_upper - mean_diff) * 0.5,  # Q3
                    ci_upper
                ]
                
                box_data.append(synthetic_data)
                positions.append(model_idx)
                colors.append(MODEL_COLORS.get(model, '#808080'))
                mean_values.append(mean_diff)
            
            if box_data:
                # Create box plot
                bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                               patch_artist=True, showmeans=False,
                               medianprops=dict(color='red', linewidth=2),
                               boxprops=dict(linewidth=1.5),
                               whiskerprops=dict(linewidth=1.5),
                               capprops=dict(linewidth=1.5))
                
                # Color boxes
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add mean value labels (optional)
                if show_values:
                    for pos, mean_val in zip(positions, mean_values):
                        ax.text(pos, mean_val, f'{mean_val:.4f}', 
                               ha='center', va='bottom', fontsize=30, rotation=0,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor='none', alpha=0.7))
                
                # Add horizontal line at zero (no difference)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
            
            # Set x-axis labels with display names
            ax.set_xticks(range(len(models)))
            model_display_names = [get_model_display_name(model) for model in models]
            ax.set_xticklabels(model_display_names, rotation=45, ha='right', fontsize=30)
            
            ax.set_title(f'{node_count} Nodes', fontsize=30, fontweight='bold')
            ax.set_ylabel('Absolute improvement', fontsize=30, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        # Main title - simple for aggregated variants  
        variant_display = variant_display_names.get(variant, variant)
        if variant == 'aggregated_TY_YT_indep':
            fig.suptitle('Improvement over Baseline (R²)', fontsize=30, fontweight='bold')
        else:
            fig.suptitle(f'R² Improvement: {variant_display}{title_suffix}', fontsize=30, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot with white background and optimized DPI
        filename = f"{output_prefix}_{variant}.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")
        plt.close(fig)


def create_aggregated_variant_data(results_dict, aggregate_variants=['TY', 'YT', 'independent_TY']):
    """
    Create aggregated data by combining specified variants.
    
    Args:
        results_dict: Dictionary of results organized by checkpoint -> variant -> node_count -> samples
        aggregate_variants: List of variant names to aggregate
    
    Returns:
        Updated results_dict with 'aggregated' variant containing combined data
    """
    aggregated_results = {}
    
    for ckpt_key, ckpt_results in results_dict.items():
        # Check if this checkpoint has any of the variants to aggregate
        available_variants = [v for v in aggregate_variants if v in ckpt_results]
        
        if not available_variants:
            continue
            
        if ckpt_key not in aggregated_results:
            aggregated_results[ckpt_key] = {}
            
        # For each node count, combine data across variants
        all_node_counts = set()
        for variant in available_variants:
            all_node_counts.update(ckpt_results[variant].keys())
            
        for node_count in all_node_counts:
            combined_samples = []
            
            # Collect all samples from available variants for this node count
            for variant in available_variants:
                if node_count in ckpt_results[variant]:
                    combined_samples.extend(ckpt_results[variant][node_count])
            
            if combined_samples:
                if 'aggregated' not in aggregated_results[ckpt_key]:
                    aggregated_results[ckpt_key]['aggregated'] = {}
                aggregated_results[ckpt_key]['aggregated'][node_count] = combined_samples
    
    # Add aggregated data to the original results_dict
    updated_results = results_dict.copy()
    for ckpt_key, ckpt_data in aggregated_results.items():
        if ckpt_key not in updated_results:
            updated_results[ckpt_key] = {}
        updated_results[ckpt_key].update(ckpt_data)
    
    return updated_results


def print_nll_difference_statistics(df, title_suffix="", output_prefix="nll_diff", output_dir=None):
    """Print summary statistics for NLL differences and save to file."""
    if output_dir is None:
        output_dir = nll_output_dir
    output_file = output_dir / f"{output_prefix}_summary.txt"
    
    with open(output_file, 'w') as f:
        def write_line(line=""):
            print(line)
            f.write(line + "\n")
        
        write_line("="*80)
        write_line(f"NLL DIFFERENCE ANALYSIS - {title_suffix}")
        write_line("Positive values indicate improvement over baseline")
        write_line("Negative values indicate degradation relative to baseline")
        write_line("90% Bootstrap Confidence Intervals")
        write_line("="*80)
        
        variants = sorted(df['variant'].unique())
        for variant in variants:
            write_line(f"\n{'='*80}")
            write_line(f"VARIANT: {variant}")
            write_line(f"{'='*80}")
            variant_data = df[df['variant'] == variant]
            
            for model in sorted(variant_data['model'].unique()):
                model_data = variant_data[variant_data['model'] == model]
                write_line(f"\n{str(model).upper()}")
                write_line("-"*80)
                
                mean_diff = model_data['nll_difference_mean'].values[0]
                ci_lower = model_data['nll_difference_ci_lower'].values[0]
                ci_upper = model_data['nll_difference_ci_upper'].values[0]
                n_samples = model_data['n_samples'].values[0]
                baseline_mean = model_data['baseline_nll_mean'].values[0]
                model_mean = model_data['model_nll_mean'].values[0]
                
                write_line(f"Mean difference: {mean_diff:.6f}")
                write_line(f"90% Bootstrap CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                write_line(f"Sample size: {n_samples}")
                write_line(f"Baseline NLL: {baseline_mean:.6f}")
                write_line(f"Model NLL: {model_mean:.6f}")
                
                # Check if CI includes zero (statistical significance at 90% level)
                significant = not (ci_lower <= 0 <= ci_upper)
                
                if significant:
                    if mean_diff > 0:
                        interpretation = "SIGNIFICANT IMPROVEMENT (90% confidence)"
                    else:
                        interpretation = "SIGNIFICANT DEGRADATION (90% confidence)"
                else:
                    interpretation = "NO SIGNIFICANT DIFFERENCE (90% confidence)"
                
                write_line(f"Statistical significance: {interpretation}")
                write_line(f"  (CI {'excludes' if significant else 'includes'} zero)")
        
        write_line("\n" + "="*80)
        write_line("Statistical significance based on 90% bootstrap confidence intervals")
        write_line("Differences computed from individual sample pairs (bootstrap resampling)")
        write_line("="*80)
    
    print(f"Saved summary: {output_file}")


def main():
    """Main function to generate NLL, MSE, and R² difference analysis with bootstrap CIs."""
    print("="*80)
    print("Generating Metric Difference Analysis with 90% Bootstrap CIs")
    print(f"Base output directory: {base_output_dir}")
    print(f"NLL plots will be saved to: {nll_output_dir}")
    print(f"MSE plots will be saved to: {mse_output_dir}")
    print(f"R² plots will be saved to: {r2_output_dir}")
    print("="*80)
    
    # Define checkpoint keys (50-node models)
    checkpoint_keys = [
        # Baseline
        "lingaus_50node_benchmarked_baseline_16715917.0",
        # Regular models
        "lingaus_50node_benchmarked_gcn_16715920.0",
        "lingaus_50node_benchmarked_gcn_and_hartatt_16715918.0",
        "lingaus_50node_benchmarked_gcn_and_softatt_16715919.0",
        "lingaus_50node_benchmarked_hardatt_16715921.0",
        "lingaus_50node_benchmarked_softatt_16715922.0",
        # Ancestor models
        "lingaus_ancestor_50node_baseline_16723333.0",
        "lingaus_ancestor_50node_gcn_16723348.0",
        "lingaus_ancestor_50node_gcn_and_hartatt_16723334.0",
        "lingaus_ancestor_50node_gcn_and_softatt_16723335.0",
        "lingaus_ancestor_50node_hardatt_16723337.0",
        "lingaus_ancestor_50node_softatt_16723338.0",
    ]
    
    print("Loading individual sample results for bootstrap analysis...")
    results_dict = load_individual_results(checkpoint_base, checkpoint_keys)
    
    if not results_dict:
        print("No results found!")
        return
    
    print("Computing NLL differences with bootstrap confidence intervals...")
    diff_df = compute_nll_differences_with_bootstrap(results_dict, confidence_level=0.90)
    
    if diff_df.empty:
        print("No difference data computed!")
        return
    
    print(f"Computed differences for {len(diff_df)} model-variant-node combinations")
    print(f"Models: {sorted(diff_df['model'].unique())}")
    print(f"Node counts: {sorted(diff_df['node_count'].unique())}")
    print(f"Variants: {sorted(diff_df['variant'].unique())}")
    
    # Generate plots for individual variants
    plot_nll_differences(diff_df, 
                        node_counts_to_compare=[2, 5, 20, 35, 50],
                        title_suffix="50-Node Training Models vs Baseline",
                        output_prefix="nll_difference",
                        show_values=False,
                        output_dir=nll_output_dir)  # By default, don't show values
    
    # Generate statistics for individual variants
    print_nll_difference_statistics(diff_df,
                                  title_suffix="50-Node Training Models vs Baseline", 
                                  output_prefix="nll_difference",
                                  output_dir=nll_output_dir)
    
    # Compute and plot aggregated data (TY + YT + Independent)
    print("\nComputing aggregated NLL differences (TY + YT + Independent)...")
    aggregated_df = compute_aggregated_nll_differences_with_bootstrap(results_dict, confidence_level=0.90)
    
    if not aggregated_df.empty:
        print(f"Computed aggregated differences for {len(aggregated_df)} model-node combinations")
        
        # Generate aggregated plots
        plot_nll_differences(aggregated_df, 
                            node_counts_to_compare=[5, 20, 50],
                            title_suffix="",
                            output_prefix="nll_difference_aggregated",
                            show_values=False,
                            output_dir=nll_output_dir)
        
        # Generate aggregated statistics
        print_nll_difference_statistics(aggregated_df,
                                      title_suffix="", 
                                      output_prefix="nll_difference_aggregated",
                                      output_dir=nll_output_dir)
    else:
        print("No aggregated data could be computed!")
    
    # MSE Analysis
    print("\n" + "="*80)
    print("MSE DIFFERENCE ANALYSIS")
    print("="*80)
    
    print("Computing MSE differences with bootstrap confidence intervals...")
    mse_diff_df = compute_mse_differences_with_bootstrap(results_dict, confidence_level=0.90)
    
    if not mse_diff_df.empty:
        print(f"Computed MSE differences for {len(mse_diff_df)} model-variant-node combinations")
        
        # Generate MSE plots for individual variants
        plot_mse_differences(mse_diff_df, 
                            node_counts_to_compare=[2, 5, 20, 35, 50],
                            title_suffix="50-Node Training Models vs Baseline",
                            output_prefix="mse_difference",
                            show_values=False,
                            output_dir=mse_output_dir)
        
        # Compute and plot aggregated MSE data
        print("Computing aggregated MSE differences (TY + YT + Independent)...")
        aggregated_mse_df = compute_aggregated_mse_differences_with_bootstrap(results_dict, confidence_level=0.90)
        
        if not aggregated_mse_df.empty:
            plot_mse_differences(aggregated_mse_df, 
                                node_counts_to_compare=[5, 20, 50],
                                title_suffix="",
                                output_prefix="mse_difference_aggregated",
                                show_values=False,
                                output_dir=mse_output_dir)
    else:
        print("No MSE difference data computed!")
    
    # R2 Analysis
    print("\n" + "="*80)
    print("R² DIFFERENCE ANALYSIS")
    print("="*80)
    
    print("Computing R² differences with bootstrap confidence intervals...")
    r2_diff_df = compute_r2_differences_with_bootstrap(results_dict, confidence_level=0.90)
    
    if not r2_diff_df.empty:
        print(f"Computed R² differences for {len(r2_diff_df)} model-variant-node combinations")
        
        # Generate R2 plots for individual variants
        plot_r2_differences(r2_diff_df, 
                           node_counts_to_compare=[2, 5, 20, 35, 50],
                           title_suffix="50-Node Training Models vs Baseline",
                           output_prefix="r2_difference",
                           show_values=False,
                           output_dir=r2_output_dir)
        
        # Compute and plot aggregated R2 data
        print("Computing aggregated R² differences (TY + YT + Independent)...")
        aggregated_r2_df = compute_aggregated_r2_differences_with_bootstrap(results_dict, confidence_level=0.90)
        
        if not aggregated_r2_df.empty:
            plot_r2_differences(aggregated_r2_df, 
                               node_counts_to_compare=[5, 20, 50],
                               title_suffix="",
                               output_prefix="r2_difference_aggregated",
                               show_values=False,
                               output_dir=r2_output_dir)
    else:
        print("No R² difference data computed!")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! All plots saved to:")
    print(f"  NLL plots: {nll_output_dir}")
    print(f"  MSE plots: {mse_output_dir}")
    print(f"  R² plots: {r2_output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()