#!/usr/bin/env python
"""
Aggregate Results Script

This script aggregates all results from experiment folders in the results directory.
It computes mean, median, and standard error for PEHE and ATE relative error
across all methods (result folders) and datasets.

Output: A matrix of shape (num_methods x 4 datasets) for each metric.
"""

import os
import pickle
import pandas as pd
import numpy as np
from scipy import stats

# Configuration
RESULTS_DIR = "<REPO_ROOT>/RealCauseEval/results"
DATASETS = ["IHDP", "ACIC", "CPS", "PSID"]
METRICS = ["pehe", "ate_rel_err"]

# Dataset information (n_train, n_test, n_features, n_realizations)
DATASET_INFO = {
    "IHDP": {"n_train": 672, "n_test": 75, "n_features": 25, "n_realizations": 100},
    "ACIC": {"n_train": 4321, "n_test": 481, "n_features": 58, "n_realizations": 10},
    "CPS": {"n_train": 14559, "n_test": 1618, "n_features": 8, "n_realizations": 100},
    "PSID": {"n_train": 2407, "n_test": 268, "n_features": 8, "n_realizations": 100},
}

# Folders to skip (not experiment results)
SKIP_FOLDERS = {"test", "test_tlearner", "test_tlearner_v2"}
SKIP_FILES = {"eval_df.csv"}


def load_results_from_folder(folder_path):
    """Load all result pickle files from a folder."""
    results = []
    
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # Skip if not a file
        if not os.path.isfile(item_path):
            continue
            
        # Skip known non-pickle files
        if item.endswith('.csv') or item.endswith('.json'):
            continue
            
        try:
            with open(item_path, 'rb') as f:
                data = pickle.load(f)
                # Remove cate_preds to save memory
                if 'cate_preds' in data:
                    data.pop('cate_preds')
                results.append(data)
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            # Skip corrupted or non-pickle files
            pass
    
    return results


def compute_stats(values):
    """Compute mean, median, and standard error for a list of values."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    
    values = np.array(values)
    mean = np.mean(values)
    median = np.median(values)
    stderr = stats.sem(values) if len(values) > 1 else np.nan
    
    return mean, median, stderr


def aggregate_results():
    """Main function to aggregate all results."""
    
    # Get all experiment folders
    all_items = os.listdir(RESULTS_DIR)
    exp_folders = []
    
    for item in all_items:
        item_path = os.path.join(RESULTS_DIR, item)
        # Only include directories, skip files and test folders
        if os.path.isdir(item_path) and item not in SKIP_FOLDERS:
            exp_folders.append(item)
    
    exp_folders = sorted(exp_folders)
    
    print(f"Found {len(exp_folders)} experiment folders:")
    for folder in exp_folders:
        print(f"  - {folder}")
    print()
    
    # Collect all results into a DataFrame
    all_results = []
    
    for folder in exp_folders:
        folder_path = os.path.join(RESULTS_DIR, folder)
        results = load_results_from_folder(folder_path)
        
        for r in results:
            r['exp_folder'] = folder
            all_results.append(r)
    
    if len(all_results) == 0:
        print("No results found!")
        return
    
    df = pd.DataFrame(all_results)
    
    print(f"Loaded {len(df)} total result records")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Create summary tables for each metric
    for metric in METRICS:
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in results, skipping...")
            continue
        
        print("=" * 100)
        print(f"METRIC: {metric.upper()}")
        print("=" * 100)
        
        # Initialize result matrices
        mean_matrix = pd.DataFrame(index=exp_folders, columns=DATASETS, dtype=float)
        median_matrix = pd.DataFrame(index=exp_folders, columns=DATASETS, dtype=float)
        stderr_matrix = pd.DataFrame(index=exp_folders, columns=DATASETS, dtype=float)
        count_matrix = pd.DataFrame(index=exp_folders, columns=DATASETS, dtype=int)
        
        for folder in exp_folders:
            for dataset in DATASETS:
                # Filter data for this folder and dataset
                mask = (df['exp_folder'] == folder) & (df['dataset'] == dataset)
                values = df.loc[mask, metric].dropna().values
                
                mean, median, stderr = compute_stats(values)
                
                mean_matrix.loc[folder, dataset] = mean
                median_matrix.loc[folder, dataset] = median
                stderr_matrix.loc[folder, dataset] = stderr
                count_matrix.loc[folder, dataset] = len(values)
        
        # Print Mean Table
        print(f"\n--- Mean {metric.upper()} ---")
        print(mean_matrix.to_string())
        
        # Print Median Table
        print(f"\n--- Median {metric.upper()} ---")
        print(median_matrix.to_string())
        
        # Print Standard Error Table
        print(f"\n--- Std Error {metric.upper()} ---")
        print(stderr_matrix.to_string())
        
        # Print Count Table
        print(f"\n--- Sample Count ---")
        print(count_matrix.to_string())
        
        # Print formatted table with mean ± stderr
        print(f"\n--- {metric.upper()}: Mean ± StdErr ---")
        combined = pd.DataFrame(index=exp_folders, columns=DATASETS)
        for folder in exp_folders:
            for dataset in DATASETS:
                mean = mean_matrix.loc[folder, dataset]
                stderr = stderr_matrix.loc[folder, dataset]
                if pd.isna(mean):
                    combined.loc[folder, dataset] = "N/A"
                elif pd.isna(stderr):
                    combined.loc[folder, dataset] = f"{mean:.4f}"
                else:
                    combined.loc[folder, dataset] = f"{mean:.4f} ± {stderr:.4f}"
        print(combined.to_string())
        print()
        
        # Print CONCISE table grouped by dataset (no N/A clutter)
        print(f"\n--- {metric.upper()}: CONCISE (Grouped by Dataset) ---")
        for dataset in DATASETS:
            # Get dataset info
            ds_info = DATASET_INFO.get(dataset, {})
            n_train = ds_info.get("n_train", "?")
            n_test = ds_info.get("n_test", "?")
            n_features = ds_info.get("n_features", "?")
            
            # Get experiments that have data for this dataset
            valid_exps = []
            for folder in exp_folders:
                mean = mean_matrix.loc[folder, dataset]
                if not pd.isna(mean):
                    stderr = stderr_matrix.loc[folder, dataset]
                    count = count_matrix.loc[folder, dataset]
                    if pd.isna(stderr):
                        val_str = f"{mean:.4f}"
                    else:
                        val_str = f"{mean:.4f} ± {stderr:.4f}"
                    # Extract method name (remove dataset prefix if present)
                    method_name = folder
                    for ds in [d.lower() for d in DATASETS]:
                        if folder.startswith(ds + "_"):
                            method_name = folder[len(ds)+1:]
                            break
                    valid_exps.append((method_name, val_str, count))
            
            if valid_exps:
                print(f"\n  {dataset} (n_train={n_train}, n_test={n_test}, n_features={n_features}):")
                # Maintain original order (sorted alphabetically by folder name)
                for method, val, count in valid_exps:
                    print(f"    {method:45s} {val:20s} (n={count})")
        print()
    
    # Save aggregated DataFrame to CSV
    output_path = os.path.join(RESULTS_DIR, "aggregated_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved raw aggregated data to: {output_path}")
    
    # Also save summary tables
    for metric in METRICS:
        if metric not in df.columns:
            continue
            
        # Recompute for saving
        mean_matrix = pd.DataFrame(index=exp_folders, columns=DATASETS, dtype=float)
        median_matrix = pd.DataFrame(index=exp_folders, columns=DATASETS, dtype=float)
        stderr_matrix = pd.DataFrame(index=exp_folders, columns=DATASETS, dtype=float)
        
        for folder in exp_folders:
            for dataset in DATASETS:
                mask = (df['exp_folder'] == folder) & (df['dataset'] == dataset)
                values = df.loc[mask, metric].dropna().values
                mean, median, stderr = compute_stats(values)
                mean_matrix.loc[folder, dataset] = mean
                median_matrix.loc[folder, dataset] = median
                stderr_matrix.loc[folder, dataset] = stderr
        
        mean_matrix.to_csv(os.path.join(RESULTS_DIR, f"summary_{metric}_mean.csv"))
        median_matrix.to_csv(os.path.join(RESULTS_DIR, f"summary_{metric}_median.csv"))
        stderr_matrix.to_csv(os.path.join(RESULTS_DIR, f"summary_{metric}_stderr.csv"))
    
    print(f"Saved summary tables to {RESULTS_DIR}/summary_*.csv")
    
    return df


if __name__ == "__main__":
    aggregate_results()
