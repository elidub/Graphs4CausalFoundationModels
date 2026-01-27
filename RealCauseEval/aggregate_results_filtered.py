#!/usr/bin/env python
"""
Filtered Aggregate Results Script

This script aggregates results from specific experiment folders.
It filters to only show selected baseline methods and DoPFN variants.

Selected methods:
- Baseline: slearner_full_context, tlearner_no_truncation, xlearner_no_truncation
- DoPFN: dofm_noclust_full_graph, dofm_noclust_all_unknown
- PSID-specific: dofm_psid_balanced_full_graph, dofm_psid_balanced_all_unknown,
                 xlearner_psid_balanced_16691166_16804588, tlearner_psid_balanced, 
                 slearner_psid_balanced
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

# Dataset information
DATASET_INFO = {
    "IHDP": {"n_train": 672, "n_test": 75, "n_features": 25, "n_realizations": 100},
    "ACIC": {"n_train": 4321, "n_test": 481, "n_features": 58, "n_realizations": 10},
    "CPS": {"n_train": 14559, "n_test": 1618, "n_features": 8, "n_realizations": 100},
    "PSID": {"n_train": 2407, "n_test": 268, "n_features": 8, "n_realizations": 100},
}

# Define which methods to keep for each dataset
SELECTED_METHODS = {
    # Methods for IHDP, ACIC, CPS (NOT PSID)
    "standard": [
        "slearner_full_context",
        "tlearner_no_truncation",
        "xlearner_no_truncation",
        "dofm_noclust_full_graph",
        "dofm_noclust_all_unknown",
    ],
    # CPS-specific S-Learner (since slearner_full_context doesn't have CPS)
    "cps_slearner": [
        "cps_slearner_evaluation",
    ],
    # Methods specifically for PSID (balanced versions only)
    "psid_only": [
        "dofm_psid_balanced_full_graph",
        "dofm_psid_balanced_all_unknown",
        "xlearner_psid_balanced_16691166_16804588",
        "tlearner_psid_balanced",
        "slearner_psid_balanced",
    ]
}


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


def is_method_selected(folder_name, dataset):
    """Check if a method should be included based on dataset."""
    # For PSID: ONLY show PSID-specific balanced methods, NOT standard methods
    if dataset == "PSID":
        # Only allow PSID-specific methods
        if folder_name in SELECTED_METHODS["psid_only"]:
            return True
        # Check if folder starts with psid_ prefix
        if folder_name.startswith("psid_"):
            method_name = folder_name[5:]  # Remove "psid_" prefix
            if method_name in SELECTED_METHODS["psid_only"]:
                return True
        return False
    
    # For CPS: also allow cps_slearner_evaluation as S-Learner
    if dataset == "CPS":
        if folder_name in SELECTED_METHODS["cps_slearner"]:
            return True
    
    # For other datasets (IHDP, ACIC, CPS): show standard methods only
    if folder_name in SELECTED_METHODS["standard"]:
        return True
    
    # Check if folder starts with dataset prefix and contains a selected method
    for ds in [d.lower() for d in DATASETS]:
        if folder_name.startswith(ds + "_"):
            method_name = folder_name[len(ds)+1:]
            if dataset.lower() == ds and method_name in SELECTED_METHODS["standard"]:
                return True
    
    return False


# Display names for methods in LaTeX tables
METHOD_DISPLAY_NAMES = {
    "slearner_full_context": "S-Learner",
    "cps_slearner_evaluation": "S-Learner",  # CPS-specific S-Learner
    "slearner_evaluation": "S-Learner",  # Alternate name after prefix strip
    "tlearner_no_truncation": "T-Learner",
    "xlearner_no_truncation": "X-Learner",
    "dofm_noclust_full_graph": "DoFM (Full Graph)",
    "dofm_noclust_all_unknown": "DoFM (Unknown Graph)",
    "dofm_psid_balanced_full_graph": "DoFM (Full Graph)",
    "dofm_psid_balanced_all_unknown": "DoFM (Unknown Graph)",
    "xlearner_psid_balanced_16691166_16804588": "X-Learner",
    "tlearner_psid_balanced": "T-Learner",
    "slearner_psid_balanced": "S-Learner",
}


def get_display_name(method_name):
    """Get display name for a method."""
    return METHOD_DISPLAY_NAMES.get(method_name, method_name)


def generate_latex_tables(df, all_folders):
    """Generate LaTeX tables for PEHE and ATE_REL_ERR metrics."""
    
    print("\n" + "=" * 100)
    print("LATEX TABLES")
    print("=" * 100)
    
    for metric in METRICS:
        if metric not in df.columns:
            continue
        
        metric_display = "PEHE" if metric == "pehe" else r"ATE Relative Error"
        
        print(f"\n% LaTeX Table for {metric.upper()}")
        print("% Copy everything between the \\begin{{table}} and \\end{{table}} lines")
        print()
        
        # Start table
        print(r"\begin{table}[htbp]")
        print(r"\centering")
        print(r"\caption{" + metric_display + r" results on benchmark datasets. Best results in \textbf{bold}.}")
        print(r"\label{tab:" + metric + r"_results}")
        print(r"\resizebox{\textwidth}{!}{%")
        print(r"\begin{tabular}{l" + "c" * len(DATASETS) + "}")
        print(r"\toprule")
        
        # Header row
        header = "Method"
        for dataset in DATASETS:
            header += f" & {dataset}"
        header += r" \\"
        print(header)
        print(r"\midrule")
        
        # Collect all results for finding best values
        results_matrix = {}  # {method_display: {dataset: (mean, stderr, original_method)}}
        best_values = {dataset: float('inf') for dataset in DATASETS}
        
        for dataset in DATASETS:
            for folder in all_folders:
                if not is_method_selected(folder, dataset):
                    continue
                
                mask = (df['exp_folder'] == folder) & (df['dataset'] == dataset)
                values = df.loc[mask, metric].dropna().values
                
                if len(values) == 0:
                    continue
                
                mean, _, stderr = compute_stats(values)
                
                # Get method name and display name
                method_name = folder
                for ds in [d.lower() for d in DATASETS]:
                    if folder.startswith(ds + "_"):
                        method_name = folder[len(ds)+1:]
                        break
                
                display_name = get_display_name(method_name)
                
                if display_name not in results_matrix:
                    results_matrix[display_name] = {}
                
                results_matrix[display_name][dataset] = (mean, stderr, method_name)
                
                if not np.isnan(mean) and mean < best_values[dataset]:
                    best_values[dataset] = mean
        
        # Define method order for consistent table layout (DoFM Full Graph last)
        method_order = [
            "S-Learner",
            "T-Learner", 
            "X-Learner",
            "DoFM (Unknown Graph)",
            "DoFM (Full Graph)",
        ]
        
        # Print rows in order
        for display_name in method_order:
            if display_name not in results_matrix:
                continue
            
            row = display_name
            for dataset in DATASETS:
                if dataset in results_matrix[display_name]:
                    mean, stderr, _ = results_matrix[display_name][dataset]
                    
                    # Format value
                    if metric == "pehe" and dataset in ["CPS", "PSID"]:
                        # Use scientific notation for large values
                        if mean > 1000:
                            val_str = f"{mean:.0f}"
                        else:
                            val_str = f"{mean:.2f}"
                    else:
                        val_str = f"{mean:.4f}"
                    
                    # Add stderr if available
                    if not pd.isna(stderr):
                        if metric == "pehe" and dataset in ["CPS", "PSID"] and mean > 1000:
                            val_str += f" $\\pm$ {stderr:.0f}"
                        else:
                            val_str += f" $\\pm$ {stderr:.4f}"
                    
                    # Bold if best
                    if abs(mean - best_values[dataset]) < 1e-6:
                        val_str = r"\textbf{" + val_str + "}"
                    
                    row += f" & {val_str}"
                else:
                    row += " & --"
            
            row += r" \\"
            print(row)
        
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"}")
        print(r"\end{table}")
        print()
    
    # Also generate a combined table (both metrics side by side for each dataset)
    print("\n% Combined LaTeX Table (PEHE and ATE Relative Error)")
    print()
    
    # Separate tables for non-PSID and PSID datasets
    print("% Table for IHDP, ACIC, CPS datasets")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Treatment effect estimation results on benchmark datasets. Best results in \textbf{bold}. Values show mean $\pm$ standard error.}")
    print(r"\label{tab:combined_results}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{l" + "cc" * 3 + "}")  # 3 datasets, 2 metrics each
    print(r"\toprule")
    
    # Header row 1: Dataset names
    print(r" & \multicolumn{2}{c}{IHDP} & \multicolumn{2}{c}{ACIC} & \multicolumn{2}{c}{CPS} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    
    # Header row 2: Metric names
    print(r"Method & PEHE & $\epsilon_{\text{ATE}}$ & PEHE & $\epsilon_{\text{ATE}}$ & PEHE & $\epsilon_{\text{ATE}}$ \\")
    print(r"\midrule")
    
    # Collect results
    combined_results = {}
    best_combined = {(ds, m): float('inf') for ds in ["IHDP", "ACIC", "CPS"] for m in METRICS}
    
    for dataset in ["IHDP", "ACIC", "CPS"]:
        for folder in all_folders:
            if not is_method_selected(folder, dataset):
                continue
            
            method_name = folder
            for ds in [d.lower() for d in DATASETS]:
                if folder.startswith(ds + "_"):
                    method_name = folder[len(ds)+1:]
                    break
            
            display_name = get_display_name(method_name)
            
            if display_name not in combined_results:
                combined_results[display_name] = {}
            
            for metric in METRICS:
                mask = (df['exp_folder'] == folder) & (df['dataset'] == dataset)
                values = df.loc[mask, metric].dropna().values
                
                if len(values) == 0:
                    continue
                
                mean, _, stderr = compute_stats(values)
                combined_results[display_name][(dataset, metric)] = (mean, stderr)
                
                if not np.isnan(mean) and mean < best_combined[(dataset, metric)]:
                    best_combined[(dataset, metric)] = mean
    
    # Print rows
    for display_name in method_order:
        if display_name not in combined_results:
            continue
        
        row = display_name
        for dataset in ["IHDP", "ACIC", "CPS"]:
            for metric in METRICS:
                key = (dataset, metric)
                if key in combined_results[display_name]:
                    mean, stderr = combined_results[display_name][key]
                    
                    # Format based on magnitude
                    if mean > 1000:
                        val_str = f"{mean:.0f}"
                        if not pd.isna(stderr):
                            val_str += f" $\\pm$ {stderr:.0f}"
                    elif mean > 10:
                        val_str = f"{mean:.2f}"
                        if not pd.isna(stderr):
                            val_str += f" $\\pm$ {stderr:.2f}"
                    else:
                        val_str = f"{mean:.3f}"
                        if not pd.isna(stderr):
                            val_str += f" $\\pm$ {stderr:.3f}"
                    
                    # Bold if best
                    if abs(mean - best_combined[key]) < 1e-6:
                        val_str = r"\textbf{" + val_str + "}"
                    
                    row += f" & {val_str}"
                else:
                    row += " & --"
        
        row += r" \\"
        print(row)
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table}")
    
    # PSID table separately
    print()
    print("% Table for PSID dataset (balanced methods)")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Treatment effect estimation results on PSID dataset (balanced sampling). Best results in \textbf{bold}.}")
    print(r"\label{tab:psid_results}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Method & PEHE & $\epsilon_{\text{ATE}}$ \\")
    print(r"\midrule")
    
    # Collect PSID results
    psid_results = {}
    best_psid = {m: float('inf') for m in METRICS}
    
    for folder in all_folders:
        if not is_method_selected(folder, "PSID"):
            continue
        
        method_name = folder
        for ds in [d.lower() for d in DATASETS]:
            if folder.startswith(ds + "_"):
                method_name = folder[len(ds)+1:]
                break
        
        display_name = get_display_name(method_name)
        psid_results[display_name] = {}
        
        for metric in METRICS:
            mask = (df['exp_folder'] == folder) & (df['dataset'] == "PSID")
            values = df.loc[mask, metric].dropna().values
            
            if len(values) == 0:
                continue
            
            mean, _, stderr = compute_stats(values)
            psid_results[display_name][metric] = (mean, stderr)
            
            if not np.isnan(mean) and mean < best_psid[metric]:
                best_psid[metric] = mean
    
    # Print PSID rows
    for display_name in method_order:
        if display_name not in psid_results:
            continue
        
        row = display_name
        for metric in METRICS:
            if metric in psid_results[display_name]:
                mean, stderr = psid_results[display_name][metric]
                
                if mean > 1000:
                    val_str = f"{mean:.0f}"
                    if not pd.isna(stderr):
                        val_str += f" $\\pm$ {stderr:.0f}"
                else:
                    val_str = f"{mean:.4f}"
                    if not pd.isna(stderr):
                        val_str += f" $\\pm$ {stderr:.4f}"
                
                if abs(mean - best_psid[metric]) < 1e-6:
                    val_str = r"\textbf{" + val_str + "}"
                
                row += f" & {val_str}"
            else:
                row += " & --"
        
        row += r" \\"
        print(row)
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def generate_concise_latex_table(df, all_folders):
    """Generate a concise LaTeX table with only S-Learner, DoFM (Full Graph), and DoFM (Unknown Graph).
    
    This produces a compact single table with all datasets and both metrics.
    """
    
    print("\n" + "=" * 100)
    print("CONCISE LATEX TABLE (S-Learner vs DoFM only)")
    print("=" * 100)
    
    # Methods to include (subset) - DoFM Full Graph last
    concise_methods = ["S-Learner", "DoFM (Unknown Graph)", "DoFM (Full Graph)"]
    
    # Collect all results
    results = {}  # {display_name: {(dataset, metric): (mean, stderr)}}
    best_values = {(ds, m): float('inf') for ds in DATASETS for m in METRICS}
    
    for dataset in DATASETS:
        for folder in all_folders:
            if not is_method_selected(folder, dataset):
                continue
            
            method_name = folder
            for ds in [d.lower() for d in DATASETS]:
                if folder.startswith(ds + "_"):
                    method_name = folder[len(ds)+1:]
                    break
            
            display_name = get_display_name(method_name)
            
            # Only include our target methods
            if display_name not in concise_methods:
                continue
            
            if display_name not in results:
                results[display_name] = {}
            
            for metric in METRICS:
                mask = (df['exp_folder'] == folder) & (df['dataset'] == dataset)
                values = df.loc[mask, metric].dropna().values
                
                if len(values) == 0:
                    continue
                
                mean, _, stderr = compute_stats(values)
                results[display_name][(dataset, metric)] = (mean, stderr)
                
                if not np.isnan(mean) and mean < best_values[(dataset, metric)]:
                    best_values[(dataset, metric)] = mean
    
    # Format helper
    def format_val(mean, stderr, is_large=False):
        if is_large:
            val_str = f"{mean:.0f}"
            if not pd.isna(stderr):
                val_str += f"$\\scriptstyle\\pm${stderr:.0f}"
        else:
            val_str = f"{mean:.2f}"
            if not pd.isna(stderr):
                val_str += f"$\\scriptstyle\\pm${stderr:.2f}"
        return val_str
    
    # Generate single compact table
    print()
    print("% Concise table: S-Learner vs DoFM comparison")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Comparison of S-Learner and DoFM variants on benchmark datasets.}")
    print(r"\label{tab:slearner_dofm_comparison}")
    print(r"\small")
    print(r"\begin{tabular}{@{}l" + "rr" * len(DATASETS) + "@{}}")
    print(r"\toprule")
    
    # Dataset header row
    header1 = ""
    for i, dataset in enumerate(DATASETS):
        header1 += f" & \\multicolumn{{2}}{{c}}{{{dataset}}}"
    header1 += r" \\"
    print(header1)
    
    # Cmidrule for each dataset
    cmidrule = ""
    for i, dataset in enumerate(DATASETS):
        start = 2 + i * 2
        end = start + 1
        cmidrule += f"\\cmidrule(lr){{{start}-{end}}} "
    print(cmidrule)
    
    # Metric header row  
    header2 = "Method"
    for dataset in DATASETS:
        header2 += r" & $\sqrt{\text{PEHE}}$ & $\epsilon_{\text{ATE}}$"
    header2 += r" \\"
    print(header2)
    print(r"\midrule")
    
    # Print rows for each method
    for display_name in concise_methods:
        if display_name not in results:
            continue
        
        row = display_name
        for dataset in DATASETS:
            for metric in METRICS:
                key = (dataset, metric)
                if key in results[display_name]:
                    mean, stderr = results[display_name][key]
                    is_large = (dataset in ["CPS", "PSID"] and metric == "pehe")
                    
                    val_str = format_val(mean, stderr, is_large)
                    
                    # Bold if best among these methods
                    if abs(mean - best_values[key]) < 1e-6:
                        val_str = r"\textbf{" + val_str + "}"
                    
                    row += f" & {val_str}"
                else:
                    row += " & --"
        
        row += r" \\"
        print(row)
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    
    # Also generate an even more compact version - transposed (datasets as rows)
    print()
    print("% Alternative: Datasets as rows (more compact)")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{S-Learner vs DoFM comparison across datasets.}")
    print(r"\label{tab:slearner_dofm_transposed}")
    print(r"\small")
    print(r"\begin{tabular}{@{}lcccccc@{}}")
    print(r"\toprule")
    print(r" & \multicolumn{2}{c}{S-Learner} & \multicolumn{2}{c}{DoFM (Unk.)} & \multicolumn{2}{c}{DoFM (Full)} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    print(r"Dataset & PEHE & $\epsilon_{\text{ATE}}$ & PEHE & $\epsilon_{\text{ATE}}$ & PEHE & $\epsilon_{\text{ATE}}$ \\")
    print(r"\midrule")
    
    for dataset in DATASETS:
        row = dataset
        for display_name in concise_methods:
            for metric in METRICS:
                key = (dataset, metric)
                if display_name in results and key in results[display_name]:
                    mean, stderr = results[display_name][key]
                    is_large = (dataset in ["CPS", "PSID"] and metric == "pehe")
                    
                    # More compact: just mean, no stderr
                    if is_large:
                        val_str = f"{mean:.0f}"
                    else:
                        val_str = f"{mean:.2f}"
                    
                    # Bold if best
                    if abs(mean - best_values[key]) < 1e-6:
                        val_str = r"\textbf{" + val_str + "}"
                    
                    row += f" & {val_str}"
                else:
                    row += " & --"
        
        row += r" \\"
        print(row)
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def aggregate_results():
    """Main function to aggregate filtered results."""
    
    # Get all experiment folders
    all_items = os.listdir(RESULTS_DIR)
    all_folders = []
    
    for item in all_items:
        item_path = os.path.join(RESULTS_DIR, item)
        if os.path.isdir(item_path):
            all_folders.append(item)
    
    all_folders = sorted(all_folders)
    
    print("=" * 100)
    print("FILTERED RESULTS - Selected Methods Only")
    print("=" * 100)
    print("\nStandard Methods (IHDP, ACIC, CPS, PSID):")
    for method in SELECTED_METHODS["standard"]:
        print(f"  - {method}")
    
    print("\nPSID-Specific Methods:")
    for method in SELECTED_METHODS["psid_only"]:
        print(f"  - {method}")
    print()
    
    # Collect all results into a DataFrame
    all_results = []
    
    for folder in all_folders:
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
    
    # Create summary tables for each metric
    for metric in METRICS:
        if metric not in df.columns:
            print(f"Metric '{metric}' not found in results, skipping...")
            continue
        
        print("\n" + "=" * 100)
        print(f"METRIC: {metric.upper()}")
        print("=" * 100)
        
        # Collect filtered results by dataset
        for dataset in DATASETS:
            # Get dataset info
            ds_info = DATASET_INFO.get(dataset, {})
            n_train = ds_info.get("n_train", "?")
            n_test = ds_info.get("n_test", "?")
            n_features = ds_info.get("n_features", "?")
            
            print(f"\n  {dataset} (n_train={n_train}, n_test={n_test}, n_features={n_features}):")
            
            # Get experiments that have data for this dataset and are selected
            valid_exps = []
            for folder in all_folders:
                # Check if this method is selected for this dataset
                if not is_method_selected(folder, dataset):
                    continue
                
                # Filter data for this folder and dataset
                mask = (df['exp_folder'] == folder) & (df['dataset'] == dataset)
                values = df.loc[mask, metric].dropna().values
                
                if len(values) == 0:
                    continue
                
                mean, median, stderr = compute_stats(values)
                count = len(values)
                
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
                
                valid_exps.append((method_name, val_str, count, mean))
            
            # Sort by mean value (ascending for better methods)
            valid_exps.sort(key=lambda x: x[3] if not np.isnan(x[3]) else float('inf'))
            
            # Print results
            if valid_exps:
                for method, val, count, _ in valid_exps:
                    print(f"    {method:50s} {val:20s} (n={count})")
            else:
                print(f"    No selected methods found for {dataset}")
        
        print()
    
    # Save filtered results
    output_path = os.path.join(RESULTS_DIR, "aggregated_results_filtered.csv")
    
    # Create filtered DataFrame
    filtered_rows = []
    for _, row in df.iterrows():
        folder = row['exp_folder']
        dataset = row.get('dataset', '')
        if is_method_selected(folder, dataset):
            filtered_rows.append(row)
    
    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.to_csv(output_path, index=False)
        print(f"\nSaved filtered aggregated data to: {output_path}")
        print(f"  Filtered from {len(df)} to {len(filtered_df)} records")
    else:
        print("\nNo filtered records to save!")
    
    # Generate LaTeX tables
    print("\n" + "=" * 100)
    print("LATEX TABLES FOR PAPER")
    print("=" * 100)
    generate_latex_tables(df, all_folders)
    
    # Generate concise table (S-Learner vs DoFM only)
    generate_concise_latex_table(df, all_folders)
    
    return df


if __name__ == "__main__":
    aggregate_results()
