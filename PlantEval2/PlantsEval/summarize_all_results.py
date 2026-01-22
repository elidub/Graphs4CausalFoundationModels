#!/usr/bin/env python3
"""
Comprehensive summary of all results in PlantEval2.
"""

import pickle as pkl
import numpy as np
from pathlib import Path


def summarize_experiment(exp_folder):
    """Summarize results from an experiment folder."""
    results_dir = Path('results') / exp_folder
    
    if not results_dir.exists():
        return None
    
    # Load all result files
    mises = []
    for result_file in sorted(results_dir.glob('*')):
        if result_file.is_file():
            try:
                with open(result_file, 'rb') as f:
                    result = pkl.load(f)
                    mises.append(result['mise'])
            except:
                continue
    
    if not mises:
        return None
    
    mises = np.array(mises)
    mean = np.mean(mises)
    stderr = np.std(mises, ddof=1) / np.sqrt(len(mises))
    
    return {
        'n': len(mises),
        'mean': mean,
        'stderr': stderr,
        'mises': mises
    }


def main():
    """Summarize all experiment results."""
    print("=" * 80)
    print("PlantEval2 Results Summary - All Experiments")
    print("=" * 80)
    print()
    
    # Find all experiment folders
    results_dir = Path('results')
    exp_folders = [f.name for f in results_dir.iterdir() if f.is_dir()]
    exp_folders = sorted(exp_folders)
    
    print(f"Found {len(exp_folders)} experiment folders:")
    for folder in exp_folders:
        print(f"  - {folder}")
    print()
    
    # Load all results
    results = {}
    for folder in exp_folders:
        results[folder] = summarize_experiment(folder)
    
    # Group by dataset
    print("=" * 80)
    print("Results by Dataset")
    print("=" * 80)
    print()
    
    # Normal dataset
    print("CID_10_ints_10_reals_normal:")
    print("-" * 80)
    normal_exps = [k for k in exp_folders if 'normal' in k]
    for exp in sorted(normal_exps):
        if results[exp] is not None:
            r = results[exp]
            model_name = exp.replace('_normal_with_graph', '').replace('_with_graph', '')
            print(f"  {model_name:30s}: {r['mean']:6.4f} ± {r['stderr']:.4f}  (n={r['n']})")
    print()
    
    # Four_std_uniform dataset
    print("CID_10_ints_10_reals_four_std_uniform:")
    print("-" * 80)
    four_std_exps = [k for k in exp_folders if 'four_std' in k]
    for exp in sorted(four_std_exps):
        if results[exp] is not None:
            r = results[exp]
            model_name = exp.replace('_four_std_with_graph', '').replace('_with_graph', '')
            print(f"  {model_name:30s}: {r['mean']:6.4f} ± {r['stderr']:.4f}  (n={r['n']})")
    print()
    
    # Comparison table
    print("=" * 80)
    print("Model Comparison (Mean MISE ± Std Error)")
    print("=" * 80)
    print()
    print(f"{'Model':<15s} {'Normal Dataset':>20s} {'Four_Std Dataset':>20s}")
    print("-" * 80)
    
    # Extract model names
    models = set()
    for exp in exp_folders:
        if 'normal_with_graph' in exp:
            model = exp.replace('_normal_with_graph', '')
            models.add(model)
        elif 'four_std_with_graph' in exp:
            model = exp.replace('_four_std_with_graph', '')
            models.add(model)
    
    for model in sorted(models):
        normal_exp = f"{model}_normal_with_graph"
        four_std_exp = f"{model}_four_std_with_graph"
        
        normal_str = "N/A"
        four_std_str = "N/A"
        
        if normal_exp in results and results[normal_exp] is not None:
            r = results[normal_exp]
            normal_str = f"{r['mean']:.4f} ± {r['stderr']:.4f}"
        
        if four_std_exp in results and results[four_std_exp] is not None:
            r = results[four_std_exp]
            four_std_str = f"{r['mean']:.4f} ± {r['stderr']:.4f}"
        
        print(f"{model:<15s} {normal_str:>20s} {four_std_str:>20s}")
    
    print()
    print("=" * 80)
    print("Summary Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
