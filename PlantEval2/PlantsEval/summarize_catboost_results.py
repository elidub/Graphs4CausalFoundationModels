#!/usr/bin/env python3
"""
Quick summary of CatBoost results on the new datasets with graph information.
"""

import pickle as pkl
import numpy as np
from pathlib import Path


def summarize_results(exp_folder):
    """Summarize results from an experiment folder."""
    results_dir = Path('results') / exp_folder
    
    if not results_dir.exists():
        print(f"Folder not found: {results_dir}")
        return None
    
    # Load all result files
    mises = []
    for result_file in sorted(results_dir.glob('*')):
        if result_file.is_file():
            with open(result_file, 'rb') as f:
                result = pkl.load(f)
                mises.append(result['mise'])
    
    if not mises:
        print(f"No results found in {exp_folder}")
        return None
    
    mises = np.array(mises)
    mean = np.mean(mises)
    stderr = np.std(mises, ddof=1) / np.sqrt(len(mises))
    median = np.median(mises)
    
    return {
        'n': len(mises),
        'mean': mean,
        'stderr': stderr,
        'median': median,
        'min': np.min(mises),
        'max': np.max(mises),
        'mises': mises
    }


def main():
    """Summarize all CatBoost results."""
    print("=" * 80)
    print("CatBoost Results Summary - PlantEval2 with Graph Information")
    print("=" * 80)
    print()
    
    experiments = [
        'catboost_normal_with_graph',
        'catboost_four_std_with_graph',
    ]
    
    results = {}
    for exp in experiments:
        print(f"Loading: {exp}")
        results[exp] = summarize_results(exp)
        print()
    
    # Display summary
    print("=" * 80)
    print("MISE Results Summary")
    print("=" * 80)
    print()
    
    for exp in experiments:
        if results[exp] is None:
            continue
        
        r = results[exp]
        dataset_name = exp.replace('catboost_', '').replace('_with_graph', '')
        
        print(f"{dataset_name}:")
        print(f"  Mean MISE:   {r['mean']:.4f} ± {r['stderr']:.4f}")
        print(f"  Median MISE: {r['median']:.4f}")
        print(f"  Range:       [{r['min']:.4f}, {r['max']:.4f}]")
        print(f"  N:           {r['n']}")
        print()
    
    # Individual realization results
    print("=" * 80)
    print("Individual Realization MISE Values")
    print("=" * 80)
    print()
    
    for exp in experiments:
        if results[exp] is None:
            continue
        
        dataset_name = exp.replace('catboost_', '').replace('_with_graph', '')
        r = results[exp]
        
        print(f"{dataset_name}:")
        for i, mise in enumerate(r['mises']):
            print(f"  Realization {i}: {mise:.4f}")
        print()


if __name__ == "__main__":
    main()
