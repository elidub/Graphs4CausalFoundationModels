#!/usr/bin/env python3
"""
Aggregate and display results from PlantEval3 experiments.
"""

import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd

# Base directory for results
BASE_DIR = Path("/fast/arikreuter/DoPFN_v2/CausalPriorFitting/PlantEval3/PlantsEval")
RESULTS_DIR = BASE_DIR / "results"


def load_experiment_results(exp_name):
    """Load all results from a single experiment."""
    exp_dir = RESULTS_DIR / exp_name
    
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return None
    
    results = []
    result_files = sorted(exp_dir.glob("*"))
    
    for result_file in result_files:
        try:
            with open(result_file, 'rb') as f:
                result = pkl.load(f)
                results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
    
    return results


def compute_statistics(results):
    """Compute statistics from a list of results."""
    if not results:
        return None
    
    mises = [r['mise'] for r in results]
    
    # Check if NLL is available (only for DOFM with BarDistribution)
    nlls = [r.get('nll') for r in results if r.get('nll') is not None]
    has_nll = len(nlls) > 0
    
    stats = {
        'model': results[0]['model'],
        'dataset': results[0]['dataset'],
        'n_realizations': len(results),
        'mise_mean': np.mean(mises),
        'mise_std': np.std(mises),
        'mise_median': np.median(mises),
        'mise_min': np.min(mises),
        'mise_max': np.max(mises),
    }
    
    if has_nll:
        stats['nll_mean'] = np.mean(nlls)
        stats['nll_std'] = np.std(nlls)
        stats['nll_median'] = np.median(nlls)
        stats['nll_min'] = np.min(nlls)
        stats['nll_max'] = np.max(nlls)
        stats['n_nll'] = len(nlls)
    
    return stats


def display_results(experiments):
    """Display results in a formatted table."""
    print("\n" + "="*80)
    print("PlantEval3 Results Summary")
    print("="*80)
    
    all_stats = []
    
    for exp_name in experiments:
        print(f"\nLoading experiment: {exp_name}")
        results = load_experiment_results(exp_name)
        
        if results:
            stats = compute_statistics(results)
            all_stats.append(stats)
            
            print(f"  Model: {stats['model']}")
            print(f"  Dataset: {stats['dataset']}")
            print(f"  Realizations: {stats['n_realizations']}")
            print(f"  MISE: {stats['mise_mean']:.4f} ± {stats['mise_std']:.4f}")
            print(f"  MISE (median): {stats['mise_median']:.4f}")
            print(f"  MISE (range): [{stats['mise_min']:.4f}, {stats['mise_max']:.4f}]")
            
            # Show NLL if available
            if 'nll_mean' in stats:
                print(f"  NLL: {stats['nll_mean']:.4f} ± {stats['nll_std']:.4f} (n={stats['n_nll']})")
                print(f"  NLL (median): {stats['nll_median']:.4f}")
                print(f"  NLL (range): [{stats['nll_min']:.4f}, {stats['nll_max']:.4f}]")
    
    # Create comparison table
    if len(all_stats) > 1:
        print("\n" + "="*80)
        print("Comparison Table")
        print("="*80)
        
        df = pd.DataFrame(all_stats)
        df = df[['model', 'n_realizations', 'mise_mean', 'mise_std', 'mise_median', 'mise_min', 'mise_max']]
        df = df.sort_values('mise_mean')
        
        print("\n" + df.to_string(index=False))
        
        # Show relative performance
        print("\n" + "="*80)
        print("Relative Performance (lower MISE is better)")
        print("="*80)
        
        best_mise = df['mise_mean'].min()
        df_relative = df.copy()
        df_relative['relative_error'] = (df_relative['mise_mean'] / best_mise - 1) * 100
        df_relative = df_relative[['model', 'mise_mean', 'relative_error']]
        df_relative['relative_error'] = df_relative['relative_error'].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")
        
        print("\n" + df_relative.to_string(index=False))
    
    print("\n" + "="*80)


def main():
    """Main function."""
    # Find all experiment directories
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return
    
    experiments = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"Found {len(experiments)} experiment(s):")
    for exp in experiments:
        print(f"  - {exp}")
    
    display_results(experiments)


if __name__ == "__main__":
    main()
