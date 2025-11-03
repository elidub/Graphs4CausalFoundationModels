"""
ablate_preprocessing.py

Ablation study script for SimplePFN preprocessing variants.

This script systematically tests different normalization methods and outlier strategies
by running the benchmark multiple times with different preprocessing configurations.
All other settings remain constant.

Each run tests:
- Individual normalization methods: 'none', 'power', 'quantile', 'robust'
- Individual outlier strategies: 'none', 'conservative', 'moderate', 'aggressive'
- Single model (n_estimators=1) to isolate preprocessing effects

Results are saved to separate CSV files for each configuration, then aggregated
into a summary comparing all preprocessing variants.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from types import SimpleNamespace
from tqdm import tqdm

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.benchmarking.Benchmark import Benchmark
from src.benchmarking.load_openml_benchmark import DEFAULT_TABULAR_NUM_REG_TASKS


def run_ablation_study(
    # Fixed settings (keep these constant across all runs)
    tasks=None,
    max_tasks=5,
    data_dir="data_cache",
    config_path=None,
    checkpoint_path=None,
    device="cpu",
    n_features=19,
    max_n_features=19,
    n_train=100,
    max_n_train=100,
    n_test=100,
    max_n_test=100,
    prefer_numeric=False,
    only_numeric=False,
    repeats=3,
    baseline_set="basic",
    bootstrap_samples=10000,
    quiet=False,
    # Ablation settings
    norm_methods_to_test=None,
    outlier_strategies_to_test=None,
    output_dir="ablation_results",
):
    """
    Run ablation study over preprocessing variants.
    
    Args:
        All standard benchmark parameters (fixed across runs)
        norm_methods_to_test: List of normalization methods to test individually
        outlier_strategies_to_test: List of outlier strategies to test individually
        output_dir: Directory to save results
    """
    
    # Set defaults
    if norm_methods_to_test is None:
        norm_methods_to_test = ["none", "power", "quantile", "robust"]
    if outlier_strategies_to_test is None:
        outlier_strategies_to_test = ["none", "conservative", "moderate", "aggressive"]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Setup environment
    import os
    os.environ.setdefault("OPENML_OFFLINE", "1")
    os.environ.setdefault("DATA_CACHE_ONLY", "1")
    if data_dir:
        os.environ.setdefault("DATA_CACHE_DIR", str(data_dir))
    
    # Resolve config and checkpoint paths
    if config_path is None:
        config_path = str(repo_root / "experiments/FirstTests/configs/early_test2.yaml")
    if checkpoint_path is None:
        checkpoint_path = str(repo_root / "experiments/FirstTests/checkpoints/simple_pfn_16561948/final_model_with_bardist.pt")
    
    # Verify checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"Warning: checkpoint not found: {checkpoint_path}")
        print("PFN will be skipped in all runs.")
        checkpoint_path = None
    
    # Initialize benchmark
    bench = Benchmark(data_dir=data_dir, device=device, verbose=not quiet)
    
    # Resolve task list
    if tasks is None or len(tasks) == 0:
        tasks = DEFAULT_TABULAR_NUM_REG_TASKS[:max_tasks]
    
    all_results = []
    
    # Calculate total number of configurations to test
    total_configs = (
        len(norm_methods_to_test) +  # Phase 1: individual norms
        len(outlier_strategies_to_test) +  # Phase 2: individual outliers
        1  # Phase 3: full ensemble
    )
    
    print("="*80)
    print("ABLATION STUDY: Testing Preprocessing Variants")
    print("="*80)
    print(f"Tasks: {len(tasks)}")
    print(f"Repeats per task: {repeats}")
    print(f"Normalization methods: {norm_methods_to_test}")
    print(f"Outlier strategies: {outlier_strategies_to_test}")
    print(f"Total configurations: {total_configs}")
    print("="*80)
    
    # Create progress bar for overall progress
    pbar = tqdm(total=total_configs, desc="Overall Progress", unit="config")
    
    # Test each normalization method (with outlier_strategy='none')
    print("\n" + "="*80)
    print("PHASE 1: Testing Normalization Methods (outlier_strategy='none')")
    print("="*80)
    
    for norm_method in norm_methods_to_test:
        print(f"\n🔬 Testing normalization: {norm_method}")
        print("-"*80)
        
        output_file = output_path / f"ablation_norm_{norm_method}.csv"
        
        df = bench.run(
            tasks=tasks,
            max_tasks=max_tasks,
            n_features=n_features,
            max_n_features=max_n_features,
            n_train=n_train,
            max_n_train=max_n_train,
            n_test=n_test,
            max_n_test=max_n_test,
            prefer_numeric=prefer_numeric,
            only_numeric=only_numeric,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_csv=str(output_file),
            device=device,
            quiet=quiet,
            repeats=repeats,
            baseline_set=baseline_set,
            bootstrap_samples=bootstrap_samples,
            # Ablation: test single normalization method
            n_estimators=1,  # Single model to isolate preprocessing effect
            norm_methods=[norm_method],
            outlier_strategies=["none"],  # Keep outlier handling constant
        )
        
        # Add configuration metadata
        df_filtered = df[df["process_id"] != "summary"].copy() if "process_id" in df.columns else df.copy()
        df_filtered["ablation_type"] = "normalization"
        df_filtered["norm_method"] = norm_method
        df_filtered["outlier_strategy"] = "none"
        
        all_results.append(df_filtered)
        pbar.update(1)
        print(f"✓ Completed {norm_method} normalization test")
    
    # Test each outlier strategy (with norm_method='none')
    print("\n" + "="*80)
    print("PHASE 2: Testing Outlier Strategies (norm_method='none')")
    print("="*80)
    
    for outlier_strategy in outlier_strategies_to_test:
        print(f"\n🔬 Testing outlier strategy: {outlier_strategy}")
        print("-"*80)
        
        output_file = output_path / f"ablation_outlier_{outlier_strategy}.csv"
        
        df = bench.run(
            tasks=tasks,
            max_tasks=max_tasks,
            n_features=n_features,
            max_n_features=max_n_features,
            n_train=n_train,
            max_n_train=max_n_train,
            n_test=n_test,
            max_n_test=max_n_test,
            prefer_numeric=prefer_numeric,
            only_numeric=only_numeric,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_csv=str(output_file),
            device=device,
            quiet=quiet,
            repeats=repeats,
            baseline_set=baseline_set,
            bootstrap_samples=bootstrap_samples,
            # Ablation: test single outlier strategy
            n_estimators=1,  # Single model to isolate preprocessing effect
            norm_methods=["none"],  # Keep normalization constant
            outlier_strategies=[outlier_strategy],
        )
        
        # Add configuration metadata
        df_filtered = df[df["process_id"] != "summary"].copy() if "process_id" in df.columns else df.copy()
        df_filtered["ablation_type"] = "outlier"
        df_filtered["norm_method"] = "none"
        df_filtered["outlier_strategy"] = outlier_strategy
        
        all_results.append(df_filtered)
        pbar.update(1)
        print(f"✓ Completed {outlier_strategy} outlier strategy test")
    
    # Test full ensemble (all combinations)
    print("\n" + "="*80)
    print("PHASE 3: Testing Full Ensemble (all norm_methods, all outlier_strategies)")
    print("="*80)
    
    print(f"\n🔬 Testing full ensemble with diverse preprocessing")
    print(f"   - Normalization methods: {norm_methods_to_test}")
    print(f"   - Outlier strategies: {outlier_strategies_to_test}")
    print(f"   - Expected ensemble size: ~{len(norm_methods_to_test) * len(outlier_strategies_to_test)} members")
    print("-"*80)
    
    output_file = output_path / f"ablation_ensemble_full.csv"
    
    # Calculate appropriate n_estimators for full diversity
    n_combinations = len(norm_methods_to_test) * len(outlier_strategies_to_test)
    # Use at least as many estimators as combinations to ensure diversity
    ensemble_size = max(n_combinations, 10)
    
    df = bench.run(
        tasks=tasks,
        max_tasks=max_tasks,
        n_features=n_features,
        max_n_features=max_n_features,
        n_train=n_train,
        max_n_train=max_n_train,
        n_test=n_test,
        max_n_test=max_n_test,
        prefer_numeric=prefer_numeric,
        only_numeric=only_numeric,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_csv=str(output_file),
        device=device,
        quiet=quiet,
        repeats=repeats,
        baseline_set=baseline_set,
        bootstrap_samples=bootstrap_samples,
        # Full ensemble: combine all preprocessing variants
        n_estimators=ensemble_size,
        norm_methods=norm_methods_to_test,
        outlier_strategies=outlier_strategies_to_test,
    )
    
    # Add configuration metadata
    df_filtered = df[df["process_id"] != "summary"].copy() if "process_id" in df.columns else df.copy()
    df_filtered["ablation_type"] = "ensemble"
    df_filtered["norm_method"] = "all"  # Indicates all methods used
    df_filtered["outlier_strategy"] = "all"  # Indicates all strategies used
    df_filtered["n_estimators"] = ensemble_size
    
    all_results.append(df_filtered)
    pbar.update(1)
    pbar.close()
    print(f"✓ Completed full ensemble test with {ensemble_size} members")
    
    # Combine all results
    print("\n" + "="*80)
    print("Aggregating Results")
    print("="*80)
    
    df_combined = pd.concat(all_results, ignore_index=True)
    
    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = output_path / f"ablation_combined_{timestamp}.csv"
    df_combined.to_csv(combined_file, index=False)
    print(f"\n✓ Saved combined results to: {combined_file}")
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if "mse_pfn" in df_combined.columns and "r2_pfn" in df_combined.columns:
        # Normalization methods summary
        print("\n📊 NORMALIZATION METHODS COMPARISON (outlier_strategy='none')")
        print("-"*80)
        df_norm = df_combined[df_combined["ablation_type"] == "normalization"]
        norm_summary = df_norm.groupby("norm_method").agg({
            "mse_pfn": ["mean", "median", "std"],
            "r2_pfn": ["mean", "median", "std"]
        }).round(4)
        print(norm_summary)
        
        # Rank by mean RMSE
        norm_ranks = df_norm.groupby("norm_method")["mse_pfn"].mean().sort_values()
        print("\nRankings by MSE (lower is better):")
        for i, (method, mse) in enumerate(norm_ranks.items(), 1):
            rmse = np.sqrt(mse)
            print(f"  {i}. {method:12s}: MSE={mse:.4f}, RMSE={rmse:.4f}")
        
        # Outlier strategies summary
        print("\n📊 OUTLIER STRATEGIES COMPARISON (norm_method='none')")
        print("-"*80)
        df_outlier = df_combined[df_combined["ablation_type"] == "outlier"]
        outlier_summary = df_outlier.groupby("outlier_strategy").agg({
            "mse_pfn": ["mean", "median", "std"],
            "r2_pfn": ["mean", "median", "std"]
        }).round(4)
        print(outlier_summary)
        
        # Rank by mean RMSE
        outlier_ranks = df_outlier.groupby("outlier_strategy")["mse_pfn"].mean().sort_values()
        print("\nRankings by MSE (lower is better):")
        for i, (strategy, mse) in enumerate(outlier_ranks.items(), 1):
            rmse = np.sqrt(mse)
            print(f"  {i}. {strategy:12s}: MSE={mse:.4f}, RMSE={rmse:.4f}")
        
        # Ensemble summary
        print("\n📊 FULL ENSEMBLE COMPARISON (all norm_methods × all outlier_strategies)")
        print("-"*80)
        df_ensemble = df_combined[df_combined["ablation_type"] == "ensemble"]
        if not df_ensemble.empty:
            ensemble_summary = df_ensemble.agg({
                "mse_pfn": ["mean", "median", "std"],
                "r2_pfn": ["mean", "median", "std"]
            }).round(4)
            print(ensemble_summary)
            
            ensemble_mse = df_ensemble["mse_pfn"].mean()
            ensemble_rmse = np.sqrt(ensemble_mse)
            ensemble_r2 = df_ensemble["r2_pfn"].mean()
            ensemble_size = df_ensemble["n_estimators"].iloc[0] if "n_estimators" in df_ensemble.columns else "N/A"
            
            print(f"\nEnsemble Performance:")
            print(f"  - Size: {ensemble_size} members")
            print(f"  - MSE: {ensemble_mse:.4f}")
            print(f"  - RMSE: {ensemble_rmse:.4f}")
            print(f"  - R²: {ensemble_r2:.4f}")
            
            # Compare ensemble to best individual methods
            best_norm_mse = norm_ranks.iloc[0]
            best_outlier_mse = outlier_ranks.iloc[0]
            
            print(f"\nComparison to Best Individual Methods:")
            print(f"  - Best normalization: {norm_ranks.index[0]} (MSE: {best_norm_mse:.4f})")
            print(f"  - Best outlier: {outlier_ranks.index[0]} (MSE: {best_outlier_mse:.4f})")
            print(f"  - Full ensemble: MSE: {ensemble_mse:.4f}")
            
            if ensemble_mse < best_norm_mse and ensemble_mse < best_outlier_mse:
                improvement_norm = ((best_norm_mse - ensemble_mse) / best_norm_mse) * 100
                improvement_outlier = ((best_outlier_mse - ensemble_mse) / best_outlier_mse) * 100
                print(f"\n✨ Ensemble IMPROVES over best individual methods:")
                print(f"  - {improvement_norm:.2f}% better than best normalization")
                print(f"  - {improvement_outlier:.2f}% better than best outlier strategy")
            elif ensemble_mse < best_norm_mse or ensemble_mse < best_outlier_mse:
                print(f"\n✓ Ensemble shows competitive performance")
            else:
                print(f"\n⚠️  Ensemble does not improve over best individual methods")
        
        # Save summary
        summary_file = output_path / f"ablation_summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write("ABLATION STUDY SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write("NORMALIZATION METHODS (outlier_strategy='none')\n")
            f.write("-"*80 + "\n")
            f.write(str(norm_summary) + "\n\n")
            f.write("Rankings by MSE:\n")
            for i, (method, mse) in enumerate(norm_ranks.items(), 1):
                rmse = np.sqrt(mse)
                f.write(f"  {i}. {method:12s}: MSE={mse:.4f}, RMSE={rmse:.4f}\n")
            f.write("\n")
            f.write("OUTLIER STRATEGIES (norm_method='none')\n")
            f.write("-"*80 + "\n")
            f.write(str(outlier_summary) + "\n\n")
            f.write("Rankings by MSE:\n")
            for i, (strategy, mse) in enumerate(outlier_ranks.items(), 1):
                rmse = np.sqrt(mse)
                f.write(f"  {i}. {strategy:12s}: MSE={mse:.4f}, RMSE={rmse:.4f}\n")
            
            # Add ensemble summary to file
            if not df_ensemble.empty:
                f.write("\n")
                f.write("FULL ENSEMBLE (all norm_methods × all outlier_strategies)\n")
                f.write("-"*80 + "\n")
                f.write(str(ensemble_summary) + "\n\n")
                f.write(f"Ensemble size: {ensemble_size} members\n")
                f.write(f"MSE: {ensemble_mse:.4f}\n")
                f.write(f"RMSE: {ensemble_rmse:.4f}\n")
                f.write(f"R²: {ensemble_r2:.4f}\n")
                f.write("\n")
                f.write("Comparison to Best Individual Methods:\n")
                f.write(f"  - Best normalization: {norm_ranks.index[0]} (MSE: {best_norm_mse:.4f})\n")
                f.write(f"  - Best outlier: {outlier_ranks.index[0]} (MSE: {best_outlier_mse:.4f})\n")
                f.write(f"  - Full ensemble: MSE: {ensemble_mse:.4f}\n")
                
                if ensemble_mse < best_norm_mse and ensemble_mse < best_outlier_mse:
                    improvement_norm = ((best_norm_mse - ensemble_mse) / best_norm_mse) * 100
                    improvement_outlier = ((best_outlier_mse - ensemble_mse) / best_outlier_mse) * 100
                    f.write(f"\nEnsemble IMPROVES over best individual methods:\n")
                    f.write(f"  - {improvement_norm:.2f}% better than best normalization\n")
                    f.write(f"  - {improvement_outlier:.2f}% better than best outlier strategy\n")
        
        print(f"\n✓ Saved summary to: {summary_file}")
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"Results directory: {output_path}")
    print(f"Combined results: {combined_file}")
    
    return df_combined


if __name__ == "__main__":
    # Configuration - matches run_benchmark.py but with ablation focus
    
    # Fixed settings (constant across all ablation runs)
    TASKS = None  # Will use first MAX_TASKS from defaults
    MAX_TASKS = 30  # Small number for faster testing
    DATA_DIR = "data_cache"
    CONFIG = str(repo_root / "experiments/FirstTests/configs/early_test2.yaml")
    CHECKPOINT = str(repo_root / "experiments/FirstTests/checkpoints/simple_pfn_16561948/final_model_with_bardist.pt")
    DEVICE = "cpu"
    
    # Data subsampling (keep constant)
    N_FEATURES = 19
    MAX_N_FEATURES = 19
    N_TRAIN = 100
    MAX_N_TRAIN = 100
    N_TEST = 100
    MAX_N_TEST = 100
    PREFER_NUMERIC = False
    ONLY_NUMERIC = False
    
    # Benchmark settings (keep constant)
    REPEATS = 10  # Repeats per task for statistical reliability
    BASELINE_SET = "basic"
    BOOTSTRAP_SAMPLES = 10000
    QUIET = False
    
    # Ablation settings (what we're testing)
    NORM_METHODS_TO_TEST = ["none", "power", "quantile", "robust"]
    OUTLIER_STRATEGIES_TO_TEST = ["none", "conservative", "moderate", "aggressive"]
    OUTPUT_DIR = "ablation_results"
    
    # Run the ablation study
    df_results = run_ablation_study(
        tasks=TASKS,
        max_tasks=MAX_TASKS,
        data_dir=DATA_DIR,
        config_path=CONFIG,
        checkpoint_path=CHECKPOINT,
        device=DEVICE,
        n_features=N_FEATURES,
        max_n_features=MAX_N_FEATURES,
        n_train=N_TRAIN,
        max_n_train=MAX_N_TRAIN,
        n_test=N_TEST,
        max_n_test=MAX_N_TEST,
        prefer_numeric=PREFER_NUMERIC,
        only_numeric=ONLY_NUMERIC,
        repeats=REPEATS,
        baseline_set=BASELINE_SET,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        quiet=QUIET,
        norm_methods_to_test=NORM_METHODS_TO_TEST,
        outlier_strategies_to_test=OUTLIER_STRATEGIES_TO_TEST,
        output_dir=OUTPUT_DIR,
    )
    
    print("\n✨ Done! Check the ablation_results directory for detailed results.")
