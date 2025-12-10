"""
run_benchmark.py

Simple runner that:
- loads a list of OpenML tasks using the SimpleOpenMLLoader
- for each dataset runs simple sklearn baselines (LinearRegression, RandomForest)
- runs the SimplePFN inference (using the sklearn-like wrapper) when possible
- reports RMSE per method and saves a CSV summary

Enhanced with BarDistribution support:
- Automatically detects BarDistribution configuration from YAML config files
- ALWAYS loads BarDistribution parameters from trained PFN model checkpoints
- Raises error if BarDistribution is expected but not found in checkpoint
- Supports both standard MSE and probabilistic BarDistribution models seamlessly

This is intentionally simple and meant for small-scale testing.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

# Fix OpenBLAS threading and memory issues BEFORE importing numpy/scipy heavy libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'

# ensure repo root is on sys.path so local imports work whether run as module or script
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.benchmarking.Benchmark import Benchmark  # noqa: E402
from src.benchmarking.load_openml_benchmark import DEFAULT_TABULAR_NUM_REG_TASKS  # noqa: E402


# we'll record MSE and R^2 for each model


def to_numpy_arrays(d: dict):
    # Kept for backward compatibility; not used in the new Benchmark runner
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
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    return X_train, y_train, X_test, y_test


def main(args):
    # Ensure offline/cache-only behavior unless explicitly overridden
    import os as _os
    _os.environ.setdefault("OPENML_OFFLINE", "1")
    _os.environ.setdefault("DATA_CACHE_ONLY", "1")
    if args.data_dir:
        _os.environ.setdefault("DATA_CACHE_DIR", str(args.data_dir))
    
    # Resolve config and checkpoint with safe defaults
    cfg_path = args.config or str(repo_root / "experiments/FirstTests/configs/early_test2.yaml")
    ckpt = args.checkpoint
    if not ckpt:
        # Try to auto-detect a checkpoint in experiments/FirstTests/checkpoints
        ckpt_dir = repo_root / "experiments/FirstTests/checkpoints"
        if ckpt_dir.exists():
            pts = sorted(ckpt_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pts:
                ckpt = str(pts[0])
    if ckpt and not Path(ckpt).exists():
        print(f"Warning: checkpoint path not found: {ckpt}. PFN will be skipped.")
        ckpt = None

    # Prepare task list
    tasks = [int(t) for t in args.tasks.split(",")] if args.tasks else None

    # Create Benchmark instance with ALL configuration
    bench = Benchmark(
        data_dir=args.data_dir,
        device=args.device,
        verbose=not args.quiet,
        # benchmark configuration
        tasks=tasks,
        max_tasks=args.max_tasks,
        # subsampling
        n_features=int(getattr(args, "n_features", 0) or 0),
        max_n_features=int(getattr(args, "max_n_features", 0) or 0),
        n_train=int(getattr(args, "n_train", 0) or 0),
        max_n_train=int(getattr(args, "max_n_train", 0) or 0),
        n_test=int(getattr(args, "n_test", 0) or 0),
        max_n_test=int(getattr(args, "max_n_test", 0) or 0),
        prefer_numeric=bool(getattr(args, "prefer_numeric", True)),
        only_numeric=bool(getattr(args, "only_numeric", False)),
        # model / output config
        config_path=cfg_path,
        output_csv=args.output,
        bootstrap_samples=int(getattr(args, "bootstrap_samples", 1000) or 1000),
        # SimplePFN ensemble parameters
        n_estimators=int(getattr(args, "n_estimators", 1) or 1),
        norm_methods=getattr(args, "norm_methods", None),
        outlier_strategies=getattr(args, "outlier_strategies", None),
        # preprocessing configuration
        negative_one_one_scaling=bool(getattr(args, "negative_one_one_scaling", True)),
        standardize=bool(getattr(args, "standardize", True)),
        yeo_johnson=bool(getattr(args, "yeo_johnson", False)),
        remove_outliers=bool(getattr(args, "remove_outliers", True)),
        outlier_quantile=float(getattr(args, "outlier_quantile", 0.90)),
        shuffle_samples=bool(getattr(args, "shuffle_samples", True)),
        shuffle_features=bool(getattr(args, "shuffle_features", True)),
        # logging
        quiet=args.quiet,
    )

    # Run benchmark with user-specified fidelity
    df = bench.run(
        fidelity=args.fidelity,
        checkpoint_path=ckpt,
    )

    # Print summary metrics robustly (ignore summary row if present, handle missing columns)
    df_metrics = df[df["process_id"] != "summary"] if "process_id" in df.columns else df
    mse_cols = [c for c in ["mse_lr", "mse_rf", "mse_pfn"] if c in df_metrics.columns]
    r2_cols = [c for c in ["r2_lr", "r2_rf", "r2_pfn"] if c in df_metrics.columns]
    if not df_metrics.empty and mse_cols and r2_cols:
        median_mse = df_metrics[mse_cols].median(numeric_only=True)
        mean_mse = df_metrics[mse_cols].mean(numeric_only=True)
        median_r2 = df_metrics[r2_cols].median(numeric_only=True)
        mean_r2 = df_metrics[r2_cols].mean(numeric_only=True)
        print("\nMedian MSE:\n", median_mse)
        print("\nMean MSE:\n", mean_mse)
        print("\nMedian R²:\n", median_r2)
        print("\nMean R²:\n", mean_r2)
    else:
        print("\nNo per-task metric rows available to summarize (0 successful tasks or missing metric columns).")

    # Print detailed per-model summaries with confidence intervals and average ranks
    if "process_id" in df.columns and (df["process_id"] == "summary_model").any():
        df_sm = df[df["process_id"] == "summary_model"]
        print("\nDetailed per-model summaries (95% CIs, IQR, std, avg ranks):")
        # Order by model then metric for stable output
        for _, row in df_sm.sort_values(["model", "metric"]).iterrows():
            model = row.get("model", "?")
            metric = row.get("metric", "?")
            mean = row.get("mean", np.nan)
            median = row.get("median", np.nan)
            std = row.get("std", np.nan)
            iqr = row.get("iqr", np.nan)
            ci_mean_low = row.get("ci95_mean_low", np.nan)
            ci_mean_high = row.get("ci95_mean_high", np.nan)
            ci_med_low = row.get("ci95_median_low", np.nan)
            ci_med_high = row.get("ci95_median_high", np.nan)
            avg_rank_mse = row.get("avg_rank_mse", np.nan)
            avg_rank_r2 = row.get("avg_rank_r2", np.nan)
            print(
                f"  - {model} [{metric}] | mean={mean:.4f} (95% CI [{ci_mean_low:.4f}, {ci_mean_high:.4f}]), "
                f"median={median:.4f} (95% CI [{ci_med_low:.4f}, {ci_med_high:.4f}]), std={std:.4f}, iqr={iqr:.4f}, "
                f"avg_rank_mse={avg_rank_mse:.2f}, avg_rank_r2={avg_rank_r2:.2f}"
            )
    else:
        print("\nNo detailed model summaries available (no 'summary_model' rows).")


if __name__ == "__main__":
    # Configuration (ALL_CAPS) - edit these constants instead of using CLI args
    TASKS = ""  # comma-separated task ids, e.g. "361072,361073" or empty to use defaults
    MAX_TASKS = 20
    DATA_DIR = str(repo_root / "data_cache")  # Use absolute path to data_cache
    CONFIG = str(repo_root / "experiments/FirstTests/configs/basic.yaml")
    #CHECKPOINT = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16631563.0/step_46000.pt"  # Leave empty to auto-detect or skip PFN
    CHECKPOINT = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16654670.0/best_model.pt"  # Leave empty to auto-detect or skip PFN
    DEVICE = "cpu"
    OUTPUT = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/src/benchmarking/results/benchmark_results.csv"  # Process ID will be automatically added: benchmark_results_pid12345.csv
    NO_TARGET_ENCODING = True
    QUIET = False

    # Build a simple args object similar to argparse.Namespace
    from types import SimpleNamespace

    # Subsampling env vars (optional) - read from ALL_CAPS environment variables so submit files can set them
    N_FEATURES = 50
    MAX_N_FEATURES = 50
    N_TRAIN = 1000
    MAX_N_TRAIN = 1000
    N_TEST = 1000
    MAX_N_TEST = 1000 
    PREFER_NUMERIC = False
    ONLY_NUMERIC = False
    FIDELITY = "low_all_baselines"  # Options: "minimal", "low", "high", "very_high"
    BOOTSTRAP_SAMPLES = 10000
    
    # SimplePFN Ensemble parameters
    N_ESTIMATORS = 1  # Number of ensemble members (1 = no ensemble)
    NORM_METHODS = ["none"]  # Normalization methods
    #OUTLIER_STRATEGIES = ["none", "moderate", "aggressive"]  # Outlier removal strategies
    OUTLIER_STRATEGIES = ["none"]
    
    # Preprocessing configuration
    NEGATIVE_ONE_ONE_SCALING = True  # Scale data to [-1, 1] range
    STANDARDIZE = True  # Standardize features (z-score normalization)
    YEO_JOHNSON = False  # Apply Yeo-Johnson transformation
    REMOVE_OUTLIERS = True  # Remove outliers based on quantile
    OUTLIER_QUANTILE = 0.99  # Quantile threshold for outlier removal
    SHUFFLE_SAMPLES = True  # Shuffle samples during preprocessing
    SHUFFLE_FEATURES = True  # Shuffle features during preprocessing

    args = SimpleNamespace(
        tasks=TASKS,
        max_tasks=MAX_TASKS,
        data_dir=DATA_DIR,
        config=CONFIG,
        checkpoint=CHECKPOINT,
        device=DEVICE,
        output=OUTPUT,
        no_target_encoding=NO_TARGET_ENCODING,
        quiet=QUIET,
        # subsampling parameters (empty string or ints)
        n_features=int(N_FEATURES) if N_FEATURES else 0,
        max_n_features=int(MAX_N_FEATURES) if MAX_N_FEATURES else 0,
        n_train=int(N_TRAIN) if N_TRAIN else 0,
        n_test=int(N_TEST) if N_TEST else 0,
        prefer_numeric=PREFER_NUMERIC,
        only_numeric=ONLY_NUMERIC,
        max_n_train=int(MAX_N_TRAIN) if MAX_N_TRAIN else 0,
        max_n_test=int(MAX_N_TEST) if MAX_N_TEST else 0,
        fidelity=FIDELITY,
        bootstrap_samples=int(BOOTSTRAP_SAMPLES) if BOOTSTRAP_SAMPLES else 1000,
        # SimplePFN ensemble parameters
        n_estimators=int(N_ESTIMATORS) if N_ESTIMATORS else 1,
        norm_methods=NORM_METHODS,
        outlier_strategies=OUTLIER_STRATEGIES,
        # preprocessing configuration
        negative_one_one_scaling=NEGATIVE_ONE_ONE_SCALING,
        standardize=STANDARDIZE,
        yeo_johnson=YEO_JOHNSON,
        remove_outliers=REMOVE_OUTLIERS,
        outlier_quantile=OUTLIER_QUANTILE,
        shuffle_samples=SHUFFLE_SAMPLES,
        shuffle_features=SHUFFLE_FEATURES,
    )

    main(args)
