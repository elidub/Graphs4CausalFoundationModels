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
import sys
import os
from pathlib import Path
import argparse
import json
import math
import numpy as np
import pandas as pd
import time
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ensure repo root is on sys.path so local imports work whether run as module or script
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from src.benchmarking.load_openml_benchmark import SimpleOpenMLLoader, DEFAULT_TABULAR_NUM_REG_TASKS
from src.models.SimplePFN_sklearn import SimplePFNSklearn


# we'll record MSE and R^2 for each model


def to_numpy_arrays(d: dict):
    # d expected to have X_train, y_train, X_test, y_test possibly as DataFrames/Series
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

    # ensure shapes: y as 1d
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    return X_train, y_train, X_test, y_test


def main(args):
    loader = SimpleOpenMLLoader(
        data_dir=args.data_dir,
        verbose=not args.quiet,
        only_numeric=getattr(args, "only_numeric", False),
        #transformation_type = "yeo_johnson"
    )

    def _serializable(obj):
        # small helper to make common types JSON serializable for logging
        try:
            if isinstance(obj, (list, dict, str, int, float, bool)) or obj is None:
                return obj
            if hasattr(obj, "isoformat"):
                return str(obj)
            if hasattr(obj, "__dict__"):
                return {k: _serializable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
            return str(obj)
        except Exception:
            return str(obj)

    if not args.quiet:
        print("[run_benchmark] Script args:", json.dumps(_serializable(args), indent=2))
        print("[run_benchmark] Loader init args:", json.dumps(_serializable(loader), indent=2))

    # compute the list of tasks to run
    if args.tasks:
        tasks = [int(t) for t in args.tasks.split(",")]
    else:
        tasks = DEFAULT_TABULAR_NUM_REG_TASKS[: args.max_tasks]

    # If user requested fixed subsampled datasets, create them once for all tasks
    results = []

    subsample_mode = bool(getattr(args, "n_train", 0) and getattr(args, "n_test", 0) and getattr(args, "n_features", 0))
    if subsample_mode:
        if not args.quiet:
            print(f"Creating subsampled datasets: n_features={args.n_features}, n_train={args.n_train}, n_test={args.n_test}")
        # create subsampled datasets across the requested tasks
        data_map = loader.create_subsampled_datasets(
            tasks,
            n_features=int(args.n_features),
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            prefer_numeric=bool(getattr(args, "prefer_numeric", True)),
            save=True,
        )
    else:
        # default: load tasks normally (download & preprocess with train/test split)
        data_map = loader.load_tasks(tasks)

    # Read model config once
    cfg_path = args.config or str(repo_root / "experiments/FirstTests/configs/early_test.yaml")
    ckpt = args.checkpoint or str(repo_root / "experiments/FirstTests/checkpoints/early_test1_32bs/step_100000.pt")

    for tid in tasks:
        if not args.quiet:
            print(f"\n=== Processing task {tid} ===")
        try:
            if tid not in data_map or not data_map.get(tid):
                print(f"Skipping task {tid}: no processed data")
                continue
            d = data_map[tid]

            X_train, y_train, X_test, y_test = to_numpy_arrays(d)

            # Apply feature selection (N_FEATURES) then pad/truncate to MAX_N_FEATURES if requested
            requested_n = int(getattr(args, "n_features", 0) or 0)
            max_n = int(getattr(args, "max_n_features", 0) or 0)

            # If a requested_n is provided and smaller than available, trim to that many features
            if requested_n and requested_n > 0:
                if X_train.shape[1] >= requested_n:
                    X_train = X_train[:, :requested_n]
                    X_test = X_test[:, :requested_n]
                else:
                    if not args.quiet:
                        print(f"Requested n_features={requested_n} > available {X_train.shape[1]}; using available features")

            # Apply max padding/truncation
            if max_n and max_n > 0:
                cur_n = X_train.shape[1]
                if cur_n > max_n:
                    # truncate to max_n
                    X_train = X_train[:, :max_n]
                    X_test = X_test[:, :max_n]
                elif cur_n < max_n:
                    # pad with zeros on the right
                    pad_train = np.zeros((X_train.shape[0], max_n - cur_n), dtype=X_train.dtype)
                    pad_test = np.zeros((X_test.shape[0], max_n - cur_n), dtype=X_test.dtype)
                    X_train = np.concatenate([X_train, pad_train], axis=1)
                    X_test = np.concatenate([X_test, pad_test], axis=1)


            if X_train.shape[0] < 2 or X_test.shape[0] < 1:
                print(f"Skipping task {tid}: too few samples")
                continue

            # Baseline: LinearRegression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            mse_lr = float(mean_squared_error(y_test, y_pred_lr))
            r2_lr = float(r2_score(y_test, y_pred_lr))

            # Baseline: RandomForest (small)
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            mse_rf = float(mean_squared_error(y_test, y_pred_rf))
            r2_rf = float(r2_score(y_test, y_pred_rf))

            # SimplePFN: try to build wrapper with model num_features set to dataset features
            mse_pfn = None
            r2_pfn = None
            try:
                # read config model defaults
                import yaml

                with open(cfg_path, "r") as f:
                    cfg = yaml.safe_load(f)
                model_cfg = cfg.get("model_config", {}) if isinstance(cfg, dict) else {}

                # Check if BarDistribution is enabled in the config
                use_bar_distribution = model_cfg.get("use_bar_distribution", {}).get("value", False)

                # prepare wrapper with config path so it can detect BarDistribution settings
                pfn = SimplePFNSklearn(config_path=cfg_path, 
                                        checkpoint_path=ckpt, 
                                        device=args.device, 
                                        verbose=not args.quiet,)
                
                # Override num_features to match dataset and read other parameters from config
                mkwargs = {
                    "num_features": int(getattr(args, "max_n_features", X_train.shape[1]) or X_train.shape[1]),
                }
                
                # Only override if the config doesn't specify these values or they need adjustment
                if not model_cfg.get("d_model"):
                    mkwargs["d_model"] = 256
                if not model_cfg.get("depth"):  
                    mkwargs["depth"] = 8
                if not model_cfg.get("heads_feat"):
                    mkwargs["heads_feat"] = 8
                if not model_cfg.get("heads_samp"):
                    mkwargs["heads_samp"] = 8
                if not model_cfg.get("dropout"):
                    mkwargs["dropout"] = 0.0

                if not args.quiet:
                    print("[run_benchmark] PFN wrapper override model_kwargs:", json.dumps(mkwargs, indent=2))
                    if use_bar_distribution:
                        print("[run_benchmark] BarDistribution detected in config")
                
                pfn.load(override_kwargs=mkwargs)
                
                # BarDistribution should always be loaded from the trained PFN checkpoint
                if use_bar_distribution and pfn.bar_distribution is not None:
                    # Check if BarDistribution was properly loaded from checkpoint
                    if (pfn.bar_distribution.centers is None or 
                        pfn.bar_distribution.edges is None or 
                        pfn.bar_distribution.widths is None):
                        if not args.quiet:
                            print("[run_benchmark] ERROR: BarDistribution not found in checkpoint!")
                            print("[run_benchmark] BarDistribution should always be loaded from trained PFN.")
                        raise ValueError("BarDistribution parameters missing from checkpoint. Cannot proceed without trained BarDistribution.")
                    else:
                        if not args.quiet:
                            print("[run_benchmark] BarDistribution parameters loaded from checkpoint")
                elif use_bar_distribution and pfn.bar_distribution is None:
                    if not args.quiet:
                        print("[run_benchmark] ERROR: BarDistribution expected but not found in model!")
                    raise ValueError("BarDistribution expected from config but not found in loaded model.")

                y_pred_pfn = pfn.predict(X_train, y_train, X_test)
                # pfn.predict returns (M,) for batch 1
                mse_pfn = float(mean_squared_error(y_test, y_pred_pfn))
                r2_pfn = float(r2_score(y_test, y_pred_pfn))
            except Exception as e:
                print(f"SimplePFN failed for task {tid}: {e}")

            results.append({
                "process_id": os.getpid(),  # Add process ID to track which run produced these results
                "timestamp": datetime.now().isoformat(),  # Add timestamp for run tracking
                "checkpoint_path": os.path.basename(ckpt),  # Track which model checkpoint was used
                "task_id": int(tid),
                "dataset_shape": d.get("data_shape"),
                "num_features": X_train.shape[1],
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "mse_lr": mse_lr,
                "mse_rf": mse_rf,
                "mse_pfn": mse_pfn,
                "r2_lr": r2_lr,
                "r2_rf": r2_rf,
                "r2_pfn": r2_pfn,
            })

            if not args.quiet:
                print(f"Task {tid} — MSE LR: {mse_lr:.4f}, RF: {mse_rf:.4f}, PFN: {mse_pfn}; R2 LR: {r2_lr:.4f}, RF: {r2_rf:.4f}, PFN: {r2_pfn}")

        except Exception as e:
            print(f"Error processing task {tid}: {e}")

    df = pd.DataFrame(results)
    
    # Calculate median and mean overall performance
    median_mse = df[['mse_lr', 'mse_rf', 'mse_pfn']].median()
    mean_mse = df[['mse_lr', 'mse_rf', 'mse_pfn']].mean()
    median_r2 = df[['r2_lr', 'r2_rf', 'r2_pfn']].median()
    mean_r2 = df[['r2_lr', 'r2_rf', 'r2_pfn']].mean()

    # Append these statistics to the DataFrame
    summary_stats = {
        'process_id': 'summary',
        'timestamp': datetime.now().isoformat(),
        'checkpoint_path': 'N/A',
        'task_id': 'N/A',
        'dataset_shape': 'N/A',
        'num_features': 'N/A',
        'n_train': 'N/A',
        'n_test': 'N/A',
        'mse_lr': median_mse['mse_lr'],
        'mse_rf': median_mse['mse_rf'],
        'mse_pfn': median_mse['mse_pfn'],
        'r2_lr': median_r2['r2_lr'],
        'r2_rf': median_r2['r2_rf'],
        'r2_pfn': median_r2['r2_pfn'],
    }
    df = pd.concat([df, pd.DataFrame([summary_stats])], ignore_index=True)

    # Print the statistics
    print("\nMedian MSE:")
    print(median_mse)
    print("\nMean MSE:")
    print(mean_mse)
    print("\nMedian R²:")
    print(median_r2)
    print("\nMean R²:")
    print(mean_r2)

    # Generate output filename with process ID
    process_id = os.getpid()
    out_file_path = Path(args.output)
    
    # Insert process ID into filename before extension
    if out_file_path.suffix:
        # e.g., "benchmark_results.csv" -> "benchmark_results_pid12345.csv"
        stem = out_file_path.stem
        suffix = out_file_path.suffix
        parent = out_file_path.parent
        out_file = parent / f"{stem}_pid{process_id}{suffix}"
    else:
        # No extension, just append process ID
        out_file = Path(f"{args.output}_pid{process_id}")
    
    df.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file} (PID: {process_id})")


if __name__ == "__main__":
    # Configuration (ALL_CAPS) - edit these constants instead of using CLI args
    TASKS = ""  # comma-separated task ids, e.g. "361072,361073" or empty to use defaults
    MAX_TASKS = 20
    DATA_DIR = "data_cache"
    CONFIG = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/configs/early_test.yaml"
    CHECKPOINT = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/simple_pfn_16423778/final_model_with_bardist.pt"
    DEVICE = "cuda"
    OUTPUT = "benchmark_results.csv"  # Process ID will be automatically added: benchmark_results_pid12345.csv
    NO_TARGET_ENCODING = False
    QUIET = False

    # Build a simple args object similar to argparse.Namespace
    from types import SimpleNamespace

    # Subsampling env vars (optional) - read from ALL_CAPS environment variables so submit files can set them
    N_FEATURES = 19
    MAX_N_FEATURES = 19
    N_TRAIN = 125
    N_TEST = 125
    PREFER_NUMERIC = True
    ONLY_NUMERIC = True

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
    )

    main(args)
