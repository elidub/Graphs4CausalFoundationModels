"""
Benchmark runner

- Uses SimpleOpenMLLoader to download raw datasets
- Handles subsampling, preprocessing (via Preprocessor), padding/truncation
- Runs baselines and optional SimplePFN
- Produces a results DataFrame and saves CSV
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))



# Robust imports so this file runs whether called as a script or a module
try:
    # When run via `python -m src.benchmarking.run_benchmark` or sys.path already has repo root
    from load_openml_benchmark import SimpleOpenMLLoader, DEFAULT_TABULAR_NUM_REG_TASKS
except Exception:
    # Fallback: import with package prefix
    from src.benchmarking.load_openml_benchmark import SimpleOpenMLLoader, DEFAULT_TABULAR_NUM_REG_TASKS

try:
    from priordata_processing.Preprocessor import Preprocessor
except Exception:
    # Fallback: import with package prefix
    from src.priordata_processing.Preprocessor import Preprocessor


class Benchmark:
    def __init__(
        self,
        data_dir: str = "data_cache",
        device: str = "cpu",
        verbose: bool = False,
    ):
        # Allow override via environment variable
        env_data_dir = os.environ.get("DATA_CACHE_DIR")
        if env_data_dir:
            data_dir = env_data_dir
            if verbose:
                print(f"[Benchmark] DATA_CACHE_DIR env detected -> {data_dir}")
        if verbose:
            try:
                print(f"[Benchmark] CWD: {os.getcwd()}")
            except Exception:
                pass
        # Robust search for data_cache directory
        data_dir_path = Path(data_dir)
        if not data_dir_path.exists():
            # Try parent directory (in case of job working dir)
            parent_data_dir = Path("../") / data_dir
            if parent_data_dir.exists():
                data_dir_path = parent_data_dir
                data_dir = str(data_dir_path)
        if not Path(data_dir).exists():
            print(f"[Benchmark] ERROR: data_cache directory not found at '{data_dir}'.")
            print("  Searched in current directory and parent.\n"
                  "  Please ensure data_cache.zip is unzipped in the working directory, "
                  "or set DATA_CACHE_DIR environment variable.")
            raise FileNotFoundError(f"data_cache directory not found at '{data_dir}'")
        else:
            if verbose:
                print(f"[Benchmark] Using data_cache directory at: {data_dir}")
        offline = os.environ.get("OPENML_OFFLINE") == "1" or os.environ.get("DATA_CACHE_ONLY") == "1"
        if verbose and offline:
            print("[Benchmark] OPENML_OFFLINE/DATA_CACHE_ONLY detected -> Loader offline mode enabled")
        self.loader = SimpleOpenMLLoader(data_dir=data_dir, verbose=verbose, offline=offline)
        self.device = device
        self.verbose = verbose

    @staticmethod
    def to_numpy_arrays(d: dict):
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

    def _preprocess_df(
        self,
        df: pd.DataFrame,
        target: str,
        n_features: Optional[int] = None,
        max_n_features: Optional[int] = None,
        n_train: Optional[int] = None,
        max_n_train: Optional[int] = None,
        n_test: Optional[int] = None,
        max_n_test: Optional[int] = None,
        only_numeric: bool = False,
        negative_one_one_scaling: bool = True,
        standardize: bool = True,
        yeo_johnson: bool = False,
        remove_outliers: bool = True,
        outlier_quantile: float = 0.90,
        shuffle_samples: bool = True,
        shuffle_features: bool = True,
    ) -> Dict[str, Any]:
        # Split X/y, simple encoding similar to previous loader
        y = df[target]
        X = df.drop(columns=[target])
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if only_numeric and cat_cols:
            X = X[num_cols]
            cat_cols = []
        elif cat_cols:
            from sklearn.preprocessing import LabelEncoder
            for col in cat_cols:
                le = LabelEncoder()
                X[col] = X[col].fillna("missing")
                X[col] = le.fit_transform(X[col])

        # impute numeric
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].mean())

        feature_names = X.columns.tolist()
        X_np = X.to_numpy().astype(np.float32)
        y_np = y.to_numpy().astype(np.float32)

        import torch
        X_tensor = torch.from_numpy(X_np).unsqueeze(0)
        y_tensor = torch.from_numpy(y_np).unsqueeze(0)
        N, F = X_np.shape

        # Determine requested sizes
        req_n_features = min(n_features or F, F)
        req_n_train = min(n_train or (N // 2), N)  # default 50/50
        req_n_test = min(n_test or (N - req_n_train), N - req_n_train)

        pre = Preprocessor(
            n_features=req_n_features,
            max_n_features=max_n_features or req_n_features,
            n_train_samples=req_n_train,
            max_n_train_samples=max_n_train or req_n_train,
            n_test_samples=req_n_test,
            max_n_test_samples=max_n_test or req_n_test,
            negative_one_one_scaling=negative_one_one_scaling,
            standardize=standardize,
            yeo_johnson=yeo_johnson,
            remove_outliers=remove_outliers,
            outlier_quantile=outlier_quantile,
            shuffle_samples=shuffle_samples,
            shuffle_features=shuffle_features,
        )
        result = pre.process(X_tensor, y_tensor)
        if result is None:
            raise ValueError("Preprocessor failed; not enough data")

        X_train_t, X_test_t, y_train_t, y_test_t = result
        X_train = X_train_t[0].numpy()
        X_test = X_test_t[0].numpy()
        y_train = y_train_t[0].numpy()
        y_test = y_test_t[0].numpy()

        # Align column names to match processed feature count exactly
        out_F = X_train.shape[1]
        if len(feature_names) >= out_F:
            feature_names_aligned = feature_names[:out_F]
        else:
            feature_names_aligned = feature_names + [f"padded_feature_{i}" for i in range(len(feature_names), out_F)]
        return {
            "X_train": pd.DataFrame(X_train, columns=feature_names_aligned),
            "X_test": pd.DataFrame(X_test, columns=feature_names_aligned),
            "y_train": pd.Series(y_train, name=target),
            "y_test": pd.Series(y_test, name=target),
            "feature_names": feature_names_aligned,
            "target_column": target,
            "data_shape": df.shape,
            "original_feature_count": len(feature_names),
            "original_sample_count": len(y_np),
        }

    def _get_baseline_models(self, baseline_set: str = "basic") -> Dict[str, Any]:
        models: Dict[str, Any] = {
            "lr": LinearRegression(),
            "rf": RandomForestRegressor(n_estimators=50, random_state=42),
        }
        if baseline_set == "extended":
            # Add a collection of simple, untuned sklearn regressors
            models.update({
                "ridge": Ridge(),
                "lasso": Lasso(max_iter=2000),
                "enet": ElasticNet(max_iter=2000),
                "huber": HuberRegressor(),
                "bayesridge": BayesianRidge(),
                "svr": SVR(),
                "knn": KNeighborsRegressor(n_neighbors=5),
                "dtr": DecisionTreeRegressor(random_state=42),
                "extratrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
                "gbrt": GradientBoostingRegressor(random_state=42),
                "adaboost": AdaBoostRegressor(random_state=42),
                "mlp": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42),
            })
        return models

    @staticmethod
    def _safe_fit_predict(model, X_train, y_train, X_test) -> Optional[np.ndarray]:
        try:
            model.fit(X_train, y_train)
            return model.predict(X_test)
        except Exception:
            return None

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
        if y_pred is None:
            return None, None
        try:
            mse = float(mean_squared_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            return mse, r2
        except Exception:
            return None, None

    @staticmethod
    def _iqr(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        q75, q25 = np.nanpercentile(values, [75, 25])
        return float(q75 - q25)

    @staticmethod
    def _bootstrap_ci(values: np.ndarray, fn, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
        values = values[~np.isnan(values)]
        if values.size == 0:
            return float("nan"), float("nan")
        rng = np.random.default_rng(42)
        stats = []
        n = values.size
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = values[idx]
            stats.append(fn(sample))
        low = float(np.nanpercentile(stats, 100 * (alpha / 2)))
        high = float(np.nanpercentile(stats, 100 * (1 - alpha / 2)))
        return low, high



    def run(
        self,
        tasks: Optional[List[int]] = None,
        max_tasks: int = 20,
        # subsampling
        n_features: int = 0,
        max_n_features: int = 0,
        n_train: int = 0,
        max_n_train: int = 0,
        n_test: int = 0,
        max_n_test: int = 0,
        prefer_numeric: bool = True,
        only_numeric: bool = False,
        # model
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        output_csv: str = "benchmark_results.csv",
        device: Optional[str] = None,
        quiet: bool = False,
        # new features
        repeats: int = 1,
        baseline_set: str = "basic",  # 'basic' or 'extended'
        bootstrap_samples: int = 1000,
        # SimplePFN ensemble parameters
        n_estimators: int = 1,
        norm_methods: Optional[List[str]] = None,
        outlier_strategies: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if tasks is None or len(tasks) == 0:
            tasks = DEFAULT_TABULAR_NUM_REG_TASKS[: max_tasks]
        raw_map = self.loader.load_tasks_raw(tasks)
        # Offline fallback: if nothing could be resolved via tasks (e.g., no mapping and offline),
        # try scanning cached datasets directly and evaluate on those.
        if len(raw_map) == 0 and getattr(self.loader, "offline", False):
            if not quiet:
                print("[Benchmark] No tasks resolved in offline mode; scanning cached datasets ...")
            cached = self.loader.list_cached_dataset_ids()
            if max_tasks and max_tasks > 0:
                cached = cached[: int(max_tasks)]
            raw_map = self.loader.load_datasets_raw_by_cache(cached)
            # Construct pseudo-task list equal to dataset ids for iteration
            tasks = list(sorted(raw_map.keys()))
        results: List[Dict[str, Any]] = []
        # choose baselines
        models = self._get_baseline_models(baseline_set)
        # lazy import of PFN wrapper
        from src.models.SimplePFN_sklearn import SimplePFNSklearn
        use_device = device or self.device

        for tid in tasks:
            if tid not in raw_map:
                if not quiet:
                    print(f"Skipping task {tid}: no raw data")
                continue
            entry = raw_map[tid]
            df = entry["df"]
            target = entry["target_column"]

            # Estimate whether repeated resampling is relevant
            N_total = df.shape[0]
            F_total = df.drop(columns=[target]).shape[1]
            wants_feat = (n_features and F_total > (n_features or 0)) or (max_n_features and F_total > (max_n_features or 0))
            wants_samp = (
                (n_train and N_total > (n_train or 0))
                or (n_test and N_total > ((n_train or 0) + (n_test or 0)))
                or (max_n_train and N_total > (max_n_train or 0))
                or (max_n_test and N_total > (max_n_test or 0))
            )
            n_repeats = max(1, int(repeats)) if (wants_feat or wants_samp) else 1

            try:
                for rep in range(n_repeats):
                    processed = self._preprocess_df(
                        df,
                        target,
                        n_features=(n_features or None),
                        max_n_features=(max_n_features or None),
                        n_train=(n_train or None),
                        max_n_train=(max_n_train or None),
                        n_test=(n_test or None),
                        max_n_test=(max_n_test or None),
                        only_numeric=only_numeric,
                        shuffle_samples=True,
                        shuffle_features=True,
                    )
                    X_train, y_train, X_test, y_test = self.to_numpy_arrays(processed)

                    # Baselines (dynamic)
                    metrics_row: Dict[str, Any] = {}
                    for name, model in models.items():
                        y_pred = self._safe_fit_predict(model, X_train, y_train, X_test)
                        mse, r2 = self._compute_metrics(y_test, y_pred)
                        metrics_row[f"mse_{name}"] = mse
                        metrics_row[f"r2_{name}"] = r2

                    # PFN
                    mse_pfn = None
                    r2_pfn = None
                    if config_path and checkpoint_path:
                        try:
                            import yaml
                            with open(config_path, "r") as f:
                                cfg = yaml.safe_load(f)
                            model_cfg = cfg.get("model_config", {}) if isinstance(cfg, dict) else {}
                            use_bar_distribution = model_cfg.get("use_bar_distribution", {}).get("value", False)

                            pfn = SimplePFNSklearn(
                                config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                device=use_device,
                                verbose=not quiet,
                                n_estimators=n_estimators,
                                norm_methods=norm_methods,
                                outlier_strategies=outlier_strategies,
                            )
                            mkwargs = {
                                "num_features": int(max_n_features or X_train.shape[1]),
                            }
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

                            pfn.load(override_kwargs=mkwargs)

                            if use_bar_distribution:
                                if pfn.bar_distribution is None:
                                    raise ValueError("BarDistribution expected from config but not found in loaded model.")
                                # Ensure BarDistribution is fitted before prediction (fit on the current split)
                                try:
                                    # Fit using this split only to avoid heavy loops; BarDistribution.fit expects iterable over batches
                                    pfn.fit_bar_distribution(
                                        X_train_data=X_train,
                                        y_train_data=y_train,
                                        X_test_data=X_test,
                                        y_test_data=y_test,
                                        max_batches=1,
                                    )
                                except Exception as _e:
                                    raise RuntimeError(f"BarDistribution fit failed before PFN inference: {_e}")
                            
                            y_pred_pfn = pfn.predict(X_train, y_train, X_test)

                            if np.isnan(y_pred_pfn).any():
                                print(f"y_pred_pfn contains NaN values for task {tid} repeat {rep}. Skipping PFN evaluation.")
                                print(f"y_pred_pfn: {y_pred_pfn}")
                                raise ValueError("NaN values found in PFN predictions.")

                            mse_pfn = float(mean_squared_error(y_test, y_pred_pfn))
                            r2_pfn = float(r2_score(y_test, y_pred_pfn))
                        except Exception as e:
                            if not quiet:
                                print(f"[Benchmark] PFN failed on task {tid} repeat {rep}: {e}")
                    if mse_pfn is not None:
                        metrics_row["mse_pfn"] = mse_pfn
                        metrics_row["r2_pfn"] = r2_pfn

                    results.append({
                        "process_id": os.getpid(),
                        "timestamp": datetime.now().isoformat(),
                        "checkpoint_path": os.path.basename(checkpoint_path) if checkpoint_path else "N/A",
                        "task_id": int(tid),
                        "repeat_idx": rep,
                        "dataset_shape": processed.get("data_shape"),
                        "num_features": X_train.shape[1],
                        "n_train": X_train.shape[0],
                        "n_test": X_test.shape[0],
                        **metrics_row,
                    })
            except Exception as e:
                print(f"Error processing task {tid}: {e}")

        df_out = pd.DataFrame(results)

        # Append summary rows
        if not df_out.empty:
            # Backward-compatible single summary row for common models if present
            common_mse = [c for c in ["mse_lr", "mse_rf", "mse_pfn"] if c in df_out.columns]
            common_r2 = [c for c in ["r2_lr", "r2_rf", "r2_pfn"] if c in df_out.columns]
            if common_mse or common_r2:
                median_mse = df_out[common_mse].median(numeric_only=True) if common_mse else pd.Series()
                median_r2 = df_out[common_r2].median(numeric_only=True) if common_r2 else pd.Series()
                summary = {
                    "process_id": "summary",
                    "timestamp": datetime.now().isoformat(),
                    "checkpoint_path": "N/A",
                    "task_id": "N/A",
                    "dataset_shape": "N/A",
                    "num_features": "N/A",
                    "n_train": "N/A",
                    "n_test": "N/A",
                }
                for k, v in median_mse.items():
                    summary[k] = v
                for k, v in median_r2.items():
                    summary[k] = v
                df_out = pd.concat([df_out, pd.DataFrame([summary])], ignore_index=True)

            # Detailed per-model summaries with stats and bootstrap CIs
            # Discover model names from columns
            model_names = sorted({col.split("_")[1] for col in df_out.columns if col.startswith("mse_")})

            detailed_rows: List[Dict[str, Any]] = []
            # Average ranks per task (aggregate repeats by mean per task)
            # Build per-task mean metrics
            if "repeat_idx" in df_out.columns:
                df_task = df_out.groupby(["task_id"]).mean(numeric_only=True).reset_index()
            else:
                df_task = df_out.copy()

            # Compute ranks per task for mse (lower is better) and r2 (higher is better)
            avg_ranks_mse: Dict[str, float] = {}
            avg_ranks_r2: Dict[str, float] = {}
            if not df_task.empty:
                # For each task, get ranks and then average
                ranks_mse_list = []
                ranks_r2_list = []
                for _, row in df_task.iterrows():
                    mse_vals = {m: row.get(f"mse_{m}", np.nan) for m in model_names}
                    r2_vals = {m: row.get(f"r2_{m}", np.nan) for m in model_names}
                    # Rank ignoring NaNs
                    valid_mse = {m: v for m, v in mse_vals.items() if pd.notna(v)}
                    valid_r2 = {m: v for m, v in r2_vals.items() if pd.notna(v)}
                    if valid_mse:
                        order = sorted(valid_mse.items(), key=lambda x: x[1])  # lower better
                        ranks = {m: i + 1 for i, (m, _) in enumerate(order)}
                        ranks_mse_list.append(ranks)
                    if valid_r2:
                        order = sorted(valid_r2.items(), key=lambda x: -x[1])  # higher better
                        ranks = {m: i + 1 for i, (m, _) in enumerate(order)}
                        ranks_r2_list.append(ranks)
                if ranks_mse_list:
                    # average ranks across tasks (fill missing with NaN then mean ignoring NaN)
                    df_ranks_mse = pd.DataFrame(ranks_mse_list)
                    avg_ranks_mse = df_ranks_mse.mean(skipna=True).to_dict()
                if ranks_r2_list:
                    df_ranks_r2 = pd.DataFrame(ranks_r2_list)
                    avg_ranks_r2 = df_ranks_r2.mean(skipna=True).to_dict()

            # For each model, compute stats on raw per-task-repeat rows
            for m in model_names:
                mse_series = df_out[f"mse_{m}"] if f"mse_{m}" in df_out.columns else pd.Series(dtype=float)
                r2_series = df_out[f"r2_{m}"] if f"r2_{m}" in df_out.columns else pd.Series(dtype=float)
                mse_vals = mse_series.to_numpy(dtype=float)
                r2_vals = r2_series.to_numpy(dtype=float)

                row_mse = {
                    "process_id": "summary_model",
                    "timestamp": datetime.now().isoformat(),
                    "model": m,
                    "metric": "mse",
                    "mean": float(np.nanmean(mse_vals)) if mse_vals.size else np.nan,
                    "median": float(np.nanmedian(mse_vals)) if mse_vals.size else np.nan,
                    "std": float(np.nanstd(mse_vals, ddof=1)) if np.sum(~np.isnan(mse_vals)) > 1 else np.nan,
                    "iqr": self._iqr(mse_vals),
                }
                ci_low, ci_high = self._bootstrap_ci(mse_vals, np.nanmean, n_boot=bootstrap_samples) if mse_vals.size else (np.nan, np.nan)
                row_mse["ci95_mean_low"], row_mse["ci95_mean_high"] = ci_low, ci_high
                ci_low, ci_high = self._bootstrap_ci(mse_vals, np.nanmedian, n_boot=bootstrap_samples) if mse_vals.size else (np.nan, np.nan)
                row_mse["ci95_median_low"], row_mse["ci95_median_high"] = ci_low, ci_high
                row_mse["avg_rank_mse"] = float(avg_ranks_mse.get(m, np.nan))
                row_mse["avg_rank_r2"] = float(avg_ranks_r2.get(m, np.nan))
                detailed_rows.append(row_mse)

                row_r2 = {
                    "process_id": "summary_model",
                    "timestamp": datetime.now().isoformat(),
                    "model": m,
                    "metric": "r2",
                    "mean": float(np.nanmean(r2_vals)) if r2_vals.size else np.nan,
                    "median": float(np.nanmedian(r2_vals)) if r2_vals.size else np.nan,
                    "std": float(np.nanstd(r2_vals, ddof=1)) if np.sum(~np.isnan(r2_vals)) > 1 else np.nan,
                    "iqr": self._iqr(r2_vals),
                }
                ci_low, ci_high = self._bootstrap_ci(r2_vals, np.nanmean, n_boot=bootstrap_samples) if r2_vals.size else (np.nan, np.nan)
                row_r2["ci95_mean_low"], row_r2["ci95_mean_high"] = ci_low, ci_high
                ci_low, ci_high = self._bootstrap_ci(r2_vals, np.nanmedian, n_boot=bootstrap_samples) if r2_vals.size else (np.nan, np.nan)
                row_r2["ci95_median_low"], row_r2["ci95_median_high"] = ci_low, ci_high
                row_r2["avg_rank_mse"] = float(avg_ranks_mse.get(m, np.nan))
                row_r2["avg_rank_r2"] = float(avg_ranks_r2.get(m, np.nan))
                detailed_rows.append(row_r2)

            if detailed_rows:
                df_out = pd.concat([df_out, pd.DataFrame(detailed_rows)], ignore_index=True)

        # Save
        out_path = Path(output_csv)
        process_id = os.getpid()
        if out_path.suffix:
            out_file = out_path.with_name(f"{out_path.stem}_pid{process_id}{out_path.suffix}")
        else:
            out_file = Path(f"{output_csv}_pid{process_id}")
        df_out.to_csv(out_file, index=False)
        if not quiet:
            print(f"Saved results to {out_file} (PID: {process_id})")

        return df_out

    def run_simplified(
        self,
        fidelity: str = "low",
        tasks: Optional[List[int]] = None,
        # subsampling
        n_features: int = 0,
        max_n_features: int = 0,
        n_train: int = 0,
        max_n_train: int = 0,
        n_test: int = 0,
        max_n_test: int = 0,
        prefer_numeric: bool = True,
        only_numeric: bool = False,
        # model
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        output_csv: str = "benchmark_results.csv",
        device: Optional[str] = None,
        quiet: bool = False,
        bootstrap_samples: int = 1000,
        # SimplePFN ensemble parameters
        n_estimators: int = 1,
        norm_methods: Optional[List[str]] = None,
        outlier_strategies: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Simplified benchmark entrypoint.

        Choose a fidelity level instead of manually setting repeats, baseline set, or max_tasks:
        - "minimal": 1 task, basic baselines, 1 repeat
        - "low":     10 tasks, basic baselines, 1 repeat
        - "high":    all tasks, extended baselines, 5 repeats
        - "very high" (or very_high): all tasks, extended baselines, 25 repeats

        Other arguments mirror `run` to control subsampling and PFN usage.
        """
        # Normalize fidelity key
        key = fidelity.strip().lower().replace("-", "_").replace(" ", "_")
        allowed = {"minimal", "low", "high", "very_high"}
        if key not in allowed:
            raise ValueError(f"Invalid fidelity '{fidelity}'. Choose one of: minimal, low, high, very high")

        # Map fidelity to settings
        if key == "minimal":
            repeats = 1
            baseline_set = "basic"
            task_cap = 1
        elif key == "low":
            repeats = 1
            baseline_set = "basic"
            task_cap = 10
        elif key == "high":
            repeats = 5
            baseline_set = "extended"
            task_cap = None  # all
        else:  # very_high
            repeats = 25
            baseline_set = "extended"
            task_cap = None  # all

        # Resolve tasks list with cap if provided/available
        if tasks is None or len(tasks) == 0:
            base_tasks = DEFAULT_TABULAR_NUM_REG_TASKS
        else:
            base_tasks = tasks

        if task_cap is not None:
            tasks_use = base_tasks[: task_cap]
        else:
            tasks_use = base_tasks

        # Delegate to the main runner with mapped settings
        return self.run(
            tasks=tasks_use,
            # subsampling
            n_features=n_features,
            max_n_features=max_n_features,
            n_train=n_train,
            max_n_train=max_n_train,
            n_test=n_test,
            max_n_test=max_n_test,
            prefer_numeric=prefer_numeric,
            only_numeric=only_numeric,
            # model
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output_csv=output_csv,
            device=device,
            quiet=quiet,
            # mapped
            repeats=repeats,
            baseline_set=baseline_set,
            bootstrap_samples=bootstrap_samples,
            # SimplePFN ensemble parameters
            n_estimators=n_estimators,
            norm_methods=norm_methods,
            outlier_strategies=outlier_strategies,
        )
