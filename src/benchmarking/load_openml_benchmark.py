"""
Simple OpenML dataset loader

- Downloads an OpenML dataset by id (once) and caches the raw CSV
- Preprocesses using the Preprocessor class for consistency
- Splits into train/test
- Caches the processed artifact (joblib) for fast reuse
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
import torch
import openml

# Import the new Preprocessor class
from priordata_processing.Preprocessor import Preprocessor

import openml

# Default Tabular numerical regression task IDs (from the user's list)
DEFAULT_TABULAR_NUM_REG_TASKS = [
    361072, 361073, 361074, 361076, 361077, 361078, 361079, 361080,
    361081, 361082, 361083, 361084, 361085, 361086, 361087, 361088,
    361279, 361280, 361281
]


class SimpleOpenMLLoader:
    def __init__(
        self,
        data_dir: str = "data",
        random_state: int = 42,
        test_size: float = 0.2,
        verbose: bool = True,
        only_numeric: bool = False,
        # Preprocessor parameters
        max_n_features: int = 100,
        max_n_train_samples: int = 1000,
        max_n_test_samples: int = 250,
        negative_one_one_scaling: bool = True,
        standardize: bool = True,
        yeo_johnson: bool = False,
        remove_outliers: bool = True,
        outlier_quantile: float = 0.95,
        shuffle_samples: bool = True,
        shuffle_features: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.random_state = int(random_state)
        self.test_size = float(test_size)
        self.only_numeric = bool(only_numeric)
        self.verbose = bool(verbose)
        
        # Preprocessor parameters
        self.max_n_features = int(max_n_features)
        self.max_n_train_samples = int(max_n_train_samples)
        self.max_n_test_samples = int(max_n_test_samples)
        self.negative_one_one_scaling = bool(negative_one_one_scaling)
        self.standardize = bool(standardize)
        self.yeo_johnson = bool(yeo_johnson)
        self.remove_outliers = bool(remove_outliers)
        self.outlier_quantile = float(outlier_quantile)
        self.shuffle_samples = bool(shuffle_samples)
        self.shuffle_features = bool(shuffle_features)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[Tuple[int, str, bool, float, bool, bool, bool, bool, bool, bool, float], Dict[str, Any]] = {}

    # ---------- public API ----------

    def load_dataset(
        self,
        dataset_id: int,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download, cache, preprocess, and return a dataset.
        Returns a dict with X_train, y_train, X_test, y_test, feature_names, target_column, etc.
        """
        # cache key includes core processing knobs
        key = (
            int(dataset_id), 
            str(target_column or ""), 
            self.test_size,
            self.standardize,
            self.yeo_johnson,
            self.negative_one_one_scaling,
            self.remove_outliers,
            self.outlier_quantile,
            self.shuffle_samples,
            self.shuffle_features
        )
        if key in self._cache:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Using in-memory cache for {key}")
            return self._cache[key]

        ds_dir = self.data_dir / f"openml_{dataset_id}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = ds_dir / "raw.csv"
        
        # Include preprocessor parameters in cache filename
        cache_suffix = f"test={self.test_size:.2f}_std={int(self.standardize)}_yj={int(self.yeo_johnson)}_scale={int(self.negative_one_one_scaling)}_outliers={int(self.remove_outliers)}_q={self.outlier_quantile:.2f}"
        proc_joblib = ds_dir / f"processed_{cache_suffix}.joblib"

        # 1) return processed if already cached on disk
        if proc_joblib.exists():
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Loading processed cache: {proc_joblib}")
            data = joblib.load(proc_joblib)
            self._cache[key] = data
            return data

        # 2) obtain raw dataframe (download once)
        if raw_csv.exists():
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Loading raw CSV: {raw_csv}")
            df = pd.read_csv(raw_csv)
            if target_column is None:
                target_column = df.columns[-1]
        else:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Downloading OpenML dataset {dataset_id} ...")
            omlds = openml.datasets.get_dataset(dataset_id)

            if target_column is None:
                target_column = omlds.default_target_attribute

            # robustly get data as dataframe(s)
            try:
                # preferred modern API
                X, y, _, _ = omlds.get_data(dataset_format="dataframe", target=target_column)
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    df = pd.concat([X, y.rename(target_column)], axis=1)
                else:
                    # rare: y as np array
                    df = pd.concat([X, pd.Series(y, name=target_column)], axis=1)
            except Exception:
                # fallback: older OpenML versions sometimes return a single df
                df = omlds.get_data()[0]
                if target_column not in df.columns:
                    target_column = df.columns[-1]

            df.to_csv(raw_csv, index=False)
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Raw saved → {raw_csv}")

        # 3) preprocess
        data = self._preprocess(df, target_column)

        # 4) cache
        joblib.dump(data, proc_joblib)
        if self.verbose:
            print(f"[SimpleOpenMLLoader] Processed saved → {proc_joblib}")

        self._cache[key] = data
        return data

    def get_data_as_numpy(
        self,
        dataset_id: int,
        target_column: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience: same as load_dataset but returns numpy arrays.
        """
        d = self.load_dataset(dataset_id, target_column)
        X_train = self._to_numpy(d["X_train"])
        X_test = self._to_numpy(d["X_test"])
        y_train = self._to_numpy(d["y_train"]).reshape(-1, 1) if d["y_train"].ndim == 1 else self._to_numpy(d["y_train"])
        y_test = self._to_numpy(d["y_test"]).reshape(-1, 1) if d["y_test"].ndim == 1 else self._to_numpy(d["y_test"])

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": np.array(d["feature_names"], dtype=object),
            "target_column": np.array([d["target_column"]], dtype=object),
        }

    def load_tasks(self, task_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Load multiple OpenML tasks (benchmark). For each OpenML task id this will:
        - resolve the underlying dataset id and target column
        - call `load_dataset(dataset_id, target_column)` which downloads, preprocesses and caches

        Returns a dict mapping task_id -> processed dataset dict (same format as `load_dataset`).

        Notes/assumptions:
        - This method tries several fallbacks to extract dataset id and target name from the
          OpenML Task object because different OpenML server versions/objects expose different
          attributes. If the target name cannot be resolved from the task, the dataset's
          default target attribute is used.
        """
        results: Dict[int, Dict[str, Any]] = {}
        for tid in task_ids:
            try:
                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Resolving task {tid} ...")

                task = openml.tasks.get_task(int(tid))

                # Try common attributes
                dataset_id = getattr(task, "dataset_id", None)
                target_name = getattr(task, "target_name", None)

                # Fallbacks: some OpenML task objects expose get_dataset() or require inspecting
                if dataset_id is None:
                    try:
                        ds_obj = task.get_dataset()
                        dataset_id = getattr(ds_obj, "dataset_id", None) or getattr(ds_obj, "id", None)
                    except Exception:
                        dataset_id = None

                # If we still don't know dataset_id, skip
                if dataset_id is None:
                    print(f"[SimpleOpenMLLoader] Warning: could not resolve dataset id for task {tid}; skipping")
                    continue

                # If target not provided by task, try dataset default
                if target_name is None:
                    try:
                        omlds = openml.datasets.get_dataset(int(dataset_id))
                        target_name = omlds.default_target_attribute
                    except Exception:
                        target_name = None

                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Task {tid} -> dataset {dataset_id}, target='{target_name}'")

                loaded = self.load_dataset(int(dataset_id), target_name)
                results[int(tid)] = loaded

            except Exception as e:
                print(f"[SimpleOpenMLLoader] Error loading task {tid}: {e}")

        return results

    def create_subsampled_datasets(
        self,
        task_ids: List[int],
        n_features: int,
        n_train: int,
        n_test: int,
        prefer_numeric: bool = True,
    save: bool = True,
    only_numeric: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Create a set of processed datasets by subsampling existing OpenML tasks/datasets.

        For each task id this method:
        - resolves the underlying dataset and target
        - downloads the raw dataframe
        - ignores the dataset if it has fewer than `n_features` (excluding target) or
          fewer than `n_train + n_test` rows
        - otherwise subsamples columns and rows (deterministic using `self.random_state`)
        - preprocesses the subsampled DataFrame (impute/encode/scale) and splits into
          train/test with sizes `n_train` and `n_test`.

        Returns a dict mapping task_id -> processed-data-dict (same format as `load_dataset`).
        If `save` is True, a joblib file will be written under the dataset cache folder.
        """
        results: Dict[int, Dict[str, Any]] = {}
        total_required = int(n_train) + int(n_test)
        rng = np.random.RandomState(self.random_state)

        for tid in task_ids:
            try:
                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Subsampling task {tid} -> need {n_features} features, {total_required} rows")

                task = openml.tasks.get_task(int(tid))
                dataset_id = getattr(task, "dataset_id", None)
                target_name = getattr(task, "target_name", None)

                if dataset_id is None:
                    try:
                        ds_obj = task.get_dataset()
                        dataset_id = getattr(ds_obj, "dataset_id", None) or getattr(ds_obj, "id", None)
                    except Exception:
                        dataset_id = None

                if dataset_id is None:
                    if self.verbose:
                        print(f"[SimpleOpenMLLoader] Could not resolve dataset for task {tid}; skipping")
                    continue

                # load raw dataframe
                omlds = openml.datasets.get_dataset(int(dataset_id))
                try:
                    X, y, _, _ = omlds.get_data(dataset_format="dataframe", target=target_name)
                    if isinstance(y, (pd.Series, pd.DataFrame)):
                        df = pd.concat([X, y.rename(target_name)], axis=1)
                    else:
                        df = pd.concat([X, pd.Series(y, name=target_name)], axis=1)
                except Exception:
                    df = omlds.get_data()[0]
                    if target_name is None or target_name not in df.columns:
                        target_name = df.columns[-1]

                n_rows, n_cols_total = df.shape
                n_cols = n_cols_total - 1  # exclude target

                if n_cols < n_features or n_rows < total_required:
                    if self.verbose:
                        print(f"[SimpleOpenMLLoader] Skipping dataset {dataset_id}: has {n_cols} features, {n_rows} rows")
                    continue

                # select feature columns (exclude target)
                feature_cols = [c for c in df.columns if c != target_name]
                num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

                if only_numeric:
                    # strictly require numeric-only selection
                    if len(num_cols) < n_features:
                        if self.verbose:
                            print(f"[SimpleOpenMLLoader] Skipping dataset {dataset_id}: needs {n_features} numeric features but has {len(num_cols)}")
                        continue
                    chosen = list(num_cols[:n_features])
                else:
                    if prefer_numeric:
                        chosen = list(num_cols[:n_features])
                        if len(chosen) < n_features:
                            # fill with remaining columns in original order
                            for c in feature_cols:
                                if c not in chosen:
                                    chosen.append(c)
                                    if len(chosen) >= n_features:
                                        break
                    else:
                        chosen = feature_cols[:n_features]

                    chosen = chosen[:n_features]

                # sample rows deterministically
                indices = rng.choice(n_rows, size=total_required, replace=False)
                df_sub = df.iloc[indices].reset_index(drop=True)

                # keep only chosen features + target
                cols_to_keep = chosen + [target_name]
                df_sub = df_sub.loc[:, cols_to_keep]

                # temporarily set test_size to match requested n_test
                original_test_size = self.test_size
                try:
                    self.test_size = float(n_test) / float(total_required)
                    processed = self._preprocess(df_sub, target_name)
                finally:
                    self.test_size = original_test_size

                # optionally save processed artifact
                ds_dir = self.data_dir / f"openml_{dataset_id}"
                ds_dir.mkdir(parents=True, exist_ok=True)
                if save:
                    out_name = ds_dir / f"subsampled_f{n_features}_n{total_required}.joblib"
                    try:
                        joblib.dump(processed, out_name)
                        if self.verbose:
                            print(f"[SimpleOpenMLLoader] Saved subsampled -> {out_name}")
                    except Exception:
                        if self.verbose:
                            print(f"[SimpleOpenMLLoader] Warning: failed to save subsampled artifact for {dataset_id}")

                results[int(tid)] = processed

            except Exception as e:
                print(f"[SimpleOpenMLLoader] Error processing task {tid}: {e}")

        return results

    # ---------- internals ----------

    def _preprocess(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        if self.verbose:
            print(f"[SimpleOpenMLLoader] Preprocessing df shape={df.shape}, target='{target}' using Preprocessor")

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")

        y = df[target]
        X = df.drop(columns=[target])

        # Basic imputation for missing values (only numeric columns supported by Preprocessor)
        # Convert categorical to numeric by label encoding if needed
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        
        # Simple preprocessing to prepare for Preprocessor
        if self.only_numeric and cat_cols:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] only_numeric=True, dropping {len(cat_cols)} categorical columns")
            X = X[num_cols]
        elif cat_cols:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Label encoding {len(cat_cols)} categorical columns")
            # Simple label encoding for categorical columns
            from sklearn.preprocessing import LabelEncoder
            for col in cat_cols:
                le = LabelEncoder()
                # Handle missing values
                X[col] = X[col].fillna('missing')
                X[col] = le.fit_transform(X[col])
        
        # Fill missing values in numeric columns
        if num_cols:
            for col in num_cols:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].mean())
        
        feature_names = X.columns.tolist()
        
        # Convert to numpy and then to torch tensors
        X_np = X.to_numpy().astype(np.float32)
        y_np = y.to_numpy().astype(np.float32)
        
        # Convert to torch tensors with batch dimension
        X_tensor = torch.from_numpy(X_np).unsqueeze(0)  # [1, N, F]
        y_tensor = torch.from_numpy(y_np).unsqueeze(0)  # [1, N]
        
        N, F = X_np.shape
        
        # Calculate train/test split sizes
        n_test = int(N * self.test_size)
        n_train = N - n_test
        
        # Ensure we don't exceed the max limits
        actual_n_train = min(n_train, self.max_n_train_samples)
        actual_n_test = min(n_test, self.max_n_test_samples)
        actual_n_features = min(F, self.max_n_features)
        
        if N < actual_n_train + actual_n_test:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Warning: Dataset has {N} samples but need {actual_n_train + actual_n_test}")
            # Adjust to what we have
            total_available = N
            train_fraction = actual_n_train / (actual_n_train + actual_n_test)
            actual_n_train = int(total_available * train_fraction)
            actual_n_test = total_available - actual_n_train
        
        if F < actual_n_features:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Warning: Dataset has {F} features but requested {actual_n_features}")
            actual_n_features = F
        
        # Create Preprocessor
        preprocessor = Preprocessor(
            n_features=actual_n_features,
            max_n_features=self.max_n_features,
            n_train_samples=actual_n_train,
            max_n_train_samples=self.max_n_train_samples,
            n_test_samples=actual_n_test,
            max_n_test_samples=self.max_n_test_samples,
            negative_one_one_scaling=self.negative_one_one_scaling,
            standardize=self.standardize,
            yeo_johnson=self.yeo_johnson,
            remove_outliers=self.remove_outliers,
            outlier_quantile=self.outlier_quantile,
            shuffle_samples=self.shuffle_samples,
            shuffle_features=self.shuffle_features,
        )
        
        if self.verbose:
            print(f"[SimpleOpenMLLoader] Using Preprocessor with: {actual_n_features} features, "
                  f"{actual_n_train} train samples, {actual_n_test} test samples")
        
        # Process with Preprocessor
        result = preprocessor.process(X_tensor, y_tensor)
        
        if result is None:
            raise ValueError(f"Preprocessor returned None - insufficient data. "
                           f"Need at least {actual_n_train + actual_n_test} samples and {actual_n_features} features, "
                           f"but got {N} samples and {F} features")
        
        X_train_processed, X_test_processed, y_train_processed, y_test_processed = result
        
        # Remove batch dimension and convert back to the expected format
        X_train = X_train_processed[0].numpy()  # [train_samples, features]
        X_test = X_test_processed[0].numpy()    # [test_samples, features]
        y_train = y_train_processed[0].numpy()  # [train_samples]
        y_test = y_test_processed[0].numpy()    # [test_samples]
        
        # Convert back to pandas for consistency with original API
        feature_names_padded = feature_names + [f"padded_feature_{i}" for i in range(len(feature_names), X_train.shape[1])]
        
        X_train_df = pd.DataFrame(X_train, columns=feature_names_padded)
        X_test_df = pd.DataFrame(X_test, columns=feature_names_padded)
        y_train_series = pd.Series(y_train, name=target)
        y_test_series = pd.Series(y_test, name=target)

        return {
            "X_train": X_train_df,
            "y_train": y_train_series,
            "X_test": X_test_df,
            "y_test": y_test_series,
            "feature_names": feature_names_padded,
            "categorical_features": cat_cols if not self.only_numeric else [],
            "numerical_features": num_cols,
            "target_column": target,
            "preprocessor_info": {
                "actual_n_features": actual_n_features,
                "actual_n_train": actual_n_train,
                "actual_n_test": actual_n_test,
                "original_n_features": F,
                "original_n_samples": N,
                "standardize": self.standardize,
                "yeo_johnson": self.yeo_johnson,
                "negative_one_one_scaling": self.negative_one_one_scaling,
                "remove_outliers": self.remove_outliers,
                "outlier_quantile": self.outlier_quantile,
            },
            "data_shape": df.shape,
        }

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.to_numpy()
        return np.asarray(x)


if __name__ == "__main__":
    print("=== SimpleOpenMLLoader demo (using Preprocessor) ===")
    loader = SimpleOpenMLLoader(
        data_dir="data_cache",
        random_state=42,
        test_size=0.2,
        verbose=True,
        only_numeric=False,
        # Preprocessor parameters
        max_n_features=50,
        max_n_train_samples=800,
        max_n_test_samples=200,
        standardize=True,
        yeo_johnson=False,
        negative_one_one_scaling=False,
        remove_outliers=True,
        outlier_quantile=0.95,
    )

    # Example 1: Boston Housing
    print("\n--- Boston Housing (531) ---")
    try:
        boston = loader.load_dataset(531)
        print(f"Train shape: {boston['X_train'].shape}, Test shape: {boston['X_test'].shape}")
        print(f"Target column: {boston['target_column']}")
        print(f"Numerical features: {len(boston['numerical_features'])}, "
              f"Categorical features: {len(boston['categorical_features'])}")

        # Convert to numpy arrays
        boston_np = loader.get_data_as_numpy(531)
        print(f"Numpy arrays: X_train {boston_np['X_train'].shape}, y_train {boston_np['y_train'].shape}")
    except Exception as e:
        print(f"Error loading Boston Housing: {e}")

    print("\nDemo completed. Preprocessor integration successful!")

