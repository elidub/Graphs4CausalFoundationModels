"""
Simple OpenML dataset loader

- Downloads an OpenML dataset by id (once) and caches the raw CSV
- Preprocesses: impute missing values, optionally target-encode categoricals, standardize numericals
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TargetEncoder is available in scikit-learn >= 1.4; make it optional
try:
    from sklearn.preprocessing import TargetEncoder  # type: ignore
    _HAS_TARGET_ENCODER = True
except Exception:
    _HAS_TARGET_ENCODER = False

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
        use_target_encoding: bool = True,
        verbose: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.random_state = int(random_state)
        self.test_size = float(test_size)
        self.use_target_encoding = bool(use_target_encoding) and _HAS_TARGET_ENCODER
        self.verbose = bool(verbose)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[Tuple[int, str, bool, float], Dict[str, Any]] = {}

        if use_target_encoding and not _HAS_TARGET_ENCODER and self.verbose:
            print("[SimpleOpenMLLoader] TargetEncoder not available; proceeding without it.")

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
        key = (int(dataset_id), str(target_column or ""), self.use_target_encoding, self.test_size)
        if key in self._cache:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Using in-memory cache for {key}")
            return self._cache[key]

        ds_dir = self.data_dir / f"openml_{dataset_id}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = ds_dir / "raw.csv"
        proc_joblib = ds_dir / f"processed_enc={int(self.use_target_encoding)}_test={self.test_size:.2f}.joblib"

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

                if prefer_numeric:
                    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
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
            print(f"[SimpleOpenMLLoader] Preprocessing df shape={df.shape}, target='{target}'")

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")

        y = df[target]
        X = df.drop(columns=[target])

        # type-based column partitions
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        feature_names = X.columns.tolist()

        # impute: mean for numeric, mode for categorical
        if num_cols:
            for c in num_cols:
                if X[c].isna().any():
                    X.loc[:, c] = X[c].fillna(X[c].mean())
        if cat_cols:
            for c in cat_cols:
                if X[c].isna().any():
                    X.loc[:, c] = X[c].fillna(X[c].mode(dropna=True).iloc[0])

        # split once (important for fitting encoders/scalers on train only)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # optional target encoding
        target_encoder = None
        if self.use_target_encoding and cat_cols:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Target-encoding {len(cat_cols)} categorical columns")
            target_encoder = TargetEncoder()
            X_train.loc[:, cat_cols] = target_encoder.fit_transform(X_train[cat_cols], y_train)
            X_test.loc[:, cat_cols] = target_encoder.transform(X_test[cat_cols])
            # after target encoding, categoricals are numeric; ensure float dtype to avoid later dtype assignment warnings
            try:
                X_train.loc[:, cat_cols] = X_train.loc[:, cat_cols].astype(float)
                X_test.loc[:, cat_cols] = X_test.loc[:, cat_cols].astype(float)
            except Exception:
                # best-effort cast; if it fails, proceed and let scaler handle types
                pass

        # standardize numerical columns (including those that remain numeric)
        scaler = None
        cols_to_scale = num_cols + ([c for c in cat_cols] if self.use_target_encoding else [])
        if cols_to_scale:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Standardizing {len(cols_to_scale)} columns")
            scaler = StandardScaler()
            # Ensure columns are float before assigning scaled values to avoid pandas dtype warnings
            try:
                X_train.loc[:, cols_to_scale] = X_train.loc[:, cols_to_scale].astype(float)
                X_test.loc[:, cols_to_scale] = X_test.loc[:, cols_to_scale].astype(float)
            except Exception:
                pass

            X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
            X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_names,
            "categorical_features": cat_cols,   # before encoding
            "numerical_features": num_cols,
            "target_column": target,
            "scaler": scaler,
            "target_encoder": target_encoder,
            "data_shape": df.shape,
        }

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.to_numpy()
        return np.asarray(x)


if __name__ == "__main__":
    print("=== SimpleOpenMLLoader demo ===")
    loader = SimpleOpenMLLoader(
        data_dir="data_cache",
        random_state=42,
        test_size=0.2,
        use_target_encoding=True,
        verbose=True,
    )

    # Example 1: Boston Housing
    print("\n--- Boston Housing (531) ---")
    boston = loader.load_dataset(531)
    print(f"Train shape: {boston['X_train'].shape}, Test shape: {boston['X_test'].shape}")
    print(f"Target column: {boston['target_column']}")
    print(f"Numerical features: {len(boston['numerical_features'])}, "
          f"Categorical features: {len(boston['categorical_features'])}")

    # Convert to numpy arrays
    boston_np = loader.get_data_as_numpy(531)
    print(f"Numpy arrays: X_train {boston_np['X_train'].shape}, y_train {boston_np['y_train'].shape}")

    # Example 2: Car Evaluation
    print("\n--- Car Evaluation (40975) ---")
    car = loader.load_dataset(40975)
    print(f"Train shape: {car['X_train'].shape}, Test shape: {car['X_test'].shape}")
    print(f"Target column: {car['target_column']}")
    print(f"Numerical features: {len(car['numerical_features'])}, "
          f"Categorical features: {len(car['categorical_features'])}")

    # Example 3: load a small benchmark (Tabular numerical regression tasks)
    print("\n--- Tabular numerical regression benchmark (sample tasks) ---")
    sample_tasks = DEFAULT_TABULAR_NUM_REG_TASKS[:5]
    print(f"Loading tasks: {sample_tasks}")
    tasks_data = loader.load_tasks(sample_tasks)
    for tid, d in tasks_data.items():
        if d:
            print(f"Task {tid}: dataset shape {d['data_shape']}, target={d['target_column']}")
        else:
            print(f"Task {tid}: failed to load")

    print("\nDemo completed successfully.")

    # Small test: create subsampled datasets (5 features, 20 train, 10 test) from the sample tasks
    print("\n--- Subsampled datasets test ---")
    subsampled = loader.create_subsampled_datasets(sample_tasks, n_features=5, n_train=250, n_test=250, prefer_numeric=True, save=True)
    for tid, d in subsampled.items():
        if d:
            print(f"Subsampled task {tid}: X_train {d['X_train'].shape}, X_test {d['X_test'].shape}, target={d['target_column']}")
        else:
            print(f"Subsampled task {tid}: skipped or failed")

