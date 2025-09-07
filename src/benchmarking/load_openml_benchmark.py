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
from scipy.stats import yeojohnson

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
        only_numeric: bool = False,
        transformation_type: str = 'standardize',
        use_heuristic_categorical_detection: bool = False,
        use_leave_one_out_encoding: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.random_state = int(random_state)
        self.test_size = float(test_size)
        self.use_target_encoding = bool(use_target_encoding) and _HAS_TARGET_ENCODER
        self.only_numeric = bool(only_numeric)
        self.verbose = bool(verbose)
        self.transformation_type = str(transformation_type)
        self.use_heuristic_categorical_detection = bool(use_heuristic_categorical_detection)
        self.use_leave_one_out_encoding = bool(use_leave_one_out_encoding)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[Tuple[int, str, bool, float, str, bool, bool], Dict[str, Any]] = {}

        if use_target_encoding and not _HAS_TARGET_ENCODER and self.verbose:
            print("[SimpleOpenMLLoader] TargetEncoder not available; proceeding without it.")
            
        if transformation_type not in ['standardize', 'yeo_johnson']:
            raise ValueError(f"transformation_type must be 'standardize' or 'yeo_johnson', got '{transformation_type}'")

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
            self.use_target_encoding, 
            self.test_size,
            self.transformation_type,
            self.use_heuristic_categorical_detection,
            self.use_leave_one_out_encoding
        )
        if key in self._cache:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Using in-memory cache for {key}")
            return self._cache[key]

        ds_dir = self.data_dir / f"openml_{dataset_id}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = ds_dir / "raw.csv"
        
        # Include new parameters in cache filename
        cache_suffix = f"enc={int(self.use_target_encoding)}_test={self.test_size:.2f}_transform={self.transformation_type}_heur={int(self.use_heuristic_categorical_detection)}_loo={int(self.use_leave_one_out_encoding)}"
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

    def _detect_categorical_features_heuristic(self, X: pd.DataFrame) -> List[str]:
        """
        Detect categorical features using heuristics similar to BasicProcessing.
        
        A feature is considered categorical if:
        1. It has a small number of unique values (≤ 10% of sample size or ≤ 20 unique values)
        2. All values are integers (within a small tolerance)
        """
        categorical_cols = []
        n_samples = len(X)
        
        for col in X.columns:
            col_data = X[col]
            
            # Skip non-numeric columns (they'll be handled by dtype detection)
            if not pd.api.types.is_numeric_dtype(col_data):
                continue
            
            # Check if values are close to integers (tolerance for floating point errors)
            is_integer_like = np.all(np.abs(col_data - np.round(col_data)) < 1e-6)
            
            # Count unique values
            n_unique = col_data.nunique()
            
            # Heuristic: categorical if integer-like and few unique values
            max_categories = min(20, max(2, int(0.1 * n_samples)))  # At most 10% of samples or 20 categories
            
            is_categorical = is_integer_like and n_unique <= max_categories
            
            if is_categorical:
                categorical_cols.append(col)
                if self.verbose:
                    unique_vals = sorted(col_data.unique())[:10]  # Show first 10 unique values
                    print(f"[SimpleOpenMLLoader] Heuristic: {col} detected as categorical ({n_unique} unique values: {unique_vals}{'...' if n_unique > 10 else ''})")
        
        return categorical_cols

    def _apply_leave_one_out_encoding(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                    y_train: pd.Series, cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Apply leave-one-out target encoding similar to BasicProcessing.
        
        This prevents overfitting by using the mean of all OTHER samples in the same category
        for encoding each sample.
        """
        encoding_params = {
            'categorical_features': cat_cols,
            'encoding_maps': {},
            'global_means': {}
        }
        
        if not cat_cols:
            return X_train.copy(), X_test.copy(), encoding_params
        
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        global_mean = y_train.mean()
        
        for col in cat_cols:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Leave-one-out encoding for column: {col}")
            
            # Get unique categories from training data
            unique_categories = X_train[col].unique()
            
            # Calculate category means for test set encoding
            category_means = {}
            for category in unique_categories:
                category_mask = X_train[col] == category
                category_targets = y_train[category_mask]
                if len(category_targets) > 0:
                    category_means[category] = category_targets.mean()
                else:
                    category_means[category] = global_mean
            
            # Leave-one-out encoding for training set
            encoded_train_values = np.zeros(len(X_train))
            for idx, (category, target) in enumerate(zip(X_train[col], y_train)):
                # Find all other samples with the same category
                same_category_mask = (X_train[col] == category)
                same_category_indices = np.where(same_category_mask)[0]
                other_indices = same_category_indices[same_category_indices != idx]
                
                if len(other_indices) > 0:
                    # Use mean of other samples in same category
                    encoded_train_values[idx] = y_train.iloc[other_indices].mean()
                else:
                    # If only one sample in category, use global mean
                    encoded_train_values[idx] = global_mean
            
            X_train_encoded[col] = encoded_train_values
            
            # Standard encoding for test set (using all training data for each category)
            X_test_encoded[col] = X_test[col].map(category_means).fillna(global_mean)
            
            # Store encoding parameters
            encoding_params['encoding_maps'][col] = category_means
            encoding_params['global_means'][col] = global_mean
        
        return X_train_encoded, X_test_encoded, encoding_params

    def _preprocess(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        if self.verbose:
            print(f"[SimpleOpenMLLoader] Preprocessing df shape={df.shape}, target='{target}'")

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")

        y = df[target]
        X = df.drop(columns=[target])

        # type-based column partitions
        cat_cols_dtype = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        feature_names = X.columns.tolist()

        # Optional: use heuristic categorical detection (similar to BasicProcessing)
        if self.use_heuristic_categorical_detection:
            cat_cols_heuristic = self._detect_categorical_features_heuristic(X)
            # Combine both methods
            cat_cols = list(set(cat_cols_dtype + cat_cols_heuristic))
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Combined categorical detection: dtype={len(cat_cols_dtype)}, heuristic={len(cat_cols_heuristic)}, total={len(cat_cols)}")
        else:
            cat_cols = cat_cols_dtype

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
        encoding_params = None
        if self.use_target_encoding and cat_cols:
            if self.use_leave_one_out_encoding:
                # Use leave-one-out encoding (similar to BasicProcessing)
                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Leave-one-out target-encoding {len(cat_cols)} categorical columns")
                X_train, X_test, encoding_params = self._apply_leave_one_out_encoding(
                    X_train, X_test, y_train, cat_cols
                )
            else:
                # Use standard sklearn TargetEncoder
                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Standard target-encoding {len(cat_cols)} categorical columns")
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

        # Apply transformations (standardization or Yeo-Johnson + standardization)
        transformation_params = {'type': self.transformation_type}
        cols_to_scale = num_cols + ([c for c in cat_cols] if self.use_target_encoding else [])
        
        if cols_to_scale:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Applying {self.transformation_type} to {len(cols_to_scale)} columns")
            
            # Ensure columns are float before transformations
            try:
                X_train.loc[:, cols_to_scale] = X_train.loc[:, cols_to_scale].astype(float)
                X_test.loc[:, cols_to_scale] = X_test.loc[:, cols_to_scale].astype(float)
            except Exception:
                pass
            
            if self.transformation_type == 'yeo_johnson':
                # Apply Yeo-Johnson transformation first (matching BasicProcessing exactly)
                lambdas = []
                for col in cols_to_scale:
                    # Fit transformation on training data
                    train_data = X_train[col].values
                    transformed_train, lambda_param = yeojohnson(train_data)
                    X_train.loc[:, col] = transformed_train
                    lambdas.append(lambda_param)
                    
                    # Apply same transformation to test data using scipy (same as BasicProcessing)
                    test_data = X_test[col].values
                    # Use the same yeojohnson function with fixed lambda - this matches BasicProcessing exactly
                    # Note: yeojohnson can accept a fixed lambda parameter
                    try:
                        # For test data, we need to apply the transformation with the fitted lambda
                        # This is more complex than the basic yeojohnson call, but matches what BasicProcessing would do
                        if lambda_param != 0:
                            transformed_test = ((test_data + 1) ** lambda_param - 1) / lambda_param
                        else:
                            transformed_test = np.log(test_data + 1)
                        # Handle negative values
                        transformed_test = np.where(test_data >= -1, transformed_test, 
                                                  np.log(np.abs(test_data) + 1) if lambda_param == 0 else
                                                  ((np.abs(test_data) + 1) ** lambda_param - 1) / lambda_param)
                    except Exception:
                        # Fallback: apply yeojohnson to test data (may not be identical but safer)
                        transformed_test = yeojohnson(test_data, lmbda=lambda_param)
                    
                    X_test.loc[:, col] = transformed_test
                
                transformation_params['lambdas'] = lambdas
            
            # Standardization (always applied after Yeo-Johnson if used)
            # Use manual standardization to match BasicProcessing exactly
            # torch.std() uses unbiased=True by default (ddof=1)
            if self.verbose:
                print("[SimpleOpenMLLoader] Applying manual standardization (ddof=1) to match BasicProcessing torch.std()")
            
            # Calculate means and stds manually with ddof=1 (sample std, like torch.std())
            train_values = X_train[cols_to_scale].values
            means = np.mean(train_values, axis=0, keepdims=True)
            stds = np.std(train_values, axis=0, keepdims=True, ddof=1)  # ddof=1 to match torch.std(unbiased=True)
            
            # Avoid division by zero (same logic as BasicProcessing)
            stds = np.where(stds == 0, 1.0, stds)
            
            # Apply standardization
            X_train_scaled = (train_values - means) / stds
            X_test_scaled = (X_test[cols_to_scale].values - means) / stds
            
            transformation_params.update({'means': means.flatten(), 'stds': stds.flatten()})
            
            X_train.loc[:, cols_to_scale] = X_train_scaled
            X_test.loc[:, cols_to_scale] = X_test_scaled

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_names": feature_names,
            "categorical_features": cat_cols,   # before encoding
            "numerical_features": num_cols,
            "target_column": target,
            "scaler": None,  # We use manual standardization now
            "target_encoder": target_encoder,
            "transformation_type": self.transformation_type,
            "transformation_params": transformation_params,
            "encoding_params": encoding_params,
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
        transformation_type='standardize',  # or 'yeo_johnson'
        use_heuristic_categorical_detection=False,  # Set to True to use BasicProcessing-style detection
        use_leave_one_out_encoding=False,  # Set to True to use BasicProcessing-style encoding
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

