"""
Simple OpenML dataset loader (download-only)

- Downloads an OpenML dataset by id (once) and caches the raw CSV
- Resolves OpenML task -> dataset id and target column
- Returns raw pandas DataFrames and metadata

All preprocessing, subsampling, and benchmarking are handled in Benchmark.py.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import openml

# Default Tabular numerical regression task IDs (from the user's list)
DEFAULT_TABULAR_NUM_REG_TASKS = [
    361072, 361073, 361074, 361075, 361076, 361077, 361078, 361079, 361080,
    361081, 361082, 361083, 361084, 361085, 361086, 361087, 361088,
    361279, 361280, 361281
]


class SimpleOpenMLLoader:
    def __init__(
        self,
        data_dir: str = "data",
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ---------- public API ----------

    def download_dataset(
        self,
        dataset_id: int,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download and cache raw dataset CSV. Return raw DataFrame and metadata.
        Returns dict with keys: df (pandas.DataFrame), target_column (str), dataset_id (int)
        """
        ds_dir = self.data_dir / f"openml_{dataset_id}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = ds_dir / "raw.csv"

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

            try:
                X, y, _, _ = omlds.get_data(dataset_format="dataframe", target=target_column)
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    df = pd.concat([X, y.rename(target_column)], axis=1)
                else:
                    df = pd.concat([X, pd.Series(y, name=target_column)], axis=1)
            except Exception:
                df = omlds.get_data()[0]
                if target_column not in df.columns:
                    target_column = df.columns[-1]

            df.to_csv(raw_csv, index=False)
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Raw saved → {raw_csv}")

        return {
            "df": df,
            "target_column": target_column,
            "dataset_id": int(dataset_id),
        }

    def resolve_task(self, task_id: int) -> Tuple[int, Optional[str]]:
        """Resolve task to (dataset_id, target_name)."""
        task = openml.tasks.get_task(int(task_id))
        dataset_id = getattr(task, "dataset_id", None)
        target_name = getattr(task, "target_name", None)

        if dataset_id is None:
            try:
                ds_obj = task.get_dataset()
                dataset_id = getattr(ds_obj, "dataset_id", None) or getattr(ds_obj, "id", None)
            except Exception:
                dataset_id = None

        if dataset_id is None:
            raise ValueError(f"Could not resolve dataset id for task {task_id}")

        if target_name is None:
            try:
                omlds = openml.datasets.get_dataset(int(dataset_id))
                target_name = omlds.default_target_attribute
            except Exception:
                target_name = None

        return int(dataset_id), target_name

    def load_tasks_raw(self, task_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Download raw datasets for the given OpenML task ids and return raw dataframes with metadata."""
        results: Dict[int, Dict[str, Any]] = {}
        for tid in task_ids:
            try:
                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Resolving task {tid} ...")
                dataset_id, target_name = self.resolve_task(int(tid))
                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Task {tid} -> dataset {dataset_id}, target='{target_name}'")
                loaded = self.download_dataset(dataset_id, target_name)
                results[int(tid)] = loaded
            except Exception as e:
                print(f"[SimpleOpenMLLoader] Error loading task {tid}: {e}")
        return results

    # Note: subsampling and preprocessing moved to Benchmark class.

    # No preprocess logic here anymore.

    # Kept for backward compatibility if needed in the future: not used here
    @staticmethod
    def _to_numpy(x):
        if isinstance(x, (pd.Series, pd.DataFrame)):
            return x.to_numpy()
        import numpy as np
        return np.asarray(x)


if __name__ == "__main__":
    print("=== SimpleOpenMLLoader download-only demo ===")
    loader = SimpleOpenMLLoader(
        data_dir="data_cache",
        random_state=42,
        verbose=True,
    )
    try:
        ds = loader.download_dataset(531)
        print(f"Raw df shape: {ds['df'].shape}, target: {ds['target_column']}")
    except Exception as e:
        print(f"Error loading dataset: {e}")

