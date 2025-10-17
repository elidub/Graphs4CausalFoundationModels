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
import json
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
try:
    import openml  # type: ignore
except Exception:
    openml = None

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
        offline: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        # Offline mode: never attempt OpenML API calls when True
        env_offline = os.environ.get("OPENML_OFFLINE") == "1" or os.environ.get("DATA_CACHE_ONLY") == "1"
        self.offline = bool(offline or env_offline)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Cached task->(dataset_id, target_name) mapping to avoid network calls when data exists
        self._map_path = self.data_dir / "task_to_dataset.json"
        self._task_map: Dict[str, Dict[str, Optional[str]]] = {}
        if self._map_path.exists():
            try:
                self._task_map = json.load(open(self._map_path, "r")) or {}
                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Loaded task map with {len(self._task_map)} entries from {self._map_path}")
            except Exception as e:
                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Failed to load task map: {e}")

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
        # Cache-first: if we have a mapping and the dataset is present on disk, avoid any network calls
        key = str(int(task_id))
        if key in self._task_map:
            entry = self._task_map[key]
            dataset_id = entry.get("dataset_id")
            target_name = entry.get("target_name")
            if dataset_id is not None:
                ds_dir = self.data_dir / f"openml_{int(dataset_id)}"
                raw_csv = ds_dir / "raw.csv"
                if raw_csv.exists():
                    if self.verbose:
                        print(f"[SimpleOpenMLLoader] Using cached mapping for task {task_id} -> dataset {dataset_id} (raw.csv found)")
                    return int(dataset_id), target_name

        # If strictly offline, do not attempt any API calls
        if self.offline:
            raise RuntimeError(
                f"Offline mode: no mapping/raw.csv for task {task_id}; cannot resolve without OpenML API."
            )

        if openml is None:
            raise RuntimeError("openml package not available and offline mapping missing; cannot resolve task")

        # Fallback to OpenML API resolution
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

        # Persist mapping for future offline/cache-first load
        try:
            self._task_map[key] = {
                "dataset_id": int(dataset_id),
                "target_name": target_name,
            }
            with open(self._map_path, "w") as f:
                json.dump(self._task_map, f)
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Updated task map: task {task_id} -> dataset {dataset_id}, target='{target_name}'")
        except Exception as e:
            if self.verbose:
                print(f"[SimpleOpenMLLoader] Failed to update task map: {e}")
        return int(dataset_id), target_name

    def load_tasks_raw(self, task_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Download raw datasets for the given OpenML task ids and return raw dataframes with metadata."""
        results: Dict[int, Dict[str, Any]] = {}
        for tid in task_ids:
            try:
                if self.verbose:
                    print(f"[SimpleOpenMLLoader] Resolving task {tid} ...")
                # Try cache-first resolution without network
                key = str(int(tid))
                dataset_id = None
                target_name = None
                if key in self._task_map:
                    entry = self._task_map[key]
                    dataset_id = entry.get("dataset_id")
                    target_name = entry.get("target_name")
                    # Only trust mapping if raw file exists
                    if dataset_id is not None:
                        ds_dir = self.data_dir / f"openml_{int(dataset_id)}"
                        raw_csv = ds_dir / "raw.csv"
                        if raw_csv.exists():
                            if self.verbose:
                                print(f"[SimpleOpenMLLoader] Cache-hit for task {tid}: dataset {dataset_id} (no API)")
                        else:
                            dataset_id = None  # force resolve

                # If no valid cache, resolve via API (will also persist mapping)
                if dataset_id is None:
                    try:
                        dataset_id, target_name = self.resolve_task(int(tid))
                    except RuntimeError as e:
                        # In offline mode with missing mapping, skip cleanly
                        print(f"[SimpleOpenMLLoader] Offline skip for task {tid}: {e}")
                        continue
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
    print("=== SimpleOpenMLLoader: load default OpenML regression tasks ===")
    loader = SimpleOpenMLLoader(
        data_dir="data_cache",
        random_state=42,
        verbose=True,
    )

    task_ids = DEFAULT_TABULAR_NUM_REG_TASKS
    print(f"Attempting to load {len(task_ids)} tasks: {task_ids}")

    results = loader.load_tasks_raw(task_ids)

    print("\nSummary:")
    print(f"Loaded {len(results)}/{len(task_ids)} tasks successfully")
    for tid in sorted(results.keys()):
        info = results[tid]
        df = info["df"]
        print(
            f"- Task {tid}: dataset_id={info['dataset_id']}, "
            f"target='{info['target_column']}', shape={df.shape}"
        )

