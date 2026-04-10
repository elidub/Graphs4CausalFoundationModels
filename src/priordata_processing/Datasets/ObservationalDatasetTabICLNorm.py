from __future__ import annotations

import sys
import os
from typing import Dict, Any, Optional

import networkx as nx
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from priors.causal_prior.scm.SCMSampler import SCMSampler
from tabicl.prior.reg2cls import Reg2Cls
from utils import FixedSampler, TorchDistributionSampler, CategoricalSampler, DiscreteUniformSampler


# ---------------------------------------------------------------------------
# Default TabICL normalisation hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_TABICL_HP: Dict[str, Any] = {
    "num_classes": 10,           # 0 = keep as regression; 2+ = convert to classification
    "multiclass_type": "rank",   # "rank" or "value"
    "balanced": False,
    "multiclass_ordered_prob": 0.0,
    "cat_prob": 0.2,
    "max_categories": 10,
    "scale_by_max_features": False,
    "permute_features": True,
    "permute_labels": True,
}


class ObservationalDatasetTabICLNorm(Dataset):
    """Synthetic causal dataset using GCFM's DAG/MLP generation pipeline
    with TabICL's Reg2Cls normalization in place of GCFM's BasicProcessing.

    Data generation:
    - Erdos-Renyi DAGs sampled via SCMSampler / GraphSampler
    - MLP (or XGBoost) mechanisms define functions between nodes
    - Mixed noise distributions (Normal, Laplace, StudentT, Gumbel)

    Normalization (replaces GCFM's Preprocessor):
    - Two-pass outlier removal (4σ clamping) on features
    - Z-score standardization clipped to [-100, 100]
    - Optional probabilistic conversion of features to categorical
    - Regression-to-classification target conversion via MulticlassAssigner
    - Feature permutation and zero-padding to max_features

    Unlike GCFM's Preprocessor, Reg2Cls operates on the full sequence
    (train + test combined), which is the standard for in-context learning.

    Parameters
    ----------
    scm_config : Dict[str, Any]
        SCM hyperparameter configuration (same format as SCMSampler / ObservationalDataset).
    dataset_config : Dict[str, Any]
        Dataset parameters: dataset_size, n_features, max_number_features,
        max_number_train_samples_per_dataset, max_number_test_samples_per_dataset,
        number_train_samples_per_dataset, number_test_samples_per_dataset.
    tabicl_hp : Dict[str, Any], optional
        Hyperparameters for Reg2Cls normalization. Merged with DEFAULT_TABICL_HP.
        max_features is automatically set from dataset_config at runtime.
    seed : int, optional
        Global random seed for reproducibility.
    """

    # Expected dataset config parameters and their types
    EXPECTED_DATASET_HYPERPARAMETERS = {
        "dataset_size": int,
        "n_features": (torch.distributions.Distribution, int),
        "max_number_features": int,
        "max_number_samples_per_dataset": int,
        "max_number_train_samples_per_dataset": int,
        "max_number_test_samples_per_dataset": int,
        "number_train_samples_per_dataset": (torch.distributions.Distribution, int),
        "number_test_samples_per_dataset": (torch.distributions.Distribution, int),
    }

    DISTRIBUTION_FACTORIES = {
        "fixed": lambda params: FixedSampler(params["value"]),
        "uniform": lambda params: TorchDistributionSampler(
            dist.Uniform(low=params["low"], high=params["high"])
        ),
        "normal": lambda params: TorchDistributionSampler(
            dist.Normal(loc=params["mean"], scale=params["std"])
        ),
        "lognormal": lambda params: TorchDistributionSampler(
            dist.LogNormal(loc=params["mean"], scale=params["std"])
        ),
        "exponential": lambda params: TorchDistributionSampler(
            dist.Exponential(rate=params["lambd"])
        ),
        "gamma": lambda params: TorchDistributionSampler(
            dist.Gamma(concentration=params["alpha"], rate=params["beta"])
        ),
        "beta": lambda params: TorchDistributionSampler(
            dist.Beta(concentration1=params["alpha"], concentration0=params["beta"])
        ),
        "categorical": lambda params: CategoricalSampler(
            params["choices"], params.get("probabilities")
        ),
        "discrete_uniform": lambda params: DiscreteUniformSampler(params["low"], params["high"]),
    }

    TORCH_DISTRIBUTION_FACTORIES = {
        "uniform": lambda params: dist.Uniform(low=params["low"], high=params["high"]),
        "normal": lambda params: dist.Normal(loc=params["mean"], scale=params["std"]),
        "lognormal": lambda params: dist.LogNormal(loc=params["mean"], scale=params["std"]),
        "exponential": lambda params: dist.Exponential(rate=params["lambd"]),
        "gamma": lambda params: dist.Gamma(concentration=params["alpha"], rate=params["beta"]),
        "beta": lambda params: dist.Beta(concentration1=params["alpha"], concentration0=params["beta"]),
    }

    def __init__(
        self,
        scm_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        tabicl_hp: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        self.scm_config = scm_config
        self.dataset_config = dataset_config
        self.tabicl_hp = {**DEFAULT_TABICL_HP, **(tabicl_hp or {})}
        self.seed = seed

        def _get_cfg_value(cfg, key, default):
            raw = cfg.get(key, default)
            if isinstance(raw, dict) and "value" in raw:
                return raw["value"]
            return raw

        self.max_resample_attempts = int(_get_cfg_value(dataset_config, "max_resample_attempts", 10) or 10)

        # Build SCM sampler
        scm_seed = None
        if seed is not None:
            scm_seed = (seed * 31 + 17) % (2**32)
        self.scm_sampler = SCMSampler(scm_config, seed=scm_seed)

        # Build dataset parameter samplers and extract fixed-size attributes
        dataset_config_filtered = {k: v for k, v in dataset_config.items() if k != "seed"}
        self.dataset_samplers = self._build_samplers(dataset_config_filtered)

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        dataset_params = self._sample_parameters(self.dataset_samplers, generator)

        self.size = dataset_params["dataset_size"]
        self.max_number_features = dataset_params.get("max_number_features")
        self.max_number_train_samples = dataset_params.get(
            "max_number_train_samples_per_dataset",
            dataset_params.get("max_number_train_samples", 0),
        )
        self.max_number_test_samples = dataset_params.get(
            "max_number_test_samples_per_dataset",
            dataset_params.get("max_number_test_samples", 0),
        )

    # ------------------------------------------------------------------
    # Sampler helpers (mirrors ObservationalDataset)
    # ------------------------------------------------------------------

    def _build_samplers(self, config: Dict[str, Any]) -> Dict[str, Any]:
        samplers = {}
        for param_name, param_config in config.items():
            if "value" in param_config and "distribution" not in param_config:
                sampler = FixedSampler(param_config["value"])
            elif "distribution" in param_config:
                dist_type = param_config["distribution"]
                if param_name in ("number_train_samples_per_dataset", "number_test_samples_per_dataset"):
                    if dist_type not in self.TORCH_DISTRIBUTION_FACTORIES:
                        raise ValueError(f"Unknown distribution type for {param_name}: {dist_type}")
                    sampler = self.TORCH_DISTRIBUTION_FACTORIES[dist_type](
                        param_config.get("distribution_parameters", {})
                    )
                else:
                    if dist_type not in self.DISTRIBUTION_FACTORIES:
                        raise ValueError(f"Unknown distribution type: {dist_type}")
                    dist_params = param_config.get("distribution_parameters", {})
                    if dist_type == "fixed":
                        dist_params = {"value": param_config["value"]}
                    sampler = self.DISTRIBUTION_FACTORIES[dist_type](dist_params)
            else:
                raise ValueError(f"Config for {param_name} must specify 'distribution' or 'value'")
            samplers[param_name] = sampler
        return samplers

    def _sample_parameters(self, samplers: Dict[str, Any], generator: torch.Generator) -> Dict[str, Any]:
        sampled = {}
        for param_name, sampler in samplers.items():
            if param_name in ("number_train_samples_per_dataset", "number_test_samples_per_dataset"):
                if isinstance(sampler, FixedSampler):
                    sampled[param_name] = sampler.sample(generator)
                else:
                    sampled[param_name] = sampler  # keep as distribution, sampled per item
            elif hasattr(sampler, "sample"):
                try:
                    value = sampler.sample(generator)
                except TypeError:
                    value = sampler.sample()
                # Coerce int/float
                expected = self.EXPECTED_DATASET_HYPERPARAMETERS.get(param_name)
                if expected is int and isinstance(value, float):
                    value = int(value)
                elif expected is float and isinstance(value, int):
                    value = float(value)
                sampled[param_name] = value
            else:
                sampled[param_name] = sampler
        return sampled

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")

        seed = self.seed + idx if self.seed is not None else idx
        torch.manual_seed(seed)

        item_generator = torch.Generator()
        item_generator.manual_seed(seed)

        # Sample per-item dataset parameters
        dataset_params = self._sample_parameters(self.dataset_samplers, item_generator)

        if "number_train_samples_per_dataset" not in dataset_params:
            dataset_params["number_train_samples_per_dataset"] = (
                self.dataset_config["max_number_train_samples_per_dataset"]["value"]
            )
        if "number_test_samples_per_dataset" not in dataset_params:
            dataset_params["number_test_samples_per_dataset"] = (
                self.dataset_config["max_number_test_samples_per_dataset"]["value"]
            )

        # Resolve n_features
        n_features_raw = dataset_params["n_features"]
        if isinstance(n_features_raw, torch.distributions.Distribution):
            n_features = int(n_features_raw.sample().item())
        elif hasattr(n_features_raw, "sample"):
            n_features = int(n_features_raw.sample(item_generator))
        else:
            n_features = int(n_features_raw)

        # Resolve n_train / n_test
        def _resolve_samples(val):
            if isinstance(val, torch.distributions.Distribution):
                return int(val.sample().item())
            if hasattr(val, "sample"):
                return int(val.sample(item_generator))
            return int(val)

        n_train = _resolve_samples(dataset_params["number_train_samples_per_dataset"])
        n_test  = _resolve_samples(dataset_params["number_test_samples_per_dataset"])
        total_samples = n_train + n_test

        # ------------------------------------------------------------------
        # SCM sampling loop (same graph validation as ObservationalDataset)
        # ------------------------------------------------------------------
        attempt = 0
        scm = None
        scm_data = None
        while True:
            scm = self.scm_sampler.sample(seed=None)

            graph_ok = nx.number_weakly_connected_components(scm.dag.g) < 3
            if not graph_ok:
                attempt += 1
                if attempt >= self.max_resample_attempts:
                    break
                continue

            scm.sample_exogenous(num_samples=total_samples)
            scm.sample_endogenous(num_samples=total_samples)
            scm_data = scm.propagate(num_samples=total_samples)
            break

        # ------------------------------------------------------------------
        # Convert raw SCM dict → (X_all, y_all)  [replaces BasicProcessing]
        # ------------------------------------------------------------------
        all_nodes = sorted(scm_data.keys())

        # Stack all nodes: [total_samples, num_nodes]
        X_all = torch.cat([scm_data[n].reshape(total_samples, 1) for n in all_nodes], dim=1)

        # Randomly select one node as target, remove it from features
        target_idx = int(torch.randint(len(all_nodes), (1,), generator=item_generator).item())
        y_all = X_all[:, target_idx]                                          # [total_samples]
        X_all = torch.cat([X_all[:, :target_idx], X_all[:, target_idx+1:]], dim=1)  # [total_samples, num_nodes-1]

        # Shuffle rows
        row_perm = torch.randperm(total_samples, generator=item_generator)
        X_all = X_all[row_perm]
        y_all = y_all[row_perm]

        # Truncate features to n_features (SCM may have more nodes than requested)
        actual_features = min(n_features, X_all.shape[1])
        X_all = X_all[:, :actual_features]

        # ------------------------------------------------------------------
        # Apply Reg2Cls normalisation on the full sequence
        # ------------------------------------------------------------------
        hp = {**self.tabicl_hp, "max_features": self.max_number_features}
        reg2cls = Reg2Cls(hp)

        # Dummy adj — we ignore the returned adj entirely
        adj_dummy = torch.zeros(actual_features + 1, actual_features + 1)

        X_norm, y_norm, _ = reg2cls(X_all, y_all, adj_dummy)
        # X_norm: [total_samples, max_number_features]
        # y_norm: [total_samples]

        # ------------------------------------------------------------------
        # Split into train / test and zero-pad to max sample counts
        # ------------------------------------------------------------------
        X_train_raw = X_norm[:n_train]           # [n_train, max_features]
        y_train_raw = y_norm[:n_train]            # [n_train]
        X_test_raw  = X_norm[n_train:n_train + n_test]
        y_test_raw  = y_norm[n_train:n_train + n_test]

        max_train = self.max_number_train_samples
        max_test  = self.max_number_test_samples

        pad_train = max_train - n_train
        pad_test  = max_test  - n_test

        X_train = F.pad(X_train_raw, (0, 0, 0, pad_train))          # [max_train, max_features]
        Y_train = F.pad(y_train_raw.unsqueeze(-1), (0, 0, 0, pad_train))  # [max_train, 1]
        X_test  = F.pad(X_test_raw,  (0, 0, 0, pad_test))
        Y_test  = F.pad(y_test_raw.unsqueeze(-1),  (0, 0, 0, pad_test))

        graph_info = {
            "scm": scm,
            "processor": None,
            "ordered_nodes": None,
        }
        dataset_info = {
            "number_train_samples": n_train,
        }

        return X_train, Y_train, X_test, Y_test, graph_info, dataset_info
