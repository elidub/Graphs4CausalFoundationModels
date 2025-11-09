import os
import sys
import math
from typing import Tuple

import torch

# Ensure project src on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from priordata_processing.Datasets.MakeInterpolatedPurelyObservationalDataset import (
    MakeInterpolatedPurelyObservationalDataset,
)


def alpha_step(t: float) -> float:
    return 0.0 if t < 0.5 else 1.0


def alpha_linear(t: float) -> float:
    return max(0.0, min(1.0, float(t)))


def make_base_configs() -> Tuple[dict, dict, dict, dict, dict, dict]:
    scm_common = {
        "num_nodes": {"value": 4},
        "graph_edge_prob": {"value": 0.3},
        "graph_seed": {"value": 101},
    }

    preproc = {
        "dropout_prob": {"value": 0.0},
        "shuffle_data": {"value": False},
        "target_feature": {"value": None},
        "random_seed": {"value": None},
        "negative_one_one_scaling": {"value": False},
        "remove_outliers": {"value": False},
        "outlier_quantile": {"value": 1.0},
        "yeo_johnson": {"value": False},
        "standardize": {"value": False},
    }

    # Distinct supports
    ds0 = {
        "dataset_size": {"value": 10},
        "max_number_features": {"value": 6},
        "max_number_train_samples": {"value": 12},
        "max_number_test_samples": {"value": 3},
        "number_train_samples_per_dataset": {"value": 12},
        "number_test_samples_per_dataset": {"value": 3},
        "seed": {"value": 1},
    }

    ds1 = {
        "dataset_size": {"value": 10},
        "max_number_features": {"value": 9},
        "max_number_train_samples": {"value": 5},
        "max_number_test_samples": {"value": 8},
        "number_train_samples_per_dataset": {"value": 5},
        "number_test_samples_per_dataset": {"value": 8},
        "seed": {"value": 1},
    }

    return scm_common, scm_common, preproc, preproc, ds0, ds1


def test_reproducibility_fixed_seed():
    configs = make_base_configs()
    factory1 = MakeInterpolatedPurelyObservationalDataset(*configs, alpha=alpha_step, seed=999)
    factory2 = MakeInterpolatedPurelyObservationalDataset(*configs, alpha=alpha_step, seed=999)

    ds1 = factory1.create_dataset()
    ds2 = factory2.create_dataset()

    for idx in range(10):
        a = ds1[idx]
        b = ds2[idx]
        # Compare tensors element-wise
        for ta, tb in zip(a, b):
            assert torch.allclose(ta, tb), f"Mismatch at idx={idx}"


def test_boundary_step_schedule_edges():
    configs = make_base_configs()
    factory = MakeInterpolatedPurelyObservationalDataset(*configs, alpha=alpha_step, seed=7)
    ds = factory.create_dataset()

    # dataset_size=10 -> t(i) = i/9, so i=4 => t~0.444, i=5 => t~0.556
    for idx in [0, 1, 2, 3, 4]:
        Xtr, Ytr, Xte, Yte = ds[idx]
        assert Xtr.shape[0] == 12 and Xte.shape[0] == 3 and Xtr.shape[1] == 6 and Xte.shape[1] == 6
    for idx in [5, 6, 7, 8, 9]:
        Xtr, Ytr, Xte, Yte = ds[idx]
        assert Xtr.shape[0] == 5 and Xte.shape[0] == 8 and Xtr.shape[1] == 9 and Xte.shape[1] == 9


def test_probabilistic_mixing_midpoint_counts():
    # At t ~ 0.5 with linear alpha, probability of t1 is ~0.5; we sample multiple independent datasets
    # and check that the proportion of t1 selections is within a tolerance.
    configs = make_base_configs()

    def build_and_get_midpoint_choice(seed: int) -> int:
        factory = MakeInterpolatedPurelyObservationalDataset(*configs, alpha=alpha_linear, seed=seed)
        ds = factory.create_dataset()
        # idx such that t ~ 0.5 -> with size=10, i=5 gives t=5/9~0.556
        Xtr, Ytr, Xte, Yte = ds[5]
        # If we got t0 or t1 based on train size (distinct 12 vs 5)
        return int(Xtr.shape[0] == 5)

    trials = 60
    got_t1 = sum(build_and_get_midpoint_choice(1000 + i) for i in range(trials))
    p_hat = got_t1 / trials
    # Expect near 0.5 with some tolerance
    assert 0.3 <= p_hat <= 0.7, f"Midpoint t1 selection rate out of expected range: {p_hat}"


def test_values_always_from_t0_or_t1():
    configs = make_base_configs()
    factory = MakeInterpolatedPurelyObservationalDataset(*configs, alpha=alpha_linear, seed=3)
    ds = factory.create_dataset()

    t0_train, t1_train = 12, 5
    t0_test, t1_test = 3, 8
    t0_feat, t1_feat = 6, 9

    for idx in range(len(ds)):
        Xtr, Ytr, Xte, Yte = ds[idx]
        assert Xtr.shape[0] in {t0_train, t1_train}
        assert Xte.shape[0] in {t0_test, t1_test}
        assert Xtr.shape[1] in {t0_feat, t1_feat}
        assert Xte.shape[1] in {t0_feat, t1_feat}


if __name__ == "__main__":
    # Run directly
    test_reproducibility_fixed_seed()
    test_boundary_step_schedule_edges()
    test_probabilistic_mixing_midpoint_counts()
    test_values_always_from_t0_or_t1()
    print("✓ Extended interpolated dataset tests passed")
