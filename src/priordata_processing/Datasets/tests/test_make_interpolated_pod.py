import os
import sys
import torch

# Ensure project src root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from priordata_processing.Datasets.MakeInterpolatedPurelyObservationalDataset import (
    MakeInterpolatedPurelyObservationalDataset,
)


def alpha_step(t: float) -> float:
    """Alpha schedule: 0 for t < 0.5, 1 for t >= 0.5."""
    return 0.0 if t < 0.5 else 1.0


def make_configs():
    # Minimal SCM configs (keep identical to avoid SCM build instability in this test)
    scm_common = {
        "num_nodes": {"value": 5},
        "graph_edge_prob": {"value": 0.2},
        "graph_seed": {"value": 42},
        # keep other params default by omission in builder
    }

    scm_config_t0 = scm_common
    scm_config_t1 = scm_common

    # Preprocessing configs (fixed and identical across t0/t1 for simplicity)
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

    preprocessing_config_t0 = preproc
    preprocessing_config_t1 = preproc

    # Dataset configs with different, non-overlapping supports (deterministic values)
    dataset_config_t0 = {
        "dataset_size": {"value": 6},
        "max_number_features": {"value": 5},
        "max_number_train_samples": {"value": 8},
        "max_number_test_samples": {"value": 4},
        "number_train_samples_per_dataset": {"value": 8},
        "number_test_samples_per_dataset": {"value": 4},
        "seed": {"value": 123},
    }

    dataset_config_t1 = {
        "dataset_size": {"value": 6},
        "max_number_features": {"value": 7},
        "max_number_train_samples": {"value": 16},
        "max_number_test_samples": {"value": 2},
        "number_train_samples_per_dataset": {"value": 16},
        "number_test_samples_per_dataset": {"value": 2},
        "seed": {"value": 123},
    }

    return (
        scm_config_t0,
        scm_config_t1,
        preprocessing_config_t0,
        preprocessing_config_t1,
        dataset_config_t0,
        dataset_config_t1,
    )


def test_interpolated_dataset_shapes_switch_with_alpha():
    (
        scm_config_t0,
        scm_config_t1,
        preprocessing_config_t0,
        preprocessing_config_t1,
        dataset_config_t0,
        dataset_config_t1,
    ) = make_configs()

    factory = MakeInterpolatedPurelyObservationalDataset(
        scm_config_t0=scm_config_t0,
        scm_config_t1=scm_config_t1,
        preprocessing_config_t0=preprocessing_config_t0,
        preprocessing_config_t1=preprocessing_config_t1,
        dataset_config_t0=dataset_config_t0,
        dataset_config_t1=dataset_config_t1,
        alpha=alpha_step,
        seed=77,
    )

    ds = factory.create_dataset()

    # Early indices (t<0.5): expect t0 shapes
    for idx in [0, 1, 2]:
        Xtr, Ytr, Xte, Yte = ds[idx]
        assert Xtr.shape[0] == 8, f"idx={idx}: expected 8 train rows, got {Xtr.shape}"
        assert Xtr.shape[1] == 5, f"idx={idx}: expected 5 features, got {Xtr.shape}"
        assert Xte.shape[0] == 4, f"idx={idx}: expected 4 test rows, got {Xte.shape}"
        assert Xte.shape[1] == 5, f"idx={idx}: expected 5 features, got {Xte.shape}"

    # Later indices (t>=0.5): expect t1 shapes
    for idx in [3, 4, 5]:
        Xtr, Ytr, Xte, Yte = ds[idx]
        assert Xtr.shape[0] == 16, f"idx={idx}: expected 16 train rows, got {Xtr.shape}"
        assert Xtr.shape[1] == 7, f"idx={idx}: expected 7 features, got {Xtr.shape}"
        assert Xte.shape[0] == 2, f"idx={idx}: expected 2 test rows, got {Xte.shape}"
        assert Xte.shape[1] == 7, f"idx={idx}: expected 7 features, got {Xte.shape}"


if __name__ == "__main__":
    # Run the test directly without pytest
    test_interpolated_dataset_shapes_switch_with_alpha()
    print("✓ Interpolated dataset test passed")
