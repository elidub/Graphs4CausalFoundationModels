"""
Test script to check the fraction of datasets with binary treatment variables.

This script samples from InterventionalDataset and counts how many have binary treatments.
"""

import sys
import os
from pathlib import Path

# Add src to path
repo_root = Path(__file__).resolve().parent
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import torch
import numpy as np
from collections import Counter

from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset


def is_binary(tensor, tolerance=1e-6):
    """
    Check if a tensor contains only two unique values (binary).
    
    Args:
        tensor: Input tensor
        tolerance: Tolerance for floating point comparison
        
    Returns:
        bool: True if binary, False otherwise
    """
    if tensor is None:
        return False
    
    # Flatten tensor
    values = tensor.flatten().numpy() if torch.is_tensor(tensor) else tensor.flatten()
    
    # Get unique values
    unique_values = np.unique(values)
    
    # Binary if exactly 2 unique values
    return len(unique_values) == 2


def count_unique_values(tensor):
    """Count number of unique values in tensor."""
    if tensor is None:
        return 0
    values = tensor.flatten().numpy() if torch.is_tensor(tensor) else tensor.flatten()
    return len(np.unique(values))


def main():
    # Configuration from final_earlytest.yaml
    scm_config = {
        "num_nodes": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 2, "high": 51}
        },
        "graph_edge_prob": {
            "distribution": "beta",
            "distribution_parameters": {"alpha": 2.0, "beta": 3.0}
        },
        "graph_seed": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 0, "high": 100000}
        },
        "xgboost_prob": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": [0.0, 0.1, 0.2, 0.3],
                "probabilities": [0.9, 0.1, 0.01, 0.001]
            }
        },
        "mechanism_seed": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 0, "high": 100000}
        },
        "mlp_nonlins": {"value": "tabicl"},
        "mlp_num_hidden_layers": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": [0, 1, 2, 3],
                "probabilities": [0.875, 0.1, 0.025, 0.01]
            }
        },
        "mlp_hidden_dim": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": [1, 2, 4, 6, 8, 10, 12, 14, 16, 32],
                "probabilities": [0.7, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01]
            }
        },
        "mlp_activation_mode": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": ["pre", "post", "mixed_in"],
                "probabilities": [0.3, 0.3, 0.3]
            }
        },
        "mlp_use_batch_norm": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": [True, False],
                "probabilities": [0.5, 0.5]
            }
        },
        "xgb_node_shape": {"value": [1]},
        "xgb_num_hidden_layers": {"value": 0},
        "xgb_hidden_dim": {
            "distribution": "categorical",
            "distribution_parameters": {"choices": [0, 16, 32, 64]}
        },
        "xgb_activation_mode": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": ["pre", "post", "mixed_in"],
                "probabilities": [0.33, 0.33, 0.34]
            }
        },
        "xgb_use_batch_norm": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": [True, False],
                "probabilities": [0.5, 0.5]
            }
        },
        "xgb_n_training_samples": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": [10, 50, 100, 200, 500],
                "probabilities": [0.1, 0.1, 0.3, 0.4, 0.5]
            }
        },
        "xgb_add_noise": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": [True, False],
                "probabilities": [0.5, 0.5]
            }
        },
        "random_additive_std": {"value": True},
        "exo_std_distribution": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": ["gamma", "pareto"],
                "probabilities": [1.0, 0.0]
            }
        },
        "endo_std_distribution": {
            "distribution": "categorical",
            "distribution_parameters": {
                "choices": ["gamma", "pareto"],
                "probabilities": [1.0, 0.0]
            }
        },
        "exo_std_mean": {
            "distribution": "lognormal",
            "distribution_parameters": {"mean": 1.0, "std": 1.0}
        },
        "exo_std_std": {
            "distribution": "uniform",
            "distribution_parameters": {"low": 0.1, "high": 0.4}
        },
        "endo_std_mean": {
            "distribution": "lognormal",
            "distribution_parameters": {"mean": -3.0, "std": 0.6}
        },
        "endo_std_std": {
            "distribution": "uniform",
            "distribution_parameters": {"low": 0.0, "high": 0.5}
        },
        "endo_p_zero": {"value": 0.0},
        "noise_mixture_proportions": {"value": [0.33, 0.33, 0.34]},
        "use_exogenous_mechanisms": {"value": True},
        "mechanism_generator_seed": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 0, "high": 100000}
        },
    }

    dataset_config = {
        "dataset_size": {"value": 1000},  # We'll sample 1000 datasets
        "max_number_samples_per_dataset": {"value": 500},
        "max_number_train_samples_per_dataset": {"value": 1000},
        "max_number_test_samples_per_dataset": {"value": 1000},
        "n_test_samples_per_dataset": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 1, "high": 500}
        },
        "return_adjacency_matrix": {"value": False},
        "return_ancestor_matrix": {"value": True},
        "use_partial_graph_format": {"value": True},
        "hide_fraction_matrix": {
            "distribution": "uniform",
            "distribution_parameters": {"low": 0.0, "high": 1.0}
        },
        "min_target_variance": {"value": 1e-2},
        "min_unique_target_fraction": {"value": 0.2},
        "max_resample_attempts": {"value": 100},
        # Add binarize_treatment_prob - 50% chance of binary treatment
        "binarize_treatment_prob": {"value": 0.5},
        "max_number_features": {"value": 50},
        "seed": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 0, "high": 100000}
        },
    }

    preprocessing_config = {
        "dropout_prob": {"value": 0.0},
        "shuffle_data": {"value": True},
        "target_feature": {"value": None},
        "feature_standardize": {"value": True},
        "feature_negative_one_one_scaling": {"value": False},
        "target_negative_one_one_scaling": {"value": True},
        "remove_outliers": {"value": True},
        "outlier_quantile": {"value": 0.99},
        "yeo_johnson": {"value": False},
        "increase_treatment_scale": {"value": False},
        "distribution_rescale_factor": {"value": 0.0},
        "interventional_distribution_type": {"value": "resample"},
        "test_feature_mask_fraction": {"value": 1.0},
        "random_seed": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 0, "high": 100000}
        },
    }

    print("=" * 80)
    print("Binary Treatment Variable Fraction Test")
    print("=" * 80)
    
    n_samples = 1000
    print(f"\nSampling {n_samples} datasets from InterventionalDataset...")
    
    # Create dataset
    dataset = InterventionalDataset(
        scm_config=scm_config,
        preprocessing_config=preprocessing_config,
        dataset_config=dataset_config,
        seed=42,
        return_scm=False,
    )
    
    # Counters
    binary_treatment_count = 0
    unique_value_counts = []
    
    # Sample datasets
    for i in range(min(n_samples, len(dataset))):
        if i % 100 == 0:
            print(f"  Processing sample {i}/{n_samples}...")
        
        try:
            # Get sample: (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, graph_matrix)
            sample = dataset[i]
            
            # T_obs is at index 1
            T_obs = sample[1]
            
            n_unique = count_unique_values(T_obs)
            unique_value_counts.append(n_unique)
            
            if is_binary(T_obs):
                binary_treatment_count += 1
                
        except Exception as e:
            print(f"  Error at sample {i}: {e}")
            continue
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    total_sampled = len(unique_value_counts)
    binary_fraction = binary_treatment_count / total_sampled if total_sampled > 0 else 0
    
    print(f"\nTotal datasets sampled: {total_sampled}")
    print(f"Datasets with binary treatment: {binary_treatment_count}")
    print(f"Binary treatment fraction: {binary_fraction:.2%}")
    
    # Distribution of unique values
    print("\nDistribution of unique values in treatment variable:")
    value_distribution = Counter(unique_value_counts)
    for n_unique in sorted(value_distribution.keys()):
        count = value_distribution[n_unique]
        pct = count / total_sampled * 100
        print(f"  {n_unique} unique values: {count} datasets ({pct:.1f}%)")
    
    # Summary statistics
    if unique_value_counts:
        print(f"\nSummary statistics for unique value counts:")
        print(f"  Min: {min(unique_value_counts)}")
        print(f"  Max: {max(unique_value_counts)}")
        print(f"  Mean: {np.mean(unique_value_counts):.1f}")
        print(f"  Median: {np.median(unique_value_counts):.1f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
