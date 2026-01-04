#!/usr/bin/env python3
"""
Generate LinGausIDK configs with fixed hide_fraction_matrix values.

Structure:
- configs/
  - {N}node/
    - base.yaml                        (keep uniform distribution)
    - path_TY/
      - hide_0.0.yaml
      - hide_0.25.yaml
      - hide_0.5.yaml
      - hide_0.75.yaml
      - hide_1.0.yaml
    - path_YT/
      - hide_0.0.yaml
      - ...
    - path_independent_TY/
      - hide_0.0.yaml
      - ...
"""

import os
import yaml
from pathlib import Path

BASE_DIR = Path("/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/GraphConditioning/Benchmarks/LinGausIDK/configs")

NODE_COUNTS = [2, 5, 10, 20, 35, 50]

# Path variants with their constraint settings
PATH_VARIANTS = {
    "path_TY": {
        "ensure_treatment_outcome_path": True,
        "ensure_outcome_treatment_path": False,
        "ensure_no_connection_treatment_outcome": False,
    },
    "path_YT": {
        "ensure_treatment_outcome_path": False,
        "ensure_outcome_treatment_path": True,
        "ensure_no_connection_treatment_outcome": False,
    },
    "path_independent_TY": {
        "ensure_treatment_outcome_path": False,
        "ensure_outcome_treatment_path": False,
        "ensure_no_connection_treatment_outcome": True,
    },
}

# Fixed hide fraction values
HIDE_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def create_config(node_count, variant_name, variant_settings, hide_fraction):
    """Create a config dictionary for given node count, variant, and hide fraction."""
    
    num_features = max(1, node_count - 3)
    
    config = {
        "experiment_name": f"lingaus_idk_{node_count}node_{variant_name}_hide{hide_fraction}",
        "description": f"{node_count}-node Linear-Gaussian SCM with partial graph knowledge ({variant_name}, hide_fraction={hide_fraction})",
        "mode": "interventional",
        
        "scm_config": {
            "num_nodes": {"value": node_count},
            "graph_edge_prob": {
                "distribution": "beta",
                "distribution_parameters": {"alpha": 2.0, "beta": 3.0}
            },
            "graph_seed": {
                "distribution": "discrete_uniform",
                "distribution_parameters": {"low": 0, "high": 100000}
            },
            "xgboost_prob": {"value": 0.0},
            "mechanism_seed": {
                "distribution": "discrete_uniform",
                "distribution_parameters": {"low": 0, "high": 100000}
            },
            "mlp_nonlins": {"value": "id"},
            "mlp_num_hidden_layers": {
                "distribution": "categorical",
                "distribution_parameters": {
                    "choices": [0],
                    "probabilities": [1.0]
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
                    "probabilities": [0.5, 0.5, 0.0]
                }
            },
            "mlp_use_batch_norm": {
                "distribution": "categorical",
                "distribution_parameters": {
                    "choices": [True, False],
                    "probabilities": [1.0, 0.0]
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
            "noise_mixture_proportions": {"value": [1.0, 0.0, 0.0]},
            "use_exogenous_mechanisms": {"value": False},
            "mechanism_generator_seed": {
                "distribution": "discrete_uniform",
                "distribution_parameters": {"low": 0, "high": 100000}
            }
        },
        
        "dataset_config": {
            "dataset_size": {"value": 100000000},
            "max_number_samples_per_dataset": {"value": 1000},
            "max_number_train_samples_per_dataset": {"value": 1000},
            "max_number_test_samples_per_dataset": {"value": 1000},
            "n_test_samples_per_dataset": {"value": 500},
            
            # PARTIAL GRAPH CONFIG - FIXED hide_fraction_matrix
            "return_adjacency_matrix": {"value": True},
            "use_partial_graph_format": {"value": True},
            "hide_fraction_matrix": {"value": hide_fraction},  # FIXED VALUE
            
            "min_target_variance": {"value": 1e-3},
            "max_resample_attempts": {"value": 10000},
            "max_number_features": {"value": 50},
            "seed": {
                "distribution": "discrete_uniform",
                "distribution_parameters": {"low": 0, "high": 100000}
            }
        },
        
        "preprocessing_config": {
            "dropout_prob": {
                "distribution": "categorical",
                "distribution_parameters": {
                    "choices": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    "probabilities": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                }
            },
            "shuffle_data": {"value": True},
            "target_feature": {"value": None},
            "feature_standardize": {"value": True},
            "feature_negative_one_one_scaling": {"value": False},
            "target_negative_one_one_scaling": {"value": True},
            "remove_outliers": {"value": True},
            "outlier_quantile": {"value": 0.99},
            "yeo_johnson": {"value": False},
            "interventional_distribution_type": {"value": "scaled_uniform"},
            "increase_treatment_scale": {"value": False},
            "distribution_rescale_factor": {"value": 0.0},
            
            # PATH CONSTRAINT SETTINGS
            "ensure_treatment_outcome_path": {"value": variant_settings["ensure_treatment_outcome_path"]},
            "ensure_outcome_treatment_path": {"value": variant_settings["ensure_outcome_treatment_path"]},
            "ensure_no_connection_treatment_outcome": {"value": variant_settings["ensure_no_connection_treatment_outcome"]},
            
            "test_feature_mask_fraction": {"value": 1.0},
            "random_seed": {
                "distribution": "discrete_uniform",
                "distribution_parameters": {"low": 0, "high": 100000}
            }
        },
        
        "model_config": {
            "d_model": {"value": 256},
            "depth": {"value": 6},
            "heads_feat": {"value": 4},
            "heads_samp": {"value": 4},
            "dropout": {"value": 0.0},
            "hidden_mult": {"value": 2},
            "num_features": {"value": num_features},
            "normalize_features": {"value": True},
            "use_graph_conditioning": {"value": False},
            "graph_conditioning_mode": {"value": "partial_gcn_and_soft_attention"},
            "gcn_use_transpose": {"value": False},
            "gcn_alpha_init": {"value": 0.1},
            "n_sample_attention_sink_rows": {"value": 10},
            "n_feature_attention_sink_cols": {"value": 0},
            "use_bar_distribution": {"value": True},
            "num_bars": {"value": 1000},
            "min_width": {"value": "1e-6"},
            "scale_floor": {"value": "1e-3"},
            "max_fit_items": {"value": 1000},
            "max_fit_batches": {"value": 1000},
            "log_prob_clip_min": {"value": -100000.0},
            "log_prob_clip_max": {"value": 100000.0}
        },
        
        "training_config": {
            "learning_rate": {"value": 0.0001},
            "weight_decay": {"value": 0.00001},
            "max_steps": {"value": 50000},
            "max_epochs": {"value": 1},
            "batch_size": {"value": 8},
            "accumulate_grad_batches": {"value": 4},
            "num_workers": {"value": 70},
            "early_stopping_patience": {"value": 0},
            "gradient_clip_val": {"value": 1.0},
            "scheduler_type": {"value": "linear_warmup_cosine_decay"},
            "warmup_ratio": {"value": 0.1},
            "min_lr_ratio": {"value": 0.1},
            "checkpoint_dir": {"value": "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/GraphConditioning/Benchmarks/LinGausIDK/checkpoints"},
            "save_every": {"value": 1000},
            "device": {"value": "cuda"},
            "use_amp": {"value": False},
            "precision": {"value": "32"},
            "experiment_name": {"value": f"lingaus_idk_{node_count}node_{variant_name}_hide{hide_fraction}"},
            "save_dir": {"value": f"./experiments/GraphConditioning/Benchmarks/LinGausIDK/logs/{node_count}node_{variant_name}_hide{hide_fraction}"},
            "log_every_n_steps": {"value": 1},
            "checkpoint_every_n_steps": {"value": 1000},
            "eval_enabled": {"value": True},
            "eval_every": {"value": 1000},
            "eval_batches": {"value": 200},
            "eval_dataset_size": {"value": 1000000},
            "benchmark_eval_fidelity": {"value": "minimal"},
            "benchmark_final_fidelity": {"value": "minimal"},
            "benchmark_offline": {"value": True},
            "benchmark_n_estimators": {"value": 1},
            "model_selection_enabled": {"value": True},
            "model_selection_metric": {"value": "eval/mse_median"},
            "model_selection_mode": {"value": "min"},
            "wandb_project": {"value": "DoPFN_v2_LinGausIDK"},
            "wandb_run_name": {"value": f"lingaus_idk_{node_count}node_{variant_name}_hide{hide_fraction}"},
            "wandb_tags": {"value": ["lingaus_idk", "partial_graph", f"{node_count}node", f"{variant_name}_variant", f"hide{hide_fraction}"]},
            "wandb_notes": {"value": f"LinGausIDK: {node_count}-node Linear-Gaussian SCM with {variant_name}, hide_fraction={hide_fraction}"},
            "wandb_offline": {"value": False}
        }
    }
    
    return config


def generate_path_variant_configs():
    """Generate configs for path variants with fixed hide fractions."""
    
    print("Generating path variant configs with fixed hide_fraction_matrix values...")
    print("=" * 80)
    
    total_generated = 0
    
    for node_count in NODE_COUNTS:
        for variant_name, variant_settings in PATH_VARIANTS.items():
            variant_dir = BASE_DIR / f"{node_count}node" / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            
            for hide_fraction in HIDE_FRACTIONS:
                # Create config
                config = create_config(node_count, variant_name, variant_settings, hide_fraction)
                
                # Write to file
                output_file = variant_dir / f"hide_{hide_fraction}.yaml"
                with open(output_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                
                print(f"✓ Generated: {node_count}node/{variant_name}/hide_{hide_fraction}.yaml")
                total_generated += 1
    
    print("=" * 80)
    print(f"✓ Successfully generated {total_generated} config files!")
    print(f"  Structure: {len(NODE_COUNTS)} node counts × {len(PATH_VARIANTS)} variants × {len(HIDE_FRACTIONS)} hide fractions")
    print(f"  Note: base.yaml files kept with uniform distribution (0.0-1.0)")


if __name__ == "__main__":
    generate_path_variant_configs()
