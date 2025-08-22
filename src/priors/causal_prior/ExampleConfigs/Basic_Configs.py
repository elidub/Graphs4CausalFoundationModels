default_sampling_config = {
    # Graph Structure Parameters
    "num_nodes": {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 3, "high": 8}
    },
    "graph_edge_prob": {
        "distribution": "beta",
        "distribution_parameters": {"alpha": 2, "beta": 3}
    },
    "graph_seed": {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 100_000_000}
    },
    
    # Mechanism Type Selection
    "xgboost_prob": {
        "distribution": "uniform",
        "distribution_parameters": {"low": 0.0, "high": 0.3}
    },
    "mechanism_seed": {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 100_000_000}
    },
    
    # MLP Mechanism Hyperparameters
    "mlp_nonlins": {
        "value": "tabicl"
    },
    "mlp_num_hidden_layers": {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 2}
    },
    "mlp_hidden_dim": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [8, 16, 32, 64]}
    },
    "mlp_activation_mode": {
        "value": "pre"
    },
    "mlp_node_shape": {
        "value": (1,)
    },
    
    # XGBoost Mechanism Hyperparameters
    "xgb_num_hidden_layers": {
        "value": 0
    },
    "xgb_hidden_dim": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0, 16, 32, 64]}
    },
    "xgb_activation_mode": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["pre", "post"]}
    },
    "xgb_node_shape": {
        "value": (1,)
    },
    "xgb_n_training_samples": {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 100, "high": 500}
    },
    "xgb_add_noise": {
        "value": False
    },
    
    # Noise Distribution Parameters
    "random_additive_std": {
        "value": True
    },
    "exo_std_distribution": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]}
    },
    "endo_std_distribution": {
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]}
    },
    
    # Random Standard Deviation Parameters (used when random_additive_std=True)
    "exo_std_mean": {
        "distribution": "lognormal",
        "distribution_parameters": {"mean": 0.0, "std": 0.5}
    },
    "exo_std_std": {
        "distribution": "uniform",
        "distribution_parameters": {"low": 0.3, "high": 1.0}
    },
    "endo_std_mean": {
        "distribution": "lognormal",
        "distribution_parameters": {"mean": -1.0, "std": 0.3}
    },
    "endo_std_std": {
        "distribution": "uniform",
        "distribution_parameters": {"low": 0.1, "high": 0.5}
    },

    
    # SCM Configuration
    "scm_fast": {
        "value": True
    },
    "use_exogenous_mechanisms": {
        "value":True
    },
    
    # Random Number Generation
    "mechanism_generator_seed": {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 100_000_000}
    },
}