default_config = {
    # Graph Structure Parameters
    "num_nodes": {   #number of nodes in the causal graph (some of them will become feautres, or targets, some of them will be unobserved)
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 3, "high": 8}
    },
    "graph_edge_prob": {  # probability that any two nodes are connected
        "distribution": "beta",
        "distribution_parameters": {"alpha": 2, "beta": 3}
    },
    "graph_seed": {   #seed to sample graph
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 100_000_000}
    },
    
    # Mechanism Type Selection
    "xgboost_prob": {   #probability of using XGBoost mechanism (vs) MLPMechanisms
        "distribution": "uniform",
        "distribution_parameters": {"low": 0.0, "high": 0.3}
    },
    "mechanism_seed": {  # seed for mechanisms 
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 100_000_000}
    },
    
    # MLP Mechanism Hyperparameters
    "mlp_nonlins": {  # nonlinearity sampling for the MLP (for now the ones from TabICL)
        "value": "tabicl"
    },
    "mlp_num_hidden_layers": {   #number of hidden layers in the MLP
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 2}
    },
    "mlp_hidden_dim": {   #number of hidden units in each layer of the MLP
        "distribution": "categorical",
        "distribution_parameters": {"choices": [8, 16, 32, 64]}
    },
    "mlp_activation_mode": {  # whether to use the activations before (pre) or after (post) adding the noise
        "value": "pre"
    },
    "mlp_node_shape": {
        "value": (1,)
    },
    
    # XGBoost Mechanism Hyperparameters
    "xgb_num_hidden_layers": {  # number of hidden layers for XGBoost Mechanisms (0 is probably already good)
        "value": 0
    },
    "xgb_hidden_dim": {  # number of hidden units in each layer for XGBoost Mechanisms (doesn't matter if xgb_num_hidden_layers=0)
        "distribution": "categorical",
        "distribution_parameters": {"choices": [0, 16, 32, 64]}
    },
    "xgb_activation_mode": {  # whether to use the activations before (pre) or after (post) adding the noise
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["pre", "post"]}
    },
    "xgb_node_shape": {
        "value": (1,)
    },
    "xgb_n_training_samples": {  # number of samples from a Gaussian to train the random XGBoost model
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 100, "high": 500}
    },
    "xgb_add_noise": {  # whether to add noise to the XGBoost mechanisms (if True, the result is not purely categorical)
        "value": False
    },
    
    # Noise Distribution Parameters
    "random_additive_std": {  # whether to use a random standard deviation for the distribution of the additive noise
        "value": True
    },
    "exo_std_distribution": {  # distribution over the standard deviation of ALL exogenous variable (only needed if random_additive_std=False)
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]}
    },
    "endo_std_distribution": {  # distribution over the standard deviation of ALL endogenous variable (only needed if random_additive_std=False)
        "distribution": "categorical",
        "distribution_parameters": {"choices": ["gamma", "pareto"], "probabilities": [1.0, 0.0]}
    },
    
    # Random Standard Deviation Parameters (used when random_additive_std=True)
    "exo_std_mean": { # random mean for the standard deviation of exogenous noise for every variable
        "distribution": "lognormal",
        "distribution_parameters": {"mean": 0.0, "std": 0.5}
    },
    "exo_std_std": { # random standard deviation for the standard deviation of exogenous noise for every variable
        "distribution": "uniform",
        "distribution_parameters": {"low": 0.3, "high": 1.0}
    },
    "endo_std_mean": { # random mean for the standard deviation of endogenous noise for every variable
        "distribution": "lognormal",
        "distribution_parameters": {"mean": -1.0, "std": 0.3}
    },
    "endo_std_std": { # random standard deviation for the standard deviation of endogenous noise for every variable
        "distribution": "uniform",
        "distribution_parameters": {"low": 0.1, "high": 0.5}
    },

    
    # SCM Configuration
    "scm_fast": { # whether to use a fast SCM implementation, can also be "safe", which is slower but does more checks
        "value": True
    },
    "use_exogenous_mechanisms": { # whether to use mechanisms ON the randomly sampled exogenous variables
        "value":True
    },
    
    # Random Number Generation
    "mechanism_generator_seed": {
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 100_000_000}
    },
}