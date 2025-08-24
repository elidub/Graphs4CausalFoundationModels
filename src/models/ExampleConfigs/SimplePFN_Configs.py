"""
Example configurations for SimplePFN model.

This module contains two fixed configurations for the SimplePFNRegressor model.
All parameters have fixed values - no sampling distributions.
"""

# Small configuration for quick experiments
small_simplepfn_config = {
    "d_model": 128,        # Model embedding dimension (feature size for internal representations)
    "depth": 4,            # Number of two-way transformer blocks in the model
    "heads_feat": 4,       # Number of attention heads for feature-wise attention (within rows)
    "heads_samp": 4,       # Number of attention heads for sample-wise attention (within columns)
    "num_bins": 64,        # Number of output bins for discretized regression
    "dropout": 0.1         # Dropout probability for regularization
}

# Medium configuration for better performance
medium_simplepfn_config = {
    "d_model": 256,        # Model embedding dimension (larger for more representational capacity)
    "depth": 6,            # Number of two-way transformer blocks (deeper for more complex patterns)
    "heads_feat": 8,       # Number of attention heads for feature-wise attention (more heads for richer attention)
    "heads_samp": 8,       # Number of attention heads for sample-wise attention (more heads for richer attention)
    "num_bins": 64,        # Number of output bins for discretized regression (same as small)
    "dropout": 0.15        # Dropout probability (slightly higher for larger model regularization)
}

