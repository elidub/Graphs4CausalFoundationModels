"""
Example configurations for MakePurelyObservationalDataset factory class.

This module contains default configurations for dataset creation and preprocessing
that follow the rigorous pattern established by SCMHyperparameterSampler.
"""

# Example configurations following the rigorous pattern
default_dataset_config = {
    "dataset_size": {  # actual number of elements in the meta-dataset (torch.dataset), i.e. number of tabular datasets in the torch.dataset!
        "value": 100_000_000
    },

    "max_number_features": { #the maximum number of samples in each dataset (rest is zero-padded)
        "value": 101
    },
    "max_number_train_samples": { # DISTRIBUTION over the number of non-zero samples in each tabular datase
        "value": 125
    },
    "number_train_samples_per_dataset": { # DISTRIBUTION over the number of non-zero samples in each tabular datase
        "distribution": "uniform",
        "distribution_parameters": {"low": 100, "high": 125}
    },
    "max_number_test_samples": {  # maximum number of test samples
        "value": 125
    },
    "number_test_samples_per_dataset": {  # number of test samples per dataset. Can be fixed because architecture is agnostic to the number of test samples.
        "value": 125
    },
    "seed": {  # Random seed for reproducibility
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 100_000_000}
    }
}

default_preprocessing_config = {
    "dropout_prob": {  # probability of masking features 
        "value": 0.1
    },
    "shuffle_data": {  #whether to shuffle the data
        "value": True
    },
    "target_feature": {
        "value": None  # Let it be randomly selected
    },
    "random_seed": {  # seed for reproducibility
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 100_000_000}
    },
    "negative_one_one_scaling": {  # whether to scale features to [-1, 1] range
        "value": False
    },
    "standardize": {
        "value": True
    },
    "yeo_johnson": {  # whether to apply Yeo-Johnson transformation
        "value": False
    },

    "remove_outliers": {  # whether to remove outliers during preprocessing
        "value": False
    },
    "outlier_quantile": {  # quantile threshold for outlier removal
        "value": 0.95
    },
}