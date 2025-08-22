"""
Example configurations for MakePurelyObservationalDataset factory class.

This module contains default configurations for dataset creation and preprocessing
that follow the rigorous pattern established by SCMHyperparameterSampler.
"""

# Example configurations following the rigorous pattern
default_dataset_config = {
    "dataset_size": {  # actual number of elements in the meta-dataset (torch.dataset), i.e. number of tabular datasets in the torch.dataset!
        "value": 100_000
    },
    "max_number_samples": { #the maximum number of samples in each dataset (rest is zero-padded)
        "value": 1000
    },
    "max_number_features": { #the maximum number of samples in each dataset (rest is zero-padded)
        "value": 101
    },
    "number_samples_per_dataset": { # DISTRIBUTION over the number of non-zero samples in each tabular datase
        "distribution": "uniform",
        "distribution_parameters": {"low": 10, "high": 1000}
    },
    "seed": {  # Random seed for reproducibility
        "distribution": "discrete_uniform",
        "distribution_parameters": {"low": 0, "high": 100_000_000}
    }
}

default_preprocessing_config = {
    "train_fraction": {  # how to split a dataset into train- and test
        "value": 0.5
    },
    "dropout_prob": {  # probability of masking features 
        "value": 0.1
    },
    "transformation_type": { # transformation for preprocessing. Can be "standardize" or "yeo_johnson"
        "value": "standardize"
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
    }
}