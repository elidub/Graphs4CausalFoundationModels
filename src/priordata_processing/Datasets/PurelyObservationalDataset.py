from torch.utils.data import Dataset
import torch
from priors.causal_prior.scm.SCMSampler import SCMSampler
from priordata_processing.BasicProcessing import BasicProcessing


class PurelyObservationalDataset(Dataset):
    """
    Class representing a purely observational dataset. I.e. no interventions are applied.
    """

    def __init__(self, 
                 scm_sampler: SCMSampler,
                 priordata_processor: BasicProcessing,
                 number_train_samples_per_dataset_distribution: torch.distributions.Distribution,
                 number_test_samples_per_dataset_distribution: torch.distributions.Distribution,
                 size: int = 10_000,
                 max_number_train_samples: int = 500,
                 max_number_test_samples: int = 500,
                 max_number_features: int = 100, 
                 seed: int = 123):
        """
        scm_sampler: the SCM sampler to use for generating scms and data 
        priordata_processor: the data processor to use for processing the data
        number_train_samples_per_dataset_distribution: the distribution to sample the number of train samples per dataset from
        number_test_samples_per_dataset_distribution: the distribution to sample the number of test samples per dataset from
        size: the number of elements in the PurelyObservationalDataset, where each element inside is a tabular dataset (!)
        max_number_train_samples: the maximum number of train samples in each dataset (rest is zero-padded)
        max_number_test_samples: the maximum number of test samples in each dataset (rest is zero-padded)
        max_number_features: the maximum number of features in each dataset (rest is zero-padded)
        seed: the random seed for reproducibility
        """

        self.scm_sampler = scm_sampler
        self.priodata_processor = priordata_processor
        self.number_train_samples_per_dataset_distribution = number_train_samples_per_dataset_distribution
        self.number_test_samples_per_dataset_distribution = number_test_samples_per_dataset_distribution
                 
        self.size = size
        self.max_number_train_samples = max_number_train_samples
        self.max_number_test_samples = max_number_test_samples
        self.max_number_features = max_number_features
        self.seed = seed

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
        
        seed = self.seed + idx
        torch.manual_seed(seed)  # Ensure reproducible sampling

        scm = self.scm_sampler.sample(
            seed=seed,
        )

        # Sample separate train and test sample counts
        number_train_samples = self.number_train_samples_per_dataset_distribution.sample()
        if hasattr(number_train_samples, 'item'):
            number_train_samples = int(number_train_samples.item())  # Convert tensor to int
        else:
            number_train_samples = int(number_train_samples)  # Already an int from FixedSampler
        
        number_test_samples = self.number_test_samples_per_dataset_distribution.sample()
        if hasattr(number_test_samples, 'item'):
            number_test_samples = int(number_test_samples.item())  # Convert tensor to int
        else:
            number_test_samples = int(number_test_samples)  # Already an int from FixedSampler
        
        # Total samples needed from SCM
        total_samples = number_train_samples + number_test_samples

        # Generate data from SCM
        scm.sample_exogenous(num_samples=total_samples)
        scm.sample_endogenous_noise(num_samples=total_samples)

        dataset = scm.propagate(num_samples=total_samples) 

        # Create a new BasicProcessing instance with the actual sampled counts
        # We need to update the processor to use the sampled counts, not the max counts
        actual_processor = BasicProcessing(
            n_features=self.priodata_processor.n_features,
            max_n_features=self.priodata_processor.max_n_features,
            n_train_samples=number_train_samples,
            max_n_train_samples=self.max_number_train_samples,
            n_test_samples=number_test_samples,
            max_n_test_samples=self.max_number_test_samples,
            dropout_prob=self.priodata_processor.dropout_prob,
            target_feature=self.priodata_processor.target_feature,
            random_seed=self.priodata_processor.random_seed,
            negative_one_one_scaling=self.priodata_processor.negative_one_one_scaling,
            standardize=self.priodata_processor.standardize,
            yeo_johnson=self.priodata_processor.yeo_johnson,
            remove_outliers=self.priodata_processor.remove_outliers,
            outlier_quantile=self.priodata_processor.outlier_quantile,
            shuffle_samples=self.priodata_processor.shuffle_samples,
            shuffle_features=self.priodata_processor.shuffle_features,
            y_clip_quantile=self.priodata_processor.y_clip_quantile,
            eps=self.priodata_processor.eps,
            device=self.priodata_processor.device,
            dtype=self.priodata_processor.dtype,
        )

        out = actual_processor.process(dataset)

        X_train, Y_train, X_test, Y_test = out

        # Compute a dummy time/alpha for compatibility with curriculum logging
        if self.size > 1:
            t = float(idx) / float(self.size - 1)
        else:
            t = 1.0
        alpha = -1.0  # indicates no interpolation

        # Return tensors directly (keeping them as PyTorch tensors) + (t, alpha)
        return X_train, Y_train, X_test, Y_test, torch.tensor(t, dtype=torch.float32), torch.tensor(alpha, dtype=torch.float32)