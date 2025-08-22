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
                 number_samples_per_dataset_distribution: torch.distributions.Distribution,
                 size: int = 10_000,
                 max_number_samples: int = 1000,
                 max_number_features: int = 100, 
                 seed:int = 123):
        """
        scm_sampler: the SCM sampler to use for generating scms and data 
        priordata_processor: the data processor to use for processing the data
        number_samples_per_dataset_distribution: the distribution to sample the number of samples per dataset from
        dataset_size: the number of elements in the PurelyObservationalDataset, where each element inside is a tabular dataset (!)
        max_number_samples: the maximum number of samples in each dataset (rest is zero-padded)
        max_number_features: the maximum number of features in each dataset (rest is zero-padded)
        seed: the random seed for reproducibility
        """

        self.scm_sampler = scm_sampler
        self.priodata_processor = priordata_processor
        self.number_samples_per_dataset_distribution = number_samples_per_dataset_distribution
                 
        self.size = size
        self.max_number_samples = max_number_samples
        self.max_number_features = max_number_features
        self.seed = seed

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
        
        seed = self.seed + idx

        scm = self.scm_sampler.sample(
            seed=seed,
        )

        number_samples_per_dataset = self.number_samples_per_dataset_distribution.sample()
        number_samples_per_dataset = int(number_samples_per_dataset.item())  # Convert tensor to int

        scm.sample_exogenous(num_samples=number_samples_per_dataset)
        scm.sample_endogenous_noise(num_samples=number_samples_per_dataset)

        dataset = scm.propagate(num_samples=number_samples_per_dataset) 

        dataset = self.priodata_processor.process(dataset)

        return dataset