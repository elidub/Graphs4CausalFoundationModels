# import dataloader 
from torch.utils.data import DataLoader

from MakePurelyObservationalDataset import MakePurelyObservationalDataset

from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config as prior_config
from priordata_processing.ExampleConfigs.BasicConfigs import default_dataset_config as dataset_config
from priordata_processing.ExampleConfigs.BasicConfigs import default_preprocessing_config as preprocessing_config



if __name__ == "__main__":
    SEED = 124
    make_dataset = MakePurelyObservationalDataset(
        scm_config=prior_config,
        preprocessing_config=preprocessing_config,
        dataset_config=dataset_config
    )

    dataset = make_dataset.create_dataset(seed=SEED)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(next(iter(dataloader))["X_test"].shape)
