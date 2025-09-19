This repository contains the code for training a prior-data fitted network (PFN) on a causal prior. Currently, only the predictive setting (i.e. no causal inference) is implemented.

Training can be run via the script src/training/simple_run.py. The configuration is done via YAML config files, see experiments/FirstTests/configs/early_test.yaml for an example.

More specifically, training can be run via:

```bash
cd src/training
python3 simple_run.py --config "../../early_test.yaml"
```

Here, the `--config` flag specifies the path to the YAML configuration file to use for training and can be changed as needed. 


### Directly using the prior to sample datasets

An example notebook for more fine-grained access to the prior and its sampling process can be found in src/priors/causal_prior/example_sample_from_prior.ipynb.


### Components of the repository

- src/priors/causal_prior: Code to define and sample from a causal prior, i.e. to sample SCMs and generate tabular data from them. See src/priors/causal_prior/README.md for more details.
- src/priordata_processing: Code to preprocess data, to select target column and to create train-and test splits. This also creates torch datasets and dataloaders that sample datasets from the causal prior on the fly. See src/priordata_processing/README.md for more details.
- src/models: Code to define the PFN torch model. See src/models/README.md for more details. This also contains an sklearn wrapper for the PFN model in src/models/SimplePFN_sklearn.py
- src/Losses: Code to define the parametrization of the posterior predictive of the model and loss functions used for training. See src/Losses/README.md for more details.
- src/training: Code to train the PFN model on datasets sampled from the causal prior. See src/training/README.md for more details.
- src/benchmarks: Preliminary code to benchmark the trained PFN model against other baselines on real-world datasets. See src/benchmarks/README.md for more details.


# Credit: 

The overall structure of the prior is inspired by the paper Do-PFN: In-Context Learning for Causal Effect Estimation (https://arxiv.org/abs/2506.06039). The sampling of nonlinearities and usage of XGboost mechansisms is inspired by the papers Hollmann, Noah, et al. "Accurate predictions on small data with a tabular foundation model." Nature 637.8045 (2025): 319-326. and Qu, Jingang, et al. "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data." ICML 2025. The architecture is inspired by "Accurate predictions on small data with a tabular foundation model." Nature 637.8045 (2025): 319-326. and Lorch, Lars, et al. "Amortized inference for causal structure learning." Neurips 2022. 

For references on PFNs please see the two original papers: 
- Hollmann, Noah, et al. "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second." ICLR 2023
- Müller, Samuel, et al. "Transformers Can Do Bayesian Inference." ICLR 2022.

