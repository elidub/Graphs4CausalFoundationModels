In this folder the code in priors is used to sample and process data for training. 

So far, the core functionality can be found in Datasets/MakePurelyObservationalDataset.py. This class takes 

- a SCM hyperparameter config, for the simulator of data. (For more details see the priors/causal_prior folder)

- a Preprocessing hyperparameter config, for the preprocessing of the data. This is implemented and explained in the priordata_processing/Preprocessing folder. This config determines the behaviour of the general preprocessing in priordata_processing/BasicProcessing.py 

- A dataset config, that determines further details of the sampled dataset (which are not already covered by the prior config).

The Datasets/MakePurelyObservationalDataset.py file outputs a PyTorch DataSet that can be used for training and evaluating the PFN. 

Furthermore, in priordata_processing/Benchmark_Dataloader, the runtime performance of the dataloader can be evaluated. 