This folder contains the actual model that is fitted on the synthetic data. 

The class SimplePFN in SimplePFN.py implements the torch model, which is a transformer with row-wise and sample-wise attention layers. 

The class SimplePFN_sklearn allows to use an SimplePFN model with sklearn-like interface (fit, predict, etc.).