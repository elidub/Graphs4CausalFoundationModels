This folder contains the implementation of the loss functions arising from the parametrization of the posterior predictive for DoPFN. For now, this includes the histogram-loss from TabPFN.

The class PosteriorPredictive contains an interface for all parametrizations of the posterior predictive. The BarDistribution implements the histogram-parametrization of the PPD from TabPFN.