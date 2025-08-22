The core element of this prior is an SCM (Structural Causal Model), which can be found in causal_prior/scm/SCM.py.

The overall goal is to (a) sample the SCM itself and (b) sample data from the SCM.

An SCM has three components: 

- A causal graph, represented by a CausalDAG object in the file causalprior/causal_graph/CausalDAG.py. (The CausalDAG is a relatively light wrapper of a networkx graph). Graphs can be sampled in causalprior/causal_graph/GraphSampler.py. 

- A dictionary of mechanisms that defines for each node, how it is computed as a function of its parents and a node-specific "endogenous" noise variable. Mechanisms are torch.nn.Module objects and need to inherit from BaseMechanism in causal_prior/mechanismsBaseMechanism.py. Currently two types of mechanisms are used: MLPMechanisms and XGBoostMechanisms. The classses causal_prior/mechanisms/SampleMLPMechanism.py and causal_prior/mechanisms/SampleXGboostMechanism.py implement them and, by instantiation, sample a specific mechanism. 

- Noise distributions in causal_prior/noise_distributions. The basic interface for a noise variable can be found in causal_prior/noise_distributions/DistributionInterface.py. Currently, the main class to sample noise is causal_prior/noise_distributions/MixedDist_RandomSTD.py that, once instatiated, allows to sample noise. 

To sample an SCM, a config, specifying distributions over all hyperparameters (of the graph sampling, mechanisms sampling and noise distributions) is needed. An example for such a config can be found in causal_prior/ExampleConfigs/Basic_Configs.py. This config is passed to a the class SCMHyperparameterSampler in causal_prior/scm/SCMHyperparameterSampler.py, which sample a hyperparameter dictionary according to the config. Such a dictionary can be passed to causal_prior/scm/SCMBuilder.py to build and return an SCM.Data from this SCM can be sampled via: 

1. `N_SAMPLES = 123` # Number of samples to draw
2. `scm.sample_exogenous(N_SAMPLES)` # sample the exogenous noise 
3. `scm.sample_endogenous(N_SAMPLES)` # sample the endogenous noise
4. `r = scm.propagate(N_SAMPLES)` # propagate the samples through the SCM

Also see the file causal_prior/scm/InspectSamplesConfig.py to see how to sample an SCM and how to, then sample data from it. This class also allows to investigate example datasets. 