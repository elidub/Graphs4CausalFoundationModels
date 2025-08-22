The core element of this prior is an SCM (Structural Causal Model), which can be found in causal_prior/scm/SCM.py.

The overall goal is to (a) sample the SCM itself and (b) sample data from the SCM.

## SCM Components

An SCM has three components: 

- **A causal graph**, represented by a CausalDAG object in the file causal_prior/causal_graph/CausalDAG.py. (The CausalDAG is a relatively light wrapper of a networkx graph). Graphs can be sampled in causal_prior/causal_graph/GraphSampler.py. 

- **A dictionary of mechanisms** that defines for each node, how it is computed as a function of its parents and a node-specific "endogenous" noise variable. Mechanisms are torch.nn.Module objects and need to inherit from BaseMechanism in causal_prior/mechanisms/BaseMechanism.py. Currently two types of mechanisms are used: MLPMechanisms and XGBoostMechanisms. The classes causal_prior/mechanisms/SampleMLPMechanism.py and causal_prior/mechanisms/SampleXGBoostMechanism.py implement them and, by instantiation, sample a specific mechanism. 

- **Noise distributions** in causal_prior/noise_distributions. The basic interface for a noise variable can be found in causal_prior/noise_distributions/DistributionInterface.py. Currently, the main class to sample noise is causal_prior/noise_distributions/MixedDist_RandomSTD.py that, once instantiated, allows to sample noise. 

## SCM Sampling Workflow

### Method 1: High-level SCMSampler (Recommended)

The easiest way to sample SCMs is using the **SCMSampler** class in causal_prior/scm/SCMSampler.py. This provides a high-level interface that encapsulates the entire sampling pipeline:

```python
from priors.causal_prior.scm.SCMSampler import SCMSampler
from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config

# Create sampler with configuration
sampler = SCMSampler(default_sampling_config, seed=42, verbose=True)

# Sample an SCM
scm = sampler.sample()

# Generate data from the SCM
N_SAMPLES = 100
scm.sample_exogenous(num_samples=N_SAMPLES)
scm.sample_endogenous_noise(num_samples=N_SAMPLES)
data = scm.propagate(num_samples=N_SAMPLES)
```

You can also sample multiple SCMs at once:
```python
scms = sampler.sample_multiple(count=5, base_seed=100)
```

### Method 2: Manual SCM Construction

To sample an SCM manually, a config specifying distributions over all hyperparameters (of the graph sampling, mechanisms sampling and noise distributions) is needed. An example for such a config can be found in causal_prior/ExampleConfigs/Basic_Configs.py. This config is passed to the class SCMHyperparameterSampler in causal_prior/scm/SCMHyperparameterSampler.py, which samples a hyperparameter dictionary according to the config. Such a dictionary can be passed to causal_prior/scm/SCMBuilder.py to build and return an SCM.

```python
from priors.causal_prior.scm.SCMHyperparameterSampler import SCMHyperparameterSampler
from priors.causal_prior.scm.SCMBuilder import SCMBuilder
from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config

# Sample hyperparameters
sampler = SCMHyperparameterSampler(default_sampling_config, seed=123)
sampled_params = sampler.sample()

# Build SCM
builder = SCMBuilder(**sampled_params)
scm = builder.build()
```

## Data Generation from SCM

Once you have an SCM (from either method above), data can be sampled via: 

1. `N_SAMPLES = 123` # Number of samples to draw
2. `scm.sample_exogenous(N_SAMPLES)` # sample the exogenous noise 
3. `scm.sample_endogenous_noise(N_SAMPLES)` # sample the endogenous noise
4. `r = scm.propagate(N_SAMPLES)` # propagate the samples through the SCM

## Example Files

- **causal_prior/scm/SCMSampler.py** - High-level SCM sampling interface (recommended for most use cases)
- **causal_prior/scm/SimpleExampleSampling.py** - Simple example of manual SCM sampling
- **causal_prior/scm/InspectSamplesConfig.py** - Example showing how to sample an SCM and investigate datasets

## Quick Start

For most users, the recommended approach is to use the SCMSampler:

```python
from priors.causal_prior.scm.SCMSampler import create_scm_sampler_from_config

# Create sampler and generate SCM
sampler = create_scm_sampler_from_config("default", seed=42)
scm = sampler.sample()

# Generate data
scm.sample_exogenous(num_samples=100)
scm.sample_endogenous_noise(num_samples=100)
data = scm.propagate(num_samples=100)
``` 