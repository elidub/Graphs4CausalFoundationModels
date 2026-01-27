# Causal Prior

This module implements the causal prior based on Structural Causal Models (SCMs). The core SCM implementation is in `scm/SCM.py`.

## Overview

The goal is to (a) sample SCMs from a prior and (b) sample data from those SCMs.

## Structure

```
causal_prior/
├── scm/                    # SCM implementation and sampling
├── causal_graph/           # DAG representation (CausalDAG)
├── mechanisms/             # Causal mechanisms (MLP, XGBoost, Linear)
├── noise_distributions/    # Noise distribution sampling
└── ExampleConfigs/         # Example configuration files
```

## SCM Components

An SCM has three components:

| Component | Location | Description |
|-----------|----------|-------------|
| Causal Graph | `causal_graph/CausalDAG.py` | DAG structure (networkx wrapper) |
| Mechanisms | `mechanisms/` | Functions computing node values from parents |
| Noise Distributions | `noise_distributions/` | Exogenous and endogenous noise |

### Mechanisms

| Mechanism | File | Description |
|-----------|------|-------------|
| MLP | `SampleMLPMechanism.py` | Neural network mechanisms with TabICL activations |
| XGBoost | `SampleXGBoostMechanism.py` | Gradient boosting mechanisms |
| Linear | `LinearMechanism.py` | Linear mechanisms (for LinGaus) |

### Noise Distributions

| Distribution | File | Description |
|--------------|------|-------------|
| Mixed | `MixedDist_RandomSTD.py` | Mixed distributions with random std |
| Base Interface | `DistributionInterface.py` | Abstract interface for noise |

## Usage

### Sampling SCMs

```python
from priors.causal_prior.scm.SCMSampler import SCMSampler
from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config

# Create sampler with configuration
sampler = SCMSampler(default_sampling_config, seed=42, verbose=True)

# Sample an SCM
scm = sampler.sample()
```

### Generating Data from SCM

```python
N_SAMPLES = 100

# Sample and fix noise
scm.sample_exogenous(num_samples=N_SAMPLES)
scm.sample_endogenous_noise(num_samples=N_SAMPLES)

# Propagate through the graph
data = scm.propagate(num_samples=N_SAMPLES)
```
 