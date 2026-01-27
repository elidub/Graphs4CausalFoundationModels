# Models

This folder contains the model implementations for causal foundation models.

## Core Models

| Model | Description |
|-------|-------------|
| `SimplePFN.py` | Base transformer with row-wise and sample-wise attention layers |
| `InterventionalPFN.py` | PFN for interventional queries |
| `GraphConditionedInterventionalPFN.py` | Graph-conditioned interventional model |
| `PartialGraphConditionedInterventionalPFN.py` | Supports partial graph knowledge (IDK edges) |
| `UltimateGraphConditionedInterventionalPFN.py` | Full-featured graph-conditioned model |

## Sklearn Wrappers

Each model has a corresponding sklearn-style wrapper for easy usage:

| Wrapper | Base Model |
|---------|------------|
| `SimplePFN_sklearn.py` | SimplePFN |
| `InterventionalPFN_sklearn.py` | InterventionalPFN |
| `InterventionalPFN_sklearn_batched.py` | InterventionalPFN (batched inference) |
| `GraphConditionedInterventionalPFN_sklearn.py` | GraphConditionedInterventionalPFN |

## Usage

```python
from models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn

# Load a trained model
model = GraphConditionedInterventionalPFNSklearn(config_path, checkpoint_path)

# Predict with graph conditioning
predictions = model.predict(X, treatment_col, adjacency_matrix)
```