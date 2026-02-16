# Causal Foundation Models with Partial Graphs

This repository contains the code for training and evaluating causal foundation models that leverage partial graph knowledge for causal effect estimation.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
- [Benchmarks](#benchmarks)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ArikReuter/Graphs4CausalFoundationModels.git
cd Graphs4CausalFoundationModels

# Install dependencies
pip install -r requirements.txt
```

---

## Project Structure

```
CausalPriorFitting/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   ├── priors/                   # Causal prior definitions
│   ├── priordata_processing/     # Data processing utilities
│   ├── training/                 # Training loops and utilities
│   ├── benchmarking/             # Evaluation and benchmarking
│   ├── Losses/                   # Loss functions
│   └── utils/                    # General utilities
├── experiments/                  # Experiment configurations
│   ├── GraphConditioning/        # Linear-Gaussian experiments
│   ├── complexmech/              # Complex mechanism experiments
│   └── Predictive/               # Predictive model experiments
├── RealCauseEval/                # Real-world causal evaluation
└── requirements.txt              # Python dependencies
```

---

## Prior

This repository contains the code for a natively causal prior that yields a tabular foundation model with competitive predictive performance on small datasets. 

This prior directly supports sampling of observational data, interventional data, as well as the corresponding SCM. 

![Model Performance Comparison (MSE)](img/model_comparison_mse.png)

On datasets up to 1000 samples, a predictive model trained on this prior achieves competitive performance with untuned tabular baselines. 

Please see the README in `src/prior` for more details on the prior.

## Model

### Checkpoints

Pre-trained model checkpoints are available in `experiments/checkpoints/`:

This includes (a) a model trained in the linear-Gaussian case (`lingaus/`) to predict $p(y \mid \text{do}(t), D)$, where $y$ is an outcome, $t$ a treatment, and $D$ an observational dataset, (b) a model to predict $p(y \mid \text{do}(t), D)$ trained on complex mechanisms (`model/`), as well as (c) a model to predict the conditional interventional distribution $p(y \mid \text{do}(t), x, D)$ (`full_conditioned_model/`).

### Inference with the Sklearn-like Wrapper

The `GraphConditionedInterventionalPFNSklearn` wrapper provides a simple interface for inference:

```python
from src.models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn

# Load model
wrapper = GraphConditionedInterventionalPFNSklearn(
    config_path="experiments/checkpoints/full_conditioned_model/config.yaml",
    checkpoint_path="experiments/checkpoints/full_conditioned_model/model.pt",
)
wrapper.load()

# Predict interventional outcomes
# adjacency_matrix shape: (L+2, L+2) with ordering [T, Y, X_0, ..., X_{L-1}]
# A[i,j] = 1 means directed edge i → j
preds = wrapper.predict(
    X_obs, T_obs, Y_obs,     # Observational data
    X_intv, T_intv,           # Interventional query
    adjacency_matrix,         # Causal graph structure
    prediction_type="mean",   # "mode", "mean", or "sample"
)

# Compute log-likelihood of interventional outcomes
log_probs = wrapper.predict_log_likelihood(
    X_obs, T_obs, Y_obs,
    X_intv, T_intv, Y_intv,
    adjacency_matrix,
)
```

The wrapper auto-detects the model architecture from the config and automatically selects GPU if available. See the docstring of `GraphConditionedInterventionalPFNSklearn` for the full adjacency matrix format specification.

## Model Architecture

The architecture for the model that conditions on partial (ancestral) information can be found in `src/models/PartialGraphConditionedInterventionalPFN.py`.

The model is a Prior-Data Fitted Network (PFN) that takes as input an observational dataset $D$, an interventional query $(x, t)$, and a (partial) causal graph, and outputs a predictive distribution over the interventional outcome $y$.

### Graph Conditioning Mechanisms

The model supports three complementary ways to condition on partial graph knowledge, which can be combined flexibly:

1. **Soft Attention Bias** — Learnable per-head biases are added to the feature-wise attention logits based on the edge state:
   - Edge exists ($A_{ij} = 1$): a positive bias $b_{\text{edge}}$ is added
   - No edge ($A_{ij} = -1$): a negative bias $b_{\text{no\_edge}}$ is subtracted
   - Unknown ($A_{ij} = 0$): no bias applied

   This allows the model to softly encourage or discourage attention between features based on the known graph structure, while remaining flexible for uncertain edges.

2. **GCN Graph Encoder** — A GCN-style message-passing encoder processes the partial adjacency matrix to produce per-node graph embeddings. Unknown edges receive a learnable weight $\alpha \in (0, 1)$, allowing the model to adaptively propagate information through uncertain connections:
$$M = \mathbf{1}[A = 1] + \alpha \cdot \mathbf{1}[A = 0], \quad M[A = -1] = 0$$
   The message matrix $M$ is symmetrically normalized ($D^{-1/2} M D^{-1/2}$) with self-loops added before message passing.

3. **Adaptive Layer Normalization (AdaLN)** — The graph embeddings from the GCN encoder are used to predict per-feature scale and shift parameters for layer normalization, allowing different features to be normalized differently based on their position in the causal graph.

### Partial Graph Format

The adjacency matrix uses a ternary encoding to represent partial causal knowledge:

| Value | Meaning |
|-------|---------|
| $1$   | Edge exists ($i \to j$) |
| $0$   | Unknown (edge status uncertain) |
| $-1$  | No edge ($i \not\to j$) |

This supports any level of graph knowledge — from fully known ($\{-1, 1\}$) to fully unknown (all $0$s) — within a single model.

### Two-Way Attention

Each transformer block performs two types of attention:
- **Feature attention**: Attends across features (columns) within each sample, conditioned on the causal graph via masking, soft biases, and/or AdaLN.
- **Sample attention**: Attends across samples (rows) within each feature. Train samples use self-attention, while test samples use cross-attention to the train set.

Additional architecture details include SwiGLU activations, pre-layer normalization, and optional attention sinks for stability.

### Other Model Variants

The repository also includes several other model variants in `src/models/`:

| Model | File | Description |
|-------|------|-------------|
| `GraphConditionedInterventionalPFN` | `GraphConditionedInterventionalPFN.py` | Hard attention masking only (binary graphs) |
| `UltimateGraphConditionedInterventionalPFN` | `UltimateGraphConditionedInterventionalPFN.py` | Flexible graph conditioning (GCN, AdaLN, soft attention) for binary graphs |
| `FlatGraphConditionedInterventionalPFN` | `FlatGraphConditionedInterventionalPFN.py` | Flat adjacency matrix appended to input |
| `InterventionalPFN` | `InterventionalPFN.py` | Base model without graph conditioning |

## Experiments


All experiments can be run using a unified interface:

```bash
python3 run.py --config "path/to/config.yaml"
```

### Linear-Gaussian Experiments

Experiments on linear-Gaussian structural causal models with varying graph knowledge:

| Experiment | Config Directory |
|------------|------------------|
| 50-node graphs | `experiments/GraphConditioning/configs_50node` |
| 50-node with ancestor info | `experiments/GraphConditioning/configs_50node_ancestor` |
| 50-node with IDK (partial knowledge) | `experiments/GraphConditioning/configs_50node_idk` |

**Example:**
```bash
python3 run.py --config experiments/GraphConditioning/configs_50node/your_config.yaml
```

### Complex Mechanism Experiments

Experiments with non-linear causal mechanisms:

```bash
python3 run.py --config experiments/complexmech/configs/your_config.yaml
```

### Predictive Model

Train and evaluate a predictive model:

```bash
python3 run.py --config experiments/Predictive/configs/predictive.yaml
```

---

## Benchmarks

The repository includes three benchmarks for evaluating causal inference models, located in `experiments/GraphConditioning/Benchmarks/`:

| Benchmark | Description | Mechanisms | Edge States |
|-----------|-------------|------------|-------------|
| [LinGaus](experiments/GraphConditioning/Benchmarks/LinGaus/) | Full graph knowledge | Linear | {0, 1} |
| [ComplexMechIDK](experiments/GraphConditioning/Benchmarks/ComplexMechIDK/) | Partial graph knowledge | MLP/XGBoost | {-1, 0, 1} |
| [ComplexMech](experiments/GraphConditioning/Benchmarks/ComplexMech/) | Sample size experiments | MLP/XGBoost | {0, 1} |

See the README in each benchmark directory for detailed usage instructions.

---
