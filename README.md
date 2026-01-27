# Causal Foundation Models with Partial Graphs

> **ICML 2026 Submission:** *"Use What You Know: Causal Foundation Models with Partial Graphs"*

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
git clone https://anonymous.4open.science/r/Graphs4CFMs-D608
cd CausalPriorFitting

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

All experiments can be run using a unified interface:

```bash
python3 run.py --config "path/to/config.yaml"
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

## Experiments

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