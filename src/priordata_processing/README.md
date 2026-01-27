# Prior Data Processing

This folder contains code to set up datasets that sample from structural causal models (SCMs).

## Datasets

| Dataset | Description |
|---------|-------------|
| `ObservationalDataset.py` | Samples observational data from SCMs |
| `InterventionalDataset.py` | Samples interventional data with do-calculus |
| `InterpolatedObservationalDataset.py` | Interpolated observational sampling |
| `Collator.py` | Batch collation utilities |

## Configuration

Each dataset takes three types of configs:

- **SCM config**: Determines hyperparameter distributions for the SCM
- **Dataset config**: Determines how datasets are generated (sample count, feature dropping, target selection)
- **Preprocessing config**: Standard preprocessing for tabular data

## Usage

```python
from priordata_processing.Datasets.InterventionalDataset import InterventionalDataset

dataset = InterventionalDataset(
    scm_config=scm_config,
    dataset_config=dataset_config,
    preprocessing_config=preprocessing_config
)

# Sample a batch
batch = dataset[0]
``` 