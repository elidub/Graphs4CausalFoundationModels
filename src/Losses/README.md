# Losses

This folder contains loss function implementations for the posterior predictive distribution.

## Files

| File | Description |
|------|-------------|
| `PosteriorPredictive.py` | Interface for parametrizations of the posterior predictive distribution (PPD) |
| `BarDistribution.py` | Histogram-based parametrization of the PPD (from TabPFN) |
| `test_BarDistribution.py` | Unit tests for BarDistribution |

## Usage

```python
from Losses.BarDistribution import BarDistribution

bar_dist = BarDistribution(num_bars=100, min_val=-10, max_val=10)
loss = bar_dist.compute_loss(predictions, targets)
```