# Benchmarking Module

This module contains tools for benchmarking models against standard datasets.

## Files

| File | Description |
|------|-------------|
| `Benchmark.py` | Core benchmark class |
| `run_benchmark.py` | Command-line benchmark runner |
| `load_openml_benchmark.py` | OpenML dataset loading utilities |
| `create_plot_paper.py` | Generate publication-ready plots |
| `create_plot_from_txt.py` | Plot generation from text results |

## Usage

```python
from benchmarking.Benchmark import Benchmark

benchmark = Benchmark(config_path, checkpoint_path)
results = benchmark.run()
```

For detailed benchmark experiments, see the benchmark directories in `experiments/GraphConditioning/Benchmarks/`.