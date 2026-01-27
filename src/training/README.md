# Training Module

This folder contains the code to train models.

## Files

| File | Description |
|------|-------------|
| `run.py` | Main runner that loads config, sets up logging, initializes dataset, dataloader, model, and trainer |
| `trainer.py` | Training loop implementation |
| `training_utils.py` | Training utilities |
| `load_utils.py` | Model and checkpoint loading utilities |

## Usage

```bash
python run.py --config path/to/config.yaml
```

## Checks

The `checks/` folder contains useful debugging tools:

| Script | Description |
|--------|-------------|
| `inspect_dataloader_samples.py` | Visualizes samples from the dataloader with statistics |
| `inspect_dataloader_sampels_curriculum.py` | Curriculum learning inspection |
| `inspect_real_world_samples.py` | Real-world data sample inspection |
| `test_run.py` | Runs training for a few iterations on CPU to verify setup |