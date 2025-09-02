# Training Module

This module provides functionality for training SimplePFN models.

## Overview

The training module includes:

1. **SimplePFNTrainer**: Low-level trainer for SimplePFN models
2. **Checkpoint Management**: Tools for saving and loading model checkpoints
3. **Load Utilities**: Functions for finding and loading saved models

## Checkpoint Organization

The model checkpoints are now organized in a hierarchical folder structure:

```
/checkpoints/
  /run_name_1/  # Run-specific directory 
    step_100.pt
    step_200.pt
    final.pt
  /run_name_2/  # Another run
    step_100.pt
    interrupted.pt
```

Each training run has its own dedicated directory named after the run ID, making it easy to find all checkpoints for a particular training run.

## Checkpoint Types

The system saves several types of checkpoints:

1. **Regular checkpoints**: Saved at specified intervals, named `step_{step_number}.pt`
2. **Final checkpoint**: Saved at the end of training, named `final.pt` 
3. **Interrupted checkpoint**: Saved when training is interrupted, named `interrupted.pt`

## Using the Checkpoint System

### Saving Models

When initializing the `SimplePFNTrainer`, provide:

```python
trainer = SimplePFNTrainer(
    # ... other parameters ...
    save_dir="/path/to/checkpoints",  # Base directory for all checkpoints
    save_every=100,                   # Save every 100 steps
    run_name="my_training_run"        # Name for this run
)
```

The trainer will:
1. Create a subdirectory `/path/to/checkpoints/my_training_run/`
2. Save models to that directory according to the specified schedule
3. Handle graceful termination with checkpoint saving

### Loading Models

To load models from the checkpoint system, use the provided utilities:

```python
from CausalPriorFitting.src.training.load_utils import load_model, find_checkpoint_in_run, find_run_directory

# Find the latest run directory
run_dir = find_run_directory("/path/to/checkpoints", latest=True)

# Find the best available checkpoint in that run
checkpoint_path = find_checkpoint_in_run(run_dir, checkpoint_type="best_available")

# Load the model
model = SimplePFNRegressor(**model_config)
model, metadata = load_model(checkpoint_path, model=model)
```

See the `examples/checkpointing_example.py` script for a complete usage example.

## Signal Handling

The trainer sets up signal handlers to catch SIGINT (Ctrl+C) and SIGTERM signals. When these signals are received, the trainer:

1. Saves an "interrupted.pt" checkpoint before exiting
2. Sets a termination flag to ensure clean shutdown

This ensures your progress is saved even when training is unexpectedly terminated.
