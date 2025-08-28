#!/usr/bin/env python3
# --- HPC/cluster preamble: MUST be first ---
#import os
#os.environ.setdefault("OMP_NUM_THREADS", "1")
#os.environ.setdefault("MKL_NUM_THREADS", "1")
#os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
#os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
#os.environ.setdefault("MALLOC_ARENA_MAX", "2")
#os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")

import torch, torch.multiprocessing as mp

# Avoid /dev/shm for IPC; safer on Condor
#torch.multiprocessing.set_sharing_strategy("file_system")

# Pick one; both are fine on clusters. Keep it consistent.
#try:
#    mp.set_start_method("spawn", force=True)   # or "forkserver"
#except RuntimeError:
#    pass
# -------------------------------------------

"""
Training with Dataloader (YAML Config) - Create dataloader in main and pass to trainer.
Uses YAML configuration files instead of default configs.
"""

import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# AFTER the preamble, do project imports
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset

# Default config path
DEFAULT_CONFIG_PATH = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/configs/early_test2.yaml"


class DataloaderTrainer:
    """Trainer that accepts a dataloader as input."""
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.device = device
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.global_step = 0
        
    def _process_batch(self, batch):
        """Process a batch - handles lists and tensors."""
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            X, y = batch
            
            # Handle case where X and y might be lists of tensors
            if isinstance(X, list):
                X = X[0] if len(X) > 0 else torch.zeros(1, 10).to(self.device)
            if isinstance(y, list):
                y = y[0] if len(y) > 0 else torch.zeros(1, 1).to(self.device)
                
            return X.to(self.device), y.to(self.device)
        else:
            # Handle single tensor/list case
            X = batch
            if isinstance(X, list):
                X = X[0] if len(X) > 0 else torch.zeros(1, 10).to(self.device)
                
            X = X.to(self.device)
            return X, X  # Auto-encoder style
    
    def fit(self):
        """Train the model using the provided dataloader."""
        print(f"Starting training with dataloader for {self.max_steps} steps...")
        
        self.model.train()
        
        for step, batch in enumerate(self.dataloader):
            if step >= self.max_steps:
                break
                
            X, y = self._process_batch(batch)
            
            self.optimizer.zero_grad()
            predictions = self.model(X)
            loss = self.criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            self.global_step += 1
            
            # Simple logging every 10 steps for debug
            if step % 10 == 0:
                print(f"Step {step}/{self.max_steps}, Loss: {loss.item():.6f}")
        
        print("Training completed!")
        return self.model


def create_simple_model(input_size: int, hidden_size: int = 64, output_size: int = 1):
    """Create a simple MLP model."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )


def load_yaml_config(config_path: str = None):
    """Load YAML configuration file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {config_path}")
    return config


def extract_config_values(config_dict):
    """Extract values from YAML config format (handles both 'value' and direct values)."""
    extracted = {}
    for key, value in config_dict.items():
        if isinstance(value, dict) and 'value' in value:
            extracted[key] = value['value']
        else:
            extracted[key] = value
    return extracted


def main():
    """Main entry point for training with dataloader using YAML config."""
    try:
        print("=== Training with Dataloader (YAML Config) ===")
        
        # Load YAML configuration
        config = load_yaml_config()
        
        # Extract config sections
        scm_config = config.get('scm_config', {})
        dataset_config = config.get('dataset_config', {})
        preprocessing_config = config.get('preprocessing_config', {})
        training_config = extract_config_values(config.get('training_config', {}))
        
        print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
        print(f"Description: {config.get('description', 'no description')}")
        print(f"Training config: {training_config}")
        
        print("Creating dataset maker...")
        dataset_maker = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config
        )
        
        print("Creating dataset...")
        dataset = dataset_maker.create_dataset(seed=42)
        print(f"Dataset created with {len(dataset)} samples")
        
        # Create DataLoader in main method
        print("Creating DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=training_config.get("batch_size", 128),
            shuffle=True,
            num_workers=training_config.get("num_workers", 4)
        )
        print(f"DataLoader created with NUM_WORKERS: {training_config.get('num_workers', 4)}")
        
        # Test first batch to get input size
        print("Determining input size from first batch...")
        first_batch = next(iter(dataloader))
        print(f"First batch type: {type(first_batch)}")
        print(f"First batch structure: {type(first_batch[0]) if isinstance(first_batch, (list, tuple)) else 'single item'}")
        
        if isinstance(first_batch, (tuple, list)) and len(first_batch) == 2:
            X_sample, y_sample = first_batch
            
            # Handle case where X_sample might be a list of tensors
            if isinstance(X_sample, list):
                X_sample = X_sample[0] if len(X_sample) > 0 else torch.zeros(1, 10)  # fallback
            if isinstance(y_sample, list):
                y_sample = y_sample[0] if len(y_sample) > 0 else torch.zeros(1, 1)  # fallback
                
            input_size = X_sample.shape[-1]
            output_size = y_sample.shape[-1] if len(y_sample.shape) > 1 else 1
        else:
            X_sample = first_batch
            
            # Handle case where X_sample might be a list
            if isinstance(X_sample, list):
                X_sample = X_sample[0] if len(X_sample) > 0 else torch.zeros(1, 10)  # fallback
                
            input_size = X_sample.shape[-1]
            output_size = input_size  # Auto-encoder style
        
        print(f"Input size: {input_size}, Output size: {output_size}")
        
        # Create model with hidden size from YAML config if available
        hidden_size = training_config.get("hidden_size", 64)  # Default fallback
        print("Creating model...")
        model = create_simple_model(
            input_size=input_size, 
            hidden_size=hidden_size,
            output_size=output_size
        )
        print(f"Model: {model}")
        
        # Create trainer with the dataloader
        print("Creating trainer...")
        trainer = DataloaderTrainer(
            model=model,
            dataloader=dataloader,
            learning_rate=training_config.get("learning_rate", 1e-3),
            max_steps=training_config.get("max_steps", 10),
            device=training_config.get("device", "cpu")
        )
        
        # Start training
        print("Starting training...")
        trained_model = trainer.fit()
        
        print("=== Training Complete ===")
        print("Training completed successfully!")
        return trained_model
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
