#!/usr/bin/env python3
# --- HPC/cluster preamble: MUST be first ---


import torch, torch.multiprocessing as mp

# -------------------------------------------

"""
Training with Dataloader - Create dataloader in main and pass to trainer.
"""

import sys
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
from priordata_processing.Datasets.ExampleConfigs.BasicConfigs import (
    default_dataset_config, 
    default_preprocessing_config
)
from priors.causal_prior.ExampleConfigs.Basic_Configs import default_sampling_config

# Default training configuration
default_training_config = {
    "batch_size": 128,
    "learning_rate": 1e-3,
    "max_steps": 1000,
    "num_workers": 4,
    "hidden_size": 64,
    "device": "cpu"
}


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
                
            #X, y = self._process_batch(batch)
            
            #self.optimizer.zero_grad()
            #predictions = self.model(X)
            #loss = self.criterion(predictions, y)
            
            # Backward pass
            #loss.backward()
            #self.optimizer.step()
            
            #self.global_step += 1
            
            # Simple logging every 100 steps
            #if step % 100 == 0:
            #    print(f"Step {step}/{self.max_steps}, Loss: {loss.item():.6f}")

            print(f"Step {step}/{self.max_steps}")

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


def main():
    """Main entry point for training with dataloader."""
    try:
        print("=== Creating Dataloader and Training ===")
        
        # Use default configs from BasicConfigs.py
        print("Using default configs from BasicConfigs.py")
        scm_config = default_sampling_config
        dataset_config = default_dataset_config
        preprocessing_config = default_preprocessing_config
        training_config = default_training_config
        
        print(f"SCM config keys: {list(scm_config.keys())}")
        print(f"Dataset config keys: {list(dataset_config.keys())}")
        print(f"Preprocessing config keys: {list(preprocessing_config.keys())}")
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
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_workers"]
        )
        print(f"DataLoader created with number of workers: {training_config['num_workers']}")
        
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
        
        # Create model
        print("Creating model...")
        model = create_simple_model(
            input_size=input_size, 
            hidden_size=training_config["hidden_size"],
            output_size=output_size
        )
        print(f"Model: {model}")
        
        # Create trainer with the dataloader
        print("Creating trainer...")
        trainer = DataloaderTrainer(
            model=model,
            dataloader=dataloader,
            learning_rate=training_config["learning_rate"],
            #max_steps=training_config["max_steps"],
            max_steps = 10,                                    ### fix later
            device=training_config["device"],   
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
