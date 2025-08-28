#!/usr/bin/env python3
# --- HPC/cluster preamble: MUST be first ---
# -------------------------------------------

"""
Simple Training Runner - Create dataloader in main and train SimplePFN models.
Uses YAML configuration files like the original approach but with clean dataloader-in-main structure.
"""

import sys
import yaml
import torch
import torch.nn as nn
import time
from pathlib import Path
from torch.utils.data import DataLoader
from copy import deepcopy

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# AFTER the preamble, do project imports
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
from models.SimplePFN import SimplePFNRegressor

# Default config path
DEFAULT_CONFIG_PATH = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/configs/early_test.yaml"


class SimplePFNTrainer:
    """Trainer for SimplePFN that accepts a dataloader as input."""
    
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
        """
        Process a batch for SimplePFN training.
        
        DataLoader returns a list of 4 tensors:
        - batch[0]: X_train (batch_size, n_train, features)
        - batch[1]: y_train (batch_size, n_train, 1)
        - batch[2]: X_test (batch_size, n_test, features)
        - batch[3]: y_test (batch_size, n_test, 1)
        
        SimplePFN expects the same format, except y_test should be flattened.
        """
        if isinstance(batch, list) and len(batch) == 4:
            X_train, y_train, X_test, y_test = batch
            
            # Print raw batch shapes before processing
            print(f"Raw batch shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

            # Move to device
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            
            # For now, SimplePFN handles only single samples, so take the first sample from batch
            # TODO: Implement proper multi-sample batch processing for SimplePFN
            actual_batch_size = X_train.shape[0]
            print(f"Processing batch with {actual_batch_size} samples, using first sample for SimplePFN")
            
            X_train = X_train[0:1]  # Take first sample: (batch_size, n_train, features) -> (1, n_train, features)
            y_train = y_train[0:1]  # Take first sample: (batch_size, n_train, 1) -> (1, n_train, 1)
            X_test = X_test[0:1]    # Take first sample: (batch_size, n_test, features) -> (1, n_test, features)
            
            # Flatten y_test for loss computation
            y_test = y_test[0].squeeze(-1)  # (n_test, 1) -> (n_test,)
            
            # Print final tensor shapes that will be passed to SimplePFN
            print(f"Final training tensors - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            return X_train, y_train, X_test, y_test
        else:
            raise ValueError(f"Expected list of 4 tensors, got {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
                
            X = X.to(self.device)
            
            # Ensure X has correct dimensions
            if len(X.shape) == 1:
                X = X.unsqueeze(0)
            if len(X.shape) > 2:
                X = X.view(X.shape[0], -1)
                
            batch_size = X.shape[0]
            split_idx = max(1, batch_size // 2)
            
            X_train = X[:split_idx].unsqueeze(0)
            y_train = torch.zeros(1, split_idx, 1).to(self.device)  # Dummy labels
            X_test = X[split_idx:].unsqueeze(0)
            y_test = torch.zeros(X.shape[0] - split_idx).to(self.device)  # Dummy test labels
            
            return X_train, y_train, X_test, y_test
    
    def fit(self):
        """Train the SimplePFN model using the provided dataloader."""
        print(f"Starting SimplePFN training for {self.max_steps} steps...")
        
        # Start total timing
        total_start_time = time.time()
        batch_times = []
        
        self.model.train()
        
        for step, batch in enumerate(self.dataloader):
            if step >= self.max_steps:
                break   

            # Start batch timing
            batch_start_time = time.time()
                
            X_train, y_train, X_test, y_test = self._process_batch(batch)
            
            self.optimizer.zero_grad()
            
            # SimplePFN forward pass expects (X_train, y_train, X_test)
            output = self.model(X_train, y_train, X_test)
            
            # Extract predictions from SimplePFN output
            if isinstance(output, dict):
                predictions = output.get('predictions', output.get('y_pred', None))
                if predictions is None:
                    # Try to get the main output tensor
                    predictions = list(output.values())[0] if output else torch.zeros_like(y_test)
            else:
                predictions = output
            
            # Ensure predictions have the right shape for loss computation
            if len(predictions.shape) > 1 and predictions.shape[0] == 1:
                predictions = predictions.squeeze(0)  # Remove batch dimension
            
            loss = self.criterion(predictions, y_test)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            self.global_step += 1
            
            # End batch timing
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            # Simple logging every 10 steps with timing info
            print(f"Step {step}/{self.max_steps}, Loss: {loss.item():.6f}, Batch Time: {batch_time:.3f}s")
        
        # Calculate timing statistics
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        
        print("SimplePFN training completed!")
        print(f"=== Training Timing Summary ===")
        print(f"Total training time: {total_time:.3f}s ({total_time/60:.2f} minutes)")
        print(f"Average time per batch: {avg_batch_time:.3f}s")
        print(f"Total batches processed: {len(batch_times)}")
        if batch_times:
            print(f"Fastest batch: {min(batch_times):.3f}s")
            print(f"Slowest batch: {max(batch_times):.3f}s")
        
        return self.model


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
    """Main entry point for SimplePFN training with YAML config."""
    try:
        print("=== SimplePFN Training with YAML Config ===")
        
        # Load YAML configuration
        config = load_yaml_config()
        
        # Extract config sections
        scm_config = config.get('scm_config', {})
        dataset_config = config.get('dataset_config', {})
        preprocessing_config = config.get('preprocessing_config', {})
        model_config = extract_config_values(config.get('model_config', {}))
        training_config = extract_config_values(config.get('training_config', {}))
        
        print(f"Experiment: {config.get('experiment_name', 'unnamed')}")
        print(f"Description: {config.get('description', 'no description')}")
        print(f"Model config: {model_config}")
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
        batch_size = training_config['batch_size']
        print(f"Creating DataLoader with batch_size={batch_size}...")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=training_config.get("num_workers", 4),
            persistent_workers=True
        )
        print(f"DataLoader created with batch_size={batch_size}, num_workers: {training_config.get('num_workers', 4)}")

        
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
        
        # Use num_features from config if available, otherwise use detected input_size
        num_features = model_config.get("num_features", input_size)
        print(f"Using num_features: {num_features}")
        
        # Create SimplePFN model
        print("Creating SimplePFN model...")

        model = SimplePFNRegressor(
            num_features=num_features,
            d_model=model_config.get("d_model", 8),
            depth=model_config.get("depth", 1), 
            heads_feat=model_config.get("heads_feat", 2),
            heads_samp=model_config.get("heads_samp", 2),
            dropout=model_config.get("dropout", 0.1)
        )
        print(f"Model: {model}")
        
        # Create trainer with the dataloader
        print("Creating SimplePFN trainer...")
        trainer = SimplePFNTrainer(
            model=model,
            dataloader=dataloader,
            learning_rate=training_config.get("learning_rate", 1e-3),
            max_steps=training_config.get("max_steps", 10),
            device=training_config.get("device", "cpu")
        )
        
        # Start training
        print("Starting SimplePFN training...")
        trained_model = trainer.fit()
        
        print("=== SimplePFN Training Complete ===")
        print("SimplePFN training completed successfully!")
        return trained_model
        
    except Exception as e:
        print(f"SimplePFN training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

