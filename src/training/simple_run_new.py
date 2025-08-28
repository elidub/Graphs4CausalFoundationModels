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
from pathlib import Path
from torch.utils.data import DataLoader

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# AFTER the preamble, do project imports
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
from models.SimplePFN import SimplePFNRegressor

# Default config path
DEFAULT_CONFIG_PATH = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/configs/early_test2.yaml"


def get_device(config_device):
    """
    Simple device setup - use what's specified in config.
    Only supports 'cpu' and 'cuda'. If cuda is requested but not available, raise error.
    """
    if config_device == "cpu":
        print("Using CPU")
        return torch.device('cpu')
    elif config_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested in config but CUDA is not available!")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda')
    else:
        raise ValueError(f"Unsupported device '{config_device}'. Only 'cpu' and 'cuda' are supported.")


class SimplePFNTrainer:
    """Trainer for SimplePFN that accepts a dataloader as input with GPU support."""
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_steps: int = 1000,
        device: torch.device = torch.device("cpu"),
        gradient_clip_val: float = 1.0
    ):
        self.device = device
        self.model = model.to(device)
        self.dataloader = dataloader
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val
        
        # Setup optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
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
            
            # Move to device
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            
            # For now, handle only batch_size=1 (first sample in batch)
            # TODO: Handle multiple samples in batch properly
            X_train = X_train[0:1]  # (batch_size, n_train, features) -> (1, n_train, features)
            y_train = y_train[0:1]  # (batch_size, n_train, 1) -> (1, n_train, 1)
            X_test = X_test[0:1]    # (batch_size, n_test, features) -> (1, n_test, features)
            y_test = y_test[0].squeeze(-1)  # (n_test, 1) -> (n_test,)
            
            return X_train, y_train, X_test, y_test
        else:
            raise ValueError(f"Expected list of 4 tensors, got {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
    
    def fit(self):
        """Train the SimplePFN model using the provided dataloader."""
        print(f"Starting SimplePFN training for {self.max_steps} steps...")
        
        self.model.train()
        
        # Initialize metrics
        total_loss = 0.0
        step_count = 0
        
        try:
            for step, batch in enumerate(self.dataloader):
                if step >= self.max_steps:
                    break
                    
                X_train, y_train, X_test, y_test = self._process_batch(batch)
                
                # Debug shapes
                print(f"Debug - X_train.shape: {X_train.shape}")
                print(f"Debug - y_train.shape: {y_train.shape}")
                print(f"Debug - X_test.shape: {X_test.shape}")
                print(f"Debug - y_test.shape: {y_test.shape}")
                
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(X_train, y_train, X_test)
                
                # Extract predictions from SimplePFN output
                if isinstance(output, dict):
                    predictions = output.get('predictions', output.get('y_pred', None))
                    if predictions is None:
                        predictions = list(output.values())[0] if output else torch.zeros_like(y_test)
                else:
                    predictions = output
                
                # Ensure predictions have the right shape for loss computation
                if len(predictions.shape) > 1 and predictions.shape[0] == 1:
                    predictions = predictions.squeeze(0)
                
                loss = self.criterion(predictions, y_test)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                step_count += 1
                self.global_step += 1
                
                # Log progress
                if step % 5 == 0:  # Log every 5 steps
                    avg_loss = total_loss / step_count
                    print(f"Step {step}/{self.max_steps}, Loss: {loss.item:.6f}")
                
        except Exception as e:
            print(f"SimplePFN training failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        print("SimplePFN training completed!")
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
        print("Creating DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=training_config.get("batch_size", 32),
            shuffle=True,
            num_workers=training_config.get("num_workers", 4)
        )
        print(f"DataLoader created with num_workers: {training_config.get('num_workers', 4)}")

        
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

        # Simple device setup from config
        config_device = training_config.get("device", "cpu")
        device = get_device(config_device)

        model = SimplePFNRegressor(
            num_features=num_features,
            d_model=model_config.get("d_model", 8),
            depth=model_config.get("depth", 1), 
            heads_feat=model_config.get("heads_feat", 2),
            heads_samp=model_config.get("heads_samp", 2),
            dropout=model_config.get("dropout", 0.1)
        )
        print(f"Model: {model}")
        
        # Create trainer
        print("Creating SimplePFN trainer...")
        trainer = SimplePFNTrainer(
            model=model,
            dataloader=dataloader,
            learning_rate=training_config.get("learning_rate", 1e-3),
            weight_decay=training_config.get("weight_decay", 1e-4),
            max_steps=training_config.get("max_steps", 10),
            device=device,
            gradient_clip_val=training_config.get("gradient_clip_val", 1.0)
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

