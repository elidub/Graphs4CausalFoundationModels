"""
SimplePFN Trainer Module
Contains the SimplePFNTrainer class for training SimplePFN models.
"""

import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader


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
        self.model = model
        self.dataloader = dataloader
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
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
        
        SimplePFN expects the same format. We now process the full batch.
        """
        if isinstance(batch, list) and len(batch) == 4:
            X_train, y_train, X_test, y_test = batch
            
            # Move to device
            X_train = X_train.to(self.device)
            y_train = y_train.to(self.device)
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            
            # Flatten y_test for loss computation while keeping batch dimension
            # y_test: (batch_size, n_test, 1) -> (batch_size, n_test)
            y_test = y_test.squeeze(-1)
            
            return X_train, y_train, X_test, y_test
        else:
            raise ValueError(f"Expected list of 4 tensors, got {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}")

    def fit(self):
        """Train the SimplePFN model."""
        print(f"Training SimplePFN for {self.max_steps} steps with learning rate {self.learning_rate}")
        print(f"Using device: {self.device}")
        
        # Start total timing
        total_start_time = time.time()
        batch_times = []
        losses = []
        
        self.model.train()
        
        # Print initial batch info
        sample_batch = next(iter(self.dataloader))
        if isinstance(sample_batch, list) and len(sample_batch) == 4:
            batch_size = sample_batch[0].shape[0]
            n_train = sample_batch[0].shape[1]
            n_test = sample_batch[2].shape[1]
            features = sample_batch[0].shape[2]
            print(f"Batch structure: {batch_size} samples, {n_train} train points, {n_test} test points, {features} features")
        
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
            if isinstance(output, dict) and 'predictions' in output:
                predictions = output['predictions']
            else:
                predictions = output
            
            # Ensure predictions have the right shape for loss computation
            # predictions should be (batch_size, n_test) to match y_test (batch_size, n_test)
            if len(predictions.shape) == 3 and predictions.shape[-1] == 1:
                # If predictions are (batch_size, n_test, 1), squeeze last dimension
                predictions = predictions.squeeze(-1)
            
            # Compute loss for the full batch
            loss = self.criterion(predictions, y_test)
            losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            self.global_step += 1
            
            # End batch timing
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            # Print progress - more frequent at start, less frequent later
            if step <= 10 or step % max(1, self.max_steps // 10) == 0:
                print(f"   Step {step:4d}/{self.max_steps} | Loss: {loss.item():.6f} | Time: {batch_time:.3f}s | Batch size: {X_train.shape[0]}")
        
        # End total timing
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # Print comprehensive timing and performance summary
        print(f"\nTRAINING SUMMARY:")
        print(f"   Total training time: {total_time:.2f}s")
        print(f"   Steps completed: {len(batch_times)}")
        if len(batch_times) > 0:
            print(f"   Average step time: {sum(batch_times)/len(batch_times):.3f}s")
            print(f"   Fastest step: {min(batch_times):.3f}s")
            print(f"   Slowest step: {max(batch_times):.3f}s")
            print(f"   Steps per second: {len(batch_times)/total_time:.2f}")
        
        if len(losses) > 0:
            print(f"   Initial loss: {losses[0]:.6f}")
            print(f"   Final loss: {losses[-1]:.6f}")
            print(f"   Loss reduction: {((losses[0] - losses[-1])/losses[0]*100):.2f}%")
            print(f"   Average loss: {sum(losses)/len(losses):.6f}")
        
        return self.model
