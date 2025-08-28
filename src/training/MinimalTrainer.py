"""
Minimal trainer implementation - only the bare essentials for training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class MinimalTrainer:
    """Minimal trainer with only essential training functionality."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        num_workers: int = 1,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.num_workers = num_workers
        self.device = device
        
        # Setup optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.global_step = 0
        
    def _create_dataloader(self, dataset, shuffle=False):
        """Create a simple DataLoader."""
        #assert 1==2, "creating dataset"
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
    
    def _process_batch(self, batch):
        """Process a batch - assumes (X, y) format."""
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            X, y = batch
            return X.to(self.device), y.to(self.device)
        else:
            # Handle single tensor case
            X = batch.to(self.device)
            return X, X  # Auto-encoder style
    
    def fit(self):
        """Train the model."""
        print(f"Starting minimal training for {self.max_steps} steps...")
        
        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            prefetch_factor=2
        )
        
        self.model.train()
        
        for step, batch in enumerate(train_loader):
            print(step)

            

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
        
        print("Training completed!")
        return self.model
