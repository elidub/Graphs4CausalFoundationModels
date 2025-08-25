"""
Core training implementation for SimplePFN models using PyTorch Lightning.

This module contains the low-level Trainer class that handles the actual training loop.
It takes explicit parameters like model, datasets, and hyperparameters directly.

Note: For PFN training, we focus on training steps rather than epochs since all data 
is synthetic and generated on-the-fly. We may train for just one "epoch" but with 
many synthetic samples per step.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping as PLEarlyStopping
from pytorch_lightning.loggers import WandbLogger
import os
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging
import numpy as np

# Optional imports with fallbacks
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SimplePFNLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for SimplePFN training.
    
    This wraps a SimplePFN model in a Lightning module for robust training.
    Focuses on training steps rather than epochs since data is synthetic.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        eta_min: float = 1e-6,
        max_steps: int = 10000,  # Focus on steps, not epochs
        criterion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.max_steps = max_steps
        self.criterion = criterion or nn.MSELoss()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model', 'criterion'])
        
        # Training metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Store for epoch-end aggregation
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'y_true': y.detach(),
            'y_pred': y_pred.detach()
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        
        # Calculate additional metrics
        with torch.no_grad():
            mse = torch.mean((y_pred - y) ** 2)
            mae = torch.mean(torch.abs(y_pred - y))
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mse', mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_mae', mae, prog_bar=False, on_step=False, on_epoch=True)
        
        # Store for epoch-end aggregation
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'mse': mse.detach(),
            'mae': mae.detach(),
            'y_true': y.detach(),
            'y_pred': y_pred.detach()
        })
        
        return loss
    
    def on_train_epoch_end(self):
        if self.training_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            self.log('train_loss_epoch', avg_loss)
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            avg_mse = torch.stack([x['mse'] for x in self.validation_step_outputs]).mean()
            avg_mae = torch.stack([x['mae'] for x in self.validation_step_outputs]).mean()
            
            # Calculate correlation if we have enough data
            all_y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs])
            all_y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs])
            
            if len(all_y_true.shape) > 1:
                # Multi-dimensional output - calculate mean correlation
                correlations = []
                for dim in range(all_y_true.shape[1]):
                    if torch.std(all_y_true[:, dim]) > 1e-6 and torch.std(all_y_pred[:, dim]) > 1e-6:
                        corr = torch.corrcoef(torch.stack([all_y_true[:, dim], all_y_pred[:, dim]]))[0, 1]
                        if not torch.isnan(corr):
                            correlations.append(corr)
                correlation = torch.mean(torch.stack(correlations)) if correlations else torch.tensor(0.0)
            else:
                # 1D output
                if torch.std(all_y_true) > 1e-6 and torch.std(all_y_pred) > 1e-6:
                    correlation = torch.corrcoef(torch.stack([all_y_true, all_y_pred]))[0, 1]
                    if torch.isnan(correlation):
                        correlation = torch.tensor(0.0)
                else:
                    correlation = torch.tensor(0.0)
            
            self.log('val_correlation', correlation, prog_bar=True)
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Use step-based scheduling since we focus on steps not epochs
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer, 
                T_max=self.max_steps,
                eta_min=self.eta_min
            ),
            'interval': 'step',  # Update every step, not epoch
            'frequency': 1,
        }
        
        return [optimizer], [scheduler]


class Trainer:
    """
    Low-level trainer for SimplePFN models using PyTorch Lightning.
    
    This class provides an interface similar to the original trainer but uses
    PyTorch Lightning underneath for robust training. It focuses on training
    steps rather than epochs since all data is synthetic.
    
    Args:
        model: The SimplePFN model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for L2 regularization
        max_steps: Maximum number of training steps (instead of epochs)
        max_epochs: Maximum epochs (usually 1 for synthetic data)
        eta_min: Minimum learning rate for cosine annealing
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        early_stopping_patience: Patience for early stopping (in steps)
        device: Training device ('cpu', 'cuda', 'auto')
        precision: Training precision ('16-mixed', '32', '64')
        gradient_clip_val: Gradient clipping value (optional)
        save_dir: Directory to save checkpoints and models
        experiment_name: Name for the experiment
        log_every_n_steps: Logging frequency
        checkpoint_every_n_steps: Checkpoint saving frequency
        wandb_project: Weights & Biases project name (optional)
        wandb_tags: List of tags for W&B (optional)
        wandb_notes: Notes for W&B experiment (optional)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_steps: int = 10000,  # Focus on steps instead of epochs
        max_epochs: int = 1,     # Usually just 1 for synthetic data
        eta_min: float = 1e-6,
        batch_size: int = 32,
        num_workers: int = 4,
        early_stopping_patience: int = 1000,  # In steps, not epochs
        device: str = 'auto',
        precision: str = '32',
        gradient_clip_val: Optional[float] = None,
        save_dir: str = './experiments/default',
        experiment_name: str = 'default_experiment',
        log_every_n_steps: int = 50,
        checkpoint_every_n_steps: int = 1000,
        wandb_project: Optional[str] = None,
        wandb_tags: Optional[list] = None,
        wandb_notes: Optional[str] = None,
    ):
        # Store parameters
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_steps = log_every_n_steps
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.wandb_project = wandb_project
        self.wandb_tags = wandb_tags or []
        self.wandb_notes = wandb_notes or ''
        self.experiment_name = experiment_name
        
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Setup precision for Lightning
        if precision == '16':
            self.precision = '16-mixed'  # Lightning format
        else:
            self.precision = precision
        
        # Setup save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Lightning components
        self.lightning_module = None
        self.trainer = None
        self.logger = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'val_correlation': []
        }
    
    def _setup_lightning_trainer(self):
        """Setup PyTorch Lightning trainer with all callbacks and loggers."""
        # Setup logger
        if self.wandb_project and WANDB_AVAILABLE:
            try:
                self.logger = WandbLogger(
                    project=self.wandb_project,
                    name=self.experiment_name,
                    tags=self.wandb_tags,
                    notes=self.wandb_notes,
                    save_dir=str(self.save_dir)
                )
            except Exception as e:
                print(f"Warning: Could not setup W&B logger: {e}")
                self.logger = None
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.save_dir / 'checkpoints',
            filename='{epoch}-{val_loss:.4f}' if self.val_dataset else '{epoch}-{train_loss:.4f}',
            monitor='val_loss' if self.val_dataset else 'train_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            every_n_train_steps=self.checkpoint_every_n_steps,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.val_dataset and self.early_stopping_patience > 0:
            early_stop_callback = PLEarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                mode='min',
                verbose=True,
            )
            callbacks.append(early_stop_callback)
        
        # Setup trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            log_every_n_steps=self.log_every_n_steps,
            callbacks=callbacks,
            logger=self.logger,
            default_root_dir=str(self.save_dir),
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
    
    def fit(self) -> nn.Module:
        """
        Train the model using PyTorch Lightning.
        
        Returns:
            The trained model (the original model is modified in-place)
        """
        # Create Lightning module
        self.lightning_module = SimplePFNLightningModule(
            model=self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            eta_min=self.eta_min,
            max_steps=self.max_steps
        )
        
        # Setup trainer
        self._setup_lightning_trainer()
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device == 'cuda',
            persistent_workers=self.num_workers > 0
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.device == 'cuda',
                persistent_workers=self.num_workers > 0
            )
        
        # Train the model
        print(f"Starting training for {self.max_steps} steps...")
        print(f"Device: {self.device}, Precision: {self.precision}")
        print(f"Batch size: {self.batch_size}, Learning rate: {self.learning_rate}")
        
        self.trainer.fit(
            self.lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Update history from Lightning logs
        if hasattr(self.trainer, 'callback_metrics'):
            for key, value in self.trainer.callback_metrics.items():
                if key.startswith(('train_', 'val_')):
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(float(value))
        
        print("Training completed!")
        return self.model
    
    def evaluate(self, dataset: torch.utils.data.Dataset) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.lightning_module is None:
            raise RuntimeError("Model must be trained first. Call fit() before evaluate().")
        
        # Create data loader
        eval_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device == 'cuda'
        )
        
        # Run evaluation
        if self.trainer is None:
            self._setup_lightning_trainer()
        
        results = self.trainer.test(
            self.lightning_module,
            dataloaders=eval_loader,
            verbose=False
        )
        
        # Extract metrics
        metrics = {}
        if results:
            for key, value in results[0].items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = float(value.item())
                else:
                    metrics[key] = float(value)
        
        return metrics
    
    def predict(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        """
        Generate predictions for a dataset.
        
        Args:
            dataset: Dataset to predict on
            
        Returns:
            Numpy array of predictions
        """
        if self.lightning_module is None:
            raise RuntimeError("Model must be trained first. Call fit() before predict().")
        
        # Create data loader
        pred_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device == 'cuda'
        )
        
        # Run prediction
        if self.trainer is None:
            self._setup_lightning_trainer()
        
        predictions = self.trainer.predict(
            self.lightning_module,
            dataloaders=pred_loader
        )
        
        # Concatenate all predictions
        if predictions:
            return torch.cat(predictions).cpu().numpy()
        else:
            return np.array([])
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if self.lightning_module is None:
            raise RuntimeError("Model must be trained first. Call fit() before save_model().")
        
        # Save the model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'lightning_state_dict': self.lightning_module.state_dict(),
            'hyperparameters': self.lightning_module.hparams,
            'history': self.history
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> nn.Module:
        """
        Load a saved model.
        
        Args:
            path: Path to load the model from
            
        Returns:
            The loaded model
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Recreate Lightning module if needed
        if 'lightning_state_dict' in checkpoint:
            self.lightning_module = SimplePFNLightningModule(
                model=self.model,
                **checkpoint.get('hyperparameters', {})
            )
            self.lightning_module.load_state_dict(checkpoint['lightning_state_dict'])
        
        # Load history
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Model loaded from {path}")
        return self.model
