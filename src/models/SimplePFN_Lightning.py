"""
PyTorch Lightning trainer for SimplePFN model.

This module contains a Lightning module for training the SimplePFNRegressor
with MSE loss for direct regression, using Adam optimizer with cosine LR scheduling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.SimplePFN import SimplePFNRegressor


class SimplePFN_Lightning(pl.LightningModule):
    """
    PyTorch Lightning module for training SimplePFN models.
    
    This class handles the training loop, validation, loss computation,
    and metrics calculation for the SimplePFN regressor using MSE loss.
    Uses Adam optimizer with cosine annealing learning rate schedule.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        eta_min: float = 1e-6,
    ):
        """
        Initialize the Lightning module.
        
        Args:
            model_config: Configuration dict for SimplePFNRegressor
            learning_rate: Initial learning rate for optimizer
            weight_decay: Weight decay for regularization
            max_epochs: Maximum number of epochs for cosine annealing
            eta_min: Minimum learning rate for cosine annealing
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Extract num_features from first batch (will be set in setup)
        self.model = None
        self.model_config = model_config
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        
        # Loss function - MSE for regression
        self.mse_loss = nn.MSELoss()
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup the model after we know the data dimensions."""
        if self.model is None:
            # This will be called after the data module is set up
            # We'll initialize the model when we see the first batch
            pass
    
    def _init_model_if_needed(self, batch: Dict[str, torch.Tensor]) -> None:
        """Initialize model if not already done, using batch to infer num_features."""
        if self.model is None:
            X_train = batch["X_train"]
            num_features = X_train.shape[-1]
            
            self.model = SimplePFNRegressor(
                num_features=num_features,
                **self.model_config
            )
            
    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        self._init_model_if_needed({"X_train": X_train})
        return self.model(X_train, y_train, X_test)
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute loss for a batch.
        
        Args:
            batch: Dictionary containing X_train, y_train, X_test, y_test
            
        Returns:
            Total loss and dictionary of individual losses
        """
        X_train = batch["X_train"]
        y_train = batch["y_train"] 
        X_test = batch["X_test"]
        y_test = batch["y_test"]
        
        # Forward pass
        outputs = self(X_train, y_train, X_test)
        predictions = outputs["predictions"]  # [batch_size, num_test_samples]
        
        # MSE loss
        mse_loss = self.mse_loss(predictions, y_test)
        
        losses = {
            "total_loss": mse_loss,
            "mse_loss": mse_loss,
        }
        
        return mse_loss, losses
    
    def _compute_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute metrics for a batch.
        
        Args:
            batch: Dictionary containing X_train, y_train, X_test, y_test
            
        Returns:
            Dictionary of metrics
        """
        X_train = batch["X_train"]
        y_train = batch["y_train"]
        X_test = batch["X_test"] 
        y_test = batch["y_test"]
        
        # Forward pass
        outputs = self(X_train, y_train, X_test)
        predictions = outputs["predictions"]
        
        # Compute regression metrics
        mse = F.mse_loss(predictions, y_test)
        mae = F.l1_loss(predictions, y_test)
        
        # R-squared
        y_test_flat = y_test.view(-1)
        y_pred_flat = predictions.view(-1)
        ss_res = torch.sum((y_test_flat - y_pred_flat) ** 2)
        ss_tot = torch.sum((y_test_flat - torch.mean(y_test_flat)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)

        metrics = {
            "mse": mse,
            "mae": mae, 
            "r2": r2,
            "rmse": torch.sqrt(mse)
        }

        return metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        loss, losses = self._compute_loss(batch)
        metrics = self._compute_metrics(batch)
        
        # Log losses
        for name, value in losses.items():
            self.log(f"train_{name}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log metrics
        for name, value in metrics.items():
            self.log(f"train_{name}", value, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        loss, losses = self._compute_loss(batch)
        metrics = self._compute_metrics(batch)
        
        # Log losses
        for name, value in losses.items():
            self.log(f"val_{name}", value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log metrics  
        for name, value in metrics.items():
            self.log(f"val_{name}", value, on_step=False, on_epoch=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        loss, losses = self._compute_loss(batch)
        metrics = self._compute_metrics(batch)
        
        # Log losses
        for name, value in losses.items():
            self.log(f"test_{name}", value, on_step=False, on_epoch=True)
        
        # Log metrics
        for name, value in metrics.items():
            self.log(f"test_{name}", value, on_step=False, on_epoch=True)
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        X_train = batch["X_train"]
        y_train = batch["y_train"]
        X_test = batch["X_test"]
        
        # Forward pass
        outputs = self(X_train, y_train, X_test)
        predictions = outputs["predictions"]
        
        return predictions
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.eta_min
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


class SimplePFNTrainer:
    """
    High-level trainer class for SimplePFN models.
    
    This class provides a simple interface for training SimplePFN models
    with PyTorch Lightning.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None,
        trainer_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model_config: Configuration for the SimplePFN model
            training_config: Training hyperparameters
            trainer_config: PyTorch Lightning trainer configuration
        """
        self.model_config = model_config
        
        # Default training configuration
        default_training_config = {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler_patience": 10,
            "scheduler_factor": 0.5,
            "bin_loss_weight": 1.0,
            "regression_loss_weight": 1.0,
            "y_min": -5.0,
            "y_max": 5.0,
        }
        
        self.training_config = default_training_config
        if training_config:
            self.training_config.update(training_config)
        
        # Default trainer configuration
        default_trainer_config = {
            "max_epochs": 100,
            "accelerator": "auto",
            "devices": "auto",
            "precision": "16-mixed",
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 1,
        }
        
        self.trainer_config = default_trainer_config
        if trainer_config:
            self.trainer_config.update(trainer_config)
    
    def create_model(self) -> SimplePFN_Lightning:
        """Create a Lightning module with the specified configuration."""
        return SimplePFN_Lightning(
            model_config=self.model_config,
            **self.training_config
        )
    
    def create_trainer(self, **kwargs) -> pl.Trainer:
        """Create a PyTorch Lightning trainer."""
        config = self.trainer_config.copy()
        config.update(kwargs)
        return pl.Trainer(**config)
    
    def fit(
        self,
        train_dataloader,
        val_dataloader=None,
        **trainer_kwargs
    ) -> Tuple[SimplePFN_Lightning, pl.Trainer]:
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            **trainer_kwargs: Additional arguments for the trainer
            
        Returns:
            Tuple of (trained_model, trainer)
        """
        model = self.create_model()
        trainer = self.create_trainer(**trainer_kwargs)
        
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        return model, trainer


class SimplePFNTrainerAdvanced:
    """
    Advanced trainer class for SimplePFN models with additional features.
    
    This class provides comprehensive training functionality including:
    - Automatic checkpointing and early stopping
    - Learning rate monitoring and logging
    - Model evaluation and testing
    - Prediction utilities
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        save_dir: str = "./checkpoints",
        experiment_name: str = "simplepfn_experiment",
        monitor_metric: str = "val_mse",
        early_stopping_patience: int = 20,
        checkpoint_top_k: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        eta_min: float = 1e-6,
    ):
        """
        Initialize the advanced trainer.
        
        Args:
            model_config: Configuration for the SimplePFN model
            save_dir: Directory to save checkpoints and logs
            experiment_name: Name for this experiment
            monitor_metric: Metric to monitor for checkpointing/early stopping
            early_stopping_patience: Patience for early stopping
            checkpoint_top_k: Number of best checkpoints to keep
            learning_rate: Initial learning rate
            weight_decay: Weight decay for regularization
            max_epochs: Maximum training epochs
            eta_min: Minimum learning rate for cosine annealing
        """
        self.model_config = model_config
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.monitor_metric = monitor_metric
        
        # Training configuration
        self.training_config = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "eta_min": eta_min,
        }
        
        # Setup callbacks
        self.callbacks = self._setup_callbacks(
            early_stopping_patience, 
            checkpoint_top_k
        )
        
        # Setup logger
        self.logger = self._setup_logger()
        
    def _setup_callbacks(self, early_stopping_patience: int, checkpoint_top_k: int):
        """Setup training callbacks."""
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
        
        callbacks = []
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor=self.monitor_metric,
            patience=early_stopping_patience,
            mode="min",
            verbose=True
        )
        callbacks.append(early_stop)
        
        # Model checkpointing
        checkpoint = ModelCheckpoint(
            dirpath=f"{self.save_dir}/{self.experiment_name}",
            filename="{epoch}-{" + self.monitor_metric + ":.4f}",
            monitor=self.monitor_metric,
            mode="min",
            save_top_k=checkpoint_top_k,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        
        return callbacks
    
    def _setup_logger(self):
        """Setup experiment logger."""
        from pytorch_lightning.loggers import TensorBoardLogger
        
        logger = TensorBoardLogger(
            save_dir=self.save_dir,
            name=self.experiment_name,
            version=None,  # Will auto-increment
        )
        
        return logger
    
    def create_model(self) -> SimplePFN_Lightning:
        """Create a SimplePFN Lightning module."""
        return SimplePFN_Lightning(
            model_config=self.model_config,
            **self.training_config
        )
    
    def create_trainer(
        self, 
        max_epochs: Optional[int] = None,
        accelerator: str = "auto",
        devices: str = "auto",
        precision: str = "16-mixed",
        **kwargs
    ) -> pl.Trainer:
        """Create a PyTorch Lightning trainer with all configurations."""
        
        trainer_config = {
            "max_epochs": max_epochs or self.training_config["max_epochs"],
            "accelerator": accelerator,
            "devices": devices,
            "precision": precision,
            "callbacks": self.callbacks,
            "logger": self.logger,
            "gradient_clip_val": 1.0,
            "log_every_n_steps": 10,
            "check_val_every_n_epoch": 1,
            "enable_progress_bar": True,
            "enable_model_summary": True,
        }
        
        # Update with any additional kwargs
        trainer_config.update(kwargs)
        
        return pl.Trainer(**trainer_config)
    
    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        **trainer_kwargs
    ) -> Tuple[SimplePFN_Lightning, pl.Trainer]:
        """
        Train the model with full configuration.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            **trainer_kwargs: Additional trainer arguments
            
        Returns:
            Tuple of (trained_model, trainer)
        """
        model = self.create_model()
        trainer = self.create_trainer(**trainer_kwargs)
        
        # Log model architecture
        print("Training SimplePFN with configuration:")
        print(f"Model config: {self.model_config}")
        print(f"Training config: {self.training_config}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Save directory: {self.save_dir}")
        
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        return model, trainer
    
    def test(
        self,
        model: SimplePFN_Lightning,
        test_dataloader,
        trainer: Optional[pl.Trainer] = None
    ) -> Dict[str, float]:
        """
        Test the trained model.
        
        Args:
            model: Trained SimplePFN model
            test_dataloader: Test data loader
            trainer: Optional trainer (will create new one if not provided)
            
        Returns:
            Dictionary of test metrics
        """
        if trainer is None:
            trainer = self.create_trainer(max_epochs=1)  # Just for testing
        
        results = trainer.test(model, test_dataloader)
        return results[0] if results else {}
    
    def predict(
        self,
        model: SimplePFN_Lightning,
        predict_dataloader,
        trainer: Optional[pl.Trainer] = None
    ) -> torch.Tensor:
        """
        Generate predictions using the trained model.
        
        Args:
            model: Trained SimplePFN model
            predict_dataloader: Data loader for prediction
            trainer: Optional trainer (will create new one if not provided)
            
        Returns:
            Tensor of predictions
        """
        if trainer is None:
            trainer = self.create_trainer(max_epochs=1)  # Just for prediction
        
        predictions = trainer.predict(model, predict_dataloader)
        return torch.cat(predictions, dim=0) if predictions else torch.tensor([])


if __name__ == "__main__":
    """
    Example usage of the SimplePFN trainers.
    """
    from models.ExampleConfigs.SimplePFN_Configs import small_simplepfn_config
    
    # Basic trainer example
    print("=== Basic Trainer Example ===")
    basic_trainer = SimplePFNTrainer(
        model_config=small_simplepfn_config,
        training_config={
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        },
        trainer_config={
            "max_epochs": 10,
            "accelerator": "cpu",  # For testing
        }
    )
    
    basic_model = basic_trainer.create_model()
    print(f"Basic model created: {type(basic_model).__name__}")
    
    # Advanced trainer example
    print("\n=== Advanced Trainer Example ===")
    advanced_trainer = SimplePFNTrainerAdvanced(
        model_config=small_simplepfn_config,
        experiment_name="test_experiment",
        learning_rate=1e-3,
        max_epochs=50,
        early_stopping_patience=10,
    )
    
    advanced_model = advanced_trainer.create_model()
    print(f"Advanced model created: {type(advanced_model).__name__}")
    print(f"Experiment name: {advanced_trainer.experiment_name}")
    print(f"Monitor metric: {advanced_trainer.monitor_metric}")
    
    print("\nTrainers created successfully!")
    print(f"Model config: {small_simplepfn_config}")
    print(f"Basic model: {basic_model}")
    print(f"Advanced model: {advanced_model}")
