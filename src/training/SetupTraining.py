"""
High-level training setup API for SimplePFN models.

This module provides the main API class that handles configuration-based training setup.
It takes configuration dictionaries and sets up the complete training pipeline.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required components
from models.SimplePFN import SimplePFNRegressor
from models.ExampleConfigs.SimplePFN_Configs import small_simplepfn_config, medium_simplepfn_config
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
from training.Trainer import Trainer
from training.configs import extract_config_values


class SetupTraining:
    """
    High-level API for SimplePFN training setup and execution.
    
    This class takes configuration dictionaries and handles:
    1. Model creation from model configs
    2. Dataset creation from data configs
    3. Trainer setup from training configs
    4. Complete training pipeline execution
    
    Example:
        ```python
        from training import SetupTraining
        
        setup = SetupTraining(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config
        )
        
        trained_model = setup.run_training()
        test_metrics = setup.evaluate_model()
        ```
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any], 
        training_config: Dict[str, Any]
    ):
        """
        Initialize training setup.
        
        Args:
            model_config: Configuration for SimplePFN model
            data_config: Configuration for dataset creation
            training_config: Configuration for training hyperparameters
        """
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        
        # Initialize components
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.trainer = None
        
        # Setup datasets and model
        self._setup_datasets()
        self._setup_model()
    
    def _setup_datasets(self):
        """Create datasets from data configuration."""
        # Extract data generation configs
        scm_config = self.data_config['scm_config']
        preprocessing_config = self.data_config['preprocessing_config']
        dataset_config = self.data_config['dataset_config']
        
        # Create dataset generator
        dataset_generator = MakePurelyObservationalDataset(
            scm_config=scm_config,
            preprocessing_config=preprocessing_config,
            dataset_config=dataset_config
        )
        
        # Generate datasets with different seeds for train/val/test
        train_seed = self.data_config.get('train_seed', 42)
        val_seed = self.data_config.get('val_seed', 123)
        test_seed = self.data_config.get('test_seed', 456)
        
        print(f"Creating training dataset with seed {train_seed}...")
        self.train_dataset = dataset_generator.create_dataset(seed=train_seed)
        
        # Create validation dataset if requested
        if self.data_config.get('create_val_dataset', True):
            print(f"Creating validation dataset with seed {val_seed}...")
            self.val_dataset = dataset_generator.create_dataset(seed=val_seed)
        
        # Create test dataset if requested
        if self.data_config.get('create_test_dataset', True):
            print(f"Creating test dataset with seed {test_seed}...")
            self.test_dataset = dataset_generator.create_dataset(seed=test_seed)
        
        print(f"✓ Datasets created:")
        print(f"  - Training: {len(self.train_dataset)} samples")
        if self.val_dataset:
            print(f"  - Validation: {len(self.val_dataset)} samples")
        if self.test_dataset:
            print(f"  - Test: {len(self.test_dataset)} samples")
    
    def _setup_model(self):
        """Create model from model configuration."""
        # Handle different ways of specifying model config
        if isinstance(self.model_config, str):
            # Model config name (e.g., 'small', 'medium')
            model_configs = {
                'small': small_simplepfn_config,
                'medium': medium_simplepfn_config
            }
            
            if self.model_config not in model_configs:
                raise ValueError(f"Unknown model config: {self.model_config}")
            
            config = model_configs[self.model_config]
        else:
            # Direct config dictionary
            config = self.model_config
        
        # Create model
        self.model = SimplePFNRegressor(**config)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model created:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
    
    def setup_trainer(self) -> Trainer:
        """
        Setup the trainer with all configurations.
        
        Returns:
            Configured Trainer instance
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call _setup_model() first.")
        if self.train_dataset is None:
            raise RuntimeError("Datasets not initialized. Call _setup_datasets() first.")
        
        # Extract values from configuration structure (handles value/distribution pattern)
        training_params = extract_config_values(self.training_config)
        
        # Create trainer with explicit parameters (step-based for PFN training)
        self.trainer = Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            learning_rate=training_params.get('learning_rate', 1e-3),
            weight_decay=training_params.get('weight_decay', 1e-4),
            max_steps=training_params.get('max_steps', 10000),  # Focus on steps
            max_epochs=training_params.get('max_epochs', 1),   # Usually 1 for synthetic data
            eta_min=training_params.get('eta_min', 1e-6),
            batch_size=training_params.get('batch_size', 32),
            num_workers=training_params.get('num_workers', 4),
            early_stopping_patience=training_params.get('early_stopping_patience', 1000),  # In steps
            device=training_params.get('device', 'auto'),
            precision=training_params.get('precision', '32'),
            gradient_clip_val=training_params.get('gradient_clip_val', None),
            save_dir=training_params.get('save_dir', './experiments/default'),
            experiment_name=training_params.get('experiment_name', 'default_experiment'),
            log_every_n_steps=training_params.get('log_every_n_steps', 50),
            checkpoint_every_n_steps=training_params.get('checkpoint_every_n_steps', 1000),
            wandb_project=training_params.get('wandb_project', None),
            wandb_tags=training_params.get('wandb_tags', []),
            wandb_notes=training_params.get('wandb_notes', ''),
        )
        
        print("✓ Trainer configured:")
        print(f"  - Experiment: {training_params.get('experiment_name', 'default_experiment')}")
        print(f"  - Learning rate: {training_params.get('learning_rate', 1e-3)}")
        print(f"  - Batch size: {training_params.get('batch_size', 32)}")
        print(f"  - Max steps: {training_params.get('max_steps', 10000)}")
        print(f"  - Max epochs: {training_params.get('max_epochs', 1)}")
        print(f"  - Device: {training_params.get('device', 'auto')}")
        print("  - PyTorch Lightning: Enabled")
        
        return self.trainer
    
    def run_training(self, resume_from_checkpoint: Optional[str] = None) -> torch.nn.Module:
        """
        Run the complete training pipeline.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
            
        Returns:
            Trained model
        """
        if self.trainer is None:
            self.setup_trainer()
        
        print("\n🚀 Starting training...")
        if resume_from_checkpoint:
            trained_model = self.trainer.fit(ckpt_path=resume_from_checkpoint)
        else:
            trained_model = self.trainer.fit()
        print("✅ Training completed!")
        
        return trained_model
    
    def evaluate_model(self, dataset: Optional[torch.utils.data.Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the trained model.
        
        Args:
            dataset: Dataset to evaluate on. If None, uses test dataset.
            
        Returns:
            Evaluation metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Run training first.")
        
        print("\n📊 Evaluating model...")
        metrics = self.trainer.evaluate(dataset=dataset)
        
        print("Evaluation results:")
        for metric, value in metrics.items():
            print(f"  - {metric.upper()}: {value:.6f}")
        
        return metrics
    
    def predict(self, dataset: torch.utils.data.Dataset) -> torch.Tensor:
        """
        Generate predictions using the trained model.
        
        Args:
            dataset: Dataset for prediction
            
        Returns:
            Model predictions
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Run training first.")
        
        print("\n🔮 Generating predictions...")
        predictions = self.trainer.predict(dataset)
        print(f"✓ Generated {len(predictions)} predictions")
        
        return predictions
    
    def save_model(self, path: str):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Run training first.")
        
        self.trainer.save_model(path)
        print(f"✓ Model saved to {path}")
    
    def get_training_history(self) -> Dict[str, list]:
        """
        Get the training history.
        
        Returns:
            Training history dictionary
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Run training first.")
        
        return self.trainer.training_history
    
    def get_model(self) -> torch.nn.Module:
        """
        Get the model instance.
        
        Returns:
            The SimplePFN model
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        
        return self.model
    
    def get_datasets(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Get all datasets.
        
        Returns:
            Dictionary containing all datasets
        """
        datasets = {'train': self.train_dataset}
        
        if self.val_dataset is not None:
            datasets['val'] = self.val_dataset
        
        if self.test_dataset is not None:
            datasets['test'] = self.test_dataset
        
        return datasets
    
    @staticmethod
    def from_config_files(
        model_config_name: str,
        scm_config: Dict[str, Any],
        preprocessing_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> 'SetupTraining':
        """
        Create SetupTraining from individual config components.
        
        Args:
            model_config_name: Name of model config ('small' or 'medium')
            scm_config: SCM configuration for data generation
            preprocessing_config: Preprocessing configuration
            dataset_config: Dataset configuration
            training_config: Training hyperparameter configuration
            
        Returns:
            Configured SetupTraining instance
        """
        # Combine data configs
        data_config = {
            'scm_config': scm_config,
            'preprocessing_config': preprocessing_config,
            'dataset_config': dataset_config
        }
        
        return SetupTraining(
            model_config=model_config_name,
            data_config=data_config,
            training_config=training_config
        )


def quick_train(
    model_config: str = 'small',
    scm_config: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    dataset_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    **training_kwargs
) -> Dict[str, Any]:
    """
    Quick training function with default configurations.
    
    Args:
        model_config: Model configuration name
        scm_config: SCM configuration (will use defaults if None)
        preprocessing_config: Preprocessing configuration (will use defaults if None)
        dataset_config: Dataset configuration (will use defaults if None)
        training_config: Training configuration (will use defaults if None)
        **training_kwargs: Additional training parameters to override
        
    Returns:
        Dictionary containing trained model, metrics, and trainer
    """
    # Import default configs if not provided
    if scm_config is None:
        from causal_prior.scm.Basic_Configs import default_sampling_config
        scm_config = default_sampling_config
    
    if preprocessing_config is None:
        from priordata_processing.ExampleConfigs.BasicConfigs import default_preprocessing_config
        preprocessing_config = default_preprocessing_config
    
    if dataset_config is None:
        from priordata_processing.ExampleConfigs.BasicConfigs import default_dataset_config
        dataset_config = default_dataset_config
    
    if training_config is None:
        # Use the new rigorous configuration structure
        from training.configs import small_config
        training_config = small_config
    
    # Override with any additional kwargs (convert to new structure if needed)
    if training_kwargs:
        # If user provides old-style kwargs, convert them to new structure
        for key, value in training_kwargs.items():
            training_config[key] = {'value': value}
    
    # Create and run training
    setup = SetupTraining.from_config_files(
        model_config_name=model_config,
        scm_config=scm_config,
        preprocessing_config=preprocessing_config,
        dataset_config=dataset_config,
        training_config=training_config
    )
    
    # Run training
    trained_model = setup.run_training()
    
    # Evaluate if test dataset exists
    metrics = {}
    if setup.test_dataset is not None:
        metrics = setup.evaluate_model()
    
    return {
        'model': trained_model,
        'trainer': setup.trainer,
        'metrics': metrics,
        'setup': setup
    }
