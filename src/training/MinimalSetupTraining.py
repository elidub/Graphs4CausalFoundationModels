"""
Minimal training setup - only the bare essentials for setting up training.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SimplePFN import SimplePFNRegressor
from models.ExampleConfigs.SimplePFN_Configs import small_simplepfn_config
from priordata_processing.Datasets.MakePurelyObservationalDataset import MakePurelyObservationalDataset
from training.MinimalTrainer import MinimalTrainer


class MinimalSetupTraining:
    """
    Minimal training setup with only essential functionality.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        data_config: Dict[str, Any], 
        training_config: Dict[str, Any],
        save_dir: Optional[str] = None
    ):
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.save_dir = Path(save_dir) if save_dir else Path("./checkpoints")
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.model = None
        self.train_dataset = None
        self.trainer = None
        
    def setup_model(self):
        """Setup the model with minimal configuration."""
        print("Setting up model...")
        
        # Use small config as default if not specified
        if 'model_type' in self.model_config:
            model_type = self.model_config['model_type']
        else:
            model_type = 'small'
        
        # Create model with basic config
        if model_type == 'small' or model_type not in ['medium', 'large']:
            config = small_simplepfn_config
        else:
            config = small_simplepfn_config  # Fallback to small
        
        # Override with any provided config
        config.update(self.model_config.get('model_params', {}))

        # Ensure num_features is set
        if 'num_features' not in config:
            # Try to get from various possible config locations
            n_vars = None
            
            # Try minimal config format first
            if 'causal_config' in self.data_config and 'n_vars' in self.data_config['causal_config']:
                n_vars = self.data_config['causal_config']['n_vars']
            # Try actual YAML format (scm_config.num_nodes)
            elif 'scm_config' in self.data_config and 'num_nodes' in self.data_config['scm_config']:
                num_nodes_config = self.data_config['scm_config']['num_nodes']
                if isinstance(num_nodes_config, dict) and 'distribution_parameters' in num_nodes_config:
                    # Use the high value from the distribution as a reasonable default
                    n_vars = num_nodes_config['distribution_parameters'].get('high', 8)
                elif isinstance(num_nodes_config, int):
                    n_vars = num_nodes_config
            
            # Set num_features or use default
            if n_vars is not None:
                config['num_features'] = n_vars
                print(f"Set num_features to {n_vars} from config")
            else:
                # Use reasonable default
                config['num_features'] = 8
                print(f"Using default num_features = 8 (no config found)")

        self.model = SimplePFNRegressor(**config)
        print(f"Model created: {type(self.model).__name__}")
        
    def setup_data(self):
        """Setup datasets with minimal configuration."""
        print("Setting up data...")
        
        # Handle both minimal and full config formats
        if 'scm_config' in self.data_config and 'dataset_config' in self.data_config:
            # Full YAML config format
            print("Using full YAML config format")
            scm_config = self.data_config['scm_config']
            dataset_config = self.data_config['dataset_config']
            preprocessing_config = self.data_config.get('preprocessing_config', {})
            
            # Create dataset maker with full configs
            dataset_maker = MakePurelyObservationalDataset(
                scm_config=scm_config,
                preprocessing_config=preprocessing_config,
                dataset_config=dataset_config
            )
            
            # Create datasets using the configured maker
            self.train_dataset = dataset_maker.create_dataset(seed=42)
            
        else:
            # Simple minimal config format - create basic datasets
            print("Using simple config format with defaults")
            n_samples_train = self.data_config.get('n_samples_train', 500)
            n_samples_val = self.data_config.get('n_samples_val', 100)
            
            # Create simple synthetic data for testing
            import torch
            from torch.utils.data import TensorDataset
            
            # Generate simple random data for testing
            n_features = 5  # Simple default
            train_X = torch.randn(n_samples_train, n_features)
            train_y = torch.sum(train_X, dim=1, keepdim=True) + 0.1 * torch.randn(n_samples_train, 1)
            
            self.train_dataset = TensorDataset(train_X, train_y)
        
        print(f"Training dataset: {len(self.train_dataset)} samples")
        
    def setup_trainer(self):
        """Setup trainer with minimal configuration."""
        print("Setting up trainer...")
        
        # Basic training config
        batch_size = self.training_config.get('batch_size', 32)
        learning_rate = self.training_config.get('learning_rate', 1e-3)
        max_steps = self.training_config.get('max_steps', 1000)
        num_workers = self.training_config.get('num_workers', 1)
        device = self.training_config.get('device', 'cpu')
        
        self.trainer = MinimalTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            num_workers=num_workers,
            device=device
        )
        
    def run_training(self):
        """Run the complete training pipeline."""
        print("=== Minimal Training Pipeline ===")
        
        # Setup everything
        self.setup_model()
        self.setup_data()
        self.setup_trainer()
        
        # Train
        trained_model = self.trainer.fit()
        
        print("=== Training Complete ===")
        return trained_model


def minimal_quick_train(config_path_or_dict):
    """
    Quick training function for minimal setup.
    
    Args:
        config_path_or_dict: Either a path to config file or a dict with configs
    
    Returns:
        Trained model
    """
    if isinstance(config_path_or_dict, (str, Path)):
        # Load from file (basic YAML loading)
        import yaml
        with open(config_path_or_dict, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = config_path_or_dict
    
    # Extract configs with defaults - handle both minimal and full YAML formats
    model_config = config.get('model', {})
    
    # For data config, try both formats
    data_config = config.get('data', {})
    if not data_config:
        # If no 'data' section, build one from the YAML structure
        data_config = {
            'scm_config': config.get('scm_config', {}),
            'dataset_config': config.get('dataset_config', {}),
            'preprocessing_config': config.get('preprocessing_config', {})
        }
    training_config = config.get('training', {})
    
    # Setup and run
    setup = MinimalSetupTraining(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config
    )
    
    return setup.run_training()
