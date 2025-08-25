"""
Utility functions for SimplePFN training.

This module contains helper functions for data loading, model utilities,
and experiment management.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Any, List
from pathlib import Path
import json

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


class SimplePFNDataset(Dataset):
    """
    Simple dataset wrapper for SimplePFN training data.
    
    This dataset handles the conversion between raw data and the format
    expected by SimplePFN models.
    """
    
    def __init__(self, samples: torch.Tensor, targets: torch.Tensor):
        """
        Initialize dataset.
        
        Args:
            samples: Input samples tensor [batch_size, seq_len, feature_dim]
            targets: Target values tensor [batch_size, 1]
        """
        assert len(samples) == len(targets), "Samples and targets must have same length"
        self.samples = samples
        self.targets = targets
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'samples': self.samples[idx],
            'targets': self.targets[idx]
        }


def create_dataloaders(
    train_samples: torch.Tensor,
    train_targets: torch.Tensor,
    val_samples: torch.Tensor = None,
    val_targets: torch.Tensor = None,
    test_samples: torch.Tensor = None,
    test_targets: torch.Tensor = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_samples: Training input samples
        train_targets: Training targets
        val_samples: Validation input samples (optional)
        val_targets: Validation targets (optional)
        test_samples: Test input samples (optional)
        test_targets: Test targets (optional)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Dictionary containing data loaders
    """
    dataloaders = {}
    
    # Training data loader
    train_dataset = SimplePFNDataset(train_samples, train_targets)
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Validation data loader
    if val_samples is not None and val_targets is not None:
        val_dataset = SimplePFNDataset(val_samples, val_targets)
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    # Test data loader
    if test_samples is not None and test_targets is not None:
        test_dataset = SimplePFNDataset(test_samples, test_targets)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    return dataloaders


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make deterministic (might impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """
    Plot training history (works with both epoch and step-based training).
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Cannot plot training history.")
        return
        
    # Use generic x-axis that works for both epochs and steps
    x_values = range(1, len(history['train_loss']) + 1)
    x_label = 'Training Steps' if len(history['train_loss']) > 1000 else 'Epochs'
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(x_values, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        val_x_values = range(1, len(history['val_loss']) + 1)
        axes[0].plot(val_x_values, history['val_loss'], 'r-', label='Validation Loss')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Learning rate plot (if available)
    if 'learning_rate' in history and history['learning_rate']:
        axes[1].plot(x_values, history['learning_rate'], 'g-', label='Learning Rate')
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_yscale('log')
    else:
        # Plot additional metrics if learning rate not available
        if 'val_mse' in history and history['val_mse']:
            axes[1].plot(x_values, history['val_mse'], 'm-', label='Validation MSE')
        if 'val_correlation' in history and history['val_correlation']:
            ax2 = axes[1].twinx()
            ax2.plot(x_values, history['val_correlation'], 'c-', label='Validation Correlation')
            ax2.set_ylabel('Correlation')
            ax2.legend(loc='upper right')
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel('MSE')
        axes[1].set_title('Additional Metrics')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def save_experiment_summary(
    config: Dict[str, Any], 
    metrics: Dict[str, float], 
    save_dir: str
):
    """
    Save experiment summary including config and final metrics.
    
    Args:
        config: Training configuration
        metrics: Final evaluation metrics
        save_dir: Directory to save summary
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'config': config,
        'final_metrics': metrics,
        'model_info': {
            'model_class': 'SimplePFNRegressor',
            'config_name': config.get('model_config_name', 'unknown')
        }
    }
    
    summary_file = save_path / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Experiment summary saved to {summary_file}")


def load_experiment_summary(summary_path: str) -> Dict[str, Any]:
    """
    Load experiment summary from file.
    
    Args:
        summary_path: Path to summary file
        
    Returns:
        Experiment summary dictionary
    """
    with open(summary_path, 'r') as f:
        return json.load(f)


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the best checkpoint in a directory based on validation loss.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to best checkpoint
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_path.glob('checkpoint_epoch_*.pt'))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    best_checkpoint = None
    best_val_loss = float('inf')
    
    for checkpoint_file in checkpoint_files:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        val_loss = checkpoint.get('val_loss', float('inf'))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = str(checkpoint_file)
    
    if best_checkpoint is None:
        # Fallback to latest checkpoint
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        best_checkpoint = str(checkpoint_files[-1])
    
    return best_checkpoint


def compare_models(model_dirs: List[str]) -> Dict[str, Any]:
    """
    Compare multiple trained models.
    
    Args:
        model_dirs: List of model directories to compare
        
    Returns:
        Comparison results dictionary
    """
    results = {}
    
    for model_dir in model_dirs:
        model_path = Path(model_dir)
        
        # Load experiment summary
        summary_file = model_path / 'experiment_summary.json'
        if summary_file.exists():
            summary = load_experiment_summary(str(summary_file))
            experiment_name = summary['config'].get('experiment_name', model_path.name)
            results[experiment_name] = {
                'metrics': summary.get('final_metrics', {}),
                'config': summary.get('config', {})
            }
    
    return results


def print_model_comparison(comparison_results: Dict[str, Any]):
    """
    Print a formatted comparison of model results.
    
    Args:
        comparison_results: Results from compare_models function
    """
    print("=== Model Comparison ===\n")
    
    # Extract metrics for comparison
    metric_names = set()
    for result in comparison_results.values():
        metric_names.update(result.get('metrics', {}).keys())
    
    # Print header
    print(f"{'Model':<20}", end="")
    for metric in sorted(metric_names):
        print(f"{metric:<12}", end="")
    print()
    print("-" * (20 + 12 * len(metric_names)))
    
    # Print results
    for model_name, result in comparison_results.items():
        print(f"{model_name:<20}", end="")
        metrics = result.get('metrics', {})
        for metric in sorted(metric_names):
            value = metrics.get(metric, 'N/A')
            if isinstance(value, float):
                print(f"{value:<12.6f}", end="")
            else:
                print(f"{str(value):<12}", end="")
        print()


class EarlyStopping:
    """
    Early stopping utility class.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
