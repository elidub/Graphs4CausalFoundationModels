"""
SimplePFN Trainer Module
Contains the SimplePFNTrainer class for training SimplePFN models.
"""

import os
import torch
import torch.nn as nn
import time
import signal
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Optional, Dict, Any, Union, Callable
from sklearn.metrics import r2_score, mean_squared_error


class SimplePFNTrainer:
    """Trainer for SimplePFN that accepts a dataloader as input."""
    
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        device: str = "cpu",
        wandb_run: Optional[object] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        save_dir: str = None,
        save_every: int = 0,
        run_name: str = None,
        bar_distribution: Optional[object] = None,  # BarDistribution for probabilistic training
        eval_dataloader: Optional[DataLoader] = None,  # Evaluation dataloader (can be same as training)
        eval_every: int = 0,  # Evaluate every N steps (0 to disable)
        eval_batches: int = 10,  # Number of batches to use for evaluation
        # Model selection parameters
        enable_model_selection: bool = False,  # Whether to enable automatic best model selection
        model_selection_metric: str = "eval/mse_median",  # Metric to use for model selection
        model_selection_mode: str = "min",  # "min" or "max" - whether lower or higher is better
    ):
        self.model = model
        self.dataloader = dataloader
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.device = device
        self.wandb_run = wandb_run
        self.scheduler_config = scheduler_config or {}
        self.save_dir = save_dir
        self.save_every = save_every
        self.run_name = run_name or f"model_{int(time.time())}"
        self.bar_distribution = bar_distribution  # Store BarDistribution
        
        # Evaluation parameters
        self.eval_dataloader = eval_dataloader
        self.eval_every = eval_every
        self.eval_batches = eval_batches
        
        # Model selection parameters
        self.enable_model_selection = enable_model_selection
        self.model_selection_metric = model_selection_metric
        self.model_selection_mode = model_selection_mode
        self.best_metric_value = float('inf') if model_selection_mode == "min" else float('-inf')
        self.best_model_state = None
        self.best_model_step = 0
        self.best_model_metadata = None
        
        # Create a run-specific subfolder for checkpoints
        self.run_save_dir = None
        if self.save_dir:
            self.run_save_dir = os.path.join(self.save_dir, self.run_name)
            try:
                os.makedirs(self.run_save_dir, exist_ok=True)
                print(f"Model checkpoints will be saved to: {os.path.abspath(self.run_save_dir)}")
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not create checkpoint directory {self.run_save_dir}: {e}")
                print(f"         Attempting to use temporary directory instead...")
                import tempfile
                self.run_save_dir = os.path.join(tempfile.gettempdir(), f"simplepfn_checkpoints_{self.run_name}")
                try:
                    os.makedirs(self.run_save_dir, exist_ok=True)
                    print(f"Model checkpoints will be saved to temporary directory: {self.run_save_dir}")
                except Exception as e2:
                    print(f"Error: Could not create temporary checkpoint directory: {e2}")
                    print(f"       Model checkpoints will be DISABLED for this run.")
                    self.run_save_dir = None
                    self.save_dir = None
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Choose loss function based on whether BarDistribution is available
        if self.bar_distribution is not None:
            self.criterion = None  # Will use BarDistribution.average_log_prob
            print(f"Using BarDistribution probabilistic loss")
        else:
            self.criterion = nn.MSELoss()
            print(f"Using MSE loss")
            
        self.global_step = 0
        
        # Setup scheduler if enabled
        self.scheduler = self._create_scheduler(self.scheduler_config)
            
        # Flag for graceful termination
        self.terminate = False
    
    def _create_scheduler(self, scheduler_config):
        sched_type = scheduler_config.get("type")
        if not sched_type:
            print(f"No scheduler specified, using constant learning rate: {self.learning_rate}")
            return None

        if sched_type == "linear_warmup_cosine_decay":
            # Get parameters with sensible defaults
            warmup_ratio = float(scheduler_config.get("warmup_ratio", 0.03))  # 3% of steps
            min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.1))   # 10% floor
            
            # Calculate warmup steps based on ratio
            warmup_steps = int(round(warmup_ratio * self.max_steps))
            warmup_steps = max(1, warmup_steps)  # avoid divide-by-zero

            def lr_lambda(current_step: int) -> float:
                # Linear warmup 0 -> 1
                if current_step < warmup_steps:
                    return float(current_step) / float(warmup_steps)

                # Cosine decay from 1 -> min_lr_ratio
                progress = (current_step - warmup_steps) / max(1, self.max_steps - warmup_steps)
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine  # 1 -> min_lr_ratio

            scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            print(f"Created linear warmup ({warmup_steps} steps) + cosine decay "
                  f"(min_lr_ratio={min_lr_ratio}) scheduler")
            return scheduler

        print(f"Unknown scheduler type: {sched_type}, using constant learning rate")
        return None

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

    def save_model(self, filename=None, metadata=None):
        """Save model to disk.
        
        Args:
            filename (str, optional): Specific filename to use. If None, auto-generated.
            metadata (dict, optional): Additional metadata to save with model.
        
        Returns:
            str: Path to the saved model file
        """
        if not self.run_save_dir:
            print("Warning: save_dir not provided, cannot save model")
            return None
            
        # Create filename if not provided
        if filename is None:
            filename = f"step_{self.global_step}.pt"
        
        # Ensure path is absolute
        path = os.path.join(self.run_save_dir, filename)
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add standard metadata
        metadata.update({
            'step': self.global_step,
            'learning_rate': self.learning_rate,
            'timestamp': time.time(),
            'device': self.device
        })
        
        # Save model state, optimizer state, and metadata
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metadata': metadata
        }
        
        # Save scheduler state if available
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            checkpoint['scheduler_config'] = self.scheduler_config
            
        # Save BarDistribution parameters if available
        if hasattr(self, 'bar_distribution') and self.bar_distribution is not None:
            bar_dist_state = {
                'num_bars': self.bar_distribution.num_bars,
                'min_width': self.bar_distribution.min_width,
                'scale_floor': self.bar_distribution.scale_floor,
                'max_fit_items': self.bar_distribution.max_fit_items,
                'log_prob_clip_min': self.bar_distribution.log_prob_clip_min,
                'log_prob_clip_max': self.bar_distribution.log_prob_clip_max,
                'device': str(self.bar_distribution.device),
                'dtype': str(self.bar_distribution.dtype),
            }
            
            # Save fitted parameters if the distribution has been fitted
            if (hasattr(self.bar_distribution, 'centers') and 
                self.bar_distribution.centers is not None):
                bar_dist_state.update({
                    'centers': self.bar_distribution.centers.cpu(),
                    'edges': self.bar_distribution.edges.cpu(),
                    'widths': self.bar_distribution.widths.cpu(),
                    'base_s_left': self.bar_distribution.base_s_left.cpu(),
                    'base_s_right': self.bar_distribution.base_s_right.cpu(),
                    'fitted': True
                })
            else:
                bar_dist_state['fitted'] = False
                
            checkpoint['bar_distribution'] = bar_dist_state
            print(f"   Saved BarDistribution parameters (fitted: {bar_dist_state['fitted']})")
            
        # Save to disk
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
        
        # Log to wandb if available
        if self.wandb_run:
            self.wandb_run.log({
                'checkpoints/saved_path': path,
                'checkpoints/global_step': self.global_step
            })
            
        return path

    def _update_best_model(self, metrics: Dict[str, float], train_loss: float = None) -> bool:
        """
        Update the best model based on the configured selection metric.
        
        Args:
            metrics: Dictionary of evaluation metrics
            train_loss: Current training loss (used as fallback)
            
        Returns:
            bool: True if this is a new best model, False otherwise
        """
        # Determine the metric value to use for selection
        metric_value = None
        metric_source = "unknown"
        
        # First try to use the configured evaluation metric
        if self.model_selection_metric in metrics:
            metric_value = metrics[self.model_selection_metric]
            metric_source = "eval"
        # Fallback options for evaluation metrics
        elif self.model_selection_metric.startswith("eval/") and not metrics:
            # If eval metrics requested but none available, try train loss
            if train_loss is not None:
                metric_value = train_loss
                metric_source = "train_fallback"
        # Direct fallback to training loss
        elif train_loss is not None and (self.model_selection_metric in ["train/loss", "loss"] or not metrics):
            metric_value = train_loss
            metric_source = "train"
        
        if metric_value is None:
            return False  # No valid metric available
            
        # Check if this is a new best
        is_better = False
        if self.model_selection_mode == "min":
            is_better = metric_value < self.best_metric_value
        else:  # "max"
            is_better = metric_value > self.best_metric_value
            
        if is_better:
            self.best_metric_value = metric_value
            self.best_model_step = self.global_step
            
            # Store the current model state
            self.best_model_state = {
                'model_state_dict': self.model.state_dict().copy(),
                'optimizer_state_dict': self.optimizer.state_dict().copy(),
            }
            
            # Store scheduler state if available
            if self.scheduler is not None:
                self.best_model_state['scheduler_state_dict'] = self.scheduler.state_dict().copy()
                self.best_model_state['scheduler_config'] = self.scheduler_config.copy()
            
            # Store BarDistribution state if available
            if hasattr(self, 'bar_distribution') and self.bar_distribution is not None:
                bar_dist_state = {
                    'num_bars': self.bar_distribution.num_bars,
                    'min_width': self.bar_distribution.min_width,
                    'scale_floor': self.bar_distribution.scale_floor,
                    'max_fit_items': self.bar_distribution.max_fit_items,
                    'log_prob_clip_min': self.bar_distribution.log_prob_clip_min,
                    'log_prob_clip_max': self.bar_distribution.log_prob_clip_max,
                    'device': str(self.bar_distribution.device),
                    'dtype': str(self.bar_distribution.dtype),
                }
                
                # Save fitted parameters if available
                if (hasattr(self.bar_distribution, 'centers') and 
                    self.bar_distribution.centers is not None):
                    bar_dist_state.update({
                        'centers': self.bar_distribution.centers.cpu().clone(),
                        'edges': self.bar_distribution.edges.cpu().clone(),
                        'widths': self.bar_distribution.widths.cpu().clone(),
                        'base_s_left': self.bar_distribution.base_s_left.cpu().clone(),
                        'base_s_right': self.bar_distribution.base_s_right.cpu().clone(),
                        'fitted': True
                    })
                else:
                    bar_dist_state['fitted'] = False
                    
                self.best_model_state['bar_distribution'] = bar_dist_state
            
            # Store metadata
            self.best_model_metadata = {
                'step': self.global_step,
                'metric_name': self.model_selection_metric,
                'metric_value': metric_value,
                'metric_source': metric_source,
                'timestamp': time.time(),
                'selection_mode': self.model_selection_mode,
            }
            
            print(f"   ✓ New best model! {self.model_selection_metric}={metric_value:.6f} (from {metric_source}) at step {self.global_step}")
            return True
        
        return False

    def save_best_model(self, filename: str = "best_model.pt") -> Optional[str]:
        """
        Save the best model encountered during training.
        
        Args:
            filename: Name of the file to save the best model to
            
        Returns:
            str: Path to saved model, or None if no best model available
        """
        if self.best_model_state is None:
            print("Warning: No best model state available to save")
            return None
            
        if not self.run_save_dir:
            print("Warning: save_dir not provided, cannot save best model")
            return None
            
        # Prepare the checkpoint
        checkpoint = self.best_model_state.copy()
        checkpoint['metadata'] = self.best_model_metadata.copy()
        checkpoint['metadata']['saved_as_best'] = True
        
        # Save to disk
        path = os.path.join(self.run_save_dir, filename)
        torch.save(checkpoint, path)
        
        print(f"Best model saved to {path}")
        print(f"   Best {self.best_model_metadata['metric_name']}: {self.best_model_metadata['metric_value']:.6f}")
        print(f"   Best step: {self.best_model_metadata['step']}")
        print(f"   Metric source: {self.best_model_metadata['metric_source']}")
        
        # Log to wandb if available
        if self.wandb_run:
            self.wandb_run.log({
                'best_model/metric_value': self.best_model_metadata['metric_value'],
                'best_model/step': self.best_model_metadata['step'],
                'best_model/saved_path': path,
            })
            
        return path

    def load_model(self, checkpoint_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        """
        Load model checkpoint including BarDistribution parameters.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            map_location: Device mapping for loading (e.g., 'cpu', 'cuda:0')
            
        Returns:
            Dict containing metadata from the checkpoint
        """
        if map_location is None:
            map_location = str(self.device)
            
        print(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("   Model state loaded")
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("   Optimizer state loaded")
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler') and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("   Scheduler state loaded")
        
        # Load BarDistribution parameters if available
        if 'bar_distribution' in checkpoint and self.bar_distribution is not None:
            bar_dist_state = checkpoint['bar_distribution']
            print(f"   Loading BarDistribution parameters (fitted: {bar_dist_state.get('fitted', False)})")
            
            # Restore basic configuration
            self.bar_distribution.num_bars = bar_dist_state['num_bars']
            self.bar_distribution.min_width = bar_dist_state['min_width']
            self.bar_distribution.scale_floor = bar_dist_state['scale_floor']
            self.bar_distribution.max_fit_items = bar_dist_state['max_fit_items']
            self.bar_distribution.log_prob_clip_min = bar_dist_state['log_prob_clip_min']
            self.bar_distribution.log_prob_clip_max = bar_dist_state['log_prob_clip_max']
            
            # Restore fitted parameters if available
            if bar_dist_state.get('fitted', False):
                self.bar_distribution.centers = bar_dist_state['centers'].to(self.device)
                self.bar_distribution.edges = bar_dist_state['edges'].to(self.device)
                self.bar_distribution.widths = bar_dist_state['widths'].to(self.device)
                self.bar_distribution.base_s_left = bar_dist_state['base_s_left'].to(self.device)
                self.bar_distribution.base_s_right = bar_dist_state['base_s_right'].to(self.device)
                print("   BarDistribution fitted parameters restored")
            else:
                print("   BarDistribution was not fitted when saved")
        
        # Return metadata
        metadata = checkpoint.get('metadata', {})
        print(f"Model checkpoint loaded successfully")
        if 'global_step' in metadata:
            print(f"   Checkpoint was saved at step {metadata['global_step']}")
        return metadata
    
    def setup_signal_handlers(self):
        """Setup signal handlers for clean shutdown and model saving on termination."""
        def signal_handler(sig, frame):
            print(f"\nReceived termination signal. Saving model before exiting...")
            self.terminate = True
            # Save final model
            self.save_model(filename="interrupted.pt", 
                           metadata={'termination_reason': 'signal_interrupted'})
            
        # Register signal handlers for common termination signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler) # Termination request

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the evaluation dataset and compute comprehensive metrics.
        
        Returns:
            Dict containing evaluation metrics including:
            - negative log likelihood (loss)
            - mean and median MSE 
            - mean and median R²
            - IQR and standard deviation for all metrics
        """
        if self.eval_dataloader is None:
            print("Warning: No evaluation dataloader provided, skipping evaluation")
            return {}
            
        print(f"Running evaluation on {self.eval_batches} batches...")
        
        self.model.eval()
        eval_losses = []
        eval_mses = []
        eval_r2s = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if batch_idx >= self.eval_batches:
                    break
                    
                X_train, y_train, X_test, y_test = self._process_batch(batch)
                
                # Forward pass
                output = self.model(X_train, y_train, X_test)
                
                # Extract predictions
                if isinstance(output, dict) and 'predictions' in output:
                    predictions = output['predictions']
                else:
                    predictions = output
                
                # Compute loss (negative log likelihood)
                if self.bar_distribution is not None:
                    # BarDistribution loss
                    log_prob = self.bar_distribution.average_log_prob(predictions, y_test)
                    loss = -log_prob.mean()
                    
                    # For MSE and R² metrics, extract mean predictions from BarDistribution
                    # predictions shape: (batch_size, n_test, K+4)
                    # We need to get the mean of the distribution
                    pred_mean = self.bar_distribution.mean(predictions)  # (batch_size, n_test)
                    pred_np = pred_mean.cpu().numpy()
                else:
                    # MSE loss
                    if len(predictions.shape) == 3 and predictions.shape[-1] == 1:
                        predictions = predictions.squeeze(-1)
                    loss = self.criterion(predictions, y_test)
                    
                    # Convert to numpy for sklearn metrics
                    pred_np = predictions.cpu().numpy()
                
                # Check for NaN/infinite values and skip problematic batches
                loss_value = loss.item()
                y_test_np = y_test.cpu().numpy()
                
                # Check if loss is NaN or infinite
                if not np.isfinite(loss_value):
                    print(f"   Skipping batch {batch_idx}: loss contains NaN/inf (loss={loss_value})")
                    continue
                    
                # Check if predictions contain NaN or infinite values
                if not np.all(np.isfinite(pred_np)):
                    nan_count = np.sum(~np.isfinite(pred_np))
                    print(f"   Skipping batch {batch_idx}: predictions contain {nan_count} NaN/inf values")
                    continue
                    
                # Check if targets contain NaN or infinite values
                if not np.all(np.isfinite(y_test_np)):
                    nan_count = np.sum(~np.isfinite(y_test_np))
                    print(f"   Skipping batch {batch_idx}: targets contain {nan_count} NaN/inf values")
                    continue
                
                # If we get here, the batch is valid
                eval_losses.append(loss_value)
                
                # Compute metrics for each sample in the batch
                batch_mses = []
                batch_r2s = []
                
                for i in range(pred_np.shape[0]):
                    # Double-check individual samples for safety
                    if not (np.all(np.isfinite(pred_np[i])) and np.all(np.isfinite(y_test_np[i]))):
                        continue  # Skip this sample if it has NaN/inf
                    
                    # MSE for this sample
                    try:
                        mse = mean_squared_error(y_test_np[i], pred_np[i])
                        if np.isfinite(mse):
                            batch_mses.append(mse)
                    except (ValueError, RuntimeWarning):
                        continue  # Skip if MSE computation fails
                    
                    # R² for this sample (handle case where y has no variance)
                    try:
                        r2 = r2_score(y_test_np[i], pred_np[i])
                        if np.isfinite(r2):
                            batch_r2s.append(r2)
                    except (ValueError, RuntimeWarning):
                        # R² is undefined when y has no variance or other issues
                        continue
                
                eval_mses.extend(batch_mses)
                eval_r2s.extend(batch_r2s)
        
        self.model.train()  # Return to training mode
        
        # Convert to numpy arrays for statistics
        eval_losses = np.array(eval_losses)
        eval_mses = np.array(eval_mses)
        eval_r2s = np.array(eval_r2s)
        
        # Check if we have any valid data
        if len(eval_losses) == 0:
            print("Warning: No valid evaluation data collected (all batches contained NaN/inf)")
            return {}
        
        if len(eval_mses) == 0:
            print("Warning: No valid MSE/R² samples collected")
            # Return only loss metrics
            metrics = {
                'eval/loss_mean': float(np.mean(eval_losses)),
                'eval/loss_median': float(np.median(eval_losses)),
                'eval/loss_std': float(np.std(eval_losses)),
                'eval/loss_iqr': float(np.percentile(eval_losses, 75) - np.percentile(eval_losses, 25)),
                'eval/num_batches': len(eval_losses),
                'eval/num_samples': 0
            }
        else:
            # Compute comprehensive statistics
            metrics = {
                # Loss (negative log likelihood)
                'eval/loss_mean': float(np.mean(eval_losses)),
                'eval/loss_median': float(np.median(eval_losses)),
                'eval/loss_std': float(np.std(eval_losses)),
                'eval/loss_iqr': float(np.percentile(eval_losses, 75) - np.percentile(eval_losses, 25)),
                
                # MSE
                'eval/mse_mean': float(np.mean(eval_mses)),
                'eval/mse_median': float(np.median(eval_mses)),
                'eval/mse_std': float(np.std(eval_mses)),
                'eval/mse_iqr': float(np.percentile(eval_mses, 75) - np.percentile(eval_mses, 25)),
                
                # R²
                'eval/r2_mean': float(np.mean(eval_r2s)),
                'eval/r2_median': float(np.median(eval_r2s)),
                'eval/r2_std': float(np.std(eval_r2s)),
                'eval/r2_iqr': float(np.percentile(eval_r2s, 75) - np.percentile(eval_r2s, 25)),
                
                # Additional summary stats
                'eval/num_batches': len(eval_losses),
                'eval/num_samples': len(eval_mses)
            }
        
        # Print evaluation summary
        print(f"Evaluation Results ({len(eval_mses)} samples from {len(eval_losses)} valid batches):")
        if len(eval_losses) > 0:
            print(f"   Loss (NLL): {metrics['eval/loss_mean']:.6f} ± {metrics['eval/loss_std']:.6f} (median: {metrics['eval/loss_median']:.6f})")
        if len(eval_mses) > 0:
            print(f"   MSE: {metrics['eval/mse_mean']:.6f} ± {metrics['eval/mse_std']:.6f} (median: {metrics['eval/mse_median']:.6f})")
            print(f"   R²: {metrics['eval/r2_mean']:.6f} ± {metrics['eval/r2_std']:.6f} (median: {metrics['eval/r2_median']:.6f})")
        
        # Update best model if model selection is enabled
        if self.enable_model_selection:
            self._update_best_model(metrics)
        
        return metrics

    def fit(self):
        """Train the SimplePFN model."""
        print(f"Training SimplePFN for {self.max_steps} steps with learning rate {self.learning_rate}")
        print(f"Using device: {self.device}")
        
        # Setup signal handlers for graceful termination
        if self.run_save_dir:
            self.setup_signal_handlers()
            print("Signal handlers set up for graceful termination with model saving")
        
        # Log scheduler info if available
        if self.scheduler is not None:
            sched_config = self.scheduler_config
            warmup_ratio = sched_config.get("warmup_ratio", 0.03)
            min_lr_ratio = sched_config.get("min_lr_ratio", 0.1)
            warmup_steps = int(round(warmup_ratio * self.max_steps))
            
            print(f"Using learning rate scheduler: enabled")
            print(f"   Type: Linear warmup + cosine decay")
            print(f"   Warmup ratio: {warmup_ratio:.2f} ({warmup_steps} steps)")
            print(f"   Min LR ratio: {min_lr_ratio:.6f} (min LR: {self.learning_rate * min_lr_ratio:.6f})")
        else:
            print(f"Using learning rate scheduler: disabled (constant LR)")
            
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
            if step >= self.max_steps or self.terminate:
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
            
            # Compute loss based on available loss function
            if self.bar_distribution is not None:
                # Use BarDistribution probabilistic loss
                # predictions should be (batch_size, n_test, K+4) for BarDistribution
                # y_test should be (batch_size, n_test)
                
                # Ensure predictions have the right shape for BarDistribution
                if len(predictions.shape) == 2:
                    # If predictions are (batch_size, n_test), this is wrong for BarDistribution
                    raise ValueError(f"BarDistribution requires high-dimensional predictions, got shape {predictions.shape}")
                
                # Use negative log probability as loss (since we want to maximize log prob)
                log_prob = self.bar_distribution.average_log_prob(predictions, y_test)  # (batch_size,)
                loss = -log_prob.mean()  # Average across batch and negate for minimization
            else:
                # Use standard MSE loss
                # Ensure predictions have the right shape for loss computation
                # predictions should be (batch_size, n_test) to match y_test (batch_size, n_test)
                if len(predictions.shape) == 3 and predictions.shape[-1] == 1:
                    # If predictions are (batch_size, n_test, 1), squeeze last dimension
                    predictions = predictions.squeeze(-1)
                
                # Compute MSE loss for the full batch
                loss = self.criterion(predictions, y_test)
                
            losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate if scheduler is enabled
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.learning_rate
            
            self.global_step += 1
            
            # End batch timing
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            # Log to Weights & Biases
            if self.wandb_run:
                self.wandb_run.log({
                    'train/loss': loss.item(),
                    'train/step_time': batch_time,
                    'train/batch_size': X_train.shape[0],
                    'train/global_step': self.global_step,
                    'train/learning_rate': current_lr
                }, step=self.global_step)
            
            # Print progress - more frequent at start, less frequent later
            if step <= 10 or step % max(1, self.max_steps // 10) == 0:
                wandb_status = "[wandb]" if self.wandb_run else "[no wandb]"
                lr_info = f"LR: {current_lr:.6f}" if self.scheduler is not None else ""
                print(f"   Step {step:4d}/{self.max_steps} | Loss: {loss.item():.6f} | Time: {batch_time:.3f}s | Batch size: {X_train.shape[0]} {lr_info} {wandb_status}")
            
            # Save model checkpoint if requested
            if self.run_save_dir and self.save_every > 0 and self.global_step % self.save_every == 0:
                self.save_model()
                
            # Run evaluation if requested
            if (self.eval_dataloader is not None and 
                self.eval_every > 0 and 
                self.global_step % self.eval_every == 0):
                
                eval_metrics = self.evaluate()
                
                # Log evaluation metrics to wandb
                if self.wandb_run and eval_metrics:
                    self.wandb_run.log(eval_metrics, step=self.global_step)
            else:
                # If evaluation is disabled but model selection is enabled,
                # update best model based on training loss
                if (self.enable_model_selection and 
                    self.eval_dataloader is None and 
                    self.global_step % max(1, self.max_steps // 20) == 0):  # Check every 5% of training
                    self._update_best_model({}, train_loss=loss.item())
        
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
        
        # Log final summary to Weights & Biases
        if self.wandb_run and len(losses) > 0 and len(batch_times) > 0:
            self.wandb_run.log({
                'summary/total_training_time': total_time,
                'summary/steps_completed': len(batch_times),
                'summary/avg_step_time': sum(batch_times)/len(batch_times),
                'summary/steps_per_second': len(batch_times)/total_time,
                'summary/initial_loss': losses[0],
                'summary/final_loss': losses[-1],
                'summary/loss_reduction_percent': ((losses[0] - losses[-1])/losses[0]*100),
                'summary/average_loss': sum(losses)/len(losses)
            })
            print(f"   Training metrics logged to Weights & Biases!")
        
        # Save final model
        if self.run_save_dir:
            self.save_model(filename="final.pt", 
                           metadata={'final': True})
            print(f"   Final model saved to {self.run_save_dir}")
            
            # Save best model if model selection was enabled
            if self.enable_model_selection and self.best_model_state is not None:
                best_path = self.save_best_model()
                if best_path:
                    print(f"   Best model saved to {best_path}")
                    
                    # Optionally load best model as the final model state
                    # (user may want to keep the last checkpoint instead)
                    print(f"   Note: Final model state represents last checkpoint.")
                    print(f"         Load '{os.path.basename(best_path)}' to use the best model.")
            elif self.enable_model_selection:
                print(f"   Warning: Model selection was enabled but no best model was tracked.")
        
        return self.model
