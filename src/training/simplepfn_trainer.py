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
        eval_dataloader: Optional[DataLoader] = None,  # Evaluation dataloader (legacy single)
        eval_dataloaders: Optional[list] = None,       # New: list of evaluation dataloaders (e.g., head & tail)
        eval_every: int = 0,  # Evaluate every N steps (0 to disable)
        eval_batches: int = 10,  # Number of batches to use for evaluation
        accumulate_grad_batches: int = 1,  # Number of micro-batches to accumulate before optimizer step
        # Model selection parameters
        enable_model_selection: bool = False,  # Whether to enable automatic best model selection
        model_selection_metric: str = "eval/mse_median",  # Metric to use for model selection
        model_selection_mode: str = "min",  # "min" or "max" - whether lower or higher is better
        # Benchmark integration
        config_path: Optional[str] = None,  # Path to YAML config for PFN wrapper
        benchmark_eval_fidelity: Optional[str] = None,  # Run benchmark at each eval with this fidelity (e.g., "minimal", "low", "high", "very high")
        benchmark_final_fidelity: Optional[str] = None,  # Run benchmark at end with this fidelity
        # Mixed precision training
        use_amp: bool = False,  # Enable automatic mixed precision (float16)
        gradient_clip_val: float = 0.0,  # Gradient clipping value (0.0 to disable)
        schedule_name: Optional[str] = None,  # curriculum schedule name for logging
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
        self.gradient_clip_val = gradient_clip_val  # Store gradient clipping value
        self.schedule_name = schedule_name or "none"
        # Gradient accumulation factor (optimizer step every N micro-batches)
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        
        # Evaluation parameters (support multiple eval dataloaders)
        if eval_dataloaders is not None:
            self.eval_dataloaders = eval_dataloaders
        elif eval_dataloader is not None:
            self.eval_dataloaders = [eval_dataloader]
        else:
            self.eval_dataloaders = None
        self.eval_every = eval_every
        self.eval_batches = eval_batches
        # Human-friendly names for eval sets (used in prints and metrics)
        if self.eval_dataloaders is not None:
            if len(self.eval_dataloaders) == 2:
                self._eval_set_names = ["head", "tail"]
            else:
                self._eval_set_names = [f"set{i}" for i in range(len(self.eval_dataloaders))]
        else:
            self._eval_set_names = []
        
        # Model selection parameters
        self.enable_model_selection = enable_model_selection
        self.model_selection_metric = model_selection_metric
        self.model_selection_mode = model_selection_mode
        self.best_metric_value = float('inf') if model_selection_mode == "min" else float('-inf')
        self.best_model_state = None
        self.best_model_step = 0
        self.best_model_metadata = None
        
        # Benchmark integration config
        self.config_path = config_path
        self.benchmark_eval_fidelity = benchmark_eval_fidelity
        self.benchmark_final_fidelity = benchmark_final_fidelity
        # Cached training shapes for aligning benchmark subsampling
        self._train_n_features = None
        self._train_n_train = None
        self._train_n_test = None
        
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
            
        self.global_step = 0  # counts optimizer steps (not micro-batches)
        
        # Setup scheduler if enabled
        self.scheduler = self._create_scheduler(self.scheduler_config)
        
        # Setup mixed precision training
        self.use_amp = use_amp and torch.cuda.is_available()  # Only enable if CUDA available
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print(f"Mixed precision training ENABLED (float16)")
        else:
            self.scaler = None
            if use_amp and not torch.cuda.is_available():
                print(f"Warning: Mixed precision requested but CUDA not available, using float32")
            else:
                print(f"Mixed precision training DISABLED (float32)")
        
        # Initialize metric caches for per-step logging (start with NaN)
        # These will be updated when eval/benchmark runs and repeated every step
        self._latest_eval_metrics = {
            'eval/mse_mean': float('nan'),
            'eval/mse_median': float('nan'),
            'eval/mse_std': float('nan'),
            'eval/r2_mean': float('nan'),
            'eval/r2_median': float('nan'),
            'eval/r2_std': float('nan'),
        }
        self._latest_benchmark_metrics = {
            'benchmark/mse_lr_median': float('nan'),
            'benchmark/mse_rf_median': float('nan'),
            'benchmark/mse_pfn_median': float('nan'),
            'benchmark/r2_lr_median': float('nan'),
            'benchmark/r2_rf_median': float('nan'),
            'benchmark/r2_pfn_median': float('nan'),
        }
        # Seed per-set and overall eval metric placeholders so keys exist from step 1
        if self.eval_dataloaders is not None:
            # Ensure names are defined
            if not hasattr(self, '_eval_set_names') or len(self._eval_set_names) != len(self.eval_dataloaders):
                if len(self.eval_dataloaders) == 2:
                    self._eval_set_names = ["head", "tail"]
                else:
                    self._eval_set_names = [f"set{i}" for i in range(len(self.eval_dataloaders))]
            # Overall placeholders (also mirror legacy top-level)
            for k in [
                'eval/overall/loss_mean','eval/overall/loss_median','eval/overall/loss_std','eval/overall/loss_iqr',
                'eval/overall/mse_mean','eval/overall/mse_median','eval/overall/mse_std','eval/overall/mse_iqr',
                'eval/overall/r2_mean','eval/overall/r2_median','eval/overall/r2_std','eval/overall/r2_iqr',
                'eval/loss_mean','eval/loss_median','eval/loss_std','eval/loss_iqr',
                'eval/mse_mean','eval/mse_median','eval/mse_std',
                'eval/r2_mean','eval/r2_median','eval/r2_std',
                'eval/num_batches','eval/num_samples']:
                if k not in self._latest_eval_metrics:
                    self._latest_eval_metrics[k] = float('nan') if not k.endswith(('num_batches','num_samples')) else 0
            # Per-set placeholders
            per_suffixes = ['loss_mean','loss_median','loss_std','loss_iqr','mse_mean','mse_median','mse_std','mse_iqr','r2_mean','r2_median','r2_std','r2_iqr','num_batches']
            for name in self._eval_set_names:
                for suf in per_suffixes:
                    key = f'eval/{name}/{suf}'
                    if key not in self._latest_eval_metrics:
                        self._latest_eval_metrics[key] = 0 if suf == 'num_batches' else float('nan')
        # Cache for latest observed (t, alpha) from batch for logging
        self._latest_schedule_info = {
            'train/t0': float('nan'),
            'train/alpha0': float('nan'),
        }
            
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
        if isinstance(batch, list) and (len(batch) == 4 or len(batch) == 6):
            X_train, y_train, X_test, y_test = batch[:4]
            # If t, alpha present, record first values for logging
            if len(batch) >= 6:
                t_batch, a_batch = batch[4], batch[5]
                try:
                    t0 = float(t_batch[0].item()) if hasattr(t_batch, 'shape') else float(t_batch)
                    a0 = float(a_batch[0].item()) if hasattr(a_batch, 'shape') else float(a_batch)
                except Exception:
                    # best effort fallback
                    try:
                        t0 = float(t_batch)
                        a0 = float(a_batch)
                    except Exception:
                        t0, a0 = float('nan'), float('nan')
                self._latest_schedule_info['train/t0'] = t0
                self._latest_schedule_info['train/alpha0'] = a0
            
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
        if self.eval_dataloaders is None:
            print("Warning: No evaluation dataloaders provided, skipping evaluation")
            return {}
            
        num_sets = len(self.eval_dataloaders)
        print(f"Running evaluation on {self.eval_batches} batches per eval set (num_sets={num_sets})...")

        # Reuse stored names to ensure consistency across calls
        if hasattr(self, '_eval_set_names') and len(self._eval_set_names) == num_sets:
            set_names = self._eval_set_names
        else:
            if num_sets == 2:
                set_names = ["head", "tail"]
            else:
                set_names = [f"set{i}" for i in range(num_sets)]

        self.model.eval()

        # Accumulate overall metrics across all sets
        overall_losses = []
        overall_mses = []
        overall_r2s = []

        metrics: Dict[str, float] = {}

        with torch.no_grad():
            for loader_idx, _loader in enumerate(self.eval_dataloaders):
                set_name = set_names[loader_idx]
                set_losses = []
                set_mses = []
                set_r2s = []

                for batch_idx, batch in enumerate(_loader):
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
                        print(f"   [{set_name}] Skipping batch {batch_idx}: loss contains NaN/inf (loss={loss_value})")
                        continue

                    # Check if predictions contain NaN or infinite values
                    if not np.all(np.isfinite(pred_np)):
                        nan_count = np.sum(~np.isfinite(pred_np))
                        print(f"   [{set_name}] Skipping batch {batch_idx}: predictions contain {nan_count} NaN/inf values")
                        continue

                    # Check if targets contain NaN or infinite values
                    if not np.all(np.isfinite(y_test_np)):
                        nan_count = np.sum(~np.isfinite(y_test_np))
                        print(f"   [{set_name}] Skipping batch {batch_idx}: targets contain {nan_count} NaN/inf values")
                        continue

                    # If we get here, the batch is valid
                    set_losses.append(loss_value)

                    # Compute metrics for each sample in the batch
                    for i in range(pred_np.shape[0]):
                        # Double-check individual samples for safety
                        if not (np.all(np.isfinite(pred_np[i])) and np.all(np.isfinite(y_test_np[i]))):
                            continue  # Skip this sample if it has NaN/inf

                        # MSE for this sample
                        try:
                            mse = mean_squared_error(y_test_np[i], pred_np[i])
                            if np.isfinite(mse):
                                set_mses.append(mse)
                        except (ValueError, RuntimeWarning):
                            continue  # Skip if MSE computation fails

                        # R² for this sample (handle case where y has no variance)
                        try:
                            r2 = r2_score(y_test_np[i], pred_np[i])
                            if np.isfinite(r2):
                                set_r2s.append(r2)
                        except (ValueError, RuntimeWarning):
                            # R² is undefined when y has no variance or other issues
                            continue

                # Aggregate per-set metrics
                set_losses_np = np.array(set_losses)
                set_mses_np = np.array(set_mses)
                set_r2s_np = np.array(set_r2s)

                # Contribute to overall aggregates
                overall_losses.extend(list(set_losses_np))
                overall_mses.extend(list(set_mses_np))
                overall_r2s.extend(list(set_r2s_np))

                # Compute per-set stats (prefix with eval/{set_name}/...)
                if len(set_losses_np) == 0 and len(set_mses_np) == 0:
                    print(f"Warning: No valid evaluation data collected for set '{set_name}'")
                else:
                    if len(set_losses_np) > 0:
                        metrics[f'eval/{set_name}/loss_mean'] = float(np.mean(set_losses_np))
                        metrics[f'eval/{set_name}/loss_median'] = float(np.median(set_losses_np))
                        metrics[f'eval/{set_name}/loss_std'] = float(np.std(set_losses_np))
                        metrics[f'eval/{set_name}/loss_iqr'] = float(np.percentile(set_losses_np, 75) - np.percentile(set_losses_np, 25))
                        metrics[f'eval/{set_name}/num_batches'] = int(len(set_losses_np))
                    else:
                        metrics[f'eval/{set_name}/num_batches'] = 0

                    if len(set_mses_np) > 0:
                        metrics[f'eval/{set_name}/mse_mean'] = float(np.mean(set_mses_np))
                        metrics[f'eval/{set_name}/mse_median'] = float(np.median(set_mses_np))
                        metrics[f'eval/{set_name}/mse_std'] = float(np.std(set_mses_np))
                        metrics[f'eval/{set_name}/mse_iqr'] = float(np.percentile(set_mses_np, 75) - np.percentile(set_mses_np, 25))
                    if len(set_r2s_np) > 0:
                        metrics[f'eval/{set_name}/r2_mean'] = float(np.mean(set_r2s_np))
                        metrics[f'eval/{set_name}/r2_median'] = float(np.median(set_r2s_np))
                        metrics[f'eval/{set_name}/r2_std'] = float(np.std(set_r2s_np))
                        metrics[f'eval/{set_name}/r2_iqr'] = float(np.percentile(set_r2s_np, 75) - np.percentile(set_r2s_np, 25))

        self.model.train()  # Return to training mode

        # Overall aggregation across all sets
        overall_losses_np = np.array(overall_losses)
        overall_mses_np = np.array(overall_mses)
        overall_r2s_np = np.array(overall_r2s)

        if len(overall_losses_np) == 0:
            print("Warning: No valid evaluation data collected (all sets contained NaN/inf)")
            return {}

        # Overall loss metrics
        metrics['eval/overall/loss_mean'] = float(np.mean(overall_losses_np))
        metrics['eval/overall/loss_median'] = float(np.median(overall_losses_np))
        metrics['eval/overall/loss_std'] = float(np.std(overall_losses_np))
        metrics['eval/overall/loss_iqr'] = float(np.percentile(overall_losses_np, 75) - np.percentile(overall_losses_np, 25))
        metrics['eval/overall/num_batches'] = int(len(overall_losses_np))

        # Overall MSE / R2 metrics (when available)
        if len(overall_mses_np) > 0:
            metrics['eval/overall/mse_mean'] = float(np.mean(overall_mses_np))
            metrics['eval/overall/mse_median'] = float(np.median(overall_mses_np))
            metrics['eval/overall/mse_std'] = float(np.std(overall_mses_np))
            metrics['eval/overall/mse_iqr'] = float(np.percentile(overall_mses_np, 75) - np.percentile(overall_mses_np, 25))
        if len(overall_r2s_np) > 0:
            metrics['eval/overall/r2_mean'] = float(np.mean(overall_r2s_np))
            metrics['eval/overall/r2_median'] = float(np.median(overall_r2s_np))
            metrics['eval/overall/r2_std'] = float(np.std(overall_r2s_np))
            metrics['eval/overall/r2_iqr'] = float(np.percentile(overall_r2s_np, 75) - np.percentile(overall_r2s_np, 25))

        # Backward-compatible top-level keys (mirror overall)
        # These preserve existing dashboards expecting 'eval/mse_median', etc.
        if 'eval/overall/mse_mean' in metrics:
            metrics['eval/mse_mean'] = metrics['eval/overall/mse_mean']
            metrics['eval/mse_median'] = metrics['eval/overall/mse_median']
            metrics['eval/mse_std'] = metrics['eval/overall/mse_std']
        if 'eval/overall/r2_mean' in metrics:
            metrics['eval/r2_mean'] = metrics['eval/overall/r2_mean']
            metrics['eval/r2_median'] = metrics['eval/overall/r2_median']
            metrics['eval/r2_std'] = metrics['eval/overall/r2_std']
        metrics['eval/loss_mean'] = metrics.get('eval/overall/loss_mean', float('nan'))
        metrics['eval/loss_median'] = metrics.get('eval/overall/loss_median', float('nan'))
        metrics['eval/loss_std'] = metrics.get('eval/overall/loss_std', float('nan'))
        metrics['eval/loss_iqr'] = metrics.get('eval/overall/loss_iqr', float('nan'))
        metrics['eval/num_batches'] = metrics.get('eval/overall/num_batches', 0)
        metrics['eval/num_samples'] = int(len(overall_mses_np)) if len(overall_mses_np) > 0 else 0

        # Print evaluation summary
        print("Evaluation Results (per set):")
        for name in set_names:
            mkeys = [
                (f'eval/{name}/loss_mean', f'eval/{name}/loss_std', f'eval/{name}/loss_median'),
                (f'eval/{name}/mse_mean', f'eval/{name}/mse_std', f'eval/{name}/mse_median'),
                (f'eval/{name}/r2_mean', f'eval/{name}/r2_std', f'eval/{name}/r2_median'),
            ]
            if any(k in metrics for ks in mkeys for k in ks):
                print(f"   [{name}] batches: {metrics.get(f'eval/{name}/num_batches', 0)}")
                if f'eval/{name}/loss_mean' in metrics:
                    print(f"      Loss (NLL): {metrics[f'eval/{name}/loss_mean']:.6f} ± {metrics.get(f'eval/{name}/loss_std', float('nan')):.6f} (median: {metrics.get(f'eval/{name}/loss_median', float('nan')):.6f})")
                if f'eval/{name}/mse_mean' in metrics:
                    print(f"      MSE: {metrics[f'eval/{name}/mse_mean']:.6f} ± {metrics.get(f'eval/{name}/mse_std', float('nan')):.6f} (median: {metrics.get(f'eval/{name}/mse_median', float('nan')):.6f})")
                if f'eval/{name}/r2_mean' in metrics:
                    print(f"      R²: {metrics[f'eval/{name}/r2_mean']:.6f} ± {metrics.get(f'eval/{name}/r2_std', float('nan')):.6f} (median: {metrics.get(f'eval/{name}/r2_median', float('nan')):.6f})")

        if len(overall_losses_np) > 0:
            print(f"Overall: Loss (NLL): {metrics['eval/overall/loss_mean']:.6f} ± {metrics['eval/overall/loss_std']:.6f} (median: {metrics['eval/overall/loss_median']:.6f})")
        if len(overall_mses_np) > 0:
            print(f"Overall: MSE: {metrics['eval/overall/mse_mean']:.6f} ± {metrics['eval/overall/mse_std']:.6f} (median: {metrics['eval/overall/mse_median']:.6f})")
        if len(overall_r2s_np) > 0:
            print(f"Overall: R²: {metrics['eval/overall/r2_mean']:.6f} ± {metrics['eval/overall/r2_std']:.6f} (median: {metrics['eval/overall/r2_median']:.6f})")

        # Update best model if model selection is enabled (use overall metrics)
        if self.enable_model_selection:
            self._update_best_model(metrics)

        # Update cached eval metrics for per-step logging (include per-set & overall keys)
        # Clear only keys related to eval to avoid stale per-set metrics from different counts
        for k in list(self._latest_eval_metrics.keys()):
            if k.startswith('eval/'):
                self._latest_eval_metrics[k] = float('nan')
        # Overall
        for k in [
            'eval/overall/mse_mean','eval/overall/mse_median','eval/overall/mse_std',
            'eval/overall/r2_mean','eval/overall/r2_median','eval/overall/r2_std',
            'eval/mse_mean','eval/mse_median','eval/mse_std',
            'eval/r2_mean','eval/r2_median','eval/r2_std']:
            if k in metrics:
                self._latest_eval_metrics[k] = metrics[k]
        # Per sets (mse/r2 + loss means)
        for name in set_names:
            for suffix in ['loss_mean','loss_median','loss_std','mse_mean','mse_median','mse_std','r2_mean','r2_median','r2_std']:
                key = f'eval/{name}/{suffix}'
                if key in metrics:
                    self._latest_eval_metrics[key] = metrics[key]

        return metrics

    def fit(self):
        """Train the SimplePFN model."""
        print(f"Training SimplePFN for {self.max_steps} steps with learning rate {self.learning_rate}")
        print(f"Using device: {self.device}")
        if self.accumulate_grad_batches > 1:
            print(f"Gradient accumulation ENABLED: accumulate_grad_batches={self.accumulate_grad_batches} (effective batch size = batch_size * {self.accumulate_grad_batches})")
        else:
            print(f"Gradient accumulation DISABLED: accumulate_grad_batches=1")
        
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
            
        # Print evaluation loaders summary if available
        if self.eval_dataloaders is not None and len(self.eval_dataloaders) > 0:
            print("\nEVALUATION LOADERS:")
            print(f"   Number of eval sets: {len(self.eval_dataloaders)}")
            for i, loader in enumerate(self.eval_dataloaders):
                name = self._eval_set_names[i] if i < len(self._eval_set_names) else f"set{i}"
                try:
                    ds_len = len(getattr(loader, 'dataset', []))
                except Exception:
                    ds_len = -1
                try:
                    num_batches = len(loader)
                except Exception:
                    num_batches = -1
                bs = getattr(loader, 'batch_size', None)
                print(f"   - {name}: dataset_size={ds_len}, batch_size={bs}, total_batches={num_batches}, using_up_to_batches={self.eval_batches}")

        # Start total timing
        total_start_time = time.time()
        batch_times = []  # per optimizer step wall time
        losses = []       # per optimizer step loss (averaged over micro-batches)
        
        self.model.train()
        
        # Print initial batch info
        sample_batch = next(iter(self.dataloader))
        if isinstance(sample_batch, list) and (len(sample_batch) == 4 or len(sample_batch) == 6):
            batch_size = sample_batch[0].shape[0]
            n_train = sample_batch[0].shape[1]
            n_test = sample_batch[2].shape[1]
            features = sample_batch[0].shape[2]
            print(f"Batch structure: {batch_size} samples, {n_train} train points, {n_test} test points, {features} features")
            # If t/alpha present, print the first values seen
            if len(sample_batch) >= 6:
                try:
                    t0 = float(sample_batch[4][0].item())
                    a0 = float(sample_batch[5][0].item())
                    print(f"First batch schedule: t={t0:.4f}, alpha={a0:.4f}, schedule={self.schedule_name}")
                except Exception:
                    pass
            # Cache shapes for benchmark alignment
            self._train_n_features = int(features)
            self._train_n_train = int(n_train)
            self._train_n_test = int(n_test)
        
        micro_count = 0
        opt_step_time_acc = 0.0
        loss_accumulator = 0.0
        current_lr = self.learning_rate
        
        for micro_idx, batch in enumerate(self.dataloader):
            if self.terminate or self.global_step >= self.max_steps:
                break

            # Start micro-batch timing
            micro_start = time.time()

            # Zero grads at the start of an accumulation window
            if micro_count % self.accumulate_grad_batches == 0:
                self.optimizer.zero_grad()

            X_train, y_train, X_test, y_test = self._process_batch(batch)

            # Forward + loss
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.float16):
                output = self.model(X_train, y_train, X_test)
                if isinstance(output, dict) and 'predictions' in output:
                    predictions = output['predictions']
                else:
                    predictions = output

                if self.bar_distribution is not None:
                    if len(predictions.shape) == 2:
                        raise ValueError(f"BarDistribution requires high-dimensional predictions, got shape {predictions.shape}")
                    log_prob = self.bar_distribution.average_log_prob(predictions, y_test)
                    loss = -log_prob.mean()
                else:
                    if len(predictions.shape) == 3 and predictions.shape[-1] == 1:
                        predictions = predictions.squeeze(-1)
                    loss = self.criterion(predictions, y_test)

                # Scale loss by accumulation factor so gradients average across micro-batches
                loss_to_backward = loss / float(self.accumulate_grad_batches)

            # Backward
            if self.use_amp:
                self.scaler.scale(loss_to_backward).backward()
            else:
                loss_to_backward.backward()

            # Accumulate timing and loss for logging
            opt_step_time_acc += (time.time() - micro_start)
            loss_accumulator += loss.item()
            micro_count += 1

            # Perform optimizer step at the end of the accumulation window or if this will reach max steps
            should_step = (micro_count % self.accumulate_grad_batches == 0)
            if should_step:
                # Gradient clipping and optimizer step
                if self.use_amp:
                    if self.gradient_clip_val > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    self.optimizer.step()

                # Scheduler step after optimizer step
                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                else:
                    current_lr = self.learning_rate

                # Update optimizer step counter (global_step) and logs
                self.global_step += 1

                # Record per-optimizer-step time and averaged loss
                batch_times.append(opt_step_time_acc)
                avg_loss = loss_accumulator / float(self.accumulate_grad_batches)
                losses.append(avg_loss)

                # Reset accumulators for next window
                opt_step_time_acc = 0.0
                loss_accumulator = 0.0

                # Log to Weights & Biases once per optimizer step
                if self.wandb_run:
                    log_dict = {
                        'train/loss': avg_loss,
                        'train/step_time': batch_times[-1],
                        'train/batch_size': X_train.shape[0],
                        'train/global_step': self.global_step,
                        'train/learning_rate': current_lr,
                    }
                    log_dict['train/schedule'] = self.schedule_name
                    log_dict.update(self._latest_schedule_info)
                    log_dict.update(self._latest_eval_metrics)
                    log_dict.update(self._latest_benchmark_metrics)
                    self.wandb_run.log(log_dict, step=self.global_step)

                # Print progress - more frequent at start, less frequent later
                if self.global_step <= 10 or self.global_step % max(1, self.max_steps // 10) == 0:
                    wandb_status = "[wandb]" if self.wandb_run else "[no wandb]"
                    lr_info = f"LR: {current_lr:.6f}" if self.scheduler is not None else ""
                    eff_bs = X_train.shape[0] * self.accumulate_grad_batches
                    print(f"   Step {self.global_step:4d}/{self.max_steps} | Loss: {avg_loss:.6f} | Time: {batch_times[-1]:.3f}s | Eff. batch size: {eff_bs} {lr_info} {wandb_status}")

                # Save model checkpoint if requested
                if self.run_save_dir and self.save_every > 0 and self.global_step % self.save_every == 0:
                    self.save_model()

                # Run evaluation if requested (triggered on optimizer steps)
                if (self.eval_dataloaders is not None and 
                    self.eval_every > 0 and 
                    self.global_step % self.eval_every == 0):
                    eval_metrics = self.evaluate()
                    if not math.isnan(self._latest_schedule_info['train/t0']) and not math.isnan(self._latest_schedule_info['train/alpha0']):
                        t0 = self._latest_schedule_info['train/t0']
                        a0 = self._latest_schedule_info['train/alpha0']
                        print(f"[Eval] First-in-batch schedule: t={t0:.4f}, alpha={a0:.4f}, schedule={self.schedule_name}")
                    if self.wandb_run and eval_metrics:
                        self.wandb_run.log(eval_metrics, step=self.global_step)
                    if self.benchmark_eval_fidelity:
                        try:
                            self._run_benchmark_with_current_model(
                                fidelity=self.benchmark_eval_fidelity,
                                tag=f"eval_step{self.global_step}"
                            )
                        except Exception as e:
                            print(f"[Trainer] Benchmark at eval step {self.global_step} failed: {e}")
                else:
                    if (self.enable_model_selection and 
                        self.eval_dataloaders is None and 
                        self.global_step % max(1, self.max_steps // 20) == 0):
                        self._update_best_model({}, train_loss=avg_loss)
            
            else:
                # If not an optimizer step: optionally update best model when no eval loaders are used
                if (self.enable_model_selection and 
                    self.eval_dataloaders is None and 
                    self.global_step % max(1, self.max_steps // 20) == 0):
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
            # Run final benchmark if requested
            if self.benchmark_final_fidelity:
                # Use the just-saved final checkpoint
                final_ckpt = os.path.join(self.run_save_dir, "final.pt")
                try:
                    self._run_benchmark_with_checkpoint(
                        fidelity=self.benchmark_final_fidelity,
                        tag="final",
                        checkpoint_path=final_ckpt,
                    )
                except Exception as e:
                    print(f"[Trainer] Final benchmark failed: {e}")
            
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

    def _run_benchmark_with_current_model(self, fidelity: str, tag: str) -> None:
        """Save a temporary checkpoint of the current model and run the OpenML benchmark with the given fidelity.

        Args:
            fidelity: One of {"minimal", "low", "high", "very high"}
            tag: Short label used in output filenames (e.g., "eval_step1000")
        """
        # Ensure we have a place to save
        if not self.run_save_dir:
            import tempfile
            self.run_save_dir = os.path.join(tempfile.gettempdir(), f"simplepfn_checkpoints_{self.run_name}")
            os.makedirs(self.run_save_dir, exist_ok=True)

        # Save a checkpoint for the benchmark to load
        ckpt_name = f"benchmark_{tag}.pt"
        ckpt_path = self.save_model(filename=ckpt_name, metadata={"stage": tag, "benchmark": True})
        if not ckpt_path:
            raise RuntimeError("Failed to save checkpoint for benchmark.")

        self._run_benchmark_with_checkpoint(fidelity=fidelity, tag=tag, checkpoint_path=ckpt_path)

    def _run_benchmark_with_checkpoint(self, fidelity: str, tag: str, checkpoint_path: str) -> None:
        """Run the OpenML benchmark with a specified checkpoint."""
        # Ensure import paths are robust regardless of invocation context
        try:
            import sys as _sys
            from pathlib import Path as _Path
            _repo_root = _Path(__file__).resolve().parents[2]
            _src_dir = _repo_root / "src"
            if str(_repo_root) not in _sys.path:
                _sys.path.insert(0, str(_repo_root))
            if str(_src_dir) not in _sys.path:
                _sys.path.insert(0, str(_src_dir))
        except Exception:
            pass

        # Robust import for Benchmark
        try:
            from src.benchmarking.Benchmark import Benchmark as _Benchmark
        except Exception:
            from benchmarking.Benchmark import Benchmark as _Benchmark

        bench = _Benchmark(data_dir="data_cache", device=self.device, verbose=True)

        # Determine subsampling to mirror training dimensions if available
        n_feat = self._train_n_features or 0
        n_tr = self._train_n_train or 0
        n_te = self._train_n_test or 0

        # Output file inside the run directory
        out_csv = os.path.join(self.run_save_dir or ".", f"benchmark_{tag}.csv")

        # Use the training config path if known for model construction
        cfg_path = self.config_path

        print(f"[Trainer] Running benchmark ({fidelity}) -> {out_csv}")
        df = bench.run_simplified(
            fidelity=fidelity,
            # subsampling to align with training
            n_features=int(n_feat),
            max_n_features=int(n_feat) if n_feat else 0,
            n_train=int(n_tr),
            max_n_train=int(n_tr) if n_tr else 0,
            n_test=int(n_te),
            max_n_test=int(n_te) if n_te else 0,
            config_path=cfg_path,
            checkpoint_path=checkpoint_path,
            output_csv=out_csv,
            device=self.device,
            quiet=False,
        )
        # Print summary metrics similar to run_benchmark.py
        try:
            df_metrics = df[df["process_id"] != "summary"] if "process_id" in df.columns else df
            mse_cols = [c for c in ["mse_lr", "mse_rf", "mse_pfn"] if c in df_metrics.columns]
            r2_cols = [c for c in ["r2_lr", "r2_rf", "r2_pfn"] if c in df_metrics.columns]
            if not df_metrics.empty and mse_cols and r2_cols:
                median_mse = df_metrics[mse_cols].median(numeric_only=True)
                mean_mse = df_metrics[mse_cols].mean(numeric_only=True)
                median_r2 = df_metrics[r2_cols].median(numeric_only=True)
                mean_r2 = df_metrics[r2_cols].mean(numeric_only=True)
                print("\nMedian MSE:\n", median_mse)
                print("\nMean MSE:\n", mean_mse)
                print("\nMedian R\u00b2:\n", median_r2)
                print("\nMean R\u00b2:\n", mean_r2)
            else:
                print("\nNo per-task metric rows available to summarize (0 successful tasks or missing metric columns).")

            if "process_id" in df.columns and (df["process_id"] == "summary_model").any():
                df_sm = df[df["process_id"] == "summary_model"]
                print("\nDetailed per-model summaries (95% CIs, IQR, std, avg ranks):")
                for _, row in df_sm.sort_values(["model", "metric"]).iterrows():
                    model = row.get("model", "?")
                    metric = row.get("metric", "?")
                    mean = row.get("mean", np.nan)
                    median = row.get("median", np.nan)
                    std = row.get("std", np.nan)
                    iqr = row.get("iqr", np.nan)
                    ci_mean_low = row.get("ci95_mean_low", np.nan)
                    ci_mean_high = row.get("ci95_mean_high", np.nan)
                    ci_med_low = row.get("ci95_median_low", np.nan)
                    ci_med_high = row.get("ci95_median_high", np.nan)
                    avg_rank_mse = row.get("avg_rank_mse", np.nan)
                    avg_rank_r2 = row.get("avg_rank_r2", np.nan)
                    print(
                        f"  - {model} [{metric}] | mean={mean:.4f} (95% CI [{ci_mean_low:.4f}, {ci_mean_high:.4f}]), "
                        f"median={median:.4f} (95% CI [{ci_med_low:.4f}, {ci_med_high:.4f}]), std={std:.4f}, iqr={iqr:.4f}, "
                        f"avg_rank_mse={avg_rank_mse:.2f}, avg_rank_r2={avg_rank_r2:.2f}"
                    )
            else:
                print("\nNo detailed model summaries available (no 'summary_model' rows).")
        except Exception as _e:
            print(f"[Trainer] Failed to print benchmark summary: {_e}")
        # Optionally log a quick summary
        if hasattr(self, 'wandb_run') and self.wandb_run is not None:
            try:
                # Log median metrics if present
                df_metrics = df[df["process_id"] != "summary"] if "process_id" in df.columns else df
                log = {}
                
                # Extract median values for caching
                mse_lr_median = float('nan')
                mse_rf_median = float('nan')
                mse_pfn_median = float('nan')
                r2_lr_median = float('nan')
                r2_rf_median = float('nan')
                r2_pfn_median = float('nan')
                
                for k in ["mse_lr", "mse_rf", "mse_pfn"]:
                    if k in df_metrics.columns:
                        median_val = float(df_metrics[k].median(skipna=True))
                        log[f"benchmark/{tag}/{k}_median"] = median_val
                        # Store in variables for cache update
                        if k == "mse_lr":
                            mse_lr_median = median_val
                        elif k == "mse_rf":
                            mse_rf_median = median_val
                        elif k == "mse_pfn":
                            mse_pfn_median = median_val
                            
                for k in ["r2_lr", "r2_rf", "r2_pfn"]:
                    if k in df_metrics.columns:
                        median_val = float(df_metrics[k].median(skipna=True))
                        log[f"benchmark/{tag}/{k}_median"] = median_val
                        # Store in variables for cache update
                        if k == "r2_lr":
                            r2_lr_median = median_val
                        elif k == "r2_rf":
                            r2_rf_median = median_val
                        elif k == "r2_pfn":
                            r2_pfn_median = median_val
                
                # Update cached benchmark metrics for per-step logging
                self._latest_benchmark_metrics['benchmark/mse_lr_median'] = mse_lr_median
                self._latest_benchmark_metrics['benchmark/mse_rf_median'] = mse_rf_median
                self._latest_benchmark_metrics['benchmark/mse_pfn_median'] = mse_pfn_median
                self._latest_benchmark_metrics['benchmark/r2_lr_median'] = r2_lr_median
                self._latest_benchmark_metrics['benchmark/r2_rf_median'] = r2_rf_median
                self._latest_benchmark_metrics['benchmark/r2_pfn_median'] = r2_pfn_median
                
                if log:
                    self.wandb_run.log(log, step=self.global_step)
            except Exception:
                pass
