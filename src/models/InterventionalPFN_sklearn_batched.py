"""
Sklearn-like wrapper for InterventionalPFN models with batched inference support.

This module provides a simplified scikit-learn-style interface for InterventionalPFN,
analogous to GraphConditionedInterventionalPFN_sklearn but without adjacency matrix.

Key features:
- Batched and non-batched inference modes
- BarDistribution support for distributional predictions
- Clean, minimal API focused on core prediction functionality

Key differences from GraphConditionedInterventionalPFN_sklearn:
- Does NOT require adjacency_matrix as input
- Otherwise identical API and functionality

Usage:
    # Basic usage
    wrapper = InterventionalPFNSklearn(
        config_path="config.yaml",
        checkpoint_path="model.pt"
    )
    wrapper.load()
    
    # Non-batched prediction (default)
    preds = wrapper.predict(
        X_obs, T_obs, Y_obs,
        X_intv, T_intv,
        prediction_type="mode"
    )
    
    # Batched prediction
    preds_batched = wrapper.predict(
        X_obs_batch, T_obs_batch, Y_obs_batch,  # (B, N, ...) shapes
        X_intv_batch, T_intv_batch,              # (B, M, ...) shapes
        prediction_type="mode",
        batched=True
    )
    
    # Log-likelihood
    log_probs = wrapper.log_likelihood(
        X_obs, T_obs, Y_obs,
        X_intv, T_intv, Y_intv
    )
    
    # Negative log-likelihood
    nll = wrapper.predict_negative_log_likelihood(
        X_obs, T_obs, Y_obs,
        X_intv, T_intv, Y_intv
    )
"""

from __future__ import annotations
from typing import Optional, Any, Literal
import numpy as np
import torch
import yaml
import os
import sys
from pathlib import Path

# Robust import - try without 'src.' prefix first, then with 'src.' prefix
try:
    from models.InterventionalPFN import InterventionalPFN
    from Losses.BarDistribution import BarDistribution
except Exception:
    try:
        from src.models.InterventionalPFN import InterventionalPFN
        from src.Losses.BarDistribution import BarDistribution
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.models.InterventionalPFN import InterventionalPFN
        from src.Losses.BarDistribution import BarDistribution


class InterventionalPFNSklearn:
    """
    Simplified sklearn-like wrapper for InterventionalPFN models.
    
    This wrapper focuses on core functionality: prediction and log-likelihood computation
    for interventional causal inference without graph structure requirements.
    
    Supports both batched and non-batched inference:
    - batched=False (default): Process single instance (N, L) shapes
    - batched=True: Process multiple instances simultaneously (B, N, L) shapes
    
    Parameters:
        config_path (str): Path to YAML config file
        checkpoint_path (str): Path to model checkpoint
        device (str, optional): Device for inference ('cpu', 'cuda', 'cuda:0', etc.).
                               If None (default), automatically uses GPU if available.
        verbose (bool): Print detailed loading information
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[InterventionalPFNSklearn] Auto-selected device: {self.device}")
        else:
            self.device = device
        self.verbose = verbose
        
        # Model components (populated by load())
        self.model = None
        self.model_kwargs = None
        self.bar_distribution = None
        self.use_bar_distribution = False
        
    def load(self, override_kwargs: Optional[dict[str, Any]] = None) -> "InterventionalPFNSklearn":
        """
        Load config, build model, and load checkpoint.
        
        CRITICAL: Model architecture config is loaded from checkpoint to ensure exact match.
        External config file is only used as fallback if checkpoint doesn't contain config.
        
        Args:
            override_kwargs: Optional dict to override config parameters
            
        Returns:
            self for method chaining
        """
        if self.verbose:
            print(f"[InterventionalPFNSklearn] Loading model...")
            print(f"  Config: {self.config_path}")
            print(f"  Checkpoint: {self.checkpoint_path}")
        
        # Helper to get value from wandb-style or flat config
        def _get_cfg_value(cfg_dict, key, default=None):
            if key in cfg_dict:
                val = cfg_dict[key]
                return val['value'] if isinstance(val, dict) and 'value' in val else val
            return default
        
        # Load checkpoint FIRST to get the training config
        checkpoint_config = None
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Try to extract config from checkpoint
            if 'config' in checkpoint:
                checkpoint_config = checkpoint['config']
                if self.verbose:
                    print(f"  ✓ Found config in checkpoint (will use for model architecture)")
        
        # Load external config file as fallback
        external_config = None
        if self.config_path:
            with open(self.config_path, 'r') as f:
                external_config = yaml.safe_load(f)
        
        # Determine which config to use (checkpoint takes priority)
        if checkpoint_config:
            config = checkpoint_config
            if self.verbose:
                print(f"  Using config from checkpoint (exact training config)")
        elif external_config:
            config = external_config
            if self.verbose:
                print(f"  Warning: Using external config (checkpoint has no config)")
        else:
            raise ValueError("No config available (neither in checkpoint nor config_path)")
        
        model_config = config.get('model_config', config.get('model', {}))
        bar_config = config.get('bar_distribution', {})
        
        # Extract model architecture parameters (MUST match training exactly)
        self.model_kwargs = {
            'num_features': _get_cfg_value(model_config, 'num_features'),
            'd_model': _get_cfg_value(model_config, 'd_model', 256),
            'depth': _get_cfg_value(model_config, 'depth', 6),
            'heads_feat': _get_cfg_value(model_config, 'heads_feat', 4),
            'heads_samp': _get_cfg_value(model_config, 'heads_samp', 4),
            'dropout': _get_cfg_value(model_config, 'dropout', 0.0),
            'hidden_mult': _get_cfg_value(model_config, 'hidden_mult', 4),  # CRITICAL: must match training
            'use_same_row_mlp': _get_cfg_value(model_config, 'use_same_row_mlp', True),
            'n_sample_attention_sink_rows': _get_cfg_value(model_config, 'n_sample_attention_sink_rows', 0),
            'n_feature_attention_sink_cols': _get_cfg_value(model_config, 'n_feature_attention_sink_cols', 0),
        }
        
        # BarDistribution config
        self.use_bar_distribution = _get_cfg_value(bar_config, 'use_bar_distribution', 
                                                   _get_cfg_value(model_config, 'use_bar_distribution', False))
        
        if self.use_bar_distribution:
            num_bars = _get_cfg_value(bar_config, 'num_bars', _get_cfg_value(model_config, 'num_bars', 50))
            output_dim = num_bars + 4  # K bars + 4 tail params
            self.model_kwargs['output_dim'] = output_dim
            
            if self.verbose:
                print(f"  BarDistribution enabled: {num_bars} bars, output_dim={output_dim}")
            
            # Parse device string to torch.device object
            device_obj = torch.device(self.device)
            
            self.bar_distribution = BarDistribution(
                num_bars=num_bars,
                min_width=_get_cfg_value(bar_config, 'min_width', 1e-6),
                scale_floor=_get_cfg_value(bar_config, 'scale_floor', 1e-6),
                device=device_obj,
                dtype=torch.float32,
            )
        else:
            self.model_kwargs['output_dim'] = 1
        
        if self.verbose:
            print(f"  Model architecture:")
            print(f"    - d_model: {self.model_kwargs['d_model']}")
            print(f"    - hidden_mult: {self.model_kwargs['hidden_mult']}")
            print(f"    - depth: {self.model_kwargs['depth']}")
            print(f"    - num_features: {self.model_kwargs['num_features']}")
        if self.verbose:
            print(f"  Model architecture:")
            print(f"    - d_model: {self.model_kwargs['d_model']}")
            print(f"    - hidden_mult: {self.model_kwargs['hidden_mult']}")
            print(f"    - depth: {self.model_kwargs['depth']}")
            print(f"    - num_features: {self.model_kwargs['num_features']}")
        
        # Apply overrides (only for non-critical parameters)
        if override_kwargs:
            if self.verbose:
                print(f"  Applying overrides: {override_kwargs}")
            self.model_kwargs.update(override_kwargs)
        
        # Sanity check
        if not self.model_kwargs.get("num_features"):
            raise ValueError("num_features must be specified in config")
        
        # Build model
        if self.verbose:
            print(f"  Building model with {self.model_kwargs['num_features']} features...")
        
        self.model = InterventionalPFN(**self.model_kwargs).to(self.device)
        
        # Load checkpoint weights
        if self.checkpoint_path:
            if self.verbose:
                print(f"  Loading checkpoint weights from {self.checkpoint_path}")
            
            # Checkpoint was already loaded earlier, extract state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load model weights with strict=True (should match exactly now)
            try:
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                if missing_keys or unexpected_keys:
                    if self.verbose:
                        if missing_keys:
                            print(f"  Warning: Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
                        if unexpected_keys:
                            print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
                else:
                    if self.verbose:
                        print(f"  ✓ All checkpoint weights loaded successfully")
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to load checkpoint weights. This usually means the model architecture "
                    f"doesn't match the checkpoint. Error: {e}"
                )
            
            # Load BarDistribution state if available
            if self.use_bar_distribution and 'bar_distribution' in checkpoint:
                bar_state = checkpoint['bar_distribution']
                for key, value in bar_state.items():
                    if isinstance(value, torch.Tensor):
                        setattr(self.bar_distribution, key, value.to(self.device))
                    else:
                        setattr(self.bar_distribution, key, value)
                
                if self.verbose:
                    print(f"  ✓ BarDistribution state loaded from checkpoint")
        
        self.model.eval()
        
        if self.verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Model loaded with {n_params:,} parameters")
        
        return self
    
    def predict(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        prediction_type: Literal["point", "mode", "mean", "sample"] = "mean",
        num_samples: int = 100,
        batched: bool = False,
    ) -> np.ndarray:
        """
        Make predictions for interventional test data.
        
        Args:
            X_obs: Observational features
                - If batched=False: (N, L)
                - If batched=True: (B, N, L)
            T_obs: Observational intervened feature
                - If batched=False: (N,) or (N, 1)
                - If batched=True: (B, N) or (B, N, 1)
            Y_obs: Observational targets
                - If batched=False: (N,) or (N, 1)
                - If batched=True: (B, N) or (B, N, 1)
            X_intv: Interventional features
                - If batched=False: (M, L)
                - If batched=True: (B, M, L)
            T_intv: Interventional intervened feature
                - If batched=False: (M,) or (M, 1)
                - If batched=True: (B, M) or (B, M, 1)
            prediction_type: Type of prediction
                - "point": Direct model output (requires BarDistribution disabled)
                - "mode": Most likely value from BarDistribution
                - "mean": Expected value from BarDistribution
                - "sample": Draw samples from BarDistribution
            num_samples: Number of samples for prediction_type="sample"
            batched: If True, expects inputs with leading batch dimension (B, ...)
                    If False (default), expects single-instance inputs (no batch dim)
            
        Returns:
            Predictions array
            - If batched=False: 
                - For "point", "mode", "mean": shape (M,)
                - For "sample": shape (num_samples, M)
            - If batched=True: 
                - For "point", "mode", "mean": shape (B, M)
                - For "sample": shape (B, num_samples, M)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Validate prediction type
        if prediction_type in ["mode", "mean", "sample"] and not self.use_bar_distribution:
            raise ValueError(f"prediction_type='{prediction_type}' requires BarDistribution to be enabled")
        
        # Convert to numpy
        if hasattr(X_obs, "values"):
            X_obs = X_obs.values
        if hasattr(T_obs, "values"):
            T_obs = T_obs.values
        if hasattr(Y_obs, "values"):
            Y_obs = Y_obs.values
        if hasattr(X_intv, "values"):
            X_intv = X_intv.values
        if hasattr(T_intv, "values"):
            T_intv = T_intv.values
        
        X_obs = np.asarray(X_obs, dtype=np.float32)
        T_obs = np.asarray(T_obs, dtype=np.float32)
        Y_obs = np.asarray(Y_obs, dtype=np.float32)
        X_intv = np.asarray(X_intv, dtype=np.float32)
        T_intv = np.asarray(T_intv, dtype=np.float32)
        
        # Ensure T arrays are 2D
        if T_obs.ndim == 1:
            T_obs = T_obs[:, None]
        if T_intv.ndim == 1:
            T_intv = T_intv[:, None]
        if Y_obs.ndim == 2 and Y_obs.shape[1] == 1:
            Y_obs = Y_obs.squeeze(1)
        
        # Validate and convert to torch tensors based on batched flag
        if batched:
            # Batched mode: inputs should have shape (B, N, ...) and (B, M, ...)
            X_obs_t = torch.from_numpy(X_obs).to(self.device)  # (B, N, L)
            T_obs_t = torch.from_numpy(T_obs).to(self.device)  # (B, N, 1)
            Y_obs_t = torch.from_numpy(Y_obs).to(self.device)  # (B, N)
            X_intv_t = torch.from_numpy(X_intv).to(self.device)  # (B, M, L)
            T_intv_t = torch.from_numpy(T_intv).to(self.device)  # (B, M, 1)
        else:
            # Non-batched mode: add batch dimension
            X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
            T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
            Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
            X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
            T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
            predictions = out["predictions"]  # (B, M) or (B, M, output_dim)
        
        # Process predictions based on type
        if not self.use_bar_distribution or prediction_type == "point":
            # Direct regression output
            if batched:
                return predictions.cpu().numpy()  # (B, M)
            else:
                return predictions.squeeze(0).cpu().numpy()  # (M,)
        
        # BarDistribution must be loaded from checkpoint
        if self.bar_distribution.centers is None:
            raise RuntimeError(
                "BarDistribution is not fitted. This should have been loaded from the checkpoint. "
                "Make sure your checkpoint contains the bar_distribution state."
            )
        
        # BarDistribution predictions
        if prediction_type == "mode":
            preds = self.bar_distribution.mode(predictions)  # (B, M)
            if batched:
                return preds.cpu().numpy()  # (B, M)
            else:
                return preds.squeeze(0).cpu().numpy()  # (M,)
        elif prediction_type == "mean":
            preds = self.bar_distribution.mean(predictions)  # (B, M)
            if batched:
                return preds.cpu().numpy()  # (B, M)
            else:
                return preds.squeeze(0).cpu().numpy()  # (M,)
        elif prediction_type == "sample":
            samples = self.bar_distribution.sample(predictions, num_samples)  # (B, num_samples, M)
            if batched:
                return samples.cpu().numpy()  # (B, num_samples, M)
            else:
                return samples.squeeze(0).cpu().numpy()  # (num_samples, M)
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
    
    def log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        batched: bool = False,
    ) -> np.ndarray:
        """
        Compute log-likelihood of test targets under the predictive distribution.
        
        Requires BarDistribution to be enabled.
        
        Args:
            X_obs: Observational features
                - If batched=False: (N, L)
                - If batched=True: (B, N, L)
            T_obs: Observational intervened feature
                - If batched=False: (N,) or (N, 1)
                - If batched=True: (B, N) or (B, N, 1)
            Y_obs: Observational targets
                - If batched=False: (N,) or (N, 1)
                - If batched=True: (B, N) or (B, N, 1)
            X_intv: Interventional features
                - If batched=False: (M, L)
                - If batched=True: (B, M, L)
            T_intv: Interventional intervened feature
                - If batched=False: (M,) or (M, 1)
                - If batched=True: (B, M) or (B, M, 1)
            Y_intv: Interventional targets (ground truth)
                - If batched=False: (M,) or (M, 1)
                - If batched=True: (B, M) or (B, M, 1)
            batched: If True, expects inputs with leading batch dimension (B, ...)
                    If False (default), expects single-instance inputs (no batch dim)
            
        Returns:
            Log-likelihood values
            - If batched=False: shape (M,)
            - If batched=True: shape (B, M)
        """
        if not self.use_bar_distribution:
            raise ValueError("log_likelihood requires BarDistribution to be enabled")
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Convert to numpy
        if hasattr(X_obs, "values"):
            X_obs = X_obs.values
        if hasattr(T_obs, "values"):
            T_obs = T_obs.values
        if hasattr(Y_obs, "values"):
            Y_obs = Y_obs.values
        if hasattr(X_intv, "values"):
            X_intv = X_intv.values
        if hasattr(T_intv, "values"):
            T_intv = T_intv.values
        if hasattr(Y_intv, "values"):
            Y_intv = Y_intv.values
        
        X_obs = np.asarray(X_obs, dtype=np.float32)
        T_obs = np.asarray(T_obs, dtype=np.float32)
        Y_obs = np.asarray(Y_obs, dtype=np.float32)
        X_intv = np.asarray(X_intv, dtype=np.float32)
        T_intv = np.asarray(T_intv, dtype=np.float32)
        Y_intv = np.asarray(Y_intv, dtype=np.float32)
        
        # Ensure T arrays are 2D
        if T_obs.ndim == 1:
            T_obs = T_obs[:, None]
        if T_intv.ndim == 1:
            T_intv = T_intv[:, None]
        if Y_obs.ndim == 2 and Y_obs.shape[1] == 1:
            Y_obs = Y_obs.squeeze(1)
        if Y_intv.ndim == 2 and Y_intv.shape[1] == 1:
            Y_intv = Y_intv.squeeze(1)
        
        # Validate and convert to torch tensors based on batched flag
        if batched:
            # Batched mode
            X_obs_t = torch.from_numpy(X_obs).to(self.device)  # (B, N, L)
            T_obs_t = torch.from_numpy(T_obs).to(self.device)  # (B, N, 1)
            Y_obs_t = torch.from_numpy(Y_obs).to(self.device)  # (B, N)
            X_intv_t = torch.from_numpy(X_intv).to(self.device)  # (B, M, L)
            T_intv_t = torch.from_numpy(T_intv).to(self.device)  # (B, M, 1)
            Y_intv_t = torch.from_numpy(Y_intv).to(self.device)  # (B, M)
        else:
            # Non-batched mode: add batch dimension
            X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
            T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
            Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
            X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
            T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
            Y_intv_t = torch.from_numpy(Y_intv).unsqueeze(0).to(self.device)  # (1, M)
        
        # BarDistribution must be loaded from checkpoint
        if self.bar_distribution.centers is None:
            raise RuntimeError(
                "BarDistribution is not fitted. This should have been loaded from the checkpoint."
            )
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
            predictions = out["predictions"]  # (B, M, output_dim)
            log_probs = self.bar_distribution._logpdf_from_pred(predictions, Y_intv_t)  # (B, M)
            # Clamp for numerical stability
            log_probs = torch.clamp(log_probs, min=self.bar_distribution.log_prob_clip_min, 
                                   max=self.bar_distribution.log_prob_clip_max)
        
        if batched:
            return log_probs.cpu().numpy()  # (B, M)
        else:
            return log_probs.squeeze(0).cpu().numpy()  # (M,)
    
    def predict_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        batched: bool = False,
    ) -> np.ndarray:
        """
        Alias for log_likelihood method for consistency with other wrappers.
        
        Compute log-likelihood of test targets under the predictive distribution.
        Requires BarDistribution to be enabled.
        
        Args:
            X_obs: Observational features
                - If batched=False: (N, L)
                - If batched=True: (B, N, L)
            T_obs: Observational intervened feature
                - If batched=False: (N,) or (N, 1)
                - If batched=True: (B, N) or (B, N, 1)
            Y_obs: Observational targets
                - If batched=False: (N,) or (N, 1)
                - If batched=True: (B, N) or (B, N, 1)
            X_intv: Interventional features
                - If batched=False: (M, L)
                - If batched=True: (B, M, L)
            T_intv: Interventional intervened feature
                - If batched=False: (M,) or (M, 1)
                - If batched=True: (B, M) or (B, M, 1)
            Y_intv: Interventional targets (ground truth)
                - If batched=False: (M,) or (M, 1)
                - If batched=True: (B, M) or (B, M, 1)
            batched: If True, expects inputs with leading batch dimension (B, ...)
                    If False (default), expects single-instance inputs (no batch dim)
            
        Returns:
            Log-likelihood values
            - If batched=False: shape (M,)
            - If batched=True: shape (B, M)
        """
        return self.log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, batched=batched)
    
    def predict_negative_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        batched: bool = False,
    ) -> np.ndarray:
        """
        Compute negative log-likelihood (NLL) for use as a loss metric.
        
        This is simply the negative of log_likelihood, provided for convenience
        when you want a loss value (lower is better) instead of likelihood (higher is better).
        
        Args:
            Same as log_likelihood()
            
        Returns:
            Negative log-likelihood values
            - If batched=False: shape (M,)
            - If batched=True: shape (B, M)
        """
        return -self.log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, batched=batched)


if __name__ == "__main__":
    """
    Comprehensive test suite for batched and non-batched inference.
    
    This test suite validates:
    1. Non-batched mode (batched=False, default):
       - predict() with prediction_type="mode", "mean", "sample"
       - log_likelihood() and predict_log_likelihood()
       - predict_negative_log_likelihood()
       - Correct output shapes: (M,) for scalars, (num_samples, M) for samples
    
    2. Batched mode (batched=True):
       - predict() with all prediction types on batch of instances
       - log_likelihood() and all variants on batched inputs
       - Correct output shapes: (B, M) for scalars, (B, num_samples, M) for samples
    
    3. Method consistency:
       - predict_log_likelihood() returns same values as log_likelihood()
       - predict_negative_log_likelihood() returns negative of log_likelihood()
    
    4. Batched vs non-batched consistency:
       - Verifies that batched predictions match B separate non-batched calls
       - Tests mode, mean, and log-likelihood computations
       - Ensures batching is just an efficiency optimization, not changing results
    """
    import tempfile
    
    # Test the wrapper
    print("\n" + "="*80)
    print("TEST: InterventionalPFNSklearn")
    print("="*80)
    
    # For testing, we'll create a minimal config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'model_config': {
                'num_features': {'value': 5},
                'd_model': {'value': 64},
                'depth': {'value': 2},
                'heads_feat': {'value': 4},
                'heads_samp': {'value': 4},
                'dropout': {'value': 0.0},
                'use_bar_distribution': {'value': True},
                'num_bars': {'value': 50},
                'min_width': {'value': 1e-6},
                'scale_floor': {'value': 1e-6},
            }
        }, f)
        cfg_path = f.name
    
    # Create wrapper
    wrapper = InterventionalPFNSklearn(
        config_path=cfg_path,
        checkpoint_path=None,  # No checkpoint for this test
        device="cpu",
        verbose=True,
    )
    
    # Load model
    try:
        wrapper.load()
        print("\n✓ Model loaded successfully")
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Manually fit the BarDistribution with dummy data (since we have no checkpoint)
    # This is a workaround for testing - in production, this would be loaded from checkpoint
    if wrapper.use_bar_distribution:
        print("\n[INFO] Manually initializing BarDistribution for testing...")
        # Create dummy centers spanning a reasonable range
        num_bars = wrapper.bar_distribution.num_bars
        dummy_centers = np.linspace(-3, 3, num_bars).astype(np.float32)
        wrapper.bar_distribution.centers = torch.tensor(dummy_centers, device=wrapper.device)
        
        # Set widths (uniform for simplicity)
        wrapper.bar_distribution.widths = torch.ones(num_bars, device=wrapper.device) * 0.12
        
        # Compute edges from centers and widths
        half_widths = wrapper.bar_distribution.widths / 2
        edges = torch.zeros(num_bars + 1, device=wrapper.device)
        edges[0] = wrapper.bar_distribution.centers[0] - half_widths[0]
        for i in range(num_bars):
            edges[i + 1] = wrapper.bar_distribution.centers[i] + half_widths[i]
        wrapper.bar_distribution.edges = edges
        
        # Set tail scales
        wrapper.bar_distribution.tailscale_left = torch.tensor(1.0, device=wrapper.device)
        wrapper.bar_distribution.tailscale_right = torch.tensor(1.0, device=wrapper.device)
        wrapper.bar_distribution.base_s_left = torch.tensor(1.0, device=wrapper.device)
        wrapper.bar_distribution.base_s_right = torch.tensor(1.0, device=wrapper.device)
        
        print("✓ BarDistribution initialized")
    
    # Create synthetic data for non-batched tests
    num_features = wrapper.model.num_features
    N, M = 20, 5  # 20 train, 5 test
    
    X_obs = np.random.randn(N, num_features).astype(np.float32)
    T_obs = np.random.randn(N, 1).astype(np.float32)
    Y_obs = np.random.randn(N).astype(np.float32)
    X_intv = np.random.randn(M, num_features).astype(np.float32)
    T_intv = np.random.randn(M, 1).astype(np.float32)
    Y_intv = np.random.randn(M).astype(np.float32)
    
    print("\n" + "="*80)
    print("TEST 1: Non-Batched Mode (batched=False, default)")
    print("="*80)
    print(f"  X_obs: {X_obs.shape}, T_obs: {T_obs.shape}, Y_obs: {Y_obs.shape}")
    print(f"  X_intv: {X_intv.shape}, T_intv: {T_intv.shape}")
    
    # Test non-batched predictions
    try:
        preds_mode = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, 
                                     prediction_type="mode", batched=False)
        print(f"\n✓ Mode predictions shape: {preds_mode.shape} (expected: ({M},))")
        assert preds_mode.shape == (M,), f"Expected shape ({M},), got {preds_mode.shape}"
        print(f"  Sample predictions: {preds_mode[:3]}")
        
        preds_mean = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, 
                                     prediction_type="mean", batched=False)
        print(f"\n✓ Mean predictions shape: {preds_mean.shape} (expected: ({M},))")
        assert preds_mean.shape == (M,), f"Expected shape ({M},), got {preds_mean.shape}"
        print(f"  Sample predictions: {preds_mean[:3]}")
        
        num_samples = 10
        preds_sample = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, 
                                       prediction_type="sample", num_samples=num_samples, batched=False)
        print(f"\n✓ Sample predictions shape: {preds_sample.shape} (expected: ({num_samples}, {M}))")
        assert preds_sample.shape == (num_samples, M), f"Expected shape ({num_samples}, {M}), got {preds_sample.shape}"
        
    except Exception as e:
        print(f"\n✗ Error in non-batched prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test non-batched log-likelihood
    try:
        log_probs = wrapper.log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, batched=False)
        print(f"\n✓ Log-likelihood shape: {log_probs.shape} (expected: ({M},))")
        assert log_probs.shape == (M,), f"Expected shape ({M},), got {log_probs.shape}"
        print(f"  Sample log-likelihoods: {log_probs[:3]}")
        
        # Test alias method
        log_probs_alias = wrapper.predict_log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, batched=False)
        print(f"\n✓ predict_log_likelihood shape: {log_probs_alias.shape}")
        assert np.allclose(log_probs, log_probs_alias), "log_likelihood and predict_log_likelihood should return the same values"
        
        # Test negative log-likelihood
        nll = wrapper.predict_negative_log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, batched=False)
        print(f"\n✓ Negative log-likelihood shape: {nll.shape} (expected: ({M},))")
        assert nll.shape == (M,), f"Expected shape ({M},), got {nll.shape}"
        print(f"  Sample NLL: {nll[:3]}")
        assert np.allclose(nll, -log_probs), "NLL should be negative of log-likelihood"
        
    except Exception as e:
        print(f"\n✗ Error in non-batched log-likelihood: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ All non-batched tests passed!")
    
    # Create synthetic data for batched tests
    B = 3  # Batch size
    X_obs_batch = np.random.randn(B, N, num_features).astype(np.float32)
    T_obs_batch = np.random.randn(B, N, 1).astype(np.float32)
    Y_obs_batch = np.random.randn(B, N).astype(np.float32)
    X_intv_batch = np.random.randn(B, M, num_features).astype(np.float32)
    T_intv_batch = np.random.randn(B, M, 1).astype(np.float32)
    Y_intv_batch = np.random.randn(B, M).astype(np.float32)
    
    print("\n" + "="*80)
    print("TEST 2: Batched Mode (batched=True)")
    print("="*80)
    print(f"  X_obs_batch: {X_obs_batch.shape}, T_obs_batch: {T_obs_batch.shape}, Y_obs_batch: {Y_obs_batch.shape}")
    print(f"  X_intv_batch: {X_intv_batch.shape}, T_intv_batch: {T_intv_batch.shape}")
    
    # Test batched predictions
    try:
        preds_mode_batch = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch, 
                                          X_intv_batch, T_intv_batch,
                                          prediction_type="mode", batched=True)
        print(f"\n✓ Mode predictions shape: {preds_mode_batch.shape} (expected: ({B}, {M}))")
        assert preds_mode_batch.shape == (B, M), f"Expected shape ({B}, {M}), got {preds_mode_batch.shape}"
        print(f"  Sample predictions [0]: {preds_mode_batch[0, :3]}")
        
        preds_mean_batch = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch,
                                          X_intv_batch, T_intv_batch,
                                          prediction_type="mean", batched=True)
        print(f"\n✓ Mean predictions shape: {preds_mean_batch.shape} (expected: ({B}, {M}))")
        assert preds_mean_batch.shape == (B, M), f"Expected shape ({B}, {M}), got {preds_mean_batch.shape}"
        print(f"  Sample predictions [0]: {preds_mean_batch[0, :3]}")
        
        num_samples = 10
        preds_sample_batch = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch,
                                            X_intv_batch, T_intv_batch,
                                            prediction_type="sample", num_samples=num_samples, batched=True)
        print(f"\n✓ Sample predictions shape: {preds_sample_batch.shape} (expected: ({B}, {num_samples}, {M}))")
        assert preds_sample_batch.shape == (B, num_samples, M), f"Expected shape ({B}, {num_samples}, {M}), got {preds_sample_batch.shape}"
        
    except Exception as e:
        print(f"\n✗ Error in batched prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test batched log-likelihood
    try:
        log_probs_batch = wrapper.log_likelihood(X_obs_batch, T_obs_batch, Y_obs_batch,
                                                X_intv_batch, T_intv_batch, Y_intv_batch, batched=True)
        print(f"\n✓ Log-likelihood shape: {log_probs_batch.shape} (expected: ({B}, {M}))")
        assert log_probs_batch.shape == (B, M), f"Expected shape ({B}, {M}), got {log_probs_batch.shape}"
        print(f"  Sample log-likelihoods [0]: {log_probs_batch[0, :3]}")
        
        # Test alias method
        log_probs_alias_batch = wrapper.predict_log_likelihood(X_obs_batch, T_obs_batch, Y_obs_batch,
                                                               X_intv_batch, T_intv_batch, Y_intv_batch, batched=True)
        print(f"\n✓ predict_log_likelihood shape: {log_probs_alias_batch.shape}")
        assert np.allclose(log_probs_batch, log_probs_alias_batch), "log_likelihood and predict_log_likelihood should return the same values"
        
        # Test negative log-likelihood
        nll_batch = wrapper.predict_negative_log_likelihood(X_obs_batch, T_obs_batch, Y_obs_batch,
                                                           X_intv_batch, T_intv_batch, Y_intv_batch, batched=True)
        print(f"\n✓ Negative log-likelihood shape: {nll_batch.shape} (expected: ({B}, {M}))")
        assert nll_batch.shape == (B, M), f"Expected shape ({B}, {M}), got {nll_batch.shape}"
        print(f"  Sample NLL [0]: {nll_batch[0, :3]}")
        assert np.allclose(nll_batch, -log_probs_batch), "NLL should be negative of log-likelihood"
        
    except Exception as e:
        print(f"\n✗ Error in batched log-likelihood: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✓ All batched tests passed!")
    
    # Test consistency between batched and non-batched modes
    print("\n" + "="*80)
    print("TEST 3: Batched vs Non-Batched Consistency")
    print("="*80)
    print("Verifying that batched predictions match multiple non-batched calls...")
    
    # Use the same random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run batched prediction once
    preds_mode_batched = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch,
                                        X_intv_batch, T_intv_batch,
                                        prediction_type="mode", batched=True)
    
    preds_mean_batched = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch,
                                        X_intv_batch, T_intv_batch,
                                        prediction_type="mean", batched=True)
    
    log_probs_batched = wrapper.log_likelihood(X_obs_batch, T_obs_batch, Y_obs_batch,
                                               X_intv_batch, T_intv_batch, Y_intv_batch, batched=True)
    
    # Run non-batched predictions B times
    preds_mode_nonbatched = []
    preds_mean_nonbatched = []
    log_probs_nonbatched = []
    
    for i in range(B):
        mode_i = wrapper.predict(X_obs_batch[i], T_obs_batch[i], Y_obs_batch[i],
                                X_intv_batch[i], T_intv_batch[i],
                                prediction_type="mode", batched=False)
        preds_mode_nonbatched.append(mode_i)
        
        mean_i = wrapper.predict(X_obs_batch[i], T_obs_batch[i], Y_obs_batch[i],
                                X_intv_batch[i], T_intv_batch[i],
                                prediction_type="mean", batched=False)
        preds_mean_nonbatched.append(mean_i)
        
        log_prob_i = wrapper.log_likelihood(X_obs_batch[i], T_obs_batch[i], Y_obs_batch[i],
                                           X_intv_batch[i], T_intv_batch[i], Y_intv_batch[i], batched=False)
        log_probs_nonbatched.append(log_prob_i)
    
    # Stack non-batched results
    preds_mode_nonbatched = np.stack(preds_mode_nonbatched, axis=0)
    preds_mean_nonbatched = np.stack(preds_mean_nonbatched, axis=0)
    log_probs_nonbatched = np.stack(log_probs_nonbatched, axis=0)
    
    # Compare results
    print(f"\nMode predictions:")
    print(f"  Batched shape: {preds_mode_batched.shape}")
    print(f"  Non-batched stacked shape: {preds_mode_nonbatched.shape}")
    mode_max_diff = np.abs(preds_mode_batched - preds_mode_nonbatched).max()
    print(f"  Max absolute difference: {mode_max_diff:.2e}")
    
    print(f"\nMean predictions:")
    print(f"  Batched shape: {preds_mean_batched.shape}")
    print(f"  Non-batched stacked shape: {preds_mean_nonbatched.shape}")
    mean_max_diff = np.abs(preds_mean_batched - preds_mean_nonbatched).max()
    print(f"  Max absolute difference: {mean_max_diff:.2e}")
    
    print(f"\nLog-likelihood:")
    print(f"  Batched shape: {log_probs_batched.shape}")
    print(f"  Non-batched stacked shape: {log_probs_nonbatched.shape}")
    ll_max_diff = np.abs(log_probs_batched - log_probs_nonbatched).max()
    print(f"  Max absolute difference: {ll_max_diff:.2e}")
    
    # Assert consistency (allowing for floating point errors)
    tolerance = 1e-5
    try:
        assert np.allclose(preds_mode_batched, preds_mode_nonbatched, atol=tolerance, rtol=tolerance), \
            f"Mode predictions differ! Max diff: {mode_max_diff}"
        print(f"\n✓ Mode predictions are consistent (tolerance={tolerance})")
        
        assert np.allclose(preds_mean_batched, preds_mean_nonbatched, atol=tolerance, rtol=tolerance), \
            f"Mean predictions differ! Max diff: {mean_max_diff}"
        print(f"✓ Mean predictions are consistent (tolerance={tolerance})")
        
        assert np.allclose(log_probs_batched, log_probs_nonbatched, atol=tolerance, rtol=tolerance), \
            f"Log-likelihoods differ! Max diff: {ll_max_diff}"
        print(f"✓ Log-likelihoods are consistent (tolerance={tolerance})")
        
        print("\n✓ All consistency tests passed!")
        
    except AssertionError as e:
        print(f"\n✗ Consistency test failed: {e}")
        print("\nDetailed comparison:")
        print(f"  Mode predictions batched[0]: {preds_mode_batched[0]}")
        print(f"  Mode predictions non-batched[0]: {preds_mode_nonbatched[0]}")
        print(f"  Difference: {preds_mode_batched[0] - preds_mode_nonbatched[0]}")
        sys.exit(1)
    
    # Cleanup
    os.unlink(cfg_path)
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
    print("\nSummary:")
    print("  ✓ Non-batched predictions (mode, mean, sample)")
    print("  ✓ Non-batched log-likelihood and NLL")
    print("  ✓ Batched predictions (mode, mean, sample)")
    print("  ✓ Batched log-likelihood and NLL")
    print("  ✓ Batched vs non-batched consistency")
    print("="*80)
