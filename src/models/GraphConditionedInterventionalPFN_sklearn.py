"""
Sklearn-like wrapper for Graph-Conditioned InterventionalPFN models.

This module provides a scikit-learn-style interface for graph-conditioned models
including GraphConditionedInterventionalPFN, UltimateGraphConditionedInterventionalPFN,
and FlatGraphConditionedInterventionalPFN.

Key differences from InterventionalPFN_sklearn:
- Requires adjacency_matrix as input for all predictions
- Simpler API: only supports basic prediction and log-likelihood
- No ensemble, clustering, entropy, or variance methods (use full wrapper for those)

Adjacency Matrix Format:
- Shape: (L+2, L+2) where L is the number of features
- Position ordering: [X_0, X_1, ..., X_{L-1}, T, Y]
  * Positions 0 to L-1: Feature variables (sorted order, kept after dropout)
  * Position L: Treatment variable (intervention node)
  * Position L+1: Outcome variable (target feature)
- Edge semantics: A[i,j] = 1 means directed edge from i to j (i causes j)
- The matrix is transposed internally so that j can attend to i (effects attend to causes)
- Self-loops are added automatically by the model for self-attention

Usage:
    # Basic usage
    wrapper = GraphConditionedInterventionalPFNSklearn(
        config_path="config.yaml",
        checkpoint_path="model.pt"
    )
    wrapper.load()
    
    # Prediction with adjacency matrix
    preds = wrapper.predict(
        X_obs, T_obs, Y_obs,
        X_intv, T_intv,
        adjacency_matrix,  # (L+2, L+2) - A[i,j]=1 means i→j
        prediction_type="mode"
    )
    
    # Log-likelihood (requires BarDistribution)
    log_probs = wrapper.log_likelihood(
        X_obs, T_obs, Y_obs,
        X_intv, T_intv, Y_intv,
        adjacency_matrix  # (L+2, L+2) - A[i,j]=1 means i→j
    )
    
    # Or use alias
    log_probs = wrapper.predict_log_likelihood(
        X_obs, T_obs, Y_obs,
        X_intv, T_intv, Y_intv,
        adjacency_matrix
    )
    
    # Negative log-likelihood (NLL) for loss computation
    nll = wrapper.predict_negative_log_likelihood(
        X_obs, T_obs, Y_obs,
        X_intv, T_intv, Y_intv,
        adjacency_matrix
    )
"""

from __future__ import annotations
from typing import Optional, Any, Literal
import sys
import os
import numpy as np
import torch
import yaml
from pathlib import Path

# Robust import - try without 'src.' prefix first, then with 'src.' prefix
try:
    from models.GraphConditionedInterventionalPFN import GraphConditionedInterventionalPFN
    from models.UltimateGraphConditionedInterventionalPFN import UltimateGraphConditionedInterventionalPFN
    from models.FlatGraphConditionedInterventionalPFN import FlatGraphConditionedInterventionalPFN
    from Losses.BarDistribution import BarDistribution
except Exception:
    try:
        from src.models.GraphConditionedInterventionalPFN import GraphConditionedInterventionalPFN
        from src.models.UltimateGraphConditionedInterventionalPFN import UltimateGraphConditionedInterventionalPFN
        from src.models.FlatGraphConditionedInterventionalPFN import FlatGraphConditionedInterventionalPFN
        from src.Losses.BarDistribution import BarDistribution
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.models.GraphConditionedInterventionalPFN import GraphConditionedInterventionalPFN
        from src.models.UltimateGraphConditionedInterventionalPFN import UltimateGraphConditionedInterventionalPFN
        from src.models.FlatGraphConditionedInterventionalPFN import FlatGraphConditionedInterventionalPFN
        from src.Losses.BarDistribution import BarDistribution


class GraphConditionedInterventionalPFNSklearn:
    """
    Simplified sklearn-like wrapper for graph-conditioned InterventionalPFN models.
    
    This wrapper focuses on core functionality: prediction and log-likelihood computation
    for models that require causal graph structure (adjacency matrix) as input.
    
    Supported models (automatically detected from config):
    - GraphConditionedInterventionalPFN (hard attention masking only) 
      → graph_conditioning_mode: not specified or basic modes
    - UltimateGraphConditionedInterventionalPFN (flexible graph conditioning)
      → graph_conditioning_mode: 'ultimate_soft_attention', 'ultimate_gcn', 
         'ultimate_gcn_and_soft_attention', or legacy modes
      → Controlled by flags: use_attention_masking, use_gcn, use_adaln, use_soft_attention_bias
      → Note: soft attention bias and hard attention masking are typically mutually exclusive
    - FlatGraphConditionedInterventionalPFN (flat adjacency append)
      → graph_conditioning_mode: "flat_append"
    
    For advanced features like ensemble, clustering, entropy, and variance,
    use the full InterventionalPFN_sklearn wrapper.
    
    Parameters:
        config_path (str): Path to YAML config file
        checkpoint_path (str): Path to model checkpoint
        device (str): Device for inference ('cpu', 'cuda', 'cuda:0', etc.)
        verbose (bool): Print detailed loading information
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        verbose: bool = False,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.verbose = verbose
        
        # Model components (populated by load())
        self.model = None
        self.model_kwargs = None
        self.bar_distribution = None
        self.use_bar_distribution = False
        self.graph_conditioning_mode = None  # Track which model type to use
        
    def load(self, override_kwargs: Optional[dict[str, Any]] = None) -> "GraphConditionedInterventionalPFNSklearn":
        """
        Load config, build model, and load checkpoint.
        
        Args:
            override_kwargs: Optional dict to override config parameters
            
        Returns:
            self for method chaining
        """
        if self.verbose:
            print(f"[GraphConditionedInterventionalPFNSklearn] Loading model...")
            print(f"  Config: {self.config_path}")
            print(f"  Checkpoint: {self.checkpoint_path}")
        
        # Load config
        if self.config_path:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Helper function to get value from config (handles both direct values and nested "value" keys)
            def get_config_value(config_dict, key, default):
                if key not in config_dict:
                    return default
                val = config_dict[key]
                # If it's a dict with "value" key, extract the value
                if isinstance(val, dict) and "value" in val:
                    return val["value"]
                return val
            
            # Extract model kwargs from config
            self.model_kwargs = {}
            
            # Core architecture parameters (look for 'model_config' key, which is the standard)
            if 'model_config' in config:
                model_cfg = config['model_config']
                self.model_kwargs['num_features'] = get_config_value(model_cfg, 'num_features', None)
                self.model_kwargs['d_model'] = get_config_value(model_cfg, 'd_model', 256)
                self.model_kwargs['depth'] = get_config_value(model_cfg, 'depth', 8)
                self.model_kwargs['heads_feat'] = get_config_value(model_cfg, 'heads_feat', 8)
                self.model_kwargs['heads_samp'] = get_config_value(model_cfg, 'heads_samp', 8)
                self.model_kwargs['dropout'] = get_config_value(model_cfg, 'dropout', 0.0)
                self.model_kwargs['hidden_mult'] = get_config_value(model_cfg, 'hidden_mult', 4)
                self.model_kwargs['normalize_features'] = get_config_value(model_cfg, 'normalize_features', True)
                self.model_kwargs['use_same_row_mlp'] = get_config_value(model_cfg, 'use_same_row_mlp', True)
                self.model_kwargs['n_sample_attention_sink_rows'] = get_config_value(model_cfg, 'n_sample_attention_sink_rows', 0)
                self.model_kwargs['n_feature_attention_sink_cols'] = get_config_value(model_cfg, 'n_feature_attention_sink_cols', 0)
                
                # Detect graph conditioning mode
                self.graph_conditioning_mode = get_config_value(model_cfg, 'graph_conditioning_mode', 'hard_attention_only')
                
                # UltimateGraphConditionedInterventionalPFN-specific parameters
                # First try to get explicit values from config
                use_attention_masking = get_config_value(model_cfg, 'use_attention_masking', None)
                use_gcn = get_config_value(model_cfg, 'use_gcn', None)
                use_adaln = get_config_value(model_cfg, 'use_adaln', None)
                use_soft_attention_bias = get_config_value(model_cfg, 'use_soft_attention_bias', None)
                
                # If not explicitly set, infer from graph_conditioning_mode
                if use_attention_masking is None or use_gcn is None or use_adaln is None or use_soft_attention_bias is None:
                    # Map mode names to flag settings: (use_attention_masking, use_gcn, use_adaln, use_soft_attention_bias)
                    # Note: soft attention bias REQUIRES attention masking to be enabled
                    mode_to_flags = {
                        'ultimate_hard_attention_only': (True, False, False, False),
                        'ultimate_gcn_only': (False, True, True, False),
                        'ultimate_gcn_and_hard_attention': (True, True, True, False),
                        'ultimate_soft_attention': (True, False, False, True),  # Soft bias requires attention masking
                        'ultimate_gcn_and_soft_attention': (True, True, True, True),  # GCN+AdaLN+soft bias requires attention masking
                        # Legacy modes
                        'hard_attention_only': (True, False, False, False),
                        'soft_learned_bias': (True, False, False, True),  # Soft bias requires attention masking
                        'hybrid_half_and_half': (True, False, False, True),  # Soft bias requires attention masking
                    }
                    
                    if self.graph_conditioning_mode in mode_to_flags:
                        inferred_masking, inferred_gcn, inferred_adaln, inferred_soft = mode_to_flags[self.graph_conditioning_mode]
                        if use_attention_masking is None:
                            use_attention_masking = inferred_masking
                        if use_gcn is None:
                            use_gcn = inferred_gcn
                        if use_adaln is None:
                            use_adaln = inferred_adaln
                        if use_soft_attention_bias is None:
                            use_soft_attention_bias = inferred_soft
                    else:
                        # Default fallback
                        if use_attention_masking is None:
                            use_attention_masking = True
                        if use_gcn is None:
                            use_gcn = False
                        if use_adaln is None:
                            use_adaln = False
                        if use_soft_attention_bias is None:
                            use_soft_attention_bias = False
                
                self.model_kwargs['use_attention_masking'] = use_attention_masking
                self.model_kwargs['use_gcn'] = use_gcn
                self.model_kwargs['use_adaln'] = use_adaln
                self.model_kwargs['use_soft_attention_bias'] = use_soft_attention_bias
                self.model_kwargs['soft_bias_init'] = get_config_value(model_cfg, 'soft_bias_init', 5.0)
                
                # BarDistribution configuration
                use_bar = get_config_value(model_cfg, 'use_bar_distribution', False)
                if use_bar:
                    self.use_bar_distribution = True
                    num_bars = get_config_value(model_cfg, 'num_bars', 100)
                    self.model_kwargs['output_dim'] = num_bars + 4  # BarDistribution requires K + 4 parameters
                    
                    # Create BarDistribution
                    self.bar_distribution = BarDistribution(
                        num_bars=num_bars,
                        min_width=float(get_config_value(model_cfg, 'min_width', 1e-6)),
                        scale_floor=float(get_config_value(model_cfg, 'scale_floor', 1e-6)),
                        device=self.device,
                        max_fit_items=get_config_value(model_cfg, 'max_fit_items', None),
                        log_prob_clip_min=float(get_config_value(model_cfg, 'log_prob_clip_min', -50.0)),
                        log_prob_clip_max=float(get_config_value(model_cfg, 'log_prob_clip_max', 50.0)),
                    )
                    if self.verbose:
                        print(f"  BarDistribution enabled: {self.bar_distribution.num_bars} bars, "
                              f"output_dim={self.model_kwargs['output_dim']}")
                else:
                    self.model_kwargs['output_dim'] = 1
                    self.use_bar_distribution = False
            else:
                raise ValueError("'model_config' not found in config file")
        else:
            raise ValueError("config_path is required")
        
        # Apply overrides
        if override_kwargs:
            self.model_kwargs.update(override_kwargs)
        
        # Sanity check
        if not self.model_kwargs.get("num_features"):
            raise ValueError("num_features not found in config")
        
        # Build model based on graph conditioning mode
        if self.verbose:
            print(f"  Graph conditioning mode: {self.graph_conditioning_mode}")
            print(f"  Building model with {self.model_kwargs['num_features']} features...")
        
        # Remove parameters not supported by specific model types
        model_kwargs_filtered = self.model_kwargs.copy()
        
        # Map graph_conditioning_mode to UltimateGraphConditionedInterventionalPFN flags
        if self.graph_conditioning_mode == 'flat_append':
            if self.verbose:
                print(f"  Creating FlatGraphConditionedInterventionalPFN (flat adjacency append)")
            self.model = FlatGraphConditionedInterventionalPFN(**model_kwargs_filtered).to(self.device)
        elif self.graph_conditioning_mode in ['ultimate_hard_attention_only', 'ultimate_gcn_only', 'ultimate_gcn_and_hard_attention',
                                               'ultimate_soft_attention', 'ultimate_gcn_and_soft_attention', 
                                               'soft_learned_bias', 'hybrid_half_and_half', 'hard_attention_only']:
            # All these modes use UltimateGraphConditionedInterventionalPFN with different flag combinations
            if self.verbose:
                print(f"  Creating UltimateGraphConditionedInterventionalPFN")
                print(f"    use_attention_masking: {model_kwargs_filtered.get('use_attention_masking', True)}")
                print(f"    use_gcn: {model_kwargs_filtered.get('use_gcn', False)}")
                print(f"    use_adaln: {model_kwargs_filtered.get('use_adaln', False)}")
                print(f"    use_soft_attention_bias: {model_kwargs_filtered.get('use_soft_attention_bias', False)}")
            self.model = UltimateGraphConditionedInterventionalPFN(**model_kwargs_filtered).to(self.device)
        else:  
            # Default: use basic GraphConditionedInterventionalPFN (hard attention masking only)
            if self.verbose:
                print(f"  Creating GraphConditionedInterventionalPFN (hard attention masking)")
            # Remove UltimateGraphConditionedInterventionalPFN-specific parameters
            for key in ['use_attention_masking', 'use_gcn', 'use_adaln', 'use_soft_attention_bias', 'soft_bias_init']:
                model_kwargs_filtered.pop(key, None)
            self.model = GraphConditionedInterventionalPFN(**model_kwargs_filtered).to(self.device)
        
        # Load checkpoint
        if self.checkpoint_path:
            if self.verbose:
                print(f"  Loading checkpoint from {self.checkpoint_path}...")
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load model state
            self.model.load_state_dict(state_dict, strict=True)
            
            # Load BarDistribution state if available
            if self.use_bar_distribution and 'bar_distribution' in checkpoint:
                bar_state = checkpoint['bar_distribution']
                # Load all BarDistribution state fields
                if 'centers' in bar_state:
                    self.bar_distribution.centers = bar_state['centers']
                if 'edges' in bar_state:
                    self.bar_distribution.edges = bar_state['edges']
                if 'widths' in bar_state:
                    self.bar_distribution.widths = bar_state['widths']
                if 'base_s_left' in bar_state:
                    self.bar_distribution.base_s_left = bar_state['base_s_left']
                if 'base_s_right' in bar_state:
                    self.bar_distribution.base_s_right = bar_state['base_s_right']
                
                if self.verbose and self.bar_distribution.centers is not None:
                    print(f"  BarDistribution state loaded from checkpoint")
                    print(f"    centers shape: {self.bar_distribution.centers.shape}")
                    print(f"    edges shape: {self.bar_distribution.edges.shape if self.bar_distribution.edges is not None else 'None'}")
            
            if self.verbose:
                print(f"  Checkpoint loaded successfully")
        
        self.model.eval()
        
        if self.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  Model loaded with {total_params:,} parameters")
        
        return self
    
    def predict(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        adjacency_matrix: Any,
        prediction_type: Literal["point", "mode", "mean", "sample"] = "mean",
        num_samples: int = 100,
    ) -> np.ndarray:
        """
        Make predictions for interventional test data with graph conditioning.
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            adjacency_matrix: Causal graph adjacency matrix (L+2, L+2)
                Position ordering (matches internal embedding order):
                  - Position 0 to L-1: Feature variables (X[:,0] to X[:,L-1])
                  - Position L: Treatment variable (T)
                  - Position L+1: Outcome variable (Y)
                
                Edge semantics:
                  - A[i,j] = 1 means directed edge from i to j (i causes j)
                  - The matrix is transposed internally so j can attend to i
                  - This ensures effects attend to their causes for causal inference
                  - Self-loops are added automatically for self-attention
            prediction_type: Type of prediction
                - "point": Direct model output (requires BarDistribution disabled)
                - "mode": Most likely value from BarDistribution
                - "mean": Expected value from BarDistribution
                - "sample": Draw samples from BarDistribution
            num_samples: Number of samples for prediction_type="sample"
            
        Returns:
            Predictions array of shape (M,) or (M, num_samples) for sampling
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
        if hasattr(adjacency_matrix, "values"):
            adjacency_matrix = adjacency_matrix.values
        
        X_obs = np.asarray(X_obs, dtype=np.float32)
        T_obs = np.asarray(T_obs, dtype=np.float32)
        Y_obs = np.asarray(Y_obs, dtype=np.float32)
        X_intv = np.asarray(X_intv, dtype=np.float32)
        T_intv = np.asarray(T_intv, dtype=np.float32)
        adjacency_matrix = np.asarray(adjacency_matrix, dtype=np.float32)
        
        # Ensure T arrays are 2D
        if T_obs.ndim == 1:
            T_obs = T_obs.reshape(-1, 1)
        if T_intv.ndim == 1:
            T_intv = T_intv.reshape(-1, 1)
        if Y_obs.ndim == 2 and Y_obs.shape[1] == 1:
            Y_obs = Y_obs.squeeze(-1)
        
        # Ensure adjacency matrix is 2D
        if adjacency_matrix.ndim != 2:
            raise ValueError(f"adjacency_matrix must be 2D, got shape {adjacency_matrix.shape}")
        
        # Convert to torch tensors
        X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
        T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
        Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
        X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
        T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
        adjacency_matrix_t = torch.from_numpy(adjacency_matrix).unsqueeze(0).to(self.device)  # (1, L+2, L+2)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t, adjacency_matrix_t)
            predictions = out["predictions"]  # (1, M) or (1, M, output_dim)
        
        # Process predictions based on type
        if not self.use_bar_distribution or prediction_type == "point":
            # Direct regression output
            return predictions.squeeze(0).cpu().numpy()  # (M,)
        
        # BarDistribution must be loaded from checkpoint
        if self.bar_distribution.centers is None:
            raise RuntimeError(
                "BarDistribution is not fitted. This should have been loaded from the checkpoint. "
                "Make sure your checkpoint contains the bar_distribution state."
            )
        
        # BarDistribution predictions
        if prediction_type == "mode":
            preds = self.bar_distribution.mode(predictions)  # (1, M)
            return preds.squeeze(0).cpu().numpy()  # (M,)
        elif prediction_type == "mean":
            preds = self.bar_distribution.mean(predictions)  # (1, M)
            return preds.squeeze(0).cpu().numpy()  # (M,)
        elif prediction_type == "sample":
            samples = self.bar_distribution.sample(predictions, num_samples)  # (1, M, num_samples)
            return samples.squeeze(0).cpu().numpy()  # (M, num_samples)
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
        adjacency_matrix: Any,
    ) -> np.ndarray:
        """
        Compute log-likelihood of test targets under the predictive distribution.
        
        Requires BarDistribution to be enabled.
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            Y_intv: Interventional targets (M,) or (M, 1) - ground truth
            adjacency_matrix: Causal graph adjacency matrix (L+2, L+2)
                Position ordering (matches internal embedding order):
                  - Position 0 to L-1: Feature variables (X[:,0] to X[:,L-1])
                  - Position L: Treatment variable (T)
                  - Position L+1: Outcome variable (Y)
                
                Edge semantics:
                  - A[i,j] = 1 means directed edge from i to j (i causes j)
                  - The matrix is transposed internally so j can attend to i
                  - This ensures effects attend to their causes for causal inference
                  - Self-loops are added automatically for self-attention
            
        Returns:
            Log-likelihood values of shape (M,)
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
        if hasattr(adjacency_matrix, "values"):
            adjacency_matrix = adjacency_matrix.values
        
        X_obs = np.asarray(X_obs, dtype=np.float32)
        T_obs = np.asarray(T_obs, dtype=np.float32)
        Y_obs = np.asarray(Y_obs, dtype=np.float32)
        X_intv = np.asarray(X_intv, dtype=np.float32)
        T_intv = np.asarray(T_intv, dtype=np.float32)
        Y_intv = np.asarray(Y_intv, dtype=np.float32)
        adjacency_matrix = np.asarray(adjacency_matrix, dtype=np.float32)
        
        # Ensure T arrays are 2D
        if T_obs.ndim == 1:
            T_obs = T_obs.reshape(-1, 1)
        if T_intv.ndim == 1:
            T_intv = T_intv.reshape(-1, 1)
        if Y_obs.ndim == 2 and Y_obs.shape[1] == 1:
            Y_obs = Y_obs.squeeze(-1)
        if Y_intv.ndim == 2 and Y_intv.shape[1] == 1:
            Y_intv = Y_intv.squeeze(-1)
        
        # Ensure adjacency matrix is 2D
        if adjacency_matrix.ndim != 2:
            raise ValueError(f"adjacency_matrix must be 2D, got shape {adjacency_matrix.shape}")
        
        # Convert to torch tensors
        X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
        T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
        Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
        X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
        T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
        Y_intv_t = torch.from_numpy(Y_intv).unsqueeze(0).to(self.device)  # (1, M)
        adjacency_matrix_t = torch.from_numpy(adjacency_matrix).unsqueeze(0).to(self.device)  # (1, L+2, L+2)
        
        # BarDistribution must be loaded from checkpoint
        if self.bar_distribution.centers is None:
            raise RuntimeError(
                "BarDistribution is not fitted. This should have been loaded from the checkpoint. "
                "Make sure your checkpoint contains the bar_distribution state."
            )
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t, adjacency_matrix_t)
            predictions = out["predictions"]  # (1, M, output_dim)
            
            # Compute log-likelihood
            log_probs = self.bar_distribution._logpdf_from_pred(predictions, Y_intv_t)  # (1, M)
        
        return log_probs.squeeze(0).cpu().numpy()  # (M,)
    
    def predict_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        adjacency_matrix: Any,
    ) -> np.ndarray:
        """
        Alias for log_likelihood method for consistency with SimplePFN_sklearn.
        
        Compute log-likelihood of test targets under the predictive distribution.
        Requires BarDistribution to be enabled.
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            Y_intv: Interventional targets (M,) or (M, 1) - ground truth
            adjacency_matrix: Causal graph adjacency matrix (L+2, L+2)
                Position ordering (matches internal embedding order):
                  - Position 0 to L-1: Feature variables (X[:,0] to X[:,L-1])
                  - Position L: Treatment variable (T)
                  - Position L+1: Outcome variable (Y)
                
                Edge semantics:
                  - A[i,j] = 1 means directed edge from i to j (i causes j)
                  - The matrix is transposed internally so j can attend to i
                  - This ensures effects attend to their causes for causal inference
                  - Self-loops are added automatically for self-attention
            
        Returns:
            Log-likelihood values of shape (M,)
        """
        return self.log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix)
    
    def predict_negative_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        adjacency_matrix: Any,
    ) -> np.ndarray:
        """
        Compute negative log-likelihood (NLL) of test targets.
        
        This is simply the negative of log_likelihood, commonly used as a loss metric.
        Requires BarDistribution to be enabled.
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            Y_intv: Interventional targets (M,) or (M, 1) - ground truth
            adjacency_matrix: Causal graph adjacency matrix (L+2, L+2)
                Position ordering (matches internal embedding order):
                  - Position 0 to L-1: Feature variables (X[:,0] to X[:,L-1])
                  - Position L: Treatment variable (T)
                  - Position L+1: Outcome variable (Y)
                
                Edge semantics:
                  - A[i,j] = 1 means directed edge from i to j (i causes j)
                  - The matrix is transposed internally so j can attend to i
                  - This ensures effects attend to their causes for causal inference
                  - Self-loops are added automatically for self-attention
            
        Returns:
            Negative log-likelihood values of shape (M,)
        """
        return -self.log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix)


if __name__ == "__main__":
    # Test the graph-conditioned wrapper
    print("\n" + "="*80)
    print("TEST: GraphConditionedInterventionalPFNSklearn Basic Usage")
    print("="*80)
    
    # Create dummy config and checkpoint paths (adjust to your actual paths)
    cfg_path = "/path/to/config.yaml"
    ckpt_path = "/path/to/checkpoint.pt"
    
    # For testing, we'll create a minimal config
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            'model': {
                'num_features': 5,
                'd_model': 64,
                'depth': 2,
                'heads_feat': 4,
                'heads_samp': 4,
                'dropout': 0.0,
            },
            'bar_distribution': {
                'use_bar_distribution': True,
                'num_bars': 50,
                'min_width': 1e-6,
                'scale_floor': 1e-6,
            }
        }, f)
        cfg_path = f.name
    
    # Create wrapper
    wrapper = GraphConditionedInterventionalPFNSklearn(
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
    
    # Create synthetic data
    num_features = wrapper.model.num_features
    N, M = 20, 5  # 20 train, 5 test
    
    X_obs = np.random.randn(N, num_features).astype(np.float32)
    T_obs = np.random.randn(N, 1).astype(np.float32)
    Y_obs = np.random.randn(N).astype(np.float32)
    X_intv = np.random.randn(M, num_features).astype(np.float32)
    T_intv = np.random.randn(M, 1).astype(np.float32)
    Y_intv = np.random.randn(M).astype(np.float32)
    
    # Create dummy adjacency matrix (fully connected for testing)
    adjacency_matrix = np.ones((num_features + 2, num_features + 2), dtype=np.float32)
    
    print(f"\n[GraphConditionedInterventionalPFNSklearn] Making predictions...")
    print(f"  X_obs: {X_obs.shape}, T_obs: {T_obs.shape}, Y_obs: {Y_obs.shape}")
    print(f"  X_intv: {X_intv.shape}, T_intv: {T_intv.shape}")
    print(f"  adjacency_matrix: {adjacency_matrix.shape}")
    
    # Test predictions
    try:
        preds_mode = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix, prediction_type="mode")
        print(f"\n✓ Mode predictions shape: {preds_mode.shape}")
        print(f"  Sample predictions: {preds_mode[:3]}")
        
        preds_mean = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix, prediction_type="mean")
        print(f"\n✓ Mean predictions shape: {preds_mean.shape}")
        print(f"  Sample predictions: {preds_mean[:3]}")
        
        preds_sample = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix, prediction_type="sample", num_samples=10)
        print(f"\n✓ Sample predictions shape: {preds_sample.shape}")
        
    except Exception as e:
        print(f"\n✗ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
    
    # Test log-likelihood
    try:
        log_probs = wrapper.log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix)
        print(f"\n✓ Log-likelihood shape: {log_probs.shape}")
        print(f"  Sample log-likelihoods: {log_probs[:3]}")
        
        # Test alias method
        log_probs_alias = wrapper.predict_log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix)
        print(f"\n✓ predict_log_likelihood shape: {log_probs_alias.shape}")
        assert np.allclose(log_probs, log_probs_alias), "log_likelihood and predict_log_likelihood should return the same values"
        
        # Test negative log-likelihood
        nll = wrapper.predict_negative_log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix)
        print(f"\n✓ Negative log-likelihood shape: {nll.shape}")
        print(f"  Sample NLL: {nll[:3]}")
        assert np.allclose(nll, -log_probs), "NLL should be negative of log-likelihood"
        
    except Exception as e:
        print(f"\n✗ Error in log-likelihood: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    os.unlink(cfg_path)
    
    print("\n" + "="*80)
    print("TEST COMPLETED!")
    print("="*80)
