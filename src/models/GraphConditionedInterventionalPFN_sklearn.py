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

Special Inference Behavior for hide_fraction_matrix >= 1.0:
- When a model is trained with dataset_config.hide_fraction_matrix = 1.0 (all graph entries hidden),
  the wrapper automatically replaces off-diagonal entries in the (L+2)×(L+2) submatrix with 0s
- Only applies to active features (features with at least one non-zero entry in their row/column)
- Diagonal entries (self-loops) and any padding (sink columns) remain unchanged
- This ensures the model sees the same "fully hidden" graph structure at inference time
- Prevents information leakage and ensures fair evaluation of models trained without graph knowledge

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
    from models.PartialGraphConditionedInterventionalPFN import PartialGraphConditionedInterventionalPFN
    from Losses.BarDistribution import BarDistribution
    from utils.graph_utils import propagate_ancestor_knowledge
except Exception:
    try:
        from src.models.GraphConditionedInterventionalPFN import GraphConditionedInterventionalPFN
        from src.models.UltimateGraphConditionedInterventionalPFN import UltimateGraphConditionedInterventionalPFN
        from src.models.FlatGraphConditionedInterventionalPFN import FlatGraphConditionedInterventionalPFN
        from src.models.PartialGraphConditionedInterventionalPFN import PartialGraphConditionedInterventionalPFN
        from src.Losses.BarDistribution import BarDistribution
        from src.utils.graph_utils import propagate_ancestor_knowledge
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        utils_path = Path(__file__).resolve().parents[1] / "utils"
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        if str(utils_path) not in sys.path:
            sys.path.insert(0, str(utils_path))
        from src.models.GraphConditionedInterventionalPFN import GraphConditionedInterventionalPFN
        from src.models.UltimateGraphConditionedInterventionalPFN import UltimateGraphConditionedInterventionalPFN
        from src.models.FlatGraphConditionedInterventionalPFN import FlatGraphConditionedInterventionalPFN
        from src.models.PartialGraphConditionedInterventionalPFN import PartialGraphConditionedInterventionalPFN
        from src.Losses.BarDistribution import BarDistribution
        from graph_utils import propagate_ancestor_knowledge


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
    - PartialGraphConditionedInterventionalPFN (partial graph support with {-1, 0, 1} edges)
      → graph_conditioning_mode: "partial_soft_attention" or "partial_gcn_and_soft_attention"
      → Requires use_partial_graph_format=true in dataset config
      → Supports adjacency matrices with three states: -1 (no edge), 0 (unknown), 1 (edge)
    
    For advanced features like ensemble, clustering, entropy, and variance,
    use the full InterventionalPFN_sklearn wrapper.
    
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
            print(f"[GraphConditionedInterventionalPFNSklearn] Auto-selected device: {self.device}")
        else:
            self.device = device
        self.verbose = verbose
        
        # Model components (populated by load())
        self.model = None
        self.model_kwargs = None
        self.bar_distribution = None
        self.use_bar_distribution = False
        self.graph_conditioning_mode = None  # Track which model type to use
        self.use_partial_graph_format = False  # Track if partial graph format is enabled
        self.propagate_partial_knowledge = True  # Auto-propagate knowledge in partial graphs
        self.hide_fraction_matrix = 0.0  # Track hide_fraction for inference behavior
        
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
                        # Partial graph modes (for three-state adjacency: -1/0/1)
                        'partial_soft_attention': (True, False, False, True),  # Soft bias for partial graphs
                        'partial_gcn_and_soft_attention': (True, True, True, True),  # Full conditioning for partial graphs
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
        
        # Check if partial graph format is enabled
        use_partial_graph = self.model_kwargs.get("use_partial_graph_format", False)
        
        # Track if we're using partial graph format (three-state matrices {-1, 0, 1})
        # This is used to enable automatic propagation of ancestor knowledge
        self.use_partial_graph_format = (
            use_partial_graph or 
            self.graph_conditioning_mode in ['partial_soft_attention', 'partial_gcn_and_soft_attention']
        )
        
        # Extract hide_fraction_matrix from dataset_config if available
        # This controls whether to force all-unknown matrices during inference
        if 'dataset_config' in config:
            dataset_cfg = config['dataset_config']
            hide_fraction_raw = get_config_value(dataset_cfg, 'hide_fraction_matrix', 0.0)
            # Handle case where hide_fraction_matrix is a distribution (dict) rather than a fixed value
            if isinstance(hide_fraction_raw, dict):
                # If it's a distribution, use a default value (0.5 for variable, or check if it's fixed)
                if 'value' in hide_fraction_raw:
                    self.hide_fraction_matrix = float(hide_fraction_raw['value'])
                else:
                    # It's a distribution - for inference, we'll use the data's actual hide fraction
                    # Set to None to indicate variable/unknown
                    self.hide_fraction_matrix = None
                    if self.verbose:
                        print(f"  hide_fraction_matrix: variable (distribution)")
            else:
                self.hide_fraction_matrix = float(hide_fraction_raw)
            if self.hide_fraction_matrix is not None and self.verbose and self.hide_fraction_matrix > 0:
                print(f"  hide_fraction_matrix: {self.hide_fraction_matrix}")
                if self.hide_fraction_matrix >= 1.0:
                    print(f"  → Inference mode: Off-diagonal entries in (L+2)×(L+2) will be hidden (set to 0)")
        
        if self.verbose and self.use_partial_graph_format:
            print(f"  Partial graph format enabled - will propagate ancestor knowledge automatically")
        
        # Map graph_conditioning_mode to appropriate model class
        if self.graph_conditioning_mode == 'flat_append':
            if self.verbose:
                print(f"  Creating FlatGraphConditionedInterventionalPFN (flat adjacency append)")
            # Remove use_partial_graph_format since it's not a model parameter
            model_kwargs_filtered.pop('use_partial_graph_format', None)
            self.model = FlatGraphConditionedInterventionalPFN(**model_kwargs_filtered).to(self.device)
        elif self.graph_conditioning_mode in ['partial_soft_attention', 'partial_gcn_and_soft_attention']:
            # PartialGraphConditionedInterventionalPFN (for partial graphs with {-1, 0, 1} edges)
            if self.verbose:
                print(f"  Creating PartialGraphConditionedInterventionalPFN")
                print(f"    Mode: {self.graph_conditioning_mode}")
            
            # Map mode to specific configuration
            if self.graph_conditioning_mode == 'partial_soft_attention':
                model_kwargs_filtered['use_attention_masking'] = True  # Soft bias requires attention masking
                model_kwargs_filtered['use_gcn'] = False
                model_kwargs_filtered['use_adaln'] = False
                model_kwargs_filtered['use_soft_attention_bias'] = True
            elif self.graph_conditioning_mode == 'partial_gcn_and_soft_attention':
                model_kwargs_filtered['use_attention_masking'] = True  # Soft bias requires attention masking
                model_kwargs_filtered['use_gcn'] = True
                model_kwargs_filtered['use_adaln'] = True
                model_kwargs_filtered['use_soft_attention_bias'] = True
            
            # Remove use_partial_graph_format since it's not a model parameter
            model_kwargs_filtered.pop('use_partial_graph_format', None)
            
            self.model = PartialGraphConditionedInterventionalPFN(**model_kwargs_filtered).to(self.device)
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
            # Remove use_partial_graph_format since it's not a model parameter
            model_kwargs_filtered.pop('use_partial_graph_format', None)
            self.model = UltimateGraphConditionedInterventionalPFN(**model_kwargs_filtered).to(self.device)
        elif use_partial_graph:
            # If use_partial_graph_format is enabled but no specific mode, use PartialGraphConditionedInterventionalPFN
            # with default configuration (full graph conditioning)
            if self.verbose:
                print(f"  Creating PartialGraphConditionedInterventionalPFN (use_partial_graph_format=True)")
                print(f"    Using default full graph conditioning mode")
            
            model_kwargs_filtered['use_attention_masking'] = True
            model_kwargs_filtered['use_gcn'] = True
            model_kwargs_filtered['use_adaln'] = True
            model_kwargs_filtered['use_soft_attention_bias'] = False
            
            if self.verbose:
                print(f"    use_attention_masking: {model_kwargs_filtered['use_attention_masking']}")
                print(f"    use_gcn: {model_kwargs_filtered['use_gcn']}")
                print(f"    use_adaln: {model_kwargs_filtered['use_adaln']}")
                print(f"    use_soft_attention_bias: {model_kwargs_filtered['use_soft_attention_bias']}")
            
            model_kwargs_filtered.pop('use_partial_graph_format', None)
            self.model = PartialGraphConditionedInterventionalPFN(**model_kwargs_filtered).to(self.device)
        else:  
            # Default: use basic GraphConditionedInterventionalPFN (hard attention masking only)
            if self.verbose:
                print(f"  Creating GraphConditionedInterventionalPFN (hard attention masking)")
            # Remove UltimateGraphConditionedInterventionalPFN-specific and partial graph parameters
            for key in ['use_attention_masking', 'use_gcn', 'use_adaln', 'use_soft_attention_bias', 'soft_bias_init', 'use_partial_graph_format']:
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
    
    def _preprocess_adjacency_matrix(self, adjacency_matrix_t: torch.Tensor) -> torch.Tensor:
        """
        Preprocess adjacency/ancestor matrix before passing to model.
        
        Special behavior when hide_fraction_matrix >= 1.0:
        - Replaces off-diagonal entries in the (L+2)×(L+2) submatrix with 0 (all unknown)
        - Only applies to active features (rows/columns with at least one non-zero entry)
        - Keeps diagonal entries and any padding (sink columns) unchanged
        - This ensures the model uses the same "fully hidden" setting it was trained with
        
        For partial graph formats (three-state matrices with {-1, 0, 1}), this automatically
        propagates known ancestor relationships to fill in as many unknown (0) entries as
        possible using transitivity and antisymmetry rules (unless hide_fraction_matrix >= 1.0).
        
        Args:
            adjacency_matrix_t: Tensor of shape (B, N, N) or (N, N)
                For partial graphs: values in {-1, 0, 1}
                  -1: no edge/ancestor relationship
                   0: unknown
                   1: edge/ancestor relationship exists
        
        Returns:
            Preprocessed matrix with same shape, potentially with fewer unknowns
            (or all off-diagonal 0s for active features in (L+2)×(L+2) block if hide_fraction_matrix >= 1.0)
        """
        # If trained with hide_fraction_matrix >= 1.0, force off-diagonal entries to 0 (unknown)
        # This ensures inference matches the training condition
        if self.hide_fraction_matrix >= 1.0:
            # Get the size of the feature adjacency part (L+2)
            # The matrix may be larger if there are sink columns added
            # We determine L+2 as the model's num_features + 2
            L_plus_2 = self.model.num_features + 2
            
            # Clone to avoid modifying the input
            result = adjacency_matrix_t.clone()
            
            # Identify active features (rows/columns with at least one non-zero entry)
            if result.ndim == 3:  # Batched: (B, N, N)
                # Check if any entry in each row or column is non-zero
                # A feature is active if either its row or column has non-zero entries
                row_active = (result[:, :L_plus_2, :L_plus_2] != 0).any(dim=2)  # (B, L+2)
                col_active = (result[:, :L_plus_2, :L_plus_2] != 0).any(dim=1)  # (B, L+2)
                active_mask = row_active | col_active  # (B, L+2) - True if feature is active
                
                # Create off-diagonal mask for (L+2)×(L+2) block
                eye_mask = torch.eye(L_plus_2, device=result.device, dtype=torch.bool)
                off_diag_mask = ~eye_mask  # (L+2, L+2)
                
                # For each batch element, zero out off-diagonal entries for active features
                for b in range(result.shape[0]):
                    active_indices = torch.where(active_mask[b])[0]
                    for i in active_indices:
                        for j in active_indices:
                            if i != j:  # Off-diagonal
                                result[b, i, j] = 0.0
            else:  # Non-batched: (N, N)
                # Check if any entry in each row or column is non-zero
                row_active = (result[:L_plus_2, :L_plus_2] != 0).any(dim=1)  # (L+2,)
                col_active = (result[:L_plus_2, :L_plus_2] != 0).any(dim=0)  # (L+2,)
                active_mask = row_active | col_active  # (L+2,) - True if feature is active
                
                # Zero out off-diagonal entries only for active features
                active_indices = torch.where(active_mask)[0]
                for i in active_indices:
                    for j in active_indices:
                        if i != j:  # Off-diagonal
                            result[i, j] = 0.0
            
            return result
        
        if not self.use_partial_graph_format or not self.propagate_partial_knowledge:
            # No preprocessing needed
            return adjacency_matrix_t
        
        # Ensure float dtype (propagate_ancestor_knowledge needs float for bmm on CUDA)
        if adjacency_matrix_t.dtype not in [torch.float32, torch.float64]:
            adjacency_matrix_t = adjacency_matrix_t.float()
        
        # propagate_ancestor_knowledge already handles both (N,N) and (B,N,N) shapes
        res =  propagate_ancestor_knowledge(adjacency_matrix_t)

        return res
    
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
        batched: bool = False,
    ) -> np.ndarray:
        """
        Make predictions for interventional test data with graph conditioning.
        
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
            adjacency_matrix: Causal graph adjacency matrix
                - If batched=False: (L+2, L+2)
                - If batched=True: (B, L+2, L+2)
                
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
        
        # Validate and convert to torch tensors based on batched flag
        if batched:
            # Batched input: expect (B, ...) shapes
            if adjacency_matrix.ndim != 3:
                raise ValueError(f"adjacency_matrix must be 3D when batched=True, got shape {adjacency_matrix.shape}")
            
            # Convert to torch tensors (already have batch dimension)
            X_obs_t = torch.from_numpy(X_obs).to(self.device)  # (B, N, L)
            T_obs_t = torch.from_numpy(T_obs).to(self.device)  # (B, N, 1)
            Y_obs_t = torch.from_numpy(Y_obs).to(self.device)  # (B, N)
            X_intv_t = torch.from_numpy(X_intv).to(self.device)  # (B, M, L)
            T_intv_t = torch.from_numpy(T_intv).to(self.device)  # (B, M, 1)
            adjacency_matrix_t = torch.from_numpy(adjacency_matrix).to(self.device)  # (B, L+2, L+2)
        else:
            # Non-batched input: add batch dimension
            if adjacency_matrix.ndim != 2:
                raise ValueError(f"adjacency_matrix must be 2D when batched=False, got shape {adjacency_matrix.shape}")
            
            # Convert to torch tensors and add batch dimension
            X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
            T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
            Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
            X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
            T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
            adjacency_matrix_t = torch.from_numpy(adjacency_matrix).unsqueeze(0).to(self.device)  # (1, L+2, L+2)
        
        # Preprocess adjacency matrix (propagate ancestor knowledge for partial graphs)
        adjacency_matrix_t = self._preprocess_adjacency_matrix(adjacency_matrix_t)

        #breakpoint()
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            #breakpoint()
            out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t, adjacency_matrix_t)
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
        adjacency_matrix: Any,
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
            adjacency_matrix: Causal graph adjacency matrix
                - If batched=False: (L+2, L+2)
                - If batched=True: (B, L+2, L+2)
                
                Position ordering (matches internal embedding order):
                  - Position 0 to L-1: Feature variables (X[:,0] to X[:,L-1])
                  - Position L: Treatment variable (T)
                  - Position L+1: Outcome variable (Y)
                
                Edge semantics:
                  - A[i,j] = 1 means directed edge from i to j (i causes j)
                  - The matrix is transposed internally so j can attend to i
                  - This ensures effects attend to their causes for causal inference
                  - Self-loops are added automatically for self-attention
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
        
        # Validate and convert to torch tensors based on batched flag
        if batched:
            # Batched input: expect (B, ...) shapes
            if adjacency_matrix.ndim != 3:
                raise ValueError(f"adjacency_matrix must be 3D when batched=True, got shape {adjacency_matrix.shape}")
            
            # Convert to torch tensors (already have batch dimension)
            X_obs_t = torch.from_numpy(X_obs).to(self.device)  # (B, N, L)
            T_obs_t = torch.from_numpy(T_obs).to(self.device)  # (B, N, 1)
            Y_obs_t = torch.from_numpy(Y_obs).to(self.device)  # (B, N)
            X_intv_t = torch.from_numpy(X_intv).to(self.device)  # (B, M, L)
            T_intv_t = torch.from_numpy(T_intv).to(self.device)  # (B, M, 1)
            Y_intv_t = torch.from_numpy(Y_intv).to(self.device)  # (B, M)
            adjacency_matrix_t = torch.from_numpy(adjacency_matrix).to(self.device)  # (B, L+2, L+2)
        else:
            # Non-batched input: add batch dimension
            if adjacency_matrix.ndim != 2:
                raise ValueError(f"adjacency_matrix must be 2D when batched=False, got shape {adjacency_matrix.shape}")
            
            # Convert to torch tensors and add batch dimension
            X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
            T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
            Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
            X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
            T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
            Y_intv_t = torch.from_numpy(Y_intv).unsqueeze(0).to(self.device)  # (1, M)
            adjacency_matrix_t = torch.from_numpy(adjacency_matrix).unsqueeze(0).to(self.device)  # (1, L+2, L+2)
        
        # Preprocess adjacency matrix (propagate ancestor knowledge for partial graphs)
        adjacency_matrix_t = self._preprocess_adjacency_matrix(adjacency_matrix_t)
        
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
            predictions = out["predictions"]  # (B, M, output_dim)
            
            # Compute log-likelihood
            log_probs = self.bar_distribution._logpdf_from_pred(predictions, Y_intv_t)  # (B, M)
        
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
        adjacency_matrix: Any,
        batched: bool = False,
    ) -> np.ndarray:
        """
        Alias for log_likelihood method for consistency with SimplePFN_sklearn.
        
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
            adjacency_matrix: Causal graph adjacency matrix
                - If batched=False: (L+2, L+2)
                - If batched=True: (B, L+2, L+2)
                
                Position ordering (matches internal embedding order):
                  - Position 0 to L-1: Feature variables (X[:,0] to X[:,L-1])
                  - Position L: Treatment variable (T)
                  - Position L+1: Outcome variable (Y)
                
                Edge semantics:
                  - A[i,j] = 1 means directed edge from i to j (i causes j)
                  - The matrix is transposed internally so j can attend to i
                  - This ensures effects attend to their causes for causal inference
                  - Self-loops are added automatically for self-attention
            batched: If True, expects inputs with leading batch dimension (B, ...)
                    If False (default), expects single-instance inputs (no batch dim)
            
        Returns:
            Log-likelihood values
            - If batched=False: shape (M,)
            - If batched=True: shape (B, M)
        """
        return self.log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix, batched=batched)
    
    def predict_negative_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        adjacency_matrix: Any,
        batched: bool = False,
    ) -> np.ndarray:
        """
        Compute negative log-likelihood (NLL) of test targets.
        
        This is simply the negative of log_likelihood, commonly used as a loss metric.
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
            adjacency_matrix: Causal graph adjacency matrix
                - If batched=False: (L+2, L+2)
                - If batched=True: (B, L+2, L+2)
                
                Position ordering (matches internal embedding order):
                  - Position 0 to L-1: Feature variables (X[:,0] to X[:,L-1])
                  - Position L: Treatment variable (T)
                  - Position L+1: Outcome variable (Y)
                
                Edge semantics:
                  - A[i,j] = 1 means directed edge from i to j (i causes j)
                  - The matrix is transposed internally so j can attend to i
                  - This ensures effects attend to their causes for causal inference
                  - Self-loops are added automatically for self-attention
            batched: If True, expects inputs with leading batch dimension (B, ...)
                    If False (default), expects single-instance inputs (no batch dim)
            
        Returns:
            Negative log-likelihood values
            - If batched=False: shape (M,)
            - If batched=True: shape (B, M)
        """
        return -self.log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix, batched=batched)


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
    
    3. Shape validation:
       - Ensures 2D adjacency matrix with batched=True raises ValueError
       - Ensures 3D adjacency matrix with batched=False raises ValueError
    
    4. Method consistency:
       - predict_log_likelihood() returns same values as log_likelihood()
       - predict_negative_log_likelihood() returns negative of log_likelihood()
    
    5. Batched vs non-batched consistency:
       - Verifies that batched predictions match B separate non-batched calls
       - Tests mode, mean, and log-likelihood computations
       - Ensures batching is just an efficiency optimization, not changing results
    """
    # Test the graph-conditioned wrapper
    print("\n" + "="*80)
    print("TEST: GraphConditionedInterventionalPFNSklearn")
    print("="*80)
    
    # For testing, we'll create a minimal config
    import tempfile
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
                'graph_conditioning_mode': {'value': 'ultimate_gcn_only'},
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
    
    # Create dummy adjacency matrix (fully connected for testing)
    adjacency_matrix = np.ones((num_features + 2, num_features + 2), dtype=np.float32)
    
    print("\n" + "="*80)
    print("TEST 1: Non-Batched Mode (batched=False, default)")
    print("="*80)
    print(f"  X_obs: {X_obs.shape}, T_obs: {T_obs.shape}, Y_obs: {Y_obs.shape}")
    print(f"  X_intv: {X_intv.shape}, T_intv: {T_intv.shape}")
    print(f"  adjacency_matrix: {adjacency_matrix.shape}")
    
    # Test non-batched predictions
    try:
        preds_mode = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix, 
                                     prediction_type="mode", batched=False)
        print(f"\n✓ Mode predictions shape: {preds_mode.shape} (expected: ({M},))")
        assert preds_mode.shape == (M,), f"Expected shape ({M},), got {preds_mode.shape}"
        print(f"  Sample predictions: {preds_mode[:3]}")
        
        preds_mean = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix, 
                                     prediction_type="mean", batched=False)
        print(f"\n✓ Mean predictions shape: {preds_mean.shape} (expected: ({M},))")
        assert preds_mean.shape == (M,), f"Expected shape ({M},), got {preds_mean.shape}"
        print(f"  Sample predictions: {preds_mean[:3]}")
        
        num_samples = 10
        preds_sample = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix, 
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
        log_probs = wrapper.log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, 
                                          adjacency_matrix, batched=False)
        print(f"\n✓ Log-likelihood shape: {log_probs.shape} (expected: ({M},))")
        assert log_probs.shape == (M,), f"Expected shape ({M},), got {log_probs.shape}"
        print(f"  Sample log-likelihoods: {log_probs[:3]}")
        
        # Test alias method
        log_probs_alias = wrapper.predict_log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, 
                                                         adjacency_matrix, batched=False)
        print(f"\n✓ predict_log_likelihood shape: {log_probs_alias.shape}")
        assert np.allclose(log_probs, log_probs_alias), "log_likelihood and predict_log_likelihood should return the same values"
        
        # Test negative log-likelihood
        nll = wrapper.predict_negative_log_likelihood(X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, 
                                                      adjacency_matrix, batched=False)
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
    
    # Create batched adjacency matrices
    adjacency_matrix_batch = np.ones((B, num_features + 2, num_features + 2), dtype=np.float32)
    
    print("\n" + "="*80)
    print("TEST 2: Batched Mode (batched=True)")
    print("="*80)
    print(f"  X_obs_batch: {X_obs_batch.shape}, T_obs_batch: {T_obs_batch.shape}, Y_obs_batch: {Y_obs_batch.shape}")
    print(f"  X_intv_batch: {X_intv_batch.shape}, T_intv_batch: {T_intv_batch.shape}")
    print(f"  adjacency_matrix_batch: {adjacency_matrix_batch.shape}")
    
    # Test batched predictions
    try:
        preds_mode_batch = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch, 
                                          X_intv_batch, T_intv_batch, adjacency_matrix_batch,
                                          prediction_type="mode", batched=True)
        print(f"\n✓ Mode predictions shape: {preds_mode_batch.shape} (expected: ({B}, {M}))")
        assert preds_mode_batch.shape == (B, M), f"Expected shape ({B}, {M}), got {preds_mode_batch.shape}"
        print(f"  Sample predictions [0]: {preds_mode_batch[0, :3]}")
        
        preds_mean_batch = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch,
                                          X_intv_batch, T_intv_batch, adjacency_matrix_batch,
                                          prediction_type="mean", batched=True)
        print(f"\n✓ Mean predictions shape: {preds_mean_batch.shape} (expected: ({B}, {M}))")
        assert preds_mean_batch.shape == (B, M), f"Expected shape ({B}, {M}), got {preds_mean_batch.shape}"
        print(f"  Sample predictions [0]: {preds_mean_batch[0, :3]}")
        
        num_samples = 10
        preds_sample_batch = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch,
                                            X_intv_batch, T_intv_batch, adjacency_matrix_batch,
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
                                                X_intv_batch, T_intv_batch, Y_intv_batch,
                                                adjacency_matrix_batch, batched=True)
        print(f"\n✓ Log-likelihood shape: {log_probs_batch.shape} (expected: ({B}, {M}))")
        assert log_probs_batch.shape == (B, M), f"Expected shape ({B}, {M}), got {log_probs_batch.shape}"
        print(f"  Sample log-likelihoods [0]: {log_probs_batch[0, :3]}")
        
        # Test alias method
        log_probs_alias_batch = wrapper.predict_log_likelihood(X_obs_batch, T_obs_batch, Y_obs_batch,
                                                               X_intv_batch, T_intv_batch, Y_intv_batch,
                                                               adjacency_matrix_batch, batched=True)
        print(f"\n✓ predict_log_likelihood shape: {log_probs_alias_batch.shape}")
        assert np.allclose(log_probs_batch, log_probs_alias_batch), "log_likelihood and predict_log_likelihood should return the same values"
        
        # Test negative log-likelihood
        nll_batch = wrapper.predict_negative_log_likelihood(X_obs_batch, T_obs_batch, Y_obs_batch,
                                                           X_intv_batch, T_intv_batch, Y_intv_batch,
                                                           adjacency_matrix_batch, batched=True)
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
    
    # Test shape validation errors
    print("\n" + "="*80)
    print("TEST 3: Shape Validation")
    print("="*80)
    
    try:
        # Should fail: 2D adjacency matrix with batched=True
        wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch, X_intv_batch, T_intv_batch,
                       adjacency_matrix, prediction_type="mean", batched=True)
        print("\n✗ Should have raised ValueError for 2D adjacency matrix with batched=True")
        sys.exit(1)
    except ValueError as e:
        print(f"\n✓ Correctly caught error for 2D adjacency matrix with batched=True: {e}")
    
    try:
        # Should fail: 3D adjacency matrix with batched=False
        wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, adjacency_matrix_batch,
                       prediction_type="mean", batched=False)
        print("\n✗ Should have raised ValueError for 3D adjacency matrix with batched=False")
        sys.exit(1)
    except ValueError as e:
        print(f"\n✓ Correctly caught error for 3D adjacency matrix with batched=False: {e}")
    
    print("\n✓ All validation tests passed!")
    
    # Test consistency between batched and non-batched modes
    print("\n" + "="*80)
    print("TEST 4: Batched vs Non-Batched Consistency")
    print("="*80)
    print("Verifying that batched predictions match multiple non-batched calls...")
    
    # Use the same random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run batched prediction once
    preds_mode_batched = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch,
                                        X_intv_batch, T_intv_batch, adjacency_matrix_batch,
                                        prediction_type="mode", batched=True)
    
    preds_mean_batched = wrapper.predict(X_obs_batch, T_obs_batch, Y_obs_batch,
                                        X_intv_batch, T_intv_batch, adjacency_matrix_batch,
                                        prediction_type="mean", batched=True)
    
    log_probs_batched = wrapper.log_likelihood(X_obs_batch, T_obs_batch, Y_obs_batch,
                                               X_intv_batch, T_intv_batch, Y_intv_batch,
                                               adjacency_matrix_batch, batched=True)
    
    # Run non-batched predictions B times
    preds_mode_nonbatched = []
    preds_mean_nonbatched = []
    log_probs_nonbatched = []
    
    for i in range(B):
        mode_i = wrapper.predict(X_obs_batch[i], T_obs_batch[i], Y_obs_batch[i],
                                X_intv_batch[i], T_intv_batch[i], adjacency_matrix_batch[i],
                                prediction_type="mode", batched=False)
        preds_mode_nonbatched.append(mode_i)
        
        mean_i = wrapper.predict(X_obs_batch[i], T_obs_batch[i], Y_obs_batch[i],
                                X_intv_batch[i], T_intv_batch[i], adjacency_matrix_batch[i],
                                prediction_type="mean", batched=False)
        preds_mean_nonbatched.append(mean_i)
        
        log_prob_i = wrapper.log_likelihood(X_obs_batch[i], T_obs_batch[i], Y_obs_batch[i],
                                           X_intv_batch[i], T_intv_batch[i], Y_intv_batch[i],
                                           adjacency_matrix_batch[i], batched=False)
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
    print("  ✓ Shape validation for batched flag")
    print("  ✓ Batched vs non-batched consistency")
    print("="*80)
