"""
Sklearn-like wrapper for InterventionalPFN with ensemble and adaptive clustering support.

This module provides a scikit-learn-style interface for InterventionalPFN, inheriting
most functionality from SimplePFNSklearn and adapting it for interventional causal data.

The InterventionalPFN architecture includes modern improvements:
- SwiGLU activation instead of GELU for better performance
- Pre-layer normalization for improved training stability
- Separate train/test attention without masking (memory efficient)
- Optional attention sinks for stability

Key differences from SimplePFNSklearn:
- predict() expects (X_obs, T_obs, Y_obs, X_intv, T_intv) instead of (X_train, y_train, X_test)
- All ensemble/clustering logic is inherited and adapted for interventional format
- Supports same features: ensemble, adaptive clustering, test batching, BarDistribution

Usage:
    # Basic usage
    wrapper = InterventionalPFNSklearn(config_path="config.yaml", checkpoint_path="model.pt")
    wrapper.load()
    preds = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type="mode")
    
    # Ensemble usage
    wrapper = InterventionalPFNSklearn(config_path="config.yaml", checkpoint_path="model.pt",
                                       n_estimators=5)
    wrapper.load()
    wrapper.fit(X_obs, T_obs, Y_obs)  # Fits ensemble preprocessors
    preds = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type="mode")
    
    # With clustering
    wrapper = InterventionalPFNSklearn(config_path="config.yaml", checkpoint_path="model.pt",
                                       max_n_train=100)
    wrapper.load()
    preds = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type="mode")
"""

from __future__ import annotations
from typing import Optional, Any, Union, Literal, List
import numpy as np
import torch
from pathlib import Path
import sys

# Robust import - try without 'src.' prefix first, then with 'src.' prefix
try:
    from models.SimplePFN_sklearn import SimplePFNSklearn
    from models.InterventionalPFN import InterventionalPFN
except Exception:
    try:
        from src.models.SimplePFN_sklearn import SimplePFNSklearn
        from src.models.InterventionalPFN import InterventionalPFN
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from src.models.SimplePFN_sklearn import SimplePFNSklearn
        from src.models.InterventionalPFN import InterventionalPFN


class InterventionalPFNSklearn(SimplePFNSklearn):
    """
    Sklearn-like wrapper for InterventionalPFN with full ensemble and clustering support.
    
    Inherits all functionality from SimplePFNSklearn and adapts it for interventional data:
    - Ensemble diversity through feature permutations
    - Adaptive hierarchical clustering for large datasets
    - Test batching for memory efficiency
    - BarDistribution support for distributional predictions
    
    The main difference is that predict() expects interventional data format:
    (X_obs, T_obs, Y_obs, X_intv, T_intv) instead of (X_train, y_train, X_test).
    
    The InterventionalPFN architecture now includes:
    - SwiGLU activation for improved performance
    - Pre-layer normalization for training stability
    - Separate train/test attention (no masking, memory efficient)
    - Optional attention sinks for stability
    
    Parameters:
        Same as SimplePFNSklearn, but model will be InterventionalPFN instead of SimplePFN.
        Additional model parameters available via config:
        - use_same_row_mlp: Whether to share row MLP between train and test (default: True)
        - n_sample_attention_sink_rows: Number of learnable sink rows (default: 0)
        - n_feature_attention_sink_cols: Number of learnable sink columns (default: 0)
    """
    
    def load(self, override_kwargs: Optional[dict[str, Any]] = None) -> "InterventionalPFNSklearn":
        """
        Load config, build InterventionalPFN model, and load checkpoint.
        
        This overrides SimplePFNSklearn.load() to instantiate InterventionalPFN
        instead of SimplePFNRegressor.
        """
        # Always print debug info if verbose
        print(f"[InterventionalPFNSklearn] Loading with config_path: {self.config_path}")
        print(f"[InterventionalPFNSklearn] verbose={self.verbose}")
        
        # Call parent's config loading logic
        if self.config_path:
            import yaml
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            
            print(f"[InterventionalPFNSklearn] Config loaded. Top-level keys: {list(cfg.keys())}")
            
            # Extract model kwargs from config
            # Note: Config uses "model_config" and each parameter has a nested "value" key
            self.model_kwargs = {}
            if "model_config" in cfg:
                model_cfg = cfg["model_config"]
                
                # Helper function to get value from config (handles both direct values and nested "value" keys)
                def get_config_value(config_dict, key, default):
                    if key not in config_dict:
                        return default
                    val = config_dict[key]
                    # If it's a dict with "value" key, extract the value
                    if isinstance(val, dict) and "value" in val:
                        return val["value"]
                    return val
                
                self.model_kwargs = {
                    "num_features": get_config_value(model_cfg, "num_features", None),
                    "d_model": get_config_value(model_cfg, "d_model", 256),
                    "depth": get_config_value(model_cfg, "depth", 8),
                    "heads_feat": get_config_value(model_cfg, "heads_feat", 8),
                    "heads_samp": get_config_value(model_cfg, "heads_samp", 8),
                    "dropout": get_config_value(model_cfg, "dropout", 0.0),
                    "output_dim": get_config_value(model_cfg, "output_dim", 1),
                    "hidden_mult": get_config_value(model_cfg, "hidden_mult", 4),
                    "normalize_features": get_config_value(model_cfg, "normalize_features", True),
                    "use_same_row_mlp": get_config_value(model_cfg, "use_same_row_mlp", True),
                    "n_sample_attention_sink_rows": get_config_value(model_cfg, "n_sample_attention_sink_rows", 0),
                    "n_feature_attention_sink_cols": get_config_value(model_cfg, "n_feature_attention_sink_cols", 0),
                }
                
                if self.verbose:
                    print(f"[InterventionalPFNSklearn] Loaded model_kwargs from config: {self.model_kwargs}")
                
                # Check for BarDistribution in config
                use_bar = get_config_value(model_cfg, "use_bar_distribution", False)
                if use_bar:
                    self.use_bar_distribution = True
                    try:
                        from Losses.BarDistribution import BarDistribution
                    except ImportError:
                        from src.Losses.BarDistribution import BarDistribution
                    self.bar_distribution = BarDistribution(
                        num_bars=get_config_value(model_cfg, "num_bars", 50),
                        min_width=float(get_config_value(model_cfg, "min_width", 0.01)),
                        scale_floor=float(get_config_value(model_cfg, "scale_floor", 0.1)),
                        max_fit_items=get_config_value(model_cfg, "max_fit_items", 10000),
                        log_prob_clip_min=get_config_value(model_cfg, "log_prob_clip_min", -10.0),
                        log_prob_clip_max=get_config_value(model_cfg, "log_prob_clip_max", 10.0),
                    )
                    # Update output_dim to match bar distribution parameters
                    self.model_kwargs["output_dim"] = self.bar_distribution.num_params
                    if self.verbose:
                        print(f"[InterventionalPFNSklearn] BarDistribution enabled with {self.bar_distribution.num_bars} bars")
                        print(f"[InterventionalPFNSklearn] Model output_dim set to {self.model_kwargs['output_dim']}")
            else:
                if self.verbose:
                    print(f"[InterventionalPFNSklearn] WARNING: 'model_config' not found in config file!")
                # Set defaults if model_config is missing
                self.model_kwargs = {
                    "num_features": None,
                    "d_model": 256,
                    "depth": 8,
                    "heads_feat": 8,
                    "heads_samp": 8,
                    "dropout": 0.0,
                    "output_dim": 1,
                    "hidden_mult": 4,
                    "normalize_features": True,
                    "use_same_row_mlp": True,
                    "n_sample_attention_sink_rows": 0,
                    "n_feature_attention_sink_cols": 0,
                }
        else:
            self.model_kwargs = {
                "num_features": override_kwargs.get("num_features") if override_kwargs else None,
                "d_model": 256,
                "depth": 8,
                "heads_feat": 8,
                "heads_samp": 8,
                "dropout": 0.0,
                "output_dim": 1,
                "hidden_mult": 4,
                "normalize_features": True,
                "use_same_row_mlp": True,
                "n_sample_attention_sink_rows": 0,
                "n_feature_attention_sink_cols": 0,
            }

        # Apply overrides
        if override_kwargs:
            self.model_kwargs.update(override_kwargs)

        # Sanity check
        if not self.model_kwargs.get("num_features"):
            raise ValueError("[InterventionalPFNSklearn] 'num_features' must be specified in config or override_kwargs")

        # Build InterventionalPFN model (not SimplePFNRegressor)
        if self.verbose:
            print(f"[InterventionalPFNSklearn] Building InterventionalPFN with kwargs: {self.model_kwargs}")
        
        self.model = InterventionalPFN(**self.model_kwargs).to(self.device)

        # Load checkpoint using parent's logic
        if self.checkpoint_path:
            if self.verbose:
                print(f"[InterventionalPFNSklearn] Loading checkpoint from {self.checkpoint_path}")
            
            state = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if "model" in state:
                model_state = state["model"]
            elif "model_state_dict" in state:
                model_state = state["model_state_dict"]
            else:
                model_state = state
            
            # Load model weights
            self.model.load_state_dict(model_state, strict=True)
            
            # Load BarDistribution parameters if present
            if self.use_bar_distribution and "bar_distribution" in state:
                if self.bar_distribution is None:
                    # Create BarDistribution if it doesn't exist yet
                    try:
                        from Losses.BarDistribution import BarDistribution
                    except ImportError:
                        from src.Losses.BarDistribution import BarDistribution
                    bar_state = state["bar_distribution"]
                    self.bar_distribution = BarDistribution(
                        num_bars=bar_state.get("num_bars", 11),
                        min_width=bar_state.get("min_width", 1e-6),
                        scale_floor=bar_state.get("scale_floor", 1e-6),
                        device=self.device,
                        max_fit_items=bar_state.get("max_fit_items", None),
                        log_prob_clip_min=bar_state.get("log_prob_clip_min", -50.0),
                        log_prob_clip_max=bar_state.get("log_prob_clip_max", 50.0),
                    )
                self._load_bar_distribution_parameters(state["bar_distribution"])
                if self.verbose:
                    print(f"[InterventionalPFNSklearn] Loaded BarDistribution parameters from checkpoint")
            elif "bar_distribution" in state and not self.use_bar_distribution:
                # Checkpoint has bar_distribution but config doesn't - auto-enable it
                if self.verbose:
                    print(f"[InterventionalPFNSklearn] Found BarDistribution in checkpoint, enabling it")
                try:
                    from Losses.BarDistribution import BarDistribution
                except ImportError:
                    from src.Losses.BarDistribution import BarDistribution
                bar_state = state["bar_distribution"]
                self.use_bar_distribution = True
                self.bar_distribution = BarDistribution(
                    num_bars=bar_state.get("num_bars", 11),
                    min_width=bar_state.get("min_width", 1e-6),
                    scale_floor=bar_state.get("scale_floor", 1e-6),
                    device=self.device,
                    max_fit_items=bar_state.get("max_fit_items", None),
                    log_prob_clip_min=bar_state.get("log_prob_clip_min", -50.0),
                    log_prob_clip_max=bar_state.get("log_prob_clip_max", 50.0),
                )
                self._load_bar_distribution_parameters(state["bar_distribution"])
                if self.verbose:
                    print(f"[InterventionalPFNSklearn] Loaded BarDistribution parameters from checkpoint")
            
            if self.verbose:
                print(f"[InterventionalPFNSklearn] Model loaded successfully")

        return self
    
    def fit(self, X_obs: Any = None, T_obs: Any = None, Y_obs: Any = None, **kwargs) -> "InterventionalPFNSklearn":
        """
        Fit ensemble preprocessors if ensemble is enabled.
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            
        Returns:
            self for method chaining
            
        Notes:
            - For ensemble: Fits preprocessors on concatenated [X_obs, T_obs] features
            - For clustering: No fitting needed, clustering happens during predict()
        """
        if self.use_ensemble:
            if X_obs is None or T_obs is None or Y_obs is None:
                raise ValueError("[InterventionalPFNSklearn] fit() requires X_obs, T_obs, Y_obs when n_estimators > 1")
            
            # Convert to numpy
            if hasattr(X_obs, "values"):
                X_obs = X_obs.values
            if hasattr(T_obs, "values"):
                T_obs = T_obs.values
            if hasattr(Y_obs, "values"):
                Y_obs = Y_obs.values
            
            X_obs = np.asarray(X_obs, dtype=np.float32)
            T_obs = np.asarray(T_obs, dtype=np.float32)
            Y_obs = np.asarray(Y_obs, dtype=np.float32)
            
            # Ensure T_obs is 2D
            if T_obs.ndim == 1:
                T_obs = T_obs.reshape(-1, 1)
            
            # Concatenate X_obs and T_obs for ensemble preprocessing
            X_train_combined = np.hstack([X_obs, T_obs])  # (N, L+1)
            
            # Fit ensemble preprocessor
            if self.verbose:
                print(f"[InterventionalPFNSklearn] Fitting ensemble preprocessor on {X_train_combined.shape} training data")
            
            from src.models.SimplePFN_sklearn import EnsemblePreprocessor
            self.ensemble_preprocessor = EnsemblePreprocessor(
                n_estimators=self.n_estimators,
                norm_methods=self.norm_methods,
                outlier_strategies=self.outlier_strategies,
                feat_shuffle_method=self.feat_shuffle_method,
                random_state=self.random_state,
            )
            self.ensemble_preprocessor.fit(X_train_combined, Y_obs)
            
            if self.verbose:
                print(f"[InterventionalPFNSklearn] Ensemble preprocessor fitted")
        else:
            if self.verbose:
                print("[InterventionalPFNSklearn] No ensemble, fit() is a no-op")
        
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
        aggregate: Literal["mean", "median", "none"] = "mean",
    ) -> np.ndarray:
        """
        Make predictions for interventional test data.
        
        Supports:
        - Single model prediction (n_estimators=1, max_n_train=None)
        - Ensemble prediction (n_estimators>1)
        - Adaptive clustering (max_n_train set)
        - Test batching (max_n_test set)
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            prediction_type: Type of prediction ("point", "mode", "mean", "sample")
            num_samples: Number of samples for prediction_type="sample"
            aggregate: How to aggregate ensemble predictions ("mean", "median", "none")
            
        Returns:
            Predictions array of shape (M,) or (n_estimators, M) or with sample dimension
        """
        if self.model is None:
            raise RuntimeError("[InterventionalPFNSklearn] Model not loaded. Call load() first.")
        
        # Validate prediction type
        if prediction_type in ["mode", "mean", "sample"] and not self.use_bar_distribution:
            raise ValueError(f"[InterventionalPFNSklearn] prediction_type='{prediction_type}' requires BarDistribution")
        
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
            T_obs = T_obs.reshape(-1, 1)
        if T_intv.ndim == 1:
            T_intv = T_intv.reshape(-1, 1)
        if Y_obs.ndim == 2 and Y_obs.shape[1] == 1:
            Y_obs = Y_obs.squeeze(1)
        
        # Auto-fit ensemble if needed
        if self.use_ensemble and self.ensemble_preprocessor is None:
            if self.verbose:
                print("[InterventionalPFNSklearn] Auto-fitting ensemble preprocessor")
            self.fit(X_obs, T_obs, Y_obs)
        
        # Route to appropriate prediction method
        if self.max_n_train is not None:
            return self._predict_with_adaptive_clustering_intv(
                X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type, num_samples, aggregate
            )
        
        if self.use_ensemble:
            return self._predict_ensemble_intv(
                X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type, num_samples, aggregate
            )
        
        # Single model prediction
        return self._predict_single_intv(X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type, num_samples)
    
    def _predict_single_intv(
        self,
        X_obs: np.ndarray,
        T_obs: np.ndarray,
        Y_obs: np.ndarray,
        X_intv: np.ndarray,
        T_intv: np.ndarray,
        prediction_type: str,
        num_samples: int,
    ) -> np.ndarray:
        """Single model prediction for interventional data."""
        # Handle test batching if needed
        if self.max_n_test is not None and X_intv.shape[0] > self.max_n_test:
            return self._split_test_data_intv(
                X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type, num_samples,
                lambda X_intv_chunk, T_intv_chunk: self._predict_single_impl_intv(
                    X_obs, T_obs, Y_obs, X_intv_chunk, T_intv_chunk, prediction_type, num_samples
                )
            )
        
        return self._predict_single_impl_intv(X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type, num_samples)
    
    def _predict_single_impl_intv(
        self,
        X_obs: np.ndarray,
        T_obs: np.ndarray,
        Y_obs: np.ndarray,
        X_intv: np.ndarray,
        T_intv: np.ndarray,
        prediction_type: str,
        num_samples: int,
    ) -> np.ndarray:
        """Core single model prediction implementation."""
        # Convert to torch tensors
        X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
        T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
        Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
        X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
        T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
            raw_preds = out["predictions"]  # (1, M) or (1, M, output_dim)
        
        # Process predictions based on type
        if not self.use_bar_distribution or prediction_type == "point":
            # Raw output
            return raw_preds.squeeze(0).cpu().numpy()  # (M,)
        
        # Check if BarDistribution needs to be fitted
        if self.bar_distribution.centers is None:
            if self.verbose:
                print("[InterventionalPFNSklearn] BarDistribution not fitted. Fitting on provided observational data...")
            # Create a simple dataloader from the observational data
            # Concatenate X_obs and T_obs for the full feature set
            X_obs_full = torch.cat([X_obs_t, T_obs_t], dim=2)  # (1, N, L+1)
            Y_obs_full = Y_obs_t.unsqueeze(2)  # (1, N, 1)
            # Use observational data for both train and test in fitting
            simple_dataloader = [(X_obs_full, Y_obs_full, X_obs_full, Y_obs_full)]
            self.bar_distribution.fit(simple_dataloader)
            if self.verbose:
                print("[InterventionalPFNSklearn] BarDistribution fitted successfully.")
        
        # BarDistribution predictions
        if prediction_type == "mode":
            preds = self.bar_distribution.mode(raw_preds)
            return preds.squeeze(0).cpu().numpy()  # (M,)
        elif prediction_type == "mean":
            preds = self.bar_distribution.mean(raw_preds)
            return preds.squeeze(0).cpu().numpy()  # (M,)
        elif prediction_type == "sample":
            samples = self.bar_distribution.sample(raw_preds, num_samples=num_samples)
            # BarDistribution.sample returns (B, num_samples, M), squeeze(0) -> (num_samples, M)
            return samples.squeeze(0).cpu().numpy()  # (num_samples, M)
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
    
    def _predict_ensemble_intv(
        self,
        X_obs: np.ndarray,
        T_obs: np.ndarray,
        Y_obs: np.ndarray,
        X_intv: np.ndarray,
        T_intv: np.ndarray,
        prediction_type: str,
        num_samples: int,
        aggregate: str,
    ) -> np.ndarray:
        """Ensemble prediction for interventional data."""
        # Handle test batching if needed
        if self.max_n_test is not None and X_intv.shape[0] > self.max_n_test:
            return self._split_test_data_intv(
                X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type, num_samples,
                lambda X_intv_chunk, T_intv_chunk: self._predict_ensemble_impl_intv(
                    X_obs, T_obs, Y_obs, X_intv_chunk, T_intv_chunk, prediction_type, num_samples, aggregate
                )
            )
        
        return self._predict_ensemble_impl_intv(X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type, num_samples, aggregate)
    
    def _predict_ensemble_impl_intv(
        self,
        X_obs: np.ndarray,
        T_obs: np.ndarray,
        Y_obs: np.ndarray,
        X_intv: np.ndarray,
        T_intv: np.ndarray,
        prediction_type: str,
        num_samples: int,
        aggregate: str,
    ) -> np.ndarray:
        """Core ensemble prediction implementation."""
        # Combine X and T for preprocessing
        X_obs_combined = np.hstack([X_obs, T_obs])  # (N, L+1)
        X_intv_combined = np.hstack([X_intv, T_intv])  # (M, L+1)
        
        # Get ensemble variants
        ensemble_data = self.ensemble_preprocessor.transform(X_obs_combined, Y_obs, X_intv_combined)
        
        # Collect predictions from all ensemble members
        all_preds = []
        
        for variant_idx in range(self.n_estimators):
            X_obs_var, Y_obs_var, X_intv_var = ensemble_data[variant_idx]
            
            # Split back into X and T
            X_obs_v = X_obs_var[:, :-1]  # (N, L)
            T_obs_v = X_obs_var[:, -1:]  # (N, 1)
            X_intv_v = X_intv_var[:, :-1]  # (M, L)
            T_intv_v = X_intv_var[:, -1:]  # (M, 1)
            
            # Make prediction for this variant
            pred = self._predict_single_impl_intv(
                X_obs_v, T_obs_v, Y_obs_var, X_intv_v, T_intv_v, prediction_type, num_samples
            )
            all_preds.append(pred)
        
        # Stack predictions
        all_preds = np.stack(all_preds, axis=0)  # (n_estimators, M) or (n_estimators, M, num_samples)
        
        # Aggregate
        if aggregate == "none":
            return all_preds
        elif aggregate == "mean":
            return np.mean(all_preds, axis=0)
        elif aggregate == "median":
            return np.median(all_preds, axis=0)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")
    
    def _predict_with_adaptive_clustering_intv(
        self,
        X_obs: np.ndarray,
        T_obs: np.ndarray,
        Y_obs: np.ndarray,
        X_intv: np.ndarray,
        T_intv: np.ndarray,
        prediction_type: str,
        num_samples: int,
        aggregate: str,
    ) -> np.ndarray:
        """Adaptive clustering prediction for interventional data."""
        # Combine X_obs and T_obs for clustering
        X_obs_combined = np.hstack([X_obs, T_obs])  # (N, L+1)
        
        # Perform hierarchical clustering
        cluster_assignments = self._hierarchical_cluster(X_obs_combined, Y_obs)
        
        # Combine X_intv and T_intv for cluster assignment
        X_intv_combined = np.hstack([X_intv, T_intv])  # (M, L+1)
        
        # Assign test samples to clusters
        test_cluster_assignments = self._assign_to_clusters(X_intv_combined, X_obs_combined, cluster_assignments)
        
        # Make predictions for each cluster
        M = X_intv.shape[0]
        # For sampling, predictions shape is (num_samples, M), otherwise (M,)
        predictions = np.zeros(M if prediction_type != "sample" else (num_samples, M), dtype=np.float32)
        
        for cluster_id in np.unique(cluster_assignments):
            # Get train samples in this cluster
            train_mask = cluster_assignments == cluster_id
            X_obs_c = X_obs[train_mask]
            T_obs_c = T_obs[train_mask]
            Y_obs_c = Y_obs[train_mask]
            
            # Get test samples assigned to this cluster
            test_mask = test_cluster_assignments == cluster_id
            if not test_mask.any():
                continue
            
            X_intv_c = X_intv[test_mask]
            T_intv_c = T_intv[test_mask]
            
            # Make prediction for this cluster
            if self.use_ensemble:
                cluster_pred = self._predict_ensemble_impl_intv(
                    X_obs_c, T_obs_c, Y_obs_c, X_intv_c, T_intv_c, prediction_type, num_samples, aggregate
                )
            else:
                cluster_pred = self._predict_single_impl_intv(
                    X_obs_c, T_obs_c, Y_obs_c, X_intv_c, T_intv_c, prediction_type, num_samples
                )
            
            # For sampling, cluster_pred has shape (num_samples, M_cluster)
            # For other types, cluster_pred has shape (M_cluster,)
            if prediction_type == "sample":
                predictions[:, test_mask] = cluster_pred
            else:
                predictions[test_mask] = cluster_pred
        
        return predictions
    
    def _split_test_data_intv(
        self,
        X_obs: np.ndarray,
        T_obs: np.ndarray,
        Y_obs: np.ndarray,
        X_intv: np.ndarray,
        T_intv: np.ndarray,
        adjacency_matrix: Optional[np.ndarray],
        prediction_type: str,
        num_samples: int,
        predict_fn,
    ) -> np.ndarray:
        """Split test data into chunks and process separately."""
        M = X_intv.shape[0]
        all_preds = []
        
        for start_idx in range(0, M, self.max_n_test):
            end_idx = min(start_idx + self.max_n_test, M)
            X_intv_chunk = X_intv[start_idx:end_idx]
            T_intv_chunk = T_intv[start_idx:end_idx]
            
            chunk_pred = predict_fn(X_intv_chunk, T_intv_chunk)
            all_preds.append(chunk_pred)
        
        return np.concatenate(all_preds, axis=0)
    
    def predict_entropy(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        num_samples: int = 1000,
    ) -> np.ndarray:
        """
        Compute the entropy of the predictive distribution for each test sample.
        
        Entropy is estimated using Monte Carlo sampling from the BarDistribution:
        H[p(y)] ≈ -E[log p(y)] where y ~ p(y)
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            num_samples: Number of MC samples for entropy estimation (default: 1000)
            
        Returns:
            Entropy values of shape (M,)
            
        Raises:
            ValueError: If BarDistribution is not enabled
        """
        if not self.use_bar_distribution:
            raise ValueError("[InterventionalPFNSklearn] predict_entropy() requires BarDistribution")
        
        if self.model is None:
            raise RuntimeError("[InterventionalPFNSklearn] Model not loaded. Call load() first.")
        
        # Convert to numpy and ensure correct shapes
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
            T_obs = T_obs.reshape(-1, 1)
        if T_intv.ndim == 1:
            T_intv = T_intv.reshape(-1, 1)
        if Y_obs.ndim == 2 and Y_obs.shape[1] == 1:
            Y_obs = Y_obs.squeeze(1)
        
        # Convert to torch tensors
        X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
        T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
        Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
        X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
        T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
        
        # Forward pass to get distribution parameters
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
            raw_preds = out["predictions"]  # (1, M, output_dim)
            
            # Sample from the distribution
            samples = self.bar_distribution.sample(raw_preds, num_samples=num_samples)  # (1, num_samples, M)
            
            # Compute log probabilities of the samples
            # Need to reshape for average_log_prob: expects (B, M) for y
            M = samples.shape[2]
            log_probs = []
            
            for i in range(num_samples):
                y_sample = samples[:, i, :]  # (1, M)
                log_prob = self.bar_distribution.average_log_prob(raw_preds, y_sample)  # (1,)
                log_probs.append(log_prob)
            
            # Average log probabilities to estimate entropy
            # H[p(y)] ≈ -E[log p(y)]
            log_probs_stacked = torch.stack(log_probs, dim=0)  # (num_samples, 1)
            entropy = -log_probs_stacked.mean(dim=0)  # (1,)
            
            # Since we have a single batch, expand to per-test-sample entropy
            # For a more accurate per-sample entropy, we need to compute it separately
            # Let's compute it properly per test sample
            entropies = []
            for m in range(M):
                y_samples_m = samples[0, :, m]  # (num_samples,)
                log_probs_m = []
                
                for sample_val in y_samples_m:
                    # Create a batch with just this test sample
                    y_single = sample_val.unsqueeze(0).unsqueeze(0)  # (1, 1)
                    pred_single = raw_preds[:, m:m+1, :]  # (1, 1, output_dim)
                    log_prob = self.bar_distribution.average_log_prob(pred_single, y_single)
                    log_probs_m.append(log_prob.item())
                
                entropy_m = -np.mean(log_probs_m)
                entropies.append(entropy_m)
            
            return np.array(entropies, dtype=np.float32)
    
    def predict_variance(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        num_samples: int = 1000,
    ) -> np.ndarray:
        """
        Compute the variance of the predictive distribution for each test sample.
        
        Variance is estimated using Monte Carlo sampling from the BarDistribution:
        Var[y] = E[y²] - E[y]²
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            num_samples: Number of MC samples for variance estimation (default: 1000)
            
        Returns:
            Variance values of shape (M,)
            
        Raises:
            ValueError: If BarDistribution is not enabled
        """
        if not self.use_bar_distribution:
            raise ValueError("[InterventionalPFNSklearn] predict_variance() requires BarDistribution")
        
        if self.model is None:
            raise RuntimeError("[InterventionalPFNSklearn] Model not loaded. Call load() first.")
        
        # Convert to numpy and ensure correct shapes
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
            T_obs = T_obs.reshape(-1, 1)
        if T_intv.ndim == 1:
            T_intv = T_intv.reshape(-1, 1)
        if Y_obs.ndim == 2 and Y_obs.shape[1] == 1:
            Y_obs = Y_obs.squeeze(1)
        
        # Convert to torch tensors
        X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
        T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
        Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
        X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
        T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
        
        # Forward pass to get distribution parameters
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
            raw_preds = out["predictions"]  # (1, M, output_dim)
            
            # Sample from the distribution
            samples = self.bar_distribution.sample(raw_preds, num_samples=num_samples)  # (1, num_samples, M)
            
            # Compute variance per test sample: Var[y] = E[y²] - E[y]²
            samples_np = samples.cpu().numpy()[0]  # (num_samples, M)
            
            mean = np.mean(samples_np, axis=0)  # (M,)
            mean_of_squares = np.mean(samples_np ** 2, axis=0)  # (M,)
            variance = mean_of_squares - mean ** 2  # (M,)
            
            return variance.astype(np.float32)

    def predict_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        aggregate: Literal["mean", "median", "none"] = "mean",
    ) -> np.ndarray:
        """
        Compute the log-likelihood of interventional targets under the model's predictive distribution.
        
        Evaluates log p(Y_intv | X_obs, T_obs, Y_obs, X_intv, T_intv) for each test sample.
        Higher log-likelihood indicates better fit of the predictive distribution to the observed values.
        Requires BarDistribution to be enabled.
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            Y_intv: Interventional targets to evaluate (M,) or (M, 1)
            aggregate: How to aggregate ensemble predictions if n_estimators > 1
                - "mean": Average log-likelihood across ensemble members (default)
                - "median": Median log-likelihood across ensemble members
                - "none": Return all log-likelihoods with shape (n_estimators, M)
                
        Returns:
            Log-likelihood for each test sample (M,) or (n_estimators, M) if aggregate="none"
            
        Raises:
            ValueError: If BarDistribution is not enabled
            ValueError: If Y_intv shape doesn't match X_intv
        """
        if not self.use_bar_distribution:
            raise ValueError("[InterventionalPFNSklearn] predict_log_likelihood() requires BarDistribution")
        
        if self.model is None:
            raise RuntimeError("[InterventionalPFNSklearn] Model not loaded. Call load() first.")
        
        # Convert inputs
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
        
        X_obs_np = np.asarray(X_obs, dtype=np.float32)
        T_obs_np = np.asarray(T_obs, dtype=np.float32)
        Y_obs_np = np.asarray(Y_obs, dtype=np.float32)
        X_intv_np = np.asarray(X_intv, dtype=np.float32)
        T_intv_np = np.asarray(T_intv, dtype=np.float32)
        Y_intv_np = np.asarray(Y_intv, dtype=np.float32)
        
        # Ensure T arrays are 2D
        if T_obs_np.ndim == 1:
            T_obs_np = T_obs_np.reshape(-1, 1)
        if T_intv_np.ndim == 1:
            T_intv_np = T_intv_np.reshape(-1, 1)
        
        # Ensure Y_intv is 1D
        if Y_intv_np.ndim == 2 and Y_intv_np.shape[1] == 1:
            Y_intv_np = Y_intv_np.squeeze(1)
        elif Y_intv_np.ndim > 1:
            raise ValueError(f"Y_intv must be (M,) or (M, 1), got shape {Y_intv_np.shape}")
        
        # Ensure Y_obs is 1D
        if Y_obs_np.ndim == 2 and Y_obs_np.shape[1] == 1:
            Y_obs_np = Y_obs_np.squeeze(1)
        
        # Check shapes match
        if X_intv_np.shape[0] != Y_intv_np.shape[0]:
            raise ValueError(f"X_intv has {X_intv_np.shape[0]} samples but Y_intv has {Y_intv_np.shape[0]}")
        
        # Handle ensemble vs single model
        if self.use_ensemble and self.n_estimators > 1:
            # Get ensemble data variants
            if self.ensemble_preprocessor is None or not self.ensemble_preprocessor.fitted:
                raise RuntimeError("[InterventionalPFNSklearn] Ensemble preprocessors not fitted. Call fit() before predict().")
            
            # Combine X and T for preprocessing
            X_obs_combined = np.hstack([X_obs_np, T_obs_np])  # (N, L+1)
            X_intv_combined = np.hstack([X_intv_np, T_intv_np])  # (M, L+1)
            
            ensemble_data = self.ensemble_preprocessor.transform(X_obs_combined, Y_obs_np, X_intv_combined)
            
            log_likelihoods = np.zeros((self.n_estimators, X_intv_np.shape[0]))
            
            for ens_idx in range(self.n_estimators):
                X_obs_var, Y_obs_var, X_intv_var = ensemble_data[ens_idx]
                
                # Split back into X and T
                X_obs_v = X_obs_var[:, :-1]  # (N, L)
                T_obs_v = X_obs_var[:, -1:]  # (N, 1)
                X_intv_v = X_intv_var[:, :-1]  # (M, L)
                T_intv_v = X_intv_var[:, -1:]  # (M, 1)
                
                # Get model predictions (distribution parameters)
                with torch.no_grad():
                    X_obs_t = torch.from_numpy(X_obs_v).unsqueeze(0).to(self.device)
                    T_obs_t = torch.from_numpy(T_obs_v).unsqueeze(0).to(self.device)
                    Y_obs_t = torch.from_numpy(Y_obs_var).unsqueeze(0).to(self.device)
                    X_intv_t = torch.from_numpy(X_intv_v).unsqueeze(0).to(self.device)
                    T_intv_t = torch.from_numpy(T_intv_v).unsqueeze(0).to(self.device)
                    
                    out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
                    pred_raw = out["predictions"]  # (1, M, num_params)
                    
                    # Compute log-likelihood using BarDistribution
                    Y_intv_t = torch.from_numpy(Y_intv_np).unsqueeze(0).to(self.device)  # (1, M)
                    
                    # Get per-sample log-likelihoods
                    logpdf = self.bar_distribution._logpdf_from_pred(pred_raw, Y_intv_t)  # (1, M)
                    log_likelihoods[ens_idx, :] = logpdf.squeeze(0).cpu().numpy()
            
            # Aggregate across ensemble members
            if aggregate == "none":
                return log_likelihoods  # (n_estimators, M)
            elif aggregate == "mean":
                return np.mean(log_likelihoods, axis=0)  # (M,)
            elif aggregate == "median":
                return np.median(log_likelihoods, axis=0)  # (M,)
            else:
                raise ValueError(f"Unknown aggregate method: {aggregate}")
        
        else:
            # Single model prediction
            with torch.no_grad():
                X_obs_t = torch.from_numpy(X_obs_np).unsqueeze(0).to(self.device)
                T_obs_t = torch.from_numpy(T_obs_np).unsqueeze(0).to(self.device)
                Y_obs_t = torch.from_numpy(Y_obs_np).unsqueeze(0).to(self.device)
                X_intv_t = torch.from_numpy(X_intv_np).unsqueeze(0).to(self.device)
                T_intv_t = torch.from_numpy(T_intv_np).unsqueeze(0).to(self.device)
                
                out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
                pred_raw = out["predictions"]  # (1, M, num_params)
                
                # Compute log-likelihood using BarDistribution
                Y_intv_t = torch.from_numpy(Y_intv_np).unsqueeze(0).to(self.device)  # (1, M)
                
                # Get per-sample log-likelihoods
                logpdf = self.bar_distribution._logpdf_from_pred(pred_raw, Y_intv_t)  # (1, M)
                
                return logpdf.squeeze(0).cpu().numpy()  # (M,)

    def get_raw_predictions(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        aggregate: Literal["mean", "median", "none"] = "none",
    ) -> torch.Tensor:
        """
        Get raw model predictions (BarDistribution parameters) without aggregation.
        
        This method is useful for visualization and custom post-processing of the 
        predictive distribution. For BarDistribution models, returns the raw parameters
        that can be passed to BarDistribution methods like plot().
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            aggregate: How to aggregate ensemble predictions if n_estimators > 1
                - "none": Return all predictions with shape (n_estimators, M, num_params) (default)
                - "mean": Average parameters across ensemble members (M, num_params)
                - "median": Median parameters across ensemble members (M, num_params)
                
        Returns:
            Raw model predictions as torch.Tensor:
            - Single model: shape (M, num_params) where num_params = K+4 for BarDistribution
            - Ensemble with aggregate="none": shape (n_estimators, M, num_params)
            - Ensemble with aggregation: shape (M, num_params)
            
        Example:
            >>> # Get raw predictions for a single test point
            >>> raw_pred = model.get_raw_predictions(X_obs, T_obs, Y_obs, X_intv[[0]], T_intv[[0]])
            >>> # Plot the predictive distribution
            >>> bar_dist = model.bar_distribution
            >>> bar_dist.plot(raw_pred, idx=0, title="Predictive Distribution")
        """
        if self.model is None:
            raise RuntimeError("[InterventionalPFNSklearn] Model not loaded. Call load() first.")
        
        # Convert inputs
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
        
        X_obs_np = np.asarray(X_obs, dtype=np.float32)
        T_obs_np = np.asarray(T_obs, dtype=np.float32)
        Y_obs_np = np.asarray(Y_obs, dtype=np.float32)
        X_intv_np = np.asarray(X_intv, dtype=np.float32)
        T_intv_np = np.asarray(T_intv, dtype=np.float32)
        
        # Ensure T arrays are 2D
        if T_obs_np.ndim == 1:
            T_obs_np = T_obs_np.reshape(-1, 1)
        if T_intv_np.ndim == 1:
            T_intv_np = T_intv_np.reshape(-1, 1)
        
        # Ensure Y_obs is 1D
        if Y_obs_np.ndim == 2 and Y_obs_np.shape[1] == 1:
            Y_obs_np = Y_obs_np.squeeze(1)
        
        # Handle ensemble vs single model
        if self.use_ensemble and self.n_estimators > 1:
            # Get ensemble data variants
            if self.ensemble_preprocessor is None or not self.ensemble_preprocessor.fitted:
                raise RuntimeError("[InterventionalPFNSklearn] Ensemble preprocessors not fitted. Call fit() before get_raw_predictions().")
            
            # Combine X and T for preprocessing
            X_obs_combined = np.hstack([X_obs_np, T_obs_np])  # (N, L+1)
            X_intv_combined = np.hstack([X_intv_np, T_intv_np])  # (M, L+1)
            
            ensemble_data = self.ensemble_preprocessor.transform(X_obs_combined, Y_obs_np, X_intv_combined)
            
            # Collect predictions from each ensemble member
            all_preds = []
            
            for ens_idx in range(self.n_estimators):
                X_obs_var, Y_obs_var, X_intv_var = ensemble_data[ens_idx]
                
                # Split back into X and T
                X_obs_v = X_obs_var[:, :-1]  # (N, L)
                T_obs_v = X_obs_var[:, -1:]  # (N, 1)
                X_intv_v = X_intv_var[:, :-1]  # (M, L)
                T_intv_v = X_intv_var[:, -1:]  # (M, 1)
                
                with torch.no_grad():
                    X_obs_t = torch.from_numpy(X_obs_v).unsqueeze(0).to(self.device)
                    T_obs_t = torch.from_numpy(T_obs_v).unsqueeze(0).to(self.device)
                    Y_obs_t = torch.from_numpy(Y_obs_var).unsqueeze(0).to(self.device)
                    X_intv_t = torch.from_numpy(X_intv_v).unsqueeze(0).to(self.device)
                    T_intv_t = torch.from_numpy(T_intv_v).unsqueeze(0).to(self.device)
                    
                    out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
                    pred_raw = out["predictions"]  # (1, M, num_params)
                    all_preds.append(pred_raw.squeeze(0))  # (M, num_params)
            
            # Stack ensemble predictions
            all_preds = torch.stack(all_preds, dim=0)  # (n_estimators, M, num_params)
            
            # Aggregate if requested
            if aggregate == "none":
                return all_preds
            elif aggregate == "mean":
                return torch.mean(all_preds, dim=0)
            elif aggregate == "median":
                return torch.median(all_preds, dim=0).values
            else:
                raise ValueError(f"Unknown aggregate method: {aggregate}")
        
        else:
            # Single model prediction
            with torch.no_grad():
                X_obs_t = torch.from_numpy(X_obs_np).unsqueeze(0).to(self.device)
                T_obs_t = torch.from_numpy(T_obs_np).unsqueeze(0).to(self.device)
                Y_obs_t = torch.from_numpy(Y_obs_np).unsqueeze(0).to(self.device)
                X_intv_t = torch.from_numpy(X_intv_np).unsqueeze(0).to(self.device)
                T_intv_t = torch.from_numpy(T_intv_np).unsqueeze(0).to(self.device)
                
                out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
                pred_raw = out["predictions"]  # (1, M, num_params)
                
                return pred_raw.squeeze(0)  # (M, num_params)

    def log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
    ) -> np.ndarray:
        """
        Compute the log-likelihood of Y_intv values under the model's predicted distribution.
        
        This method evaluates how well the model's predicted distribution matches the observed
        interventional targets. It uses the BarDistribution's log-likelihood method when enabled,
        or falls back to MSE-based approximation (assuming Gaussian with fixed variance) when not.
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational intervened feature (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional intervened feature (M,) or (M, 1)
            Y_intv: Interventional targets to evaluate log-likelihood for (M,) or (M, 1)
            
        Returns:
            Log-likelihood values of shape (M,)
            - With BarDistribution: exact log p(Y_intv | X_intv, T_intv, X_obs, T_obs, Y_obs)
            - Without BarDistribution: approximate log-likelihood assuming Gaussian with unit variance
            
        Raises:
            RuntimeError: If model not loaded
            
        Notes:
            - When BarDistribution is enabled, this computes the exact log-likelihood under
              the learned piecewise-uniform + Gaussian tail distribution
            - When BarDistribution is disabled, falls back to Gaussian approximation:
              log p(y|x) ≈ -0.5 * (y - pred)^2 - 0.5*log(2π) (assumes unit variance)
            - This is the interventional version - evaluates likelihood of interventional outcomes
        """
        if self.model is None:
            raise RuntimeError("[InterventionalPFNSklearn] Model not loaded. Call load() first.")
        
        # Convert to numpy and ensure correct shapes
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
            T_obs = T_obs.reshape(-1, 1)
        if T_intv.ndim == 1:
            T_intv = T_intv.reshape(-1, 1)
        if Y_obs.ndim == 2 and Y_obs.shape[1] == 1:
            Y_obs = Y_obs.squeeze(1)
        if Y_intv.ndim == 2 and Y_intv.shape[1] == 1:
            Y_intv = Y_intv.squeeze(1)
        
        M = X_intv.shape[0]
        
        # Check if BarDistribution is available and ready
        if not self.use_bar_distribution:
            if self.verbose:
                print("[InterventionalPFNSklearn] BarDistribution not enabled. Using Gaussian approximation for log-likelihood.")
        elif not self._bar_distribution_is_ready():
            if self.verbose:
                print("[InterventionalPFNSklearn] BarDistribution not fitted. Using Gaussian approximation for log-likelihood.")
        
        # Convert to torch tensors
        X_obs_t = torch.from_numpy(X_obs).unsqueeze(0).to(self.device)  # (1, N, L)
        T_obs_t = torch.from_numpy(T_obs).unsqueeze(0).to(self.device)  # (1, N, 1)
        Y_obs_t = torch.from_numpy(Y_obs).unsqueeze(0).to(self.device)  # (1, N)
        X_intv_t = torch.from_numpy(X_intv).unsqueeze(0).to(self.device)  # (1, M, L)
        T_intv_t = torch.from_numpy(T_intv).unsqueeze(0).to(self.device)  # (1, M, 1)
        Y_intv_t = torch.from_numpy(Y_intv).unsqueeze(0).to(self.device)  # (1, M)
        
        # Forward pass to get distribution parameters
        self.model.eval()
        with torch.no_grad():
            out = self.model(X_obs_t, T_obs_t, Y_obs_t, X_intv_t, T_intv_t)
            raw_preds = out["predictions"]  # (1, M, output_dim)
            
            if self.use_bar_distribution and self._bar_distribution_is_ready():
                # Use BarDistribution's exact log-likelihood
                log_probs = self.bar_distribution._logpdf_from_pred(raw_preds, Y_intv_t)  # (1, M)
                return log_probs.squeeze(0).cpu().numpy()  # (M,)
            else:
                # Fallback: Gaussian approximation with unit variance
                # raw_preds contains point predictions (mode or mean depending on model output)
                # For MSE models: raw_preds is the direct prediction
                # log p(y|x) ≈ -0.5 * (y - pred)^2 - 0.5*log(2π)
                if raw_preds.shape[-1] == 1:
                    # Direct prediction output
                    pred_mean = raw_preds.squeeze(-1)  # (1, M)
                else:
                    # Multi-parameter output - use first parameter as mean
                    pred_mean = raw_preds[..., 0]  # (1, M)
                
                squared_error = (Y_intv_t - pred_mean) ** 2
                log_likelihood = -0.5 * squared_error - 0.5 * np.log(2 * np.pi)  # (1, M)
                return log_likelihood.squeeze(0).cpu().numpy()  # (M,)


if __name__ == "__main__":
    # Test the interventional wrapper
    import numpy as _np
    
    print("\n" + "="*80)
    print("TEST: InterventionalPFNSklearn Basic Usage")
    print("="*80)
    
    # Create dummy config and checkpoint paths (adjust to your actual paths)
    cfg_path = "<REPO_ROOT>/experiments/FirstTests/configs/early_test.yaml"
    ckpt_path = "<REPO_ROOT>/experiments/FirstTests/checkpoints/early_test1_32bs/step_100000.pt"
    
    # Create wrapper
    wrapper = InterventionalPFNSklearn(
        config_path=cfg_path,
        checkpoint_path=ckpt_path,
        device="cpu",
        verbose=True,
        n_estimators=1,
    )
    
    # Load model
    try:
        wrapper.load()
        print(f"[InterventionalPFNSklearn] Model loaded successfully")
        print(f"[InterventionalPFNSklearn] num_features = {wrapper.model.num_features}")
    except Exception as e:
        print(f"[InterventionalPFNSklearn] Could not load model (expected if paths don't exist): {e}")
        print("[InterventionalPFNSklearn] Creating dummy model for testing interface")
        
        # Create a dummy model for testing the interface
        wrapper = InterventionalPFNSklearn(device="cpu", verbose=True)
        wrapper.model_kwargs = {"num_features": 5, "d_model": 64, "depth": 2}
        wrapper.model = InterventionalPFN(**wrapper.model_kwargs)
    
    # Create synthetic data
    num_features = wrapper.model.num_features
    N, M = 20, 5  # 20 train, 5 test
    
    X_obs = _np.random.randn(N, num_features).astype(_np.float32)
    T_obs = _np.random.randn(N, 1).astype(_np.float32)
    Y_obs = _np.random.randn(N).astype(_np.float32)
    X_intv = _np.random.randn(M, num_features).astype(_np.float32)
    T_intv = _np.random.randn(M, 1).astype(_np.float32)
    
    print(f"\n[InterventionalPFNSklearn] Making predictions...")
    print(f"  X_obs: {X_obs.shape}, T_obs: {T_obs.shape}, Y_obs: {Y_obs.shape}")
    print(f"  X_intv: {X_intv.shape}, T_intv: {T_intv.shape}")
    
    # Make predictions
    preds = wrapper.predict(X_obs, T_obs, Y_obs, X_intv, T_intv, prediction_type="point")
    
    print(f"\n[InterventionalPFNSklearn] Predictions shape: {preds.shape}")
    print(f"[InterventionalPFNSklearn] Sample predictions: {preds[:3]}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
