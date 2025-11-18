from __future__ import annotations
from typing import Optional, Dict, Any, Union, Literal, List
from collections import OrderedDict
from copy import deepcopy
import itertools
import random

import yaml
import torch
import numpy as np
import re
from pathlib import Path
from sklearn.cluster import KMeans
# Removed sklearn preprocessing imports - using identity preprocessing only

# Robust import: try package-style, then relative, then add repo root to sys.path
try:
    from src.models.SimplePFN import SimplePFNRegressor
    from src.Losses.BarDistribution import BarDistribution
except Exception:
    try:
        from models.SimplePFN import SimplePFNRegressor
        from Losses.BarDistribution import BarDistribution
    except Exception:
        import sys
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.append(str(repo_root))
        from src.models.SimplePFN import SimplePFNRegressor
        from src.Losses.BarDistribution import BarDistribution


class SimplePreprocessor:
    """
    Simple preprocessing pipeline for tabular data.
    
    Simplified version that only supports identity transformation (no preprocessing).
    Maintains the same interface for compatibility.
    """
    
    def __init__(self, normalization_method: str = "none", outlier_strategy: str = "none",
                 random_state: Optional[int] = None):
        """
        Args:
            normalization_method: Only 'none' is supported (for compatibility)
            outlier_strategy: Only 'none' is supported (for compatibility)
            random_state: Random seed for reproducibility (kept for compatibility)
        """
        if normalization_method != "none":
            raise ValueError(f"Only normalization_method='none' is supported, got '{normalization_method}'")
        if outlier_strategy != "none":
            raise ValueError(f"Only outlier_strategy='none' is supported, got '{outlier_strategy}'")
            
        self.normalization_method = normalization_method
        self.outlier_strategy = outlier_strategy
        self.random_state = random_state
        self.fitted = False
        
    def fit(self, X, y=None):
        """Fit the preprocessing pipeline on training data (identity transform - no-op)."""
        X = np.asarray(X, dtype=np.float32)
        
        # Identity preprocessing - just store a copy for consistency
        self.X_transformed_ = X.copy()
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted preprocessor (identity transform - returns copy)."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        
        # Identity transformation - return as-is
        return X.copy()


class FeatureShuffler:
    """Generate feature permutations for ensemble diversity."""
    
    def __init__(self, n_features: int, method: str = "random", random_state: Optional[int] = None):
        """
        Args:
            n_features: Number of features
            method: 'none', 'random', or 'shift' (circular shift)
            random_state: Random seed
        """
        self.n_features = n_features
        self.method = method
        self.random_state = random_state
        
    def generate(self, n_patterns: int) -> List[np.ndarray]:
        """
        Generate n_patterns feature permutations.
        
        Returns:
            List of feature index arrays for shuffling
        """
        rng = random.Random(self.random_state)
        feature_indices = list(range(self.n_features))
        
        if self.method == "none" or n_patterns == 1:
            return [np.array(feature_indices)]
        elif self.method == "shift":
            # Circular shifts
            patterns = [np.array(feature_indices[-i:] + feature_indices[:-i]) 
                       for i in range(min(n_patterns, self.n_features))]
            return patterns
        elif self.method == "random":
            # Random permutations
            if self.n_features <= 5:
                all_perms = [list(perm) for perm in itertools.permutations(feature_indices)]
                patterns = [np.array(p) for p in rng.sample(all_perms, min(n_patterns, len(all_perms)))]
            else:
                patterns = [np.array(rng.sample(feature_indices, self.n_features)) 
                           for _ in range(n_patterns)]
            return patterns
        else:
            raise ValueError(f"Unknown method: {self.method}")


class EnsemblePreprocessor:
    """
    Create ensemble variants through feature permutations.
    
    Simplified version that only applies feature shuffling (no normalization/outlier removal).
    Maintains the same interface for compatibility.
    """
    
    def __init__(self, n_estimators: int = 1, 
                 norm_methods: Optional[Union[str, List[str]]] = None,
                 outlier_strategies: Optional[Union[str, List[str]]] = None,
                 feat_shuffle_method: str = "random",
                 random_state: Optional[int] = None):
        """
        Args:
            n_estimators: Number of ensemble variants to create
            norm_methods: Only 'none' is supported (for compatibility)
            outlier_strategies: Only 'none' is supported (for compatibility)
            feat_shuffle_method: Feature permutation method ('none', 'random', 'shift')
            random_state: Random seed
        """
        # Validate that only "none" is used for norm_methods and outlier_strategies
        if norm_methods is not None:
            methods = [norm_methods] if isinstance(norm_methods, str) else norm_methods
            if any(m != "none" for m in methods):
                raise ValueError(f"Only norm_methods='none' is supported, got {methods}")
        
        if outlier_strategies is not None:
            strategies = [outlier_strategies] if isinstance(outlier_strategies, str) else outlier_strategies
            if any(s != "none" for s in strategies):
                raise ValueError(f"Only outlier_strategies='none' is supported, got {strategies}")
        
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.outlier_strategies = outlier_strategies
        self.feat_shuffle_method = feat_shuffle_method
        self.random_state = random_state
        self.fitted = False
        
    def fit(self, X_train, y_train=None):
        """
        Fit ensemble preprocessors on training data (simplified - only feature shuffling).
        
        Args:
            X_train: Training features (N, F)
            y_train: Training targets (N,) - not used but kept for compatibility
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        
        # Only "none" is supported for normalization and outlier strategies
        self.norm_methods_ = ["none"]
        self.outlier_strategies_ = ["none"]
        
        self.n_features_ = X_train.shape[1]
        
        # Generate ensemble configurations (only feature shuffling)
        shuffler = FeatureShuffler(self.n_features_, self.feat_shuffle_method, self.random_state)
        shuffle_patterns = shuffler.generate(self.n_estimators)
        
        # Store shuffle patterns (no actual preprocessing needed since we use identity)
        self.ensemble_configs_ = OrderedDict()
        key = ("none", "none")  # Only one preprocessing variant
        self.ensemble_configs_[key] = shuffle_patterns
        
        # Create a single identity preprocessor
        self.preprocessors_ = {}
        preprocessor = SimplePreprocessor(
            normalization_method="none",
            outlier_strategy="none",
            random_state=self.random_state
        )
        preprocessor.fit(X_train)
        self.preprocessors_[key] = preprocessor
        
        self.fitted = True
        return self
    
    def transform(self, X_train, y_train, X_test):
        """
        Transform data into ensemble variants.
        
        Args:
            X_train: Training features (N, F)
            y_train: Training targets (N,)
            X_test: Test features (M, F)
            
        Returns:
            Dictionary mapping ensemble indices to (X_train_variant, y_train, X_test_variant) tuples
            Each variant has shuffled features and potentially different preprocessing
        """
        if not self.fitted:
            raise RuntimeError("EnsemblePreprocessor not fitted. Call fit() first.")
        
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        
        ensemble_data = {}
        variant_idx = 0
        
        for (norm_method, outlier_strategy), shuffle_patterns in self.ensemble_configs_.items():
            # Apply preprocessing for this (normalization, outlier) combination
            preprocessor = self.preprocessors_[(norm_method, outlier_strategy)]
            X_train_preprocessed = preprocessor.X_transformed_
            X_test_preprocessed = preprocessor.transform(X_test)
            
            # Apply each shuffle pattern
            for shuffle_pattern in shuffle_patterns:
                X_train_variant = X_train_preprocessed[:, shuffle_pattern]
                X_test_variant = X_test_preprocessed[:, shuffle_pattern]
                
                ensemble_data[variant_idx] = (X_train_variant, y_train.copy(), X_test_variant)
                variant_idx += 1
        
        return ensemble_data


class SimplePFNSklearn:
    """
    A small scikit-learn-like wrapper around the SimplePFN PyTorch model with BarDistribution and ensemble support.

    Usage:
      # Basic usage
      wrapper = SimplePFNSklearn(config_path="path/to/config.yaml", checkpoint_path="model.pt")
      wrapper.load()
      preds = wrapper.predict(X_train, y_train, X_test, prediction_type="mode")
      
      # Ensemble usage (feature shuffling only)
      wrapper = SimplePFNSklearn(config_path="path/to/config.yaml", checkpoint_path="model.pt",
                                 n_estimators=5)
      wrapper.load()
      wrapper.fit(X_train, y_train)  # Fits ensemble preprocessors
      preds = wrapper.predict(X_train, y_train, X_test, prediction_type="mode", aggregate="mean")
      
      # Adaptive clustering usage
      wrapper = SimplePFNSklearn(config_path="path/to/config.yaml", checkpoint_path="model.pt",
                                 max_n_train=100)
      wrapper.load()
      preds = wrapper.predict(X_train, y_train, X_test, prediction_type="mode")
      
      # Adaptive clustering + test batching
      wrapper = SimplePFNSklearn(config_path="path/to/config.yaml", checkpoint_path="model.pt",
                                 max_n_train=100, max_n_test=50)
      wrapper.load()
      preds = wrapper.predict(X_train, y_train, X_test, prediction_type="mode")

    Parameters:
      n_estimators: Number of ensemble members (default: 10)
      norm_methods: Only 'none' is supported (for compatibility)
      outlier_strategies: Only 'none' is supported (for compatibility)
      feat_shuffle_method: Feature permutation method ('none', 'random', 'shift')
      max_n_train: If set, uses adaptive hierarchical clustering to ensure no cluster exceeds
                   this size. Training set is deduplicated first, then clustered if N > max_n_train.
                   (default: None, which means no clustering)
      max_n_test: If set, test data is split into chunks of at most this many samples. Each chunk
                  is processed separately with the same training data. If None but max_n_train is
                  set, defaults to max_n_train. (default: None, which means no test splitting)
      
    Ensemble Diversity:
      The ensemble creates diverse variants through feature permutations only.
      This results in n_estimators different feature orderings of the data, which are processed in a single
      batched forward pass for efficiency, then aggregated (mean/median) or returned individually.
      
    Adaptive Clustering (max_n_train):
      When max_n_train is set, the model uses a hierarchical clustering strategy to ensure
      no cluster exceeds the specified maximum size:
      
      1. **Deduplication**: Removes duplicate (X, y) pairs from training data
      2. **Initial Clustering**: If N > max_n_train, applies k-means with k = N // max_n_train + 1
      3. **Cluster Splitting**: Any cluster with > max_n_train samples is split using k-means (k=2)
      4. **Random Splitting**: If sub-clusters still exceed max_n_train, they are randomly split
         into equal-sized parts until all clusters satisfy the size constraint
      5. **Prediction**: Each test sample is assigned to its nearest cluster centroid and predictions
         are made using only that cluster's training data
      
      Benefits:
      - Ensures computational efficiency by limiting context size
      - Improves predictions by using only relevant local training data
      - Handles large datasets gracefully through adaptive partitioning
      - Compatible with ensemble mode for combined clustering + ensemble predictions
      
    Test Batching (max_n_test):
      When max_n_test is set, test data exceeding this size is automatically split into chunks:
      - Each chunk contains at most max_n_test samples
      - All chunks are processed with the same training data
      - Results are concatenated seamlessly
      - Prevents memory issues with large test sets
      - Compatible with both clustering and ensemble modes
      
    Notes:
    - With n_estimators > 1, the model creates diverse feature permutation variants
    - All ensemble variants are processed in a SINGLE batched forward pass (efficient!)
    - Ensemble predictions are aggregated using mean, median, or returned as array
    - When BarDistribution is enabled, ensemble works with all prediction types
    - Input shapes are automatically converted to match training format
    - Normalization and outlier handling have been removed (only identity preprocessing)
    - Clustering and ensemble can be combined for clustered ensemble predictions
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        verbose: bool = False,
        n_estimators: int = 10,
        norm_methods: Optional[Union[str, List[str]]] = None,
        outlier_strategies: Optional[Union[str, List[str]]] = None,
        feat_shuffle_method: str = "random",
        random_state: Optional[int] = None,
        max_n_train: Optional[int] = 500,
        max_n_test: Optional[int] = None,
    ):
        """
        Initialize SimplePFN sklearn-like wrapper.
        
        Args:
            config_path: Path to model configuration YAML file
            checkpoint_path: Path to trained model checkpoint (.pt file)
            device: Device to run model on ("cpu" or "cuda")
            verbose: If True, prints detailed information during operations
            n_estimators: Number of ensemble members (default: 10). Set to 1 for single model.
            norm_methods: Deprecated. Only 'none' is supported (kept for compatibility)
            outlier_strategies: Deprecated. Only 'none' is supported (kept for compatibility)
            feat_shuffle_method: Feature permutation strategy for ensemble diversity:
                - 'none': No feature shuffling (all ensemble members see same feature order)
                - 'random': Random permutation of features for each ensemble member
                - 'shift': Circular shift of features for each ensemble member
            random_state: Random seed for reproducibility of clustering and ensemble generation
            max_n_train: Maximum training samples per cluster. When set, enables adaptive
                hierarchical clustering that ensures no cluster exceeds this size.
                The strategy deduplicates data and applies k-means clustering with automatic
                splitting of oversized clusters. If None, no clustering is performed.
            max_n_test: Maximum test samples to pass to model at once. When set, test data
                is split into chunks of at most max_n_test samples. If None but max_n_train
                is set, defaults to max_n_train. If both are None, no test splitting occurs.
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.verbose = verbose

        self.model: Optional[SimplePFNRegressor] = None
        self.model_kwargs: Dict[str, Any] = {}
        self.bar_distribution: Optional[BarDistribution] = None
        self.use_bar_distribution: bool = False
        
        # Ensemble parameters
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.outlier_strategies = outlier_strategies
        self.feat_shuffle_method = feat_shuffle_method
        self.random_state = random_state
        self.ensemble_preprocessor: Optional[EnsemblePreprocessor] = None
        self.use_ensemble = n_estimators > 1
        
        # Adaptive clustering parameters
        self.max_n_train = max_n_train
        self.cluster_assignments_: Optional[np.ndarray] = None  # Final cluster labels for each training sample
        
        # Test batching parameters
        # If max_n_train is set but max_n_test is not, default max_n_test to max_n_train
        if max_n_test is None and max_n_train is not None:
            self.max_n_test = max_n_train
        else:
            self.max_n_test = max_n_test
    def load(self, override_kwargs: Optional[Dict[str, Any]] = None) -> "SimplePFNSklearn":
        """Load config (if provided), apply optional override_kwargs, build the model and load checkpoint (if provided).

        override_kwargs: a dict of model kwargs (e.g., {'num_features': 9}) that will update values from the YAML.
        """
        if self.config_path:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            # model config lives under top-level 'model_config'
            mcfg = cfg.get("model_config", {}) if isinstance(cfg, dict) else {}
            
            # Check for BarDistribution configuration
            self.use_bar_distribution = mcfg.get("use_bar_distribution", {}).get("value", False)
            
            # map expected keys
            self.model_kwargs = {
                "num_features": int(mcfg.get("num_features", {}).get("value", 0)),
                "d_model": int(mcfg.get("d_model", {}).get("value", 256)),
                "depth": int(mcfg.get("depth", {}).get("value", 8)),
                "heads_feat": int(mcfg.get("heads_feat", {}).get("value", 8)),
                "heads_samp": int(mcfg.get("heads_samp", {}).get("value", 8)),
                "dropout": float(mcfg.get("dropout", {}).get("value", 0.0)),
                "hidden_mult": int(mcfg.get("hidden_mult", {}).get("value", 4)),
                # Note: Feature positional encodings are now always enabled in SimplePFNRegressor
            }
            
            # Set output_dim based on BarDistribution configuration
            if self.use_bar_distribution:
                num_bars = int(mcfg.get("num_bars", {}).get("value", 11))
                output_dim = num_bars + 4  # BarDistribution requires K + 4 parameters
                self.model_kwargs["output_dim"] = output_dim
                
                # Create BarDistribution instance
                self.bar_distribution = BarDistribution(
                    num_bars=num_bars,
                    min_width=float(mcfg.get("min_width", {}).get("value", 1e-6)),
                    scale_floor=float(mcfg.get("scale_floor", {}).get("value", 1e-6)),
                    device=self.device,
                    max_fit_items=mcfg.get("max_fit_items", {}).get("value", None),
                    log_prob_clip_min=float(mcfg.get("log_prob_clip_min", {}).get("value", -50.0)),
                    log_prob_clip_max=float(mcfg.get("log_prob_clip_max", {}).get("value", 50.0)),
                )
                
                if self.verbose:
                    print(f"[SimplePFNSklearn] BarDistribution enabled with {num_bars} bars, output_dim={output_dim}")
            else:
                self.model_kwargs["output_dim"] = 1
                if self.verbose:
                    print("[SimplePFNSklearn] Using standard MSE output (output_dim=1)")
        else:
            # If the user provided model_kwargs before load(), that's acceptable.
            if self.model_kwargs and self.model_kwargs.get("num_features"):
                if self.verbose:
                    print("[SimplePFNSklearn] Using pre-specified model_kwargs.")
            else:
                if self.verbose:
                    print("[SimplePFNSklearn] No config_path provided; you must supply model kwargs manually before load().")

        # apply overrides if present (these take precedence over config)
        if override_kwargs:
            self.model_kwargs.update({k: v for k, v in override_kwargs.items() if v is not None})

        # sanity check for num_features
        if not self.model_kwargs.get("num_features"):
            # If num_features is missing, build a minimal placeholder (user must ensure correctness)
            self.model_kwargs.setdefault("num_features", 1)

        # build model
        self.model = SimplePFNRegressor(**self.model_kwargs).to(self.device)

        # load weights
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            # common wrappers store state in ['state_dict'] or ['model_state_dict']
            state = None
            if isinstance(ckpt, dict):
                for key in ("state_dict", "model_state_dict", "net", "model"):
                    if key in ckpt:
                        state = ckpt[key]
                        break
                if state is None:
                    # maybe the dict is already the state_dict
                    state = ckpt
            else:
                state = ckpt

            # some state_dicts have 'module.' prefixes from DataParallel; strip if needed
            def _strip_module(sdict):
                new = {}
                for k, v in sdict.items():
                    nk = k.replace("module.", "")
                    new[nk] = v
                return new

            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = _strip_module(state)

            # Try loading; if size mismatch occurs, attempt to infer d_model/depth from checkpoint and rebuild
            try:
                # If checkpoint contains feature_pos_* with different rank than current config, adapt before loading
                try:
                    if any(k.startswith("feature_pos_A") for k in state.keys()) and hasattr(self.model, "feature_pos_A") and self.model.feature_pos_A is not None:
                        ck_A = state.get("feature_pos_A")
                        ck_B = state.get("feature_pos_B")
                        if ck_A is not None and ck_B is not None:
                            rank_ck = int(ck_A.shape[1])
                            if getattr(self.model, "feature_pos_rank", None) and int(self.model.feature_pos_rank) != rank_ck:
                                if self.verbose:
                                    print(f"[SimplePFNSklearn] Adapting feature_pos_rank from {self.model.feature_pos_rank} to {rank_ck} to match checkpoint")
                                # Rebuild model with adapted rank
                                self.model_kwargs["feature_pos_rank"] = rank_ck
                                self.model = SimplePFNRegressor(**self.model_kwargs).to(self.device)
                except Exception:
                    pass
                self.model.load_state_dict(state, strict=False)
                if self.verbose:
                    print("[SimplePFNSklearn] Loaded checkpoint into model (partial loads allowed).")
                    
                # Load BarDistribution parameters if available and BarDistribution is enabled
                if self.use_bar_distribution and self.bar_distribution is not None and 'bar_distribution' in ckpt:
                    self._load_bar_distribution_parameters(ckpt['bar_distribution'])
                    
            except Exception as e:
                if self.verbose:
                    print(f"[SimplePFNSklearn] Warning: initial load failed: {e}")
                # attempt to infer d_model from common param shapes
                try:
                    inferred = {}
                    # label_mask_embed -> shape (1,1,d)
                    if "label_mask_embed" in state:
                        v = state["label_mask_embed"]
                        if v.ndim >= 3:
                            inferred["d_model"] = int(v.shape[-1])
                    # value_encoder.weight -> shape (d_model, 1)
                    if "value_encoder.weight" in state and "d_model" not in inferred:
                        v = state["value_encoder.weight"]
                        inferred["d_model"] = int(v.shape[0])
                    # count blocks to infer depth
                    block_idxs = [int(m.group(1)) for k in state.keys() for m in [re.match(r"blocks\.(\d+)\.", k)] if m]
                    if block_idxs:
                        inferred["depth"] = int(max(block_idxs) + 1)

                    if inferred:
                        if self.verbose:
                            print(f"[SimplePFNSklearn] Inferred from checkpoint: {inferred}")
                        # update kwargs and rebuild model
                        if "d_model" in inferred:
                            self.model_kwargs["d_model"] = inferred["d_model"]
                        if "depth" in inferred:
                            self.model_kwargs["depth"] = inferred["depth"]
                        # rebuild model with inferred d_model/depth
                        self.model = SimplePFNRegressor(**self.model_kwargs).to(self.device)
                        try:
                            # adapt rank again for rebuilt model if necessary
                            try:
                                if any(k.startswith("feature_pos_A") for k in state.keys()) and hasattr(self.model, "feature_pos_A") and self.model.feature_pos_A is not None:
                                    ck_A = state.get("feature_pos_A")
                                    ck_B = state.get("feature_pos_B")
                                    if ck_A is not None and ck_B is not None:
                                        rank_ck = int(ck_A.shape[1])
                                        if getattr(self.model, "feature_pos_rank", None) and int(self.model.feature_pos_rank) != rank_ck:
                                            if self.verbose:
                                                print(f"[SimplePFNSklearn] Adapting feature_pos_rank from {self.model.feature_pos_rank} to {rank_ck} to match checkpoint (rebuild)")
                                            self.model_kwargs["feature_pos_rank"] = rank_ck
                                            self.model = SimplePFNRegressor(**self.model_kwargs).to(self.device)
                            except Exception:
                                pass
                            self.model.load_state_dict(state, strict=False)
                            if self.verbose:
                                print("[SimplePFNSklearn] Successfully rebuilt model with inferred shape and loaded checkpoint.")
                                
                            # Load BarDistribution parameters if available and BarDistribution is enabled
                            if self.use_bar_distribution and self.bar_distribution is not None and 'bar_distribution' in ckpt:
                                self._load_bar_distribution_parameters(ckpt['bar_distribution'])
                                
                        except Exception as e2:
                            if self.verbose:
                                print(f"[SimplePFNSklearn] Warning: failed to load even after inferring shapes: {e2}")
                    else:
                        if self.verbose:
                            print("[SimplePFNSklearn] Could not infer model shapes from checkpoint; skipping weight load.")
                except Exception as e3:
                    if self.verbose:
                        print(f"[SimplePFNSklearn] Error while attempting to infer shapes: {e3}")

        return self
    
    def _load_bar_distribution_parameters(self, bar_dist_state: Dict[str, Any]) -> None:
        """
        Load BarDistribution parameters from checkpoint state.
        
        Args:
            bar_dist_state: Dictionary containing BarDistribution parameters from checkpoint
        """
        if self.verbose:
            print(f"[SimplePFNSklearn] Loading BarDistribution parameters (fitted: {bar_dist_state.get('fitted', False)})")
        
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
            if self.verbose:
                print("[SimplePFNSklearn] BarDistribution fitted parameters restored from checkpoint")
        else:
            if self.verbose:
                print("[SimplePFNSklearn] BarDistribution was not fitted in the saved checkpoint")

    def fit(self, X: Any = None, y: Any = None, **kwargs) -> "SimplePFNSklearn":
        """
        Fit ensemble preprocessors if ensemble is enabled. Otherwise placeholder.

        Args:
            X: Training features (N, F) - numpy array, pandas DataFrame, or torch tensor
            y: Training targets (N,) or (N, 1) - numpy array, pandas Series, or torch tensor
            
        Returns:
            self: Returns the instance for method chaining
            
        Notes:
            - If n_estimators > 1: Fits ensemble preprocessors (feature shuffling) on training data
            - If n_estimators == 1: No-op placeholder (model training done via project's trainer)
            - Adaptive clustering does NOT require calling fit() - clustering happens automatically
              during predict() based on the provided training data
            - BarDistribution is loaded from checkpoint, never fitted here
        """
        if self.use_ensemble:
            if X is None or y is None:
                raise ValueError("fit() requires X and y when using ensemble (n_estimators > 1)")
            
            if self.verbose:
                print(f"[SimplePFNSklearn] Fitting ensemble preprocessors with {self.n_estimators} variants...")
            
            # Create and fit ensemble preprocessor
            self.ensemble_preprocessor = EnsemblePreprocessor(
                n_estimators=self.n_estimators,
                norm_methods=self.norm_methods,
                outlier_strategies=self.outlier_strategies,
                feat_shuffle_method=self.feat_shuffle_method,
                random_state=self.random_state
            )
            
            # Convert to numpy
            if hasattr(X, "values"):
                X = X.values
            if hasattr(y, "values"):
                y = y.values
            
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            
            self.ensemble_preprocessor.fit(X, y)
            
            if self.verbose:
                print(f"[SimplePFNSklearn] Ensemble preprocessors fitted successfully")
                print(f"[SimplePFNSklearn] Using {len(self.ensemble_preprocessor.ensemble_configs_)} unique (normalization, outlier) combinations:")
                for (norm_method, outlier_strategy), patterns in self.ensemble_preprocessor.ensemble_configs_.items():
                    print(f"  - {norm_method} normalization + {outlier_strategy} outlier removal: {len(patterns)} variants")
        else:
            if self.verbose:
                print("[SimplePFNSklearn] fit() is a no-op placeholder (n_estimators=1). Use training scripts to train the model.")
        
        return self

    def predict(self, X_train: Any, y_train: Any, X_test: Any, 
                prediction_type: Literal["point", "mode", "mean", "sample"] = "mean",
                num_samples: int = 100,
                aggregate: Literal["mean", "median", "none"] = "mean") -> np.ndarray:
        """
        Run the PFN forward pass and return predictions as a numpy array.
        
        Supports three prediction modes:
        1. Single model prediction (n_estimators=1, max_n_train=None)
        2. Ensemble prediction (n_estimators>1, max_n_train=None)
        3. Adaptive clustering (max_n_train set, with or without ensemble)

        Args:
            X_train: Training features (numpy arrays, pandas DataFrame, or torch tensors)
                - Shape: (N, F) where N is number of samples, F is number of features
                - Will be deduplicated if max_n_train is set
                
            y_train: Training targets (numpy arrays, pandas Series, or torch tensors)
                - Shape: (N,) or (N, 1)
                - Will be deduplicated along with X_train if max_n_train is set
                
            X_test: Test features (same type as X_train)
                - Shape: (M, F) where M is number of test samples
                
            prediction_type: Type of prediction to return
                - "point": Raw model output (default for MSE) or mode (default for BarDistribution)
                - "mode": Posterior mode (requires BarDistribution)
                - "mean": Posterior mean (requires BarDistribution)
                - "sample": Samples from posterior (requires BarDistribution)
                
            num_samples: Number of samples to return when prediction_type="sample" (default: 100)
            
            aggregate: How to aggregate ensemble predictions (only used when n_estimators > 1)
                - "mean": Average predictions across ensemble members (default)
                - "median": Median of predictions across ensemble members
                - "none": Return all predictions with shape (n_estimators, M) or (n_estimators, M, num_samples)

        Returns:
            numpy array of predictions:
            - Single model or aggregated ensemble: shape (M,)
            - Ensemble with aggregate="none": shape (n_estimators, M)
            - For prediction_type="sample": adds sample dimension
            
        Behavior with max_n_train:
            When max_n_train is set, the method automatically:
            1. Deduplicates (X_train, y_train) pairs
            2. If N > max_n_train, performs hierarchical k-means clustering
            3. Splits oversized clusters until all clusters ≤ max_n_train
            4. Assigns each test sample to nearest cluster centroid
            5. Makes predictions using only the relevant cluster's training data
            
        Raises:
            ValueError: If input shapes are inconsistent or incompatible
            ValueError: If prediction_type requires BarDistribution but it's not enabled
            RuntimeError: If model not loaded or ensemble not fitted when required
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call load() before predict().")
        
        # Validate prediction type
        if prediction_type in ["mode", "mean", "sample"] and not self.use_bar_distribution:
            raise ValueError(f"prediction_type='{prediction_type}' requires BarDistribution to be enabled in config.")
        
        if self.use_bar_distribution and self.bar_distribution is None:
            raise RuntimeError("BarDistribution enabled but not loaded from checkpoint. "
                             "The model checkpoint must include fitted BarDistribution parameters.")

        # Convert to numpy if pandas
        if hasattr(X_train, "values"):
            X_train = X_train.values
        if hasattr(X_test, "values"):
            X_test = X_test.values
        if hasattr(y_train, "values"):
            y_train = y_train.values
        
        X_train_orig = np.asarray(X_train, dtype=np.float32)
        X_test_orig = np.asarray(X_test, dtype=np.float32)
        y_train_orig = np.asarray(y_train, dtype=np.float32)
        
        # Auto-fit ensemble preprocessor if needed
        if self.use_ensemble and self.ensemble_preprocessor is None:
            if self.verbose:
                print(f"[SimplePFNSklearn] Auto-fitting ensemble preprocessors with {self.n_estimators} variants...")
            self.fit(X_train_orig, y_train_orig)
        
        # Check if adaptive clustering should be used
        if self.max_n_train is not None:
            return self._predict_with_adaptive_clustering(
                X_train_orig, y_train_orig, X_test_orig,
                prediction_type, num_samples, aggregate
            )
        
        # Handle ensemble predictions
        if self.use_ensemble:
            return self._predict_ensemble(
                X_train_orig, y_train_orig, X_test_orig,
                prediction_type, num_samples, aggregate
            )
        
        # Single model prediction (original implementation)
        return self._predict_single(
            X_train_orig, y_train_orig, X_test_orig,
            prediction_type, num_samples
        )
    
    def _predict_with_adaptive_clustering(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                                          prediction_type: str, num_samples: int, aggregate: str) -> np.ndarray:
        """
        Make predictions using adaptive hierarchical clustering on training data.
        
        Hierarchical Strategy:
        1. Deduplicate training set
        2. If N > max_n_train, apply k-means with k = N // max_n_train + 1
        3. For clusters exceeding max_n_train, split with k-means (k=2)
        4. For sub-clusters still exceeding max_n_train, randomly split into equal parts
        5. Assign each test sample to nearest cluster and predict using that cluster's data
        
        This ensures no cluster exceeds max_n_train while minimizing data fragmentation.
        
        NOTE: Clustering happens on the ORIGINAL unbatched data (N, F), before ensemble preprocessing.
        
        Args:
            X_train: Training features (N, F) - original unbatched
            y_train: Training targets (N,) or (N, 1) - original unbatched
            X_test: Test features (M, F) - original unbatched
            prediction_type: Type of prediction
            num_samples: Number of samples for "sample" prediction type
            aggregate: Aggregation method for ensemble (if used with ensemble)
            
        Returns:
            Predictions for X_test (M,) or appropriate shape based on prediction_type
        """
        if self.verbose:
            print(f"[SimplePFNSklearn] Using adaptive clustering with max_n_train={self.max_n_train}")
            print(f"[SimplePFNSklearn] Input shapes (pre-batching): X_train{X_train.shape}, y_train{y_train.shape}, X_test{X_test.shape}")
        
        # Validate input dimensions
        if X_train.ndim != 2:
            raise ValueError(f"_predict_with_adaptive_clustering expects unbatched X_train of shape (N, F), got {X_train.shape}")
        if X_test.ndim != 2:
            raise ValueError(f"_predict_with_adaptive_clustering expects unbatched X_test of shape (M, F), got {X_test.shape}")
        
        N = X_train.shape[0]
        
        # Step 1: Deduplicate training set
        # Combine X and y for deduplication
        y_train_1d = y_train.squeeze() if y_train.ndim > 1 else y_train
        Xy_train = np.column_stack([X_train, y_train_1d.reshape(-1, 1)])
        Xy_unique, unique_indices = np.unique(Xy_train, axis=0, return_index=True)
        
        X_train_dedup = Xy_unique[:, :-1]
        y_train_dedup = Xy_unique[:, -1]
        
        # Restore original y_train shape if needed
        if y_train.ndim > 1:
            y_train_dedup = y_train_dedup.reshape(-1, *y_train.shape[1:])
        
        N_dedup = X_train_dedup.shape[0]
        
        if self.verbose:
            print(f"[SimplePFNSklearn] Deduplication: {N} -> {N_dedup} samples ({N - N_dedup} duplicates removed)")
        
        # Step 2: Check if clustering is needed
        if N_dedup <= self.max_n_train:
            if self.verbose:
                print(f"[SimplePFNSklearn] No clustering needed (N={N_dedup} <= max_n_train={self.max_n_train})")
            # No clustering needed, use all deduplicated data
            if self.use_ensemble:
                return self._predict_ensemble(
                    X_train_dedup, y_train_dedup, X_test,
                    prediction_type, num_samples, aggregate
                )
            else:
                return self._predict_single(
                    X_train_dedup, y_train_dedup, X_test,
                    prediction_type, num_samples
                )
        
        # Step 3: Apply hierarchical clustering
        cluster_assignments = self._hierarchical_cluster(X_train_dedup, y_train_dedup)
        n_final_clusters = len(np.unique(cluster_assignments))
        
        if self.verbose:
            unique_clusters, counts = np.unique(cluster_assignments, return_counts=True)
            print(f"[SimplePFNSklearn] Final clustering: {n_final_clusters} clusters")
            print(f"[SimplePFNSklearn] Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        
        # Step 4: Assign test samples to clusters and make predictions
        test_cluster_assignments = self._assign_to_clusters(X_test, X_train_dedup, cluster_assignments)
        
        # Initialize output array
        M = X_test.shape[0]
        if prediction_type == "sample":
            predictions = np.zeros((M, num_samples), dtype=np.float32)
        else:
            predictions = np.zeros(M, dtype=np.float32)
        
        # Process each cluster
        for cluster_id in np.unique(cluster_assignments):
            # Get training indices for this cluster
            train_mask = cluster_assignments == cluster_id
            train_count = np.sum(train_mask)
            
            # Get test indices for this cluster
            test_mask = test_cluster_assignments == cluster_id
            test_count = np.sum(test_mask)
            
            if test_count == 0:
                continue
            
            # Extract cluster data
            X_train_cluster = X_train_dedup[train_mask]
            y_train_cluster = y_train_dedup[train_mask]
            X_test_cluster = X_test[test_mask]
            
            if self.verbose:
                print(f"[SimplePFNSklearn] Cluster {cluster_id}: {train_count} train samples, {test_count} test samples")
            
            # Make predictions for this cluster
            if self.use_ensemble:
                cluster_preds = self._predict_ensemble(
                    X_train_cluster, y_train_cluster, X_test_cluster,
                    prediction_type, num_samples, aggregate
                )
            else:
                cluster_preds = self._predict_single(
                    X_train_cluster, y_train_cluster, X_test_cluster,
                    prediction_type, num_samples
                )
            
            # Store predictions
            predictions[test_mask] = cluster_preds
        
        return predictions
    
    def _hierarchical_cluster(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """
        Apply hierarchical clustering to keep cluster sizes <= max_n_train.
        
        Strategy:
        1. Initial k-means with k = N // max_n_train + 1
        2. For oversized clusters: split with k-means (k=2)
        3. For still-oversized sub-clusters: random split into equal parts
        
        Args:
            X_train: Deduplicated training features (N, F)
            y_train: Deduplicated training targets (N,) or (N, 1)
            
        Returns:
            Cluster assignments for each training sample (N,)
        """
        N = X_train.shape[0]
        k_initial = N // self.max_n_train + 1
        
        if self.verbose:
            print(f"[SimplePFNSklearn] Initial k-means with k={k_initial}")
        
        # Initial clustering
        rng = np.random.RandomState(self.random_state)
        kmeans = KMeans(n_clusters=k_initial, random_state=self.random_state, n_init=10)
        initial_labels = kmeans.fit_predict(X_train)
        
        # Track final cluster assignments (will be renumbered)
        cluster_assignments = np.copy(initial_labels)
        next_cluster_id = k_initial
        
        # Check each initial cluster
        for cluster_id in range(k_initial):
            cluster_mask = initial_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size <= self.max_n_train:
                continue
            
            if self.verbose:
                print(f"[SimplePFNSklearn]   Cluster {cluster_id} has {cluster_size} samples (> {self.max_n_train}), splitting...")
            
            # Extract cluster data
            X_cluster = X_train[cluster_mask]
            
            # Try k-means split (k=2)
            kmeans_split = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            sub_labels = kmeans_split.fit_predict(X_cluster)
            
            # Check sub-cluster sizes
            sub_sizes = [np.sum(sub_labels == 0), np.sum(sub_labels == 1)]
            
            if max(sub_sizes) <= self.max_n_train:
                # K-means split was successful
                if self.verbose:
                    print(f"[SimplePFNSklearn]     K-means split successful: {sub_sizes[0]} + {sub_sizes[1]} samples")
                
                # Relabel: keep first sub-cluster with original ID, assign new ID to second
                cluster_indices = np.where(cluster_mask)[0]
                sub_0_indices = cluster_indices[sub_labels == 0]
                sub_1_indices = cluster_indices[sub_labels == 1]
                
                cluster_assignments[sub_0_indices] = cluster_id
                cluster_assignments[sub_1_indices] = next_cluster_id
                next_cluster_id += 1
            else:
                # K-means split failed, use random split
                n_subclusters = (cluster_size + self.max_n_train - 1) // self.max_n_train  # Ceiling division
                
                if self.verbose:
                    print(f"[SimplePFNSklearn]     K-means split failed, random splitting into {n_subclusters} sub-clusters")
                
                # Random permutation and split
                cluster_indices = np.where(cluster_mask)[0]
                shuffled_indices = rng.permutation(cluster_indices)
                
                # Assign to sub-clusters
                for i, idx in enumerate(shuffled_indices):
                    subcluster_id = cluster_id if i % n_subclusters == 0 else next_cluster_id + (i % n_subclusters) - 1
                    cluster_assignments[idx] = subcluster_id
                
                next_cluster_id += n_subclusters - 1
        
        # Renumber clusters to be consecutive starting from 0
        unique_labels = np.unique(cluster_assignments)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        cluster_assignments = np.array([label_map[label] for label in cluster_assignments])
        
        return cluster_assignments
    
    def _assign_to_clusters(self, X_test: np.ndarray, X_train: np.ndarray, 
                           cluster_assignments: np.ndarray) -> np.ndarray:
        """
        Assign test samples to nearest training cluster centroid.
        
        Computes the centroid of each training cluster and assigns each test sample
        to the cluster with the nearest centroid (using Euclidean distance).
        
        Args:
            X_test: Test features (M, F)
            X_train: Training features (N, F) - deduplicated training data
            cluster_assignments: Cluster labels for training samples (N,)
                Should be consecutive integers starting from 0
            
        Returns:
            Cluster assignments for test samples (M,) with same label space as cluster_assignments
        """
        # Compute cluster centroids
        unique_clusters = np.unique(cluster_assignments)
        centroids = np.zeros((len(unique_clusters), X_train.shape[1]))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_assignments == cluster_id
            centroids[i] = X_train[cluster_mask].mean(axis=0)
        
        # Assign each test sample to nearest centroid
        M = X_test.shape[0]
        test_assignments = np.zeros(M, dtype=int)
        
        for i in range(M):
            distances = np.linalg.norm(centroids - X_test[i], axis=1)
            nearest_cluster_idx = np.argmin(distances)
            test_assignments[i] = unique_clusters[nearest_cluster_idx]
        
        return test_assignments
    
    def _split_test_data(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        prediction_type: str, num_samples: int, 
                        predict_fn) -> np.ndarray:
        """
        Split test data into chunks and process separately if max_n_test is set.
        
        Args:
            X_train: Training features (N, F)
            y_train: Training targets (N,) or (N, 1)
            X_test: Test features (M, F)
            prediction_type: Type of prediction
            num_samples: Number of samples for "sample" prediction type
            predict_fn: Function to call for making predictions (either _predict_single or _predict_ensemble)
                Should have signature: fn(X_train, y_train, X_test_chunk, prediction_type, num_samples, [aggregate])
            
        Returns:
            Predictions for all test samples, properly concatenated
        """
        M = X_test.shape[0]
        
        # Check if test splitting is needed
        if self.max_n_test is None or M <= self.max_n_test:
            # No splitting needed
            return predict_fn(X_train, y_train, X_test, prediction_type, num_samples)
        
        if self.verbose:
            n_chunks = (M + self.max_n_test - 1) // self.max_n_test
            print(f"[SimplePFNSklearn] Splitting {M} test samples into {n_chunks} chunks of max {self.max_n_test} samples")
        
        # Split test data into chunks
        chunk_predictions = []
        for start_idx in range(0, M, self.max_n_test):
            end_idx = min(start_idx + self.max_n_test, M)
            X_test_chunk = X_test[start_idx:end_idx]
            
            if self.verbose:
                print(f"[SimplePFNSklearn]   Processing chunk [{start_idx}:{end_idx}] ({end_idx - start_idx} samples)")
            
            # Make predictions for this chunk
            chunk_pred = predict_fn(X_train, y_train, X_test_chunk, prediction_type, num_samples)
            chunk_predictions.append(chunk_pred)
        
        # Concatenate results
        if prediction_type == "sample":
            # Shape is (chunk_size, num_samples) for each chunk
            result = np.concatenate(chunk_predictions, axis=0)
        else:
            # Shape is (chunk_size,) for each chunk
            result = np.concatenate(chunk_predictions, axis=0)
        
        return result
    
    def _predict_single(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                       prediction_type: str, num_samples: int) -> np.ndarray:
        """
        Make predictions with a single model (no ensemble).
        
        Automatically splits test data into chunks if max_n_test is set.
        
        Args:
            X_train: Training features (N, F)
            y_train: Training targets (N,) or (N, 1)
            X_test: Test features (M, F)
            prediction_type: Type of prediction ("point", "mode", "mean", "sample")
            num_samples: Number of samples for prediction_type="sample"
            
        Returns:
            Predictions for X_test (M,) or (M, num_samples) for "sample" type
        """
        # Use test batching if needed
        if self.max_n_test is not None and X_test.shape[0] > self.max_n_test:
            return self._split_test_data(
                X_train, y_train, X_test, prediction_type, num_samples,
                lambda Xtr, ytr, Xte, pred_type, n_samp: self._predict_single_impl(Xtr, ytr, Xte, pred_type, n_samp)
            )
        
        return self._predict_single_impl(X_train, y_train, X_test, prediction_type, num_samples)
    
    def _predict_single_impl(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                            prediction_type: str, num_samples: int) -> np.ndarray:
        """
        Implementation of single model prediction (called by _predict_single).
        
        Args:
            X_train: Training features (N, F)
            y_train: Training targets (N,) or (N, 1)
            X_test: Test features (M, F)
            prediction_type: Type of prediction ("point", "mode", "mean", "sample")
            num_samples: Number of samples for prediction_type="sample"
            
        Returns:
            Predictions for X_test (M,) or (M, num_samples) for "sample" type
        """
        
        Xtr = torch.as_tensor(X_train, dtype=torch.float32)
        Xte = torch.as_tensor(X_test, dtype=torch.float32)
        ytr = torch.as_tensor(y_train, dtype=torch.float32)

        # normalize dims to (B, N, F) and (B, N) for y
        if Xtr.ndim == 2:
            Xtr = Xtr.unsqueeze(0)
        if Xte.ndim == 2:
            Xte = Xte.unsqueeze(0)

        if ytr.ndim == 1:
            ytr = ytr.unsqueeze(0)

        # CRITICAL FIX: Ensure y_train has the same shape as during training
        # Training expects y_train with shape (B, N, 1), but sklearn interface was providing (B, N)
        # This ensures consistency between training and inference
        if ytr.ndim == 2:
            # Add the feature dimension that training expects: (B, N) -> (B, N, 1)
            ytr = ytr.unsqueeze(-1)
        elif ytr.ndim == 3 and ytr.shape[-1] == 1:
            # Already correct shape (B, N, 1)
            pass
        else:
            # Try to reshape to correct format
            if ytr.ndim == 3:
                ytr = ytr.squeeze(-1).unsqueeze(-1)  # Ensure (B, N, 1)
            else:
                raise ValueError(f"Cannot handle y_train shape {ytr.shape}. Expected shapes that can be converted to (B, N, 1).")

        # move to device
        Xtr = Xtr.to(self.device)
        Xte = Xte.to(self.device)
        ytr = ytr.to(self.device)

        # shape checks
        B, N, F = Xtr.shape
        _, M, F2 = Xte.shape
        B_y, N_y, F_y = ytr.shape
        
        # Validate consistency
        if F != F2:
            raise ValueError(f"X_train and X_test must have same number of features: {F} vs {F2}")
        if F != self.model.num_features:
            raise ValueError(f"Model expects num_features={self.model.num_features}, but got input with {F} features")
        if B != B_y:
            raise ValueError(f"X_train and y_train must have same batch size: {B} vs {B_y}")
        if N != N_y:
            raise ValueError(f"X_train and y_train must have same number of training samples: {N} vs {N_y}")
        if F_y != 1:
            raise ValueError(f"y_train must have shape (B, N, 1), got shape {ytr.shape} with last dim = {F_y}")
            
        if self.verbose:
            print(f"[SimplePFNSklearn] Input shapes validated: X_train{Xtr.shape}, y_train{ytr.shape}, X_test{Xte.shape}")
            
        # Warning for potentially problematic context lengths
        if N < 10 or M < 5:
            if self.verbose:
                print(f"[SimplePFNSklearn] Warning: Small context sizes (N_train={N}, N_test={M}) may differ from training distribution")
        if N > 1000 or M > 1000:
            if self.verbose:
                print(f"[SimplePFNSklearn] Warning: Large context sizes (N_train={N}, N_test={M}) may differ from training distribution")

        # run model
        self.model.eval()
        with torch.no_grad():
            out = self.model(Xtr, ytr, Xte)
            raw_predictions = out["predictions"]  # Shape depends on output_dim

        # Process predictions based on type and BarDistribution usage
        if self.use_bar_distribution:
            # Model outputs BarDistribution parameters
            if prediction_type in ["point", "mode"]:
                result = self.bar_distribution.mode(raw_predictions).cpu().numpy()
            elif prediction_type == "mean":
                result = self.bar_distribution.mean(raw_predictions).cpu().numpy()
            elif prediction_type == "sample":
                result = self.bar_distribution.sample(raw_predictions, num_samples).cpu().numpy()
                # BarDistribution returns (B, num_samples, M), which is what we want
            else:
                raise ValueError(f"Unknown prediction_type: {prediction_type}")
        else:
            # Standard MSE model output
            if prediction_type != "point":
                raise ValueError(f"prediction_type='{prediction_type}' not supported without BarDistribution")
            result = raw_predictions.cpu().numpy()
            
        # Handle single batch case for consistency
        if result.shape[0] == 1 and prediction_type != "sample":
            return result[0]
        elif result.shape[0] == 1 and prediction_type == "sample":
            return result[0]  # Return (num_samples, M) for single batch
        
        return result
    
    def _predict_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                         prediction_type: str, num_samples: int, aggregate: str) -> np.ndarray:
        """
        Make predictions with ensemble and aggregate results.
        
        Efficiently batches all ensemble variants together in a single model forward pass,
        then aggregates predictions according to the specified method.
        
        Automatically splits test data into chunks if max_n_test is set.
        
        Args:
            X_train: Training features (N, F)
            y_train: Training targets (N,) or (N, 1)
            X_test: Test features (M, F)
            prediction_type: Type of prediction ("point", "mode", "mean", "sample")
            num_samples: Number of samples for prediction_type="sample"
            aggregate: Aggregation method ("mean", "median", "none")
            
        Returns:
            Aggregated predictions:
            - aggregate="mean" or "median": shape (M,) or (M, num_samples) for "sample" type
            - aggregate="none": shape (n_estimators, M) or (n_estimators, M, num_samples)
        """
        # Use test batching if needed
        if self.max_n_test is not None and X_test.shape[0] > self.max_n_test:
            return self._split_test_data(
                X_train, y_train, X_test, prediction_type, num_samples,
                lambda Xtr, ytr, Xte, pred_type, n_samp: self._predict_ensemble_impl(Xtr, ytr, Xte, pred_type, n_samp, aggregate)
            )
        
        return self._predict_ensemble_impl(X_train, y_train, X_test, prediction_type, num_samples, aggregate)
    
    def _predict_ensemble_impl(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                              prediction_type: str, num_samples: int, aggregate: str) -> np.ndarray:
        """
        Implementation of ensemble prediction (called by _predict_ensemble).
        
        Efficiently batches all ensemble variants together in a single model forward pass,
        then aggregates predictions according to the specified method.
        
        Args:
            X_train: Training features (N, F)
            y_train: Training targets (N,) or (N, 1)
            X_test: Test features (M, F)
            prediction_type: Type of prediction ("point", "mode", "mean", "sample")
            num_samples: Number of samples for prediction_type="sample"
            aggregate: Aggregation method ("mean", "median", "none")
            
        Returns:
            Aggregated predictions:
            - aggregate="mean" or "median": shape (M,) or (M, num_samples) for "sample" type
            - aggregate="none": shape (n_estimators, M) or (n_estimators, M, num_samples)
        """
        
        if self.verbose:
            print(f"[SimplePFNSklearn] Making ensemble predictions with {self.n_estimators} variants...")
        
        # Generate ensemble variants
        ensemble_data = self.ensemble_preprocessor.transform(X_train, y_train, X_test)
        
        # Stack all variants into a single batch for efficient processing
        X_train_batch = []
        y_train_batch = []
        X_test_batch = []
        
        for variant_idx in range(self.n_estimators):
            X_train_var, y_train_var, X_test_var = ensemble_data[variant_idx]
            X_train_batch.append(X_train_var)
            y_train_batch.append(y_train_var)
            X_test_batch.append(X_test_var)
        
        # Stack into single batch: each becomes (n_estimators, N, F) or (n_estimators, M, F)
        X_train_batch = np.stack(X_train_batch, axis=0)  # (n_estimators, N, F)
        y_train_batch = np.stack(y_train_batch, axis=0)  # (n_estimators, N)
        X_test_batch = np.stack(X_test_batch, axis=0)    # (n_estimators, M, F)
        
        if self.verbose:
            print(f"[SimplePFNSklearn] Batched ensemble data: X_train{X_train_batch.shape}, "
                  f"y_train{y_train_batch.shape}, X_test{X_test_batch.shape}")
        
        # Convert to tensors
        Xtr = torch.as_tensor(X_train_batch, dtype=torch.float32)
        Xte = torch.as_tensor(X_test_batch, dtype=torch.float32)
        ytr = torch.as_tensor(y_train_batch, dtype=torch.float32)
        
        # Ensure y_train has shape (B, N, 1) as expected by model
        if ytr.ndim == 2:
            ytr = ytr.unsqueeze(-1)  # (B, N) -> (B, N, 1)
        
        # Move to device
        Xtr = Xtr.to(self.device)
        Xte = Xte.to(self.device)
        ytr = ytr.to(self.device)
        
        # Shape validation
        B, N, F = Xtr.shape
        _, M, F2 = Xte.shape
        B_y, N_y, F_y = ytr.shape
        
        if F != F2:
            raise ValueError(f"X_train and X_test must have same number of features: {F} vs {F2}")
        if F != self.model.num_features:
            raise ValueError(f"Model expects num_features={self.model.num_features}, but got input with {F} features")
        if B != B_y:
            raise ValueError(f"X_train and y_train must have same batch size: {B} vs {B_y}")
        if N != N_y:
            raise ValueError(f"X_train and y_train must have same number of training samples: {N} vs {N_y}")
        if F_y != 1:
            raise ValueError(f"y_train must have shape (B, N, 1), got shape {ytr.shape}")
        
        # Run model once with all ensemble variants batched together
        self.model.eval()
        with torch.no_grad():
            out = self.model(Xtr, ytr, Xte)
            raw_predictions = out["predictions"]  # Shape: (n_estimators, M, output_dim)
        
        # Process predictions based on type and BarDistribution usage
        if self.use_bar_distribution:
            # Model outputs BarDistribution parameters
            if prediction_type in ["point", "mode"]:
                all_predictions = self.bar_distribution.mode(raw_predictions).cpu().numpy()  # (n_estimators, M)
            elif prediction_type == "mean":
                all_predictions = self.bar_distribution.mean(raw_predictions).cpu().numpy()  # (n_estimators, M)
            elif prediction_type == "sample":
                all_predictions = self.bar_distribution.sample(raw_predictions, num_samples).cpu().numpy()  # (n_estimators, num_samples, M)
            else:
                raise ValueError(f"Unknown prediction_type: {prediction_type}")
        else:
            # Standard MSE model output
            if prediction_type != "point":
                raise ValueError(f"prediction_type='{prediction_type}' not supported without BarDistribution")
            all_predictions = raw_predictions.cpu().numpy()  # (n_estimators, M)
        
        if self.verbose:
            print(f"[SimplePFNSklearn] Ensemble predictions shape: {all_predictions.shape}")
        
        # Aggregate predictions
        if aggregate == "none":
            return all_predictions
        elif aggregate == "mean":
            return np.mean(all_predictions, axis=0)
        elif aggregate == "median":
            return np.median(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")


if __name__ == "__main__":
    # Run a small end-to-end inference test using the project config and checkpoint
    import sys
    sys.path.append("/fast/arikreuter/DoPFN_v2/CausalPriorFitting")

    cfg_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/configs/early_test.yaml"
    ckpt_path = "/fast/arikreuter/DoPFN_v2/CausalPriorFitting/experiments/FirstTests/checkpoints/early_test1_32bs/step_100000.pt"

    print("\n" + "="*80)
    print("TEST 1: Single Model (no ensemble)")
    print("="*80)
    w = SimplePFNSklearn(config_path=cfg_path, checkpoint_path=ckpt_path, device="cpu", verbose=True)
    w.load()
    print("[SimplePFNSklearn] Model built. model_kwargs:", w.model_kwargs)
    print(f"[SimplePFNSklearn] BarDistribution enabled: {w.use_bar_distribution}")

    # prepare a tiny synthetic batch: 10 train samples, 3 test samples
    num_features = int(w.model.num_features)
    print(f"[SimplePFNSklearn] num_features = {num_features}")

    import numpy as _np
    Xtr = _np.random.randn(10, num_features).astype(_np.float32)
    ytr = _np.random.randn(10).astype(_np.float32)
    Xte = _np.random.randn(3, num_features).astype(_np.float32)
    yte_true = _np.random.randn(3).astype(_np.float32)

    # NOTE: BarDistribution is now a core property of the model checkpoint.
    # It should be fitted ONCE during training and saved with the model.
    # During inference, it is loaded from the checkpoint - never refitted!
    
    if w.use_bar_distribution:
        print("[SimplePFNSklearn] BarDistribution loaded from checkpoint")
        
        # Test different prediction types
        print("\n[SimplePFNSklearn] Testing different prediction types...")
        
        mode_preds = w.predict(Xtr, ytr, Xte, prediction_type="mode")
        print(f"Mode predictions shape: {mode_preds.shape}, values: {mode_preds}")
        
        mean_preds = w.predict(Xtr, ytr, Xte, prediction_type="mean") 
        print(f"Mean predictions shape: {mean_preds.shape}, values: {mean_preds}")
        
        samples = w.predict(Xtr, ytr, Xte, prediction_type="sample", num_samples=10)
        print(f"Sample predictions shape: {samples.shape}")
        print(f"First test point samples: {samples[:, 0]}")  # Show samples for first test point
        
    else:
        # Standard MSE prediction
        preds = w.predict(Xtr, ytr, Xte, prediction_type="point")
        print("[SimplePFNSklearn] Point prediction shape:", preds.shape)
        print("[SimplePFNSklearn] Point predictions:", preds)

    print("[SimplePFNSklearn] Test 1 completed successfully!")
    
    # TEST 2: Ensemble
    print("\n" + "="*80)
    print("TEST 2: Ensemble Model (n_estimators=5)")
    print("="*80)
    w_ensemble = SimplePFNSklearn(
        config_path=cfg_path, 
        checkpoint_path=ckpt_path, 
        device="cpu", 
        verbose=True,
        n_estimators=5,
        norm_methods=["none", "power"],
        feat_shuffle_method="random",
        random_state=42
    )
    w_ensemble.load()
    
    # Fit ensemble preprocessors
    print("\n[SimplePFNSklearn] Fitting ensemble preprocessors...")
    w_ensemble.fit(Xtr, ytr)
    
    # Make ensemble predictions
    print("\n[SimplePFNSklearn] Making ensemble predictions...")
    
    if w_ensemble.use_bar_distribution:
        # BarDistribution is loaded from checkpoint - no refitting needed!
        print("[SimplePFNSklearn] BarDistribution loaded from checkpoint for ensemble")
        
        # Test different aggregation methods
        mean_preds_ens = w_ensemble.predict(Xtr, ytr, Xte, prediction_type="mode", aggregate="mean")
        print(f"Ensemble mean predictions shape: {mean_preds_ens.shape}, values: {mean_preds_ens}")
        
        median_preds_ens = w_ensemble.predict(Xtr, ytr, Xte, prediction_type="mode", aggregate="median")
        print(f"Ensemble median predictions shape: {median_preds_ens.shape}, values: {median_preds_ens}")
        
        all_preds_ens = w_ensemble.predict(Xtr, ytr, Xte, prediction_type="mode", aggregate="none")
        print(f"All ensemble predictions shape: {all_preds_ens.shape}")
        print(f"Individual predictions:\n{all_preds_ens}")
    else:
        preds_ens = w_ensemble.predict(Xtr, ytr, Xte, prediction_type="point", aggregate="mean")
        print(f"Ensemble predictions shape: {preds_ens.shape}, values: {preds_ens}")

    print("\n[SimplePFNSklearn] Test 2 completed successfully!")
    
    # TEST 3: Adaptive Clustering
    print("\n" + "="*80)
    print("TEST 3: Adaptive Clustering Model (max_n_train=20)")
    print("="*80)
    w_cluster = SimplePFNSklearn(
        config_path=cfg_path, 
        checkpoint_path=ckpt_path, 
        device="cpu", 
        verbose=True,
        max_n_train=20,  # Changed from n_clusters to max_n_train
        random_state=42
    )
    w_cluster.load()
    
    # Create larger dataset for clustering demonstration
    Xtr_large = _np.random.randn(50, num_features).astype(_np.float32)
    ytr_large = _np.random.randn(50).astype(_np.float32)
    Xte_large = _np.random.randn(15, num_features).astype(_np.float32)
    
    # Add some duplicates to test deduplication
    Xtr_large = _np.vstack([Xtr_large, Xtr_large[:5]])  # Add 5 duplicates
    ytr_large = _np.hstack([ytr_large, ytr_large[:5]])
    
    print(f"\n[SimplePFNSklearn] Dataset with duplicates: {Xtr_large.shape[0]} samples")
    print("\n[SimplePFNSklearn] Making predictions with adaptive clustering...")
    
    if w_cluster.use_bar_distribution:
        preds_cluster = w_cluster.predict(Xtr_large, ytr_large, Xte_large, prediction_type="mode")
        print(f"Clustered predictions shape: {preds_cluster.shape}, values: {preds_cluster}")
    else:
        preds_cluster = w_cluster.predict(Xtr_large, ytr_large, Xte_large, prediction_type="point")
        print(f"Clustered predictions shape: {preds_cluster.shape}, values: {preds_cluster}")
    
    print("\n[SimplePFNSklearn] Test 3 completed successfully!")
    
    # TEST 4: Adaptive Clustering + Ensemble
    print("\n" + "="*80)
    print("TEST 4: Adaptive Clustering + Ensemble (max_n_train=20, n_estimators=3)")
    print("="*80)
    w_cluster_ensemble = SimplePFNSklearn(
        config_path=cfg_path, 
        checkpoint_path=ckpt_path, 
        device="cpu", 
        verbose=True,
        max_n_train=20,
        n_estimators=3,
        feat_shuffle_method="random",
        random_state=42
    )
    w_cluster_ensemble.load()
    
    print("\n[SimplePFNSklearn] Fitting ensemble preprocessors...")
    w_cluster_ensemble.fit(Xtr_large, ytr_large)
    
    print("\n[SimplePFNSklearn] Making predictions with clustering + ensemble...")
    
    if w_cluster_ensemble.use_bar_distribution:
        preds_cluster_ens = w_cluster_ensemble.predict(Xtr_large, ytr_large, Xte_large, 
                                                        prediction_type="mode", aggregate="mean")
        print(f"Clustered ensemble predictions shape: {preds_cluster_ens.shape}")
        print(f"Values: {preds_cluster_ens}")
    else:
        preds_cluster_ens = w_cluster_ensemble.predict(Xtr_large, ytr_large, Xte_large, 
                                                        prediction_type="point", aggregate="mean")
        print(f"Clustered ensemble predictions shape: {preds_cluster_ens.shape}")
        print(f"Values: {preds_cluster_ens}")
    
    print("\n[SimplePFNSklearn] Test 4 completed successfully!")
    
    # TEST 5: Test Batching
    print("\n" + "="*80)
    print("TEST 5: Test Batching (max_n_test=10)")
    print("="*80)
    w_test_batch = SimplePFNSklearn(
        config_path=cfg_path, 
        checkpoint_path=ckpt_path, 
        device="cpu", 
        verbose=True,
        max_n_test=10,
        random_state=42
    )
    w_test_batch.load()
    
    # Create dataset with many test samples
    Xtr_small = _np.random.randn(20, num_features).astype(_np.float32)
    ytr_small = _np.random.randn(20).astype(_np.float32)
    Xte_many = _np.random.randn(35, num_features).astype(_np.float32)  # Will be split into 4 chunks
    
    print("\n[SimplePFNSklearn] Making predictions with test batching...")
    print(f"[SimplePFNSklearn] Test set has {Xte_many.shape[0]} samples, will be split into chunks of {w_test_batch.max_n_test}")
    
    if w_test_batch.use_bar_distribution:
        preds_batched = w_test_batch.predict(Xtr_small, ytr_small, Xte_many, prediction_type="mode")
        print(f"Batched test predictions shape: {preds_batched.shape}")
        print(f"First 5 predictions: {preds_batched[:5]}")
    else:
        preds_batched = w_test_batch.predict(Xtr_small, ytr_small, Xte_many, prediction_type="point")
        print(f"Batched test predictions shape: {preds_batched.shape}")
        print(f"First 5 predictions: {preds_batched[:5]}")
    
    print("\n[SimplePFNSklearn] Test 5 completed successfully!")
    
    # TEST 6: Combined - Clustering + Test Batching
    print("\n" + "="*80)
    print("TEST 6: Adaptive Clustering + Test Batching (max_n_train=15, max_n_test=8)")
    print("="*80)
    w_combined = SimplePFNSklearn(
        config_path=cfg_path, 
        checkpoint_path=ckpt_path, 
        device="cpu", 
        verbose=True,
        max_n_train=15,
        max_n_test=8,
        random_state=42
    )
    w_combined.load()
    
    print("\n[SimplePFNSklearn] Making predictions with both clustering and test batching...")
    
    if w_combined.use_bar_distribution:
        preds_combined = w_combined.predict(Xtr_large, ytr_large, Xte_many, prediction_type="mode")
        print(f"Combined predictions shape: {preds_combined.shape}")
    else:
        preds_combined = w_combined.predict(Xtr_large, ytr_large, Xte_many, prediction_type="point")
        print(f"Combined predictions shape: {preds_combined.shape}")
    
    print("\n[SimplePFNSklearn] Test 6 completed successfully!")
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)

