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
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, RobustScaler

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
    Simple preprocessing pipeline for tabular data based on TabICL's approach.
    
    Applies scaling, normalization, and outlier handling with diverse strategies.
    """
    
    def __init__(self, normalization_method: str = "power", outlier_strategy: str = "moderate",
                 random_state: Optional[int] = None):
        """
        Args:
            normalization_method: 'power', 'quantile', 'robust', or 'none'
            outlier_strategy: Outlier handling strategy
                - 'none': No outlier removal (threshold = infinity)
                - 'conservative': Remove extreme outliers (threshold = 6.0 std)
                - 'moderate': Remove moderate outliers (threshold = 4.0 std)
                - 'aggressive': Remove more outliers (threshold = 2.5 std)
            random_state: Random seed for reproducibility
        """
        self.normalization_method = normalization_method
        self.outlier_strategy = outlier_strategy
        self.random_state = random_state
        self.fitted = False
        
        # Map outlier strategy to threshold
        self.outlier_thresholds = {
            'none': float('inf'),
            'conservative': 6.0,
            'moderate': 4.0,
            'aggressive': 2.5
        }
        self.outlier_threshold = self.outlier_thresholds.get(outlier_strategy, 4.0)
        
    def fit(self, X, y=None):
        """Fit the preprocessing pipeline on training data."""
        X = np.asarray(X, dtype=np.float32)
        
        # 1. Standard scaling
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # 2. Normalization
        if self.normalization_method != "none":
            if self.normalization_method == "power":
                self.normalizer_ = PowerTransformer(method="yeo-johnson", standardize=True)
            elif self.normalization_method == "quantile":
                self.normalizer_ = QuantileTransformer(output_distribution="normal", random_state=self.random_state)
            elif self.normalization_method == "robust":
                self.normalizer_ = RobustScaler(unit_variance=True)
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            
            self.X_min_ = np.min(X_scaled, axis=0, keepdims=True)
            self.X_max_ = np.max(X_scaled, axis=0, keepdims=True)
            X_normalized = self.normalizer_.fit_transform(X_scaled)
        else:
            self.normalizer_ = None
            X_normalized = X_scaled
        
        # 3. Outlier handling - compute bounds
        if self.outlier_threshold != float('inf'):
            self.outlier_mean_ = np.mean(X_normalized, axis=0)
            self.outlier_std_ = np.std(X_normalized, axis=0, ddof=1 if X.shape[0] > 1 else 0)
            self.outlier_std_ = np.maximum(self.outlier_std_, 1e-6)
            
            self.outlier_lower_ = self.outlier_mean_ - self.outlier_threshold * self.outlier_std_
            self.outlier_upper_ = self.outlier_mean_ + self.outlier_threshold * self.outlier_std_
            
            # Store transformed training data with outlier clipping
            self.X_transformed_ = np.clip(X_normalized, self.outlier_lower_, self.outlier_upper_)
        else:
            # No outlier removal
            self.outlier_lower_ = None
            self.outlier_upper_ = None
            self.X_transformed_ = X_normalized
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted preprocessor."""
        if not self.fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        
        # Standard scaling
        X = self.scaler_.transform(X)
        
        # Normalization
        if self.normalizer_ is not None:
            try:
                X = self.normalizer_.transform(X)
            except ValueError:
                # Clip to training range and retry
                X = np.clip(X, self.X_min_, self.X_max_)
                X = self.normalizer_.transform(X)
        
        # Outlier clipping (if applicable)
        if self.outlier_lower_ is not None and self.outlier_upper_ is not None:
            X = np.clip(X, self.outlier_lower_, self.outlier_upper_)
        
        return X


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
    Create ensemble variants through diverse preprocessing, similar to TabICL.
    
    Generates multiple data variants by:
    1. Applying different normalization methods
    2. Varying outlier removal strategies
    3. Permuting feature orders
    """
    
    def __init__(self, n_estimators: int = 1, 
                 norm_methods: Optional[Union[str, List[str]]] = None,
                 outlier_strategies: Optional[Union[str, List[str]]] = None,
                 feat_shuffle_method: str = "random",
                 random_state: Optional[int] = None):
        """
        Args:
            n_estimators: Number of ensemble variants to create
            norm_methods: Normalization methods to use ('none', 'power', 'quantile', 'robust')
                         If None, uses ['none', 'power', 'quantile']
            outlier_strategies: Outlier handling strategies ('none', 'conservative', 'moderate', 'aggressive')
                               If None, uses ['none', 'moderate', 'aggressive']
            feat_shuffle_method: Feature permutation method ('none', 'random', 'shift')
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.outlier_strategies = outlier_strategies
        self.feat_shuffle_method = feat_shuffle_method
        self.random_state = random_state
        self.fitted = False
        
    def fit(self, X_train, y_train=None):
        """
        Fit ensemble preprocessors on training data.
        
        Args:
            X_train: Training features (N, F)
            y_train: Training targets (N,) - not used but kept for compatibility
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        
        # Default normalization methods
        if self.norm_methods is None:
            self.norm_methods_ = ["none", "power", "quantile"]
        elif isinstance(self.norm_methods, str):
            self.norm_methods_ = [self.norm_methods]
        else:
            self.norm_methods_ = self.norm_methods
        
        # Default outlier strategies
        if self.outlier_strategies is None:
            self.outlier_strategies_ = ["none", "moderate", "aggressive"]
        elif isinstance(self.outlier_strategies, str):
            self.outlier_strategies_ = [self.outlier_strategies]
        else:
            self.outlier_strategies_ = self.outlier_strategies
        
        self.n_features_ = X_train.shape[1]
        
        # Generate ensemble configurations
        rng = random.Random(self.random_state)
        
        # Create feature shuffle patterns
        shuffler = FeatureShuffler(self.n_features_, self.feat_shuffle_method, self.random_state)
        shuffle_patterns = shuffler.generate(self.n_estimators)
        
        # Combine normalization methods, outlier strategies, and shuffle patterns
        all_configs = list(itertools.product(self.norm_methods_, self.outlier_strategies_, shuffle_patterns))
        rng.shuffle(all_configs)
        selected_configs = all_configs[:self.n_estimators]
        
        # Organize by (norm_method, outlier_strategy) pair
        self.ensemble_configs_ = OrderedDict()
        for norm_method, outlier_strategy, shuffle_pattern in selected_configs:
            key = (norm_method, outlier_strategy)
            if key not in self.ensemble_configs_:
                self.ensemble_configs_[key] = []
            self.ensemble_configs_[key].append(shuffle_pattern)
        
        # Fit preprocessors for each (normalization, outlier) combination
        self.preprocessors_ = {}
        for (norm_method, outlier_strategy) in self.ensemble_configs_:
            preprocessor = SimplePreprocessor(
                normalization_method=norm_method,
                outlier_strategy=outlier_strategy,
                random_state=self.random_state
            )
            preprocessor.fit(X_train)
            self.preprocessors_[(norm_method, outlier_strategy)] = preprocessor
        
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
      
      # Ensemble usage (similar to TabICL)
      wrapper = SimplePFNSklearn(config_path="path/to/config.yaml", checkpoint_path="model.pt",
                                 n_estimators=5, norm_methods=["none", "power"])
      wrapper.load()
      wrapper.fit(X_train, y_train)  # Fits ensemble preprocessors
      preds = wrapper.predict(X_train, y_train, X_test, prediction_type="mode", aggregate="mean")

    Parameters:
      n_estimators: Number of ensemble members (default: 4)
      norm_methods: Normalization methods for ensemble ('none', 'power', 'quantile', 'robust')
                    Default: ['none', 'power', 'quantile', 'robust']
      outlier_strategies: Outlier removal strategies ('none', 'conservative', 'moderate', 'aggressive')
                         Default: ['none', 'moderate', 'aggressive']
                         - 'none': No outlier removal
                         - 'conservative': Remove extreme outliers (6.0 std)
                         - 'moderate': Remove moderate outliers (4.0 std)
                         - 'aggressive': Remove more outliers (2.5 std)
      feat_shuffle_method: Feature permutation method ('none', 'random', 'shift')
      
    Ensemble Diversity:
      The ensemble creates diverse preprocessing variants by combining:
      1. Multiple normalization methods (power transform, quantile transform, robust scaling, or none)
      2. Different outlier removal strategies (none, conservative, moderate, aggressive)
      3. Feature permutations (random shuffling or circular shifts)
      
      This results in n_estimators different views of the data, which are processed in a single
      batched forward pass for efficiency, then aggregated (mean/median) or returned individually.
      
    Notes:
    - With n_estimators > 1, the model creates diverse preprocessing variants
    - All ensemble variants are processed in a SINGLE batched forward pass (efficient!)
    - Ensemble predictions are aggregated using mean, median, or returned as array
    - When BarDistribution is enabled, ensemble works with all prediction types
    - Input shapes are automatically converted to match training format
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = "cpu",
        verbose: bool = False,
        n_estimators: int = 10,
        norm_methods: Optional[Union[str, List[str]]] = ["none", "power", "quantile", "robust"],
        outlier_strategies: Optional[Union[str, List[str]]] = ["none", "moderate", "aggressive"],
        feat_shuffle_method: str = "random",
        random_state: Optional[int] = None,
    ):
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
                # Respect feature positional encodings flag and rank from YAML config if present
                "use_feature_positional": bool(mcfg.get("use_feature_positional", {}).get("value", False)),
                "feature_pos_rank": int(mcfg.get("feature_pos_rank", {}).get("value", 16)),
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

    def fit_bar_distribution(self, X_train_data, y_train_data, X_test_data, y_test_data, 
                           max_batches: Optional[int] = None) -> "SimplePFNSklearn":
        """
        Fit the BarDistribution to data. This must be called before prediction if BarDistribution is enabled.
        
        Args:
            X_train_data, y_train_data, X_test_data, y_test_data: Training and test data arrays
                Can be lists of arrays (multiple datasets) or single arrays (single dataset)
            max_batches: Maximum number of batches to use for fitting
        """
        if not self.use_bar_distribution or self.bar_distribution is None:
            if self.verbose:
                print("[SimplePFNSklearn] BarDistribution not enabled, skipping fit.")
            return self
            
        # Convert data to proper format for BarDistribution fitting
        if not isinstance(X_train_data, list):
            X_train_data = [X_train_data]
            y_train_data = [y_train_data] 
            X_test_data = [X_test_data]
            y_test_data = [y_test_data]
            
        # Create a simple iterator that yields (X_train, y_train, X_test, y_test) tuples
        class SimpleDataIterator:
            def __init__(self, X_tr_list, y_tr_list, X_te_list, y_te_list):
                self.data = list(zip(X_tr_list, y_tr_list, X_te_list, y_te_list))
                
            def __iter__(self):
                for X_tr, y_tr, X_te, y_te in self.data:
                    # Convert to tensors and ensure proper shapes
                    X_tr_tensor = torch.as_tensor(np.asarray(X_tr), dtype=torch.float32)
                    y_tr_tensor = torch.as_tensor(np.asarray(y_tr), dtype=torch.float32)
                    X_te_tensor = torch.as_tensor(np.asarray(X_te), dtype=torch.float32)
                    y_te_tensor = torch.as_tensor(np.asarray(y_te), dtype=torch.float32)
                    
                    # Ensure batch dimensions
                    if X_tr_tensor.ndim == 2:
                        X_tr_tensor = X_tr_tensor.unsqueeze(0)
                    if X_te_tensor.ndim == 2:
                        X_te_tensor = X_te_tensor.unsqueeze(0)
                    if y_tr_tensor.ndim == 1:
                        y_tr_tensor = y_tr_tensor.unsqueeze(0)
                    if y_te_tensor.ndim == 1:
                        y_te_tensor = y_te_tensor.unsqueeze(0)
                        
                    # CRITICAL FIX: Ensure y tensors have the training-expected shape (B, N, 1)
                    if y_tr_tensor.ndim == 2:
                        y_tr_tensor = y_tr_tensor.unsqueeze(-1)  # (B, N) -> (B, N, 1)
                    if y_te_tensor.ndim == 2:
                        y_te_tensor = y_te_tensor.unsqueeze(-1)  # (B, M) -> (B, M, 1)
                        
                    yield (X_tr_tensor, y_tr_tensor, X_te_tensor, y_te_tensor)
        
        data_iter = SimpleDataIterator(X_train_data, y_train_data, X_test_data, y_test_data)
        
        if self.verbose:
            print(f"[SimplePFNSklearn] Fitting BarDistribution with {len(X_train_data)} datasets...")
            
        self.bar_distribution.fit(data_iter, max_batches=max_batches)
        
        if self.verbose:
            print("[SimplePFNSklearn] BarDistribution fitted successfully.")
            
        return self

    def fit(self, X: Any = None, y: Any = None, **kwargs) -> "SimplePFNSklearn":
        """Fit ensemble preprocessors if ensemble is enabled. Otherwise placeholder.

        Args:
            X: Training features (N, F)
            y: Training targets (N,)
            
        If n_estimators > 1, this fits the ensemble preprocessors on the training data.
        If n_estimators == 1, this is a no-op placeholder (training done with project's trainer).
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
        
        With ensemble (n_estimators > 1), creates multiple preprocessed variants and aggregates predictions.

        Args:
            X_train, y_train, X_test: Input data (numpy arrays or torch tensors)
                Accepted input shapes (automatically converted to training format):
                - X_train: (N, F) or (B, N, F) -> converted to (B, N, F)
                - y_train: (N,) or (N,1) or (B, N) or (B,N,1) -> converted to (B, N, 1)  
                - X_test: (M, F) or (B, M, F) -> converted to (B, M, F)
                
                Note: All shapes are validated to ensure consistency with training format.
                
            prediction_type: Type of prediction to return
                - "point": Raw model output (default for MSE) or mode (default for BarDistribution)
                - "mode": Posterior mode (requires BarDistribution)
                - "mean": Posterior mean (requires BarDistribution)
                - "sample": Samples from posterior (requires BarDistribution)
            num_samples: Number of samples to return when prediction_type="sample"
            aggregate: How to aggregate ensemble predictions
                - "mean": Average predictions across ensemble members (default)
                - "median": Median of predictions across ensemble members
                - "none": Return all predictions as array (B, n_estimators, M) or (n_estimators, M)

        Returns:
            numpy array of predictions:
            - For "point", "mode", "mean": shape (B, M) or (M,) for batch size 1
            - For "sample": shape (B, num_samples, M) or (num_samples, M) for batch size 1
            - For aggregate="none": adds ensemble dimension
            
        Raises:
            ValueError: If input shapes are inconsistent or incompatible with training format
            RuntimeError: If model not loaded or ensemble not fitted when required
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call load() before predict().")
        
        # Validate prediction type
        if prediction_type in ["mode", "mean", "sample"] and not self.use_bar_distribution:
            raise ValueError(f"prediction_type='{prediction_type}' requires BarDistribution to be enabled in config.")
        
        if self.use_bar_distribution and self.bar_distribution is None:
            raise RuntimeError("BarDistribution enabled but not fitted. Call fit_bar_distribution() first.")

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
    
    def _predict_single(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                       prediction_type: str, num_samples: int) -> np.ndarray:
        """Make predictions with a single model (no ensemble)."""
        
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
        """Make predictions with ensemble and aggregate results.
        
        Efficiently batches all ensemble variants together in a single model forward pass.
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
    yte_true = _np.random.randn(3).astype(_np.float32)  # For BarDistribution fitting

    # If BarDistribution is enabled, fit it first
    if w.use_bar_distribution:
        print("[SimplePFNSklearn] Fitting BarDistribution...")
        # Create multiple synthetic datasets for fitting
        train_datasets = [_np.random.randn(10, num_features).astype(_np.float32) for _ in range(5)]
        train_targets = [_np.random.randn(10).astype(_np.float32) for _ in range(5)]
        test_datasets = [_np.random.randn(5, num_features).astype(_np.float32) for _ in range(5)]
        test_targets = [_np.random.randn(5).astype(_np.float32) for _ in range(5)]
        
        w.fit_bar_distribution(train_datasets, train_targets, test_datasets, test_targets, max_batches=5)
        
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
        # Need to fit BarDistribution first
        w_ensemble.fit_bar_distribution(train_datasets, train_targets, test_datasets, test_targets, max_batches=5)
        
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
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
