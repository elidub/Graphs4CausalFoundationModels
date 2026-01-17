"""
Preprocessing wrapper for Graph-Conditioned InterventionalPFN models.

This module extends GraphConditionedInterventionalPFNSklearn with automatic
preprocessing of input data to match the training configuration.

Key features:
- Automatic feature truncation/padding to match model's expected feature count
- Outlier removal (clipping to quantiles)
- Feature standardization
- Target scaling to [-1, 1] range
- Sample truncation/padding to match model's expected sample counts
- Automatic adjacency matrix construction for causal inference

Usage:
    wrapper = PreprocessingGraphConditionedPFN(
        config_path="config.yaml",
        checkpoint_path="model.pt"
    )
    wrapper.load()
    
    # Fit preprocessing on training data
    wrapper.fit(X_train, T_train, Y_train)
    
    # Prediction with automatic preprocessing
    preds = wrapper.predict(
        X_obs, T_obs, Y_obs,
        X_intv, T_intv,
        adjacency_matrix=None,  # Auto-constructed if None
        prediction_type="mean"
    )
"""

from __future__ import annotations
from typing import Optional, Any, Literal, Dict, Tuple
import numpy as np
import yaml

# Import base class
try:
    from models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn
except ImportError:
    try:
        from src.models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn
    except ImportError:
        from GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn


class PreprocessingGraphConditionedPFN(GraphConditionedInterventionalPFNSklearn):
    """
    Graph-Conditioned InterventionalPFN with automatic preprocessing.
    
    This class extends GraphConditionedInterventionalPFNSklearn to automatically
    apply the same preprocessing that was used during training. This includes:
    
    - Feature truncation/padding to match model's num_features
    - Outlier removal via quantile clipping
    - Feature standardization (zero mean, unit variance)
    - Target scaling to [-1, 1] range
    - Sample truncation/padding to match model's expected sample counts
    
    The preprocessing parameters are extracted from the model's config file.
    
    Parameters:
        config_path (str): Path to YAML config file
        checkpoint_path (str): Path to model checkpoint
        device (str, optional): Device for inference ('cpu', 'cuda', etc.)
        verbose (bool): Print detailed information
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            verbose=verbose,
        )
        
        # Preprocessing config (loaded from config file)
        self.preprocessing_config: Dict[str, Any] = {}
        self.dataset_config: Dict[str, Any] = {}
        
        # Fitted preprocessing parameters (computed from training data)
        self._fitted = False
        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._lower_bounds: Optional[np.ndarray] = None
        self._upper_bounds: Optional[np.ndarray] = None
        self._y_min: float = 0.0
        self._y_max: float = 1.0
        self._y_range: float = 1.0
        self._n_original_features: int = 0
        
    def _get_config_value(self, config_dict: Dict, key: str, default: Any = None) -> Any:
        """Extract value from config entry that may be plain or dict with 'value' key."""
        raw = config_dict.get(key, default)
        if isinstance(raw, dict) and "value" in raw:
            return raw["value"]
        return raw
    
    def load(self, override_kwargs: Optional[dict[str, Any]] = None) -> "PreprocessingGraphConditionedPFN":
        """
        Load config, build model, load checkpoint, and extract preprocessing config.
        
        Args:
            override_kwargs: Optional dict to override config parameters
            
        Returns:
            self for method chaining
        """
        # Call parent load
        super().load(override_kwargs)
        
        # Load preprocessing and dataset config from config file
        if self.config_path:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.preprocessing_config = config.get('preprocessing_config', {})
            self.dataset_config = config.get('dataset_config', {})
            
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Loaded preprocessing config:")
                print(f"  remove_outliers: {self._get_config_value(self.preprocessing_config, 'remove_outliers', True)}")
                print(f"  outlier_quantile: {self._get_config_value(self.preprocessing_config, 'outlier_quantile', 0.99)}")
                print(f"  feature_standardize: {self._get_config_value(self.preprocessing_config, 'feature_standardize', True)}")
        
        return self
    
    def fit(
        self,
        X: np.ndarray,
        T: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
    ) -> "PreprocessingGraphConditionedPFN":
        """
        Fit preprocessing parameters on training data.
        
        This computes the statistics needed for preprocessing:
        - Outlier bounds (quantiles) from X
        - Feature mean and std from X (after outlier removal)
        - Target range from Y (for scaling to [-1, 1])
        
        Args:
            X: Training features, shape (N, L)
            T: Training treatment, shape (N,) or (N, 1) - optional, not used for fitting
            Y: Training targets, shape (N,) or (N, 1) - used for target scaling
            
        Returns:
            self for method chaining
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        X = np.asarray(X, dtype=np.float32)
        self._n_original_features = X.shape[1]
        
        # Get model's expected number of features
        model_n_features = self.model.num_features
        
        # Truncate or pad features to match model size
        if X.shape[1] > model_n_features:
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Truncating features from {X.shape[1]} to {model_n_features}")
            X = X[:, :model_n_features]
        elif X.shape[1] < model_n_features:
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Padding features from {X.shape[1]} to {model_n_features}")
            X = np.hstack([X, np.zeros((X.shape[0], model_n_features - X.shape[1]), dtype=np.float32)])
        
        # Compute outlier bounds
        remove_outliers = self._get_config_value(self.preprocessing_config, 'remove_outliers', True)
        outlier_quantile = self._get_config_value(self.preprocessing_config, 'outlier_quantile', 0.99)
        
        if remove_outliers:
            lower_quantile = 1.0 - outlier_quantile
            upper_quantile = outlier_quantile
            self._lower_bounds = np.quantile(X, lower_quantile, axis=0)
            self._upper_bounds = np.quantile(X, upper_quantile, axis=0)
            
            # Apply outlier removal for computing standardization stats
            X = np.clip(X, self._lower_bounds, self._upper_bounds)
            
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Computed outlier bounds at quantile {outlier_quantile}")
        
        # Compute standardization stats
        feature_standardize = self._get_config_value(self.preprocessing_config, 'feature_standardize', True)
        
        if feature_standardize:
            self._X_mean = np.mean(X, axis=0, keepdims=True)
            self._X_std = np.std(X, axis=0, keepdims=True)
            self._X_std = np.where(self._X_std < 1e-8, 1.0, self._X_std)  # Avoid division by zero
            
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Computed standardization stats")
        
        # Compute target scaling stats
        if Y is not None:
            Y = np.asarray(Y, dtype=np.float32).flatten()
            self._y_min = float(np.min(Y))
            self._y_max = float(np.max(Y))
            self._y_range = max(self._y_max - self._y_min, 1e-8)
            
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Target range: [{self._y_min:.4f}, {self._y_max:.4f}]")
        
        self._fitted = True
        return self
    
    def _preprocess_features(
        self,
        X: np.ndarray,
        fit_mode: bool = False,
    ) -> np.ndarray:
        """
        Apply feature preprocessing (truncation/padding, outlier removal, standardization).
        
        Args:
            X: Features, shape (N, L) or (B, N, L)
            fit_mode: If True, use the data itself for stats (for training data)
            
        Returns:
            Preprocessed features with shape matching model's num_features
        """
        X = np.asarray(X, dtype=np.float32)
        original_shape = X.shape
        is_batched = X.ndim == 3
        
        if is_batched:
            B, N, L = X.shape
        else:
            N, L = X.shape
            B = 1
            X = X[np.newaxis, ...]  # Add batch dim for uniform processing
        
        model_n_features = self.model.num_features
        
        # Truncate or pad features
        if L > model_n_features:
            X = X[:, :, :model_n_features]
        elif L < model_n_features:
            pad_width = model_n_features - L
            X = np.concatenate([X, np.zeros((B, N, pad_width), dtype=np.float32)], axis=2)
        
        # Apply outlier removal
        remove_outliers = self._get_config_value(self.preprocessing_config, 'remove_outliers', True)
        if remove_outliers and self._lower_bounds is not None and self._upper_bounds is not None:
            X = np.clip(X, self._lower_bounds, self._upper_bounds)
        
        # Apply standardization
        feature_standardize = self._get_config_value(self.preprocessing_config, 'feature_standardize', True)
        if feature_standardize and self._X_mean is not None and self._X_std is not None:
            X = (X - self._X_mean) / self._X_std
        
        # Remove batch dim if input wasn't batched
        if not is_batched:
            X = X[0]
        
        return X
    
    def _preprocess_targets(self, Y: np.ndarray) -> np.ndarray:
        """
        Scale targets to [-1, 1] range.
        
        Args:
            Y: Targets, shape (N,) or (N, 1) or (B, N) or (B, N, 1)
            
        Returns:
            Scaled targets in [-1, 1] range
        """
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim > 1 and Y.shape[-1] == 1:
            Y = Y.squeeze(-1)
        
        # Scale to [-1, 1]: Y_scaled = 2.0 * (Y - ymin) / range - 1.0
        Y_scaled = 2.0 * (Y - self._y_min) / self._y_range - 1.0
        return Y_scaled
    
    def _inverse_transform_predictions(self, Y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions from [-1, 1] back to original scale.
        
        Args:
            Y_scaled: Scaled predictions in [-1, 1] range
            
        Returns:
            Predictions in original scale
        """
        # Inverse: Y_original = (Y_scaled + 1.0) * range / 2.0 + ymin
        Y_original = (Y_scaled + 1.0) * self._y_range / 2.0 + self._y_min
        return Y_original
    
    def _inverse_transform_cate(self, cate_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform CATE predictions from scaled to original scale.
        
        For CATE (difference), we only need to scale by the range.
        
        Args:
            cate_scaled: Scaled CATE predictions
            
        Returns:
            CATE in original scale
        """
        # CATE_original = CATE_scaled * range / 2.0
        return cate_scaled * self._y_range / 2.0
    
    def _pad_or_truncate_samples(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: Optional[np.ndarray],
        max_samples: int,
        is_train: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
        """
        Pad or truncate samples to match model's expected sample count.
        
        Args:
            X: Features, shape (N, L)
            T: Treatment, shape (N,) or (N, 1)
            Y: Targets, shape (N,) or None
            max_samples: Maximum number of samples
            is_train: Whether this is training data (affects truncation strategy)
            
        Returns:
            Tuple of (X, T, Y, n_original) where n_original is the original sample count
        """
        n_samples = X.shape[0]
        n_original = n_samples
        
        # Ensure T is 2D
        T = np.asarray(T, dtype=np.float32)
        if T.ndim == 1:
            T = T.reshape(-1, 1)
        
        # Truncate if needed
        if n_samples > max_samples:
            if is_train:
                # Random subsample for training
                indices = np.random.choice(n_samples, max_samples, replace=False)
                X = X[indices]
                T = T[indices]
                if Y is not None:
                    Y = Y[indices] if Y.ndim == 1 else Y[indices]
            else:
                # Just truncate for test (caller should handle batching)
                X = X[:max_samples]
                T = T[:max_samples]
                if Y is not None:
                    Y = Y[:max_samples] if Y.ndim == 1 else Y[:max_samples]
            n_samples = max_samples
        
        # Pad if needed
        if n_samples < max_samples:
            pad_size = max_samples - n_samples
            X = np.vstack([X, np.zeros((pad_size, X.shape[1]), dtype=np.float32)])
            T = np.vstack([T, np.zeros((pad_size, T.shape[1]), dtype=np.float32)])
            if Y is not None:
                Y = np.concatenate([Y, np.zeros(pad_size, dtype=np.float32)])
        
        return X, T, Y, n_original
    
    def _build_adjacency_matrix(
        self,
        n_real_features: int,
    ) -> np.ndarray:
        """
        Build a partial adjacency matrix for causal inference.
        
        Creates a matrix with known causal structure:
        - T -> Y (treatment causes outcome)
        - Real features -> T (features cause treatment)
        - Real features -> Y (features cause outcome)
        - Padded features have no edges (-1)
        
        Args:
            n_real_features: Number of real (non-padded) features
            
        Returns:
            Adjacency matrix of shape (model_n_features + 2, model_n_features + 2)
        """
        model_n_features = self.model.num_features
        
        # Initialize all as unknown (0)
        adjacency_matrix = np.zeros((model_n_features + 2, model_n_features + 2), dtype=np.float32)
        
        T_idx = model_n_features
        Y_idx = model_n_features + 1
        
        # Known edges
        # 1. T -> Y
        adjacency_matrix[T_idx, Y_idx] = 1.0
        
        # 2. Real features -> T
        for i in range(n_real_features):
            adjacency_matrix[i, T_idx] = 1.0
        
        # 3. Real features -> Y
        for i in range(n_real_features):
            adjacency_matrix[i, Y_idx] = 1.0
        
        # 4. Padded features have no edges (-1)
        for i in range(n_real_features, model_n_features):
            adjacency_matrix[i, :] = -1.0
            adjacency_matrix[:, i] = -1.0
            adjacency_matrix[i, i] = -1.0
        
        return adjacency_matrix
    
    def predict(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        prediction_type: Literal["point", "mode", "mean", "sample"] = "mean",
        num_samples: int = 100,
        batched: bool = False,
        inverse_transform: bool = True,
    ) -> np.ndarray:
        """
        Make predictions with automatic preprocessing.
        
        This method:
        1. Preprocesses features (truncation/padding, outlier removal, standardization)
        2. Scales targets to [-1, 1]
        3. Pads/truncates samples to match model capacity
        4. Builds adjacency matrix if not provided
        5. Calls parent predict()
        6. Inverse transforms predictions to original scale
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational treatment (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional treatment (M,) or (M, 1)
            adjacency_matrix: Causal graph (L+2, L+2). If None, auto-constructed.
            prediction_type: "point", "mode", "mean", or "sample"
            num_samples: Number of samples for prediction_type="sample"
            batched: If True, expects batched inputs
            inverse_transform: If True, inverse transform predictions to original scale
            
        Returns:
            Predictions in original scale (if inverse_transform=True)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessing not fitted. Call fit() first.")
        
        if batched:
            raise NotImplementedError("Batched mode not yet implemented for PreprocessingGraphConditionedPFN")
        
        # Convert to numpy
        X_obs = np.asarray(X_obs, dtype=np.float32)
        T_obs = np.asarray(T_obs, dtype=np.float32)
        Y_obs = np.asarray(Y_obs, dtype=np.float32)
        X_intv = np.asarray(X_intv, dtype=np.float32)
        T_intv = np.asarray(T_intv, dtype=np.float32)
        
        n_real_features = min(X_obs.shape[1], self.model.num_features)
        
        # Preprocess features
        X_obs = self._preprocess_features(X_obs)
        X_intv = self._preprocess_features(X_intv)
        
        # Scale targets
        Y_obs_scaled = self._preprocess_targets(Y_obs)
        
        # Get max sample counts from config
        max_n_train = self._get_config_value(self.dataset_config, 'max_number_train_samples_per_dataset', X_obs.shape[0])
        max_n_test = self._get_config_value(self.dataset_config, 'max_number_test_samples_per_dataset', X_intv.shape[0])
        
        # Pad/truncate samples
        X_obs, T_obs, Y_obs_scaled, n_train = self._pad_or_truncate_samples(
            X_obs, T_obs, Y_obs_scaled, max_n_train, is_train=True
        )
        
        n_test_original = X_intv.shape[0]
        
        # Handle test samples in batches if needed
        if n_test_original > max_n_test:
            if self.verbose:
                print(f"[PreprocessingGraphConditionedPFN] Processing {n_test_original} test samples in batches of {max_n_test}")
            
            all_preds = []
            for batch_start in range(0, n_test_original, max_n_test):
                batch_end = min(batch_start + max_n_test, n_test_original)
                batch_size = batch_end - batch_start
                
                X_intv_batch = X_intv[batch_start:batch_end]
                T_intv_batch = T_intv[batch_start:batch_end]
                
                X_intv_batch, T_intv_batch, _, _ = self._pad_or_truncate_samples(
                    X_intv_batch, T_intv_batch, None, max_n_test, is_train=False
                )
                
                # Build adjacency matrix
                adj_matrix = adjacency_matrix if adjacency_matrix is not None else self._build_adjacency_matrix(n_real_features)
                
                # Call parent predict
                preds_batch = super().predict(
                    X_obs=X_obs,
                    T_obs=T_obs,
                    Y_obs=Y_obs_scaled,
                    X_intv=X_intv_batch,
                    T_intv=T_intv_batch,
                    adjacency_matrix=adj_matrix,
                    prediction_type=prediction_type,
                    num_samples=num_samples,
                    batched=False,
                )
                
                # Only keep non-padded predictions
                all_preds.append(preds_batch[:batch_size])
            
            preds = np.concatenate(all_preds)
        else:
            X_intv, T_intv, _, n_test = self._pad_or_truncate_samples(
                X_intv, T_intv, None, max_n_test, is_train=False
            )
            
            # Build adjacency matrix
            if adjacency_matrix is None:
                adjacency_matrix = self._build_adjacency_matrix(n_real_features)
            
            # Call parent predict
            preds = super().predict(
                X_obs=X_obs,
                T_obs=T_obs,
                Y_obs=Y_obs_scaled,
                X_intv=X_intv,
                T_intv=T_intv,
                adjacency_matrix=adjacency_matrix,
                prediction_type=prediction_type,
                num_samples=num_samples,
                batched=False,
            )
            
            # Only keep non-padded predictions
            preds = preds[:n_test_original]
        
        # Inverse transform
        if inverse_transform:
            preds = self._inverse_transform_predictions(preds)
        
        return preds
    
    def predict_cate(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        prediction_type: Literal["point", "mode", "mean"] = "mean",
    ) -> np.ndarray:
        """
        Predict Conditional Average Treatment Effect (CATE).
        
        CATE = E[Y|T=1, X] - E[Y|T=0, X]
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational treatment (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features to predict CATE for (M, L)
            adjacency_matrix: Causal graph (L+2, L+2). If None, auto-constructed.
            prediction_type: "point", "mode", or "mean"
            
        Returns:
            CATE predictions of shape (M,)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessing not fitted. Call fit() first.")
        
        n_test = X_intv.shape[0]
        
        # Predict Y for T=1
        T_intv_1 = np.ones((n_test, 1), dtype=np.float32)
        y_pred_1 = self.predict(
            X_obs=X_obs,
            T_obs=T_obs,
            Y_obs=Y_obs,
            X_intv=X_intv,
            T_intv=T_intv_1,
            adjacency_matrix=adjacency_matrix,
            prediction_type=prediction_type,
            inverse_transform=False,  # Don't inverse transform yet
        )
        
        # Predict Y for T=0
        T_intv_0 = np.zeros((n_test, 1), dtype=np.float32)
        y_pred_0 = self.predict(
            X_obs=X_obs,
            T_obs=T_obs,
            Y_obs=Y_obs,
            X_intv=X_intv,
            T_intv=T_intv_0,
            adjacency_matrix=adjacency_matrix,
            prediction_type=prediction_type,
            inverse_transform=False,  # Don't inverse transform yet
        )
        
        # CATE = E[Y|T=1,X] - E[Y|T=0,X] (in scaled space)
        cate_scaled = y_pred_1 - y_pred_0
        
        # Inverse transform CATE to original scale
        cate = self._inverse_transform_cate(cate_scaled)
        
        return cate
    
    def log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        batched: bool = False,
    ) -> np.ndarray:
        """
        Compute log-likelihood with automatic preprocessing.
        
        Args:
            X_obs: Observational features (N, L)
            T_obs: Observational treatment (N,) or (N, 1)
            Y_obs: Observational targets (N,) or (N, 1)
            X_intv: Interventional features (M, L)
            T_intv: Interventional treatment (M,) or (M, 1)
            Y_intv: Interventional targets (ground truth) (M,) or (M, 1)
            adjacency_matrix: Causal graph (L+2, L+2). If None, auto-constructed.
            batched: If True, expects batched inputs
            
        Returns:
            Log-likelihood values of shape (M,)
        """
        if not self._fitted:
            raise RuntimeError("Preprocessing not fitted. Call fit() first.")
        
        if batched:
            raise NotImplementedError("Batched mode not yet implemented for PreprocessingGraphConditionedPFN")
        
        # Convert to numpy
        X_obs = np.asarray(X_obs, dtype=np.float32)
        T_obs = np.asarray(T_obs, dtype=np.float32)
        Y_obs = np.asarray(Y_obs, dtype=np.float32)
        X_intv = np.asarray(X_intv, dtype=np.float32)
        T_intv = np.asarray(T_intv, dtype=np.float32)
        Y_intv = np.asarray(Y_intv, dtype=np.float32)
        
        n_real_features = min(X_obs.shape[1], self.model.num_features)
        
        # Preprocess features
        X_obs = self._preprocess_features(X_obs)
        X_intv = self._preprocess_features(X_intv)
        
        # Scale targets
        Y_obs_scaled = self._preprocess_targets(Y_obs)
        Y_intv_scaled = self._preprocess_targets(Y_intv)
        
        # Get max sample counts from config
        max_n_train = self._get_config_value(self.dataset_config, 'max_number_train_samples_per_dataset', X_obs.shape[0])
        max_n_test = self._get_config_value(self.dataset_config, 'max_number_test_samples_per_dataset', X_intv.shape[0])
        
        # Pad/truncate samples
        X_obs, T_obs, Y_obs_scaled, n_train = self._pad_or_truncate_samples(
            X_obs, T_obs, Y_obs_scaled, max_n_train, is_train=True
        )
        
        n_test_original = X_intv.shape[0]
        X_intv, T_intv, Y_intv_scaled, n_test = self._pad_or_truncate_samples(
            X_intv, T_intv, Y_intv_scaled, max_n_test, is_train=False
        )
        
        # Build adjacency matrix
        if adjacency_matrix is None:
            adjacency_matrix = self._build_adjacency_matrix(n_real_features)
        
        # Call parent log_likelihood
        log_probs = super().log_likelihood(
            X_obs=X_obs,
            T_obs=T_obs,
            Y_obs=Y_obs_scaled,
            X_intv=X_intv,
            T_intv=T_intv,
            Y_intv=Y_intv_scaled,
            adjacency_matrix=adjacency_matrix,
            batched=False,
        )
        
        # Only keep non-padded results
        return log_probs[:n_test_original]
    
    def predict_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        batched: bool = False,
    ) -> np.ndarray:
        """Alias for log_likelihood."""
        return self.log_likelihood(
            X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix, batched
        )
    
    def predict_negative_log_likelihood(
        self,
        X_obs: Any,
        T_obs: Any,
        Y_obs: Any,
        X_intv: Any,
        T_intv: Any,
        Y_intv: Any,
        adjacency_matrix: Optional[Any] = None,
        batched: bool = False,
    ) -> np.ndarray:
        """Compute negative log-likelihood."""
        return -self.log_likelihood(
            X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv, adjacency_matrix, batched
        )


if __name__ == "__main__":
    """Test the PreprocessingGraphConditionedPFN class."""
    print("\n" + "="*80)
    print("TEST: PreprocessingGraphConditionedPFN")
    print("="*80)
    
    # Example usage (requires actual config and checkpoint)
    config_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist_config.yaml"
    checkpoint_path = "/Users/arikreuter/Documents/PhD/CausalPriorFitting/experiments/FirstTests/checkpoints/final_earlytest_16773250.0/final_model_with_bardist.pt"
    
    import os
    if os.path.exists(config_path) and os.path.exists(checkpoint_path):
        print(f"\nLoading model from:")
        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        wrapper = PreprocessingGraphConditionedPFN(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device="cpu",
            verbose=True,
        )
        wrapper.load()
        
        # Create synthetic data
        np.random.seed(42)
        N_train, N_test, L = 100, 20, 10
        
        X_train = np.random.randn(N_train, L).astype(np.float32)
        T_train = np.random.randint(0, 2, (N_train, 1)).astype(np.float32)
        Y_train = np.random.randn(N_train).astype(np.float32)
        
        X_test = np.random.randn(N_test, L).astype(np.float32)
        T_test = np.random.randint(0, 2, (N_test, 1)).astype(np.float32)
        Y_test = np.random.randn(N_test).astype(np.float32)
        
        print(f"\nFitting preprocessing on training data...")
        wrapper.fit(X_train, T_train, Y_train)
        
        print(f"\nMaking predictions...")
        preds = wrapper.predict(
            X_obs=X_train,
            T_obs=T_train,
            Y_obs=Y_train,
            X_intv=X_test,
            T_intv=T_test,
            prediction_type="mean",
        )
        print(f"  Predictions shape: {preds.shape}")
        print(f"  Predictions range: [{preds.min():.4f}, {preds.max():.4f}]")
        
        print(f"\nPredicting CATE...")
        cate = wrapper.predict_cate(
            X_obs=X_train,
            T_obs=T_train,
            Y_obs=Y_train,
            X_intv=X_test,
        )
        print(f"  CATE shape: {cate.shape}")
        print(f"  CATE range: [{cate.min():.4f}, {cate.max():.4f}]")
        
        print(f"\nComputing log-likelihood...")
        log_probs = wrapper.log_likelihood(
            X_obs=X_train,
            T_obs=T_train,
            Y_obs=Y_train,
            X_intv=X_test,
            T_intv=T_test,
            Y_intv=Y_test,
        )
        print(f"  Log-likelihood shape: {log_probs.shape}")
        print(f"  Mean log-likelihood: {log_probs.mean():.4f}")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
    else:
        print(f"\nSkipping tests - config or checkpoint not found:")
        print(f"  Config exists: {os.path.exists(config_path)}")
        print(f"  Checkpoint exists: {os.path.exists(checkpoint_path)}")
