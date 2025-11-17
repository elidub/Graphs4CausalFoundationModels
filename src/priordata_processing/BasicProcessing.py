
import torch
import random
import numpy as np
from typing import Dict, Tuple, Any, Optional

from .Preprocessor import Preprocessor


class BasicProcessing:
    """Wrapper around `Preprocessor` exposing its full capability set.

    This class ONLY adds:
      - Target feature selection (user-specified or random).
      - Optional feature hiding (dropout probability) excluding the target.

    All other preprocessing (shuffling, train/test split, outlier winsorization,
    Yeo-Johnson, standardization, optional [-1,1] scaling, padding) is done by
    `Preprocessor` using parameters passed directly here.

    Parameters (mirror `Preprocessor` + wrapper extras)
    --------------------------------------------------
    n_features : int
        Number of (non-target) features to keep (before padding). Must be <= max_n_features.
    max_n_features : int
        Maximum number of features for padding.
    n_train_samples : int
        Number of training samples (non-zero) required.
    max_n_train_samples : int
        Maximum padded training samples.
    n_test_samples : int
        Number of test samples (non-zero) required.
    max_n_test_samples : int
        Maximum padded test samples.
    dropout_prob : float, default 0.0
        Probability of dropping each non-target feature.
    target_feature : Optional[int]
        Fixed target feature index; if None choose randomly.
    random_seed : Optional[int]
        Reproducibility for target selection, dropout & shuffling.
    negative_one_one_scaling : bool
        Apply [-1,1] scaling after other transforms.
    standardize : bool
        Apply per-feature standardization.
    yeo_johnson : bool
        Apply Yeo-Johnson before standardization.
    remove_outliers : bool
        Winsorize using (1-outlier_quantile, outlier_quantile) on train.
    outlier_quantile : float
        Upper quantile for winsorization (e.g. 0.95).
    shuffle_samples : bool
        Shuffle rows before split.
    shuffle_features : bool
        Shuffle columns before feature selection (affects which features kept when overcomplete).
    y_clip_quantile : Optional[float]
        Optional winsorization of target.
    eps : float
        Numerical epsilon.
    device / dtype : torch.device / torch.dtype
        Optional output casting.
    """

    def __init__(
        self,
        n_features: int,
        max_n_features: int,
        n_train_samples: int,
        max_n_train_samples: int,
        n_test_samples: int,
        max_n_test_samples: int,
        dropout_prob: float = 0.0,
        target_feature: Optional[int] = None,
        random_seed: Optional[int] = None,
        negative_one_one_scaling: bool = True,
        standardize: bool = True,
        yeo_johnson: bool = False,
        remove_outliers: bool = True,
        outlier_quantile: float = 0.95,
        shuffle_samples: bool = True,
        shuffle_features: bool = True,
        y_clip_quantile: Optional[float] = None,
        eps: float = 1e-8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # Store wrapper-specific
        self.dropout_prob = dropout_prob
        self.target_feature = target_feature
        self.random_seed = random_seed

        # Store Preprocessor-compatible config
        self.n_features = n_features
        self.max_n_features = max_n_features
        self.n_train_samples = n_train_samples
        self.max_n_train_samples = max_n_train_samples
        self.n_test_samples = n_test_samples
        self.max_n_test_samples = max_n_test_samples
        self.negative_one_one_scaling = negative_one_one_scaling
        self.standardize = standardize
        self.yeo_johnson = yeo_johnson
        self.remove_outliers = remove_outliers
        self.outlier_quantile = outlier_quantile
        self.shuffle_samples = shuffle_samples
        self.shuffle_features = shuffle_features
        self.y_clip_quantile = y_clip_quantile
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Basic validations mirroring Preprocessor expectations
        assert 0 < outlier_quantile <= 1.0, "outlier_quantile must be in (0,1]"
        assert n_features <= max_n_features, "n_features <= max_n_features"
        assert n_train_samples <= max_n_train_samples, "n_train_samples <= max_n_train_samples"
        assert n_test_samples <= max_n_test_samples, "n_test_samples <= max_n_test_samples"

    # ------------------------------------------------------------------
    def process(
        self,
        dataset: Dict[int, torch.Tensor],
        mode: str = 'fast'
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        if mode not in {"fast", "safe"}:
            raise ValueError("mode must be 'fast' or 'safe'")

        self._validate(dataset, mode)

        feature_indices = sorted(dataset.keys())
        original_num_features = len(feature_indices)
        original_num_samples = next(iter(dataset.values())).shape[0]

        # Build data tensor [N,F_total]
        data_tensor = torch.cat([dataset[i] for i in feature_indices], dim=1)

        # Target selection (exclude features with zero variance)
        target_feat = self._select_target_feature(feature_indices, data_tensor)
        target_col = feature_indices.index(target_feat)
        target_values = data_tensor[:, target_col]

        # Candidate (non-target) columns
        remaining_cols = [i for i in range(len(feature_indices)) if i != target_col]

        # Feature dropout
        kept_cols, dropped_original_indices = self._apply_feature_dropout(remaining_cols, feature_indices)
        
        # Enforce requested n_features (truncate if surplus)
        kept_cols = kept_cols[: self.n_features]

        X_no_target = data_tensor[:, kept_cols]  # (N, n_features)
        Y_tensor = target_values.unsqueeze(-1)

        # Batch dimension for Preprocessor
        X_batch = X_no_target.unsqueeze(0)  # [1,N,F]
        Y_batch = Y_tensor.squeeze(-1).unsqueeze(0)  # [1,N]
        preproc = Preprocessor(
            n_features=self.n_features,
            max_n_features=self.max_n_features,
            n_train_samples=self.n_train_samples,
            max_n_train_samples=self.max_n_train_samples,
            n_test_samples=self.n_test_samples,
            max_n_test_samples=self.max_n_test_samples,
            negative_one_one_scaling=self.negative_one_one_scaling,
            standardize=self.standardize,
            yeo_johnson=self.yeo_johnson,
            remove_outliers=self.remove_outliers,
            outlier_quantile=self.outlier_quantile,
            shuffle_samples=self.shuffle_samples,
            shuffle_features=self.shuffle_features,
            y_clip_quantile=self.y_clip_quantile,
            eps=self.eps,
            device=self.device,
            dtype=self.dtype,
        )

        processed = preproc.process(X_batch, Y_batch)
        if processed is None:
            raise RuntimeError("Preprocessor returned None (internal size validation failed).")

        X_train_b, X_test_b, Y_train_b, Y_test_b = processed
        X_train = X_train_b[0]
        X_test = X_test_b[0]
        Y_train = Y_train_b[0].unsqueeze(-1)
        Y_test = Y_test_b[0].unsqueeze(-1)


        return X_train, Y_train, X_test, Y_test

    # ------------------------------------------------------------------
    def _validate(self, dataset: Dict[int, torch.Tensor], mode: str) -> None:
        if not dataset:
            raise ValueError("dataset empty")
        n_samples_set = {v.shape[0] for v in dataset.values()}
        if len(n_samples_set) != 1:
            raise ValueError("All features must have same number of samples")
        for k, v in dataset.items():
            if v.dim() != 2 or v.shape[1] != 1:
                raise ValueError(f"Feature {k} must have shape (N,1)")
            if mode == 'safe':
                if torch.isnan(v).any():
                    raise ValueError(f"Feature {k} has NaNs")
                if torch.isinf(v).any():
                    raise ValueError(f"Feature {k} has infs")

    def _select_target_feature(self, feature_indices: list, data_tensor: torch.Tensor) -> int:
        """
        Select target feature with preference for high-variance features.
        
        Selection strategy:
        - If user specified target_feature: use that (with zero-variance check)
        - Otherwise:
          - With probability 0.9: Select uniformly at random from features with variance > 0.8 quantile
          - With probability 0.1: Select uniformly at random from features with sufficient variance (> 1e-5)
          - If no features have sufficient variance: Select random feature
        
        Parameters
        ----------
        feature_indices : list
            List of feature indices available in the dataset
        data_tensor : torch.Tensor
            Data tensor of shape (N, F) where F is len(feature_indices)
            
        Returns
        -------
        int
            Selected target feature index
            
        Raises
        ------
        ValueError
            If provided target_feature is not in dataset, has zero variance,
            or all features have zero variance (when user-specified)
        """
        # Compute variance for each feature
        variances = torch.var(data_tensor, dim=0, unbiased=False)  # (F,)
        
        # Filter out features with zero variance (using small epsilon for numerical stability)
        eps = 1e-5
        valid_mask = variances > eps
        valid_cols = [i for i, valid in enumerate(valid_mask) if valid]
        valid_features = [feature_indices[i] for i in valid_cols]
        
        # If user specified a target, check it's valid
        if self.target_feature is not None:
            if self.target_feature not in feature_indices:
                raise ValueError("Provided target_feature not in dataset")
            
            # Check if specified target has zero variance
            target_col = feature_indices.index(self.target_feature)
            if not valid_mask[target_col]:
                raise ValueError(
                    f"Provided target_feature {self.target_feature} has zero variance "
                    f"(trivial prediction task) - cannot be used as target"
                )
            return self.target_feature
        
        # Random selection with variance-based preference
        # Case 1: No features with sufficient variance - just pick randomly
        if len(valid_features) == 0:
            return random.choice(feature_indices)
        
        # Case 2: Features with sufficient variance exist
        # Extract variances for valid features only
        valid_variances = variances[valid_cols]
        
        # Compute 0.8 quantile of variance distribution
        variance_80_quantile = torch.quantile(valid_variances, 0.9).item()
        
        # Find features with variance above 0.8 quantile
        high_variance_mask = valid_variances > variance_80_quantile
        high_variance_cols = [valid_cols[i] for i, is_high in enumerate(high_variance_mask) if is_high]
        high_variance_features = [feature_indices[i] for i in high_variance_cols]
        
        # With probability 0.9, select uniformly from high-variance features
        # With probability 0.1, select uniformly from all valid features
        if random.random() < 0.9 and len(high_variance_features) > 0:
            return random.choice(high_variance_features)
        else:
            return random.choice(valid_features)

    def _apply_feature_dropout(self, remaining_cols: list, feature_indices: list) -> Tuple[list, list]:
        """Randomly drop (hide) some features (by column indices) excluding target.

        Returns
        -------
        kept_cols : list[int]
            Column indices (w.r.t concatenated tensor) kept (excluding target column).
        dropped_original_indices : list[int]
            Original feature indices that were dropped.
        """
        if self.dropout_prob <= 0.0:
            return remaining_cols, []

        kept = []
        dropped = []
        for col in remaining_cols:
            if random.random() < self.dropout_prob:
                dropped.append(feature_indices[col])
            else:
                kept.append(col)

        # Ensure at least one feature kept
        if len(kept) == 0:
            kept = [remaining_cols[0]]
            # remove it from dropped list if present
            dropped = [d for d in dropped if d != feature_indices[remaining_cols[0]]]

        return kept, dropped