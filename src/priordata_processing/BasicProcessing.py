
import torch
import random
import numpy as np
from typing import Dict, Tuple, Any, Optional

from .Preprocessor import Preprocessor


class BasicProcessing:
    """Lightweight wrapper around Preprocessor.

    Responsibilities kept here (as requested):
      1. Select the target feature (user specified or random).
      2. Hide (drop) some features via dropout_prob (never dropping the target).

    All *actual* preprocessing (shuffling, splitting, winsorization, Yeo-Johnson,
    standardization, optional scaling and padding) is delegated to `Preprocessor`.

    Previous functionality such as target encoding, manual transforms, manual padding
    has been removed to avoid duplication.

    Parameters
    ----------
    max_num_samples : int
        Maximum total samples (train+test) before per-split padding.
    max_num_features : int
        Maximum number of (non-target) features after padding.
    train_fraction : float, default 0.5
        Fraction of samples used for training (rest for test).
    dropout_prob : float, default 0.0
        Probability of dropping EACH non-target feature (feature hiding). Ensures at least one remains.
    transformation_type : str, {'standardize','yeo_johnson'}
        Chooses which transformation pipeline the underlying Preprocessor applies.
        'standardize' -> standardization only.
        'yeo_johnson' -> Yeo-Johnson then standardization.
    shuffle_data : bool, default True
        If True, enables both sample and feature shuffling inside the Preprocessor.
    target_feature : Optional[int]
        If provided, use this feature index as target; else choose randomly.
    random_seed : Optional[int]
        Reproducibility for target selection, feature dropout, and underlying shuffles.

    Notes
    -----
    - The target column is removed from the feature tensor before calling the Preprocessor.
    - We wrap data into a single batch (B=1) for the Preprocessor which expects [B,N,F].
    - Output batch dimension is stripped to keep the legacy BasicProcessing API.
    """

    def __init__(
        self,
        max_num_samples: int,
        max_num_features: int,
        train_fraction: float = 0.5,
        dropout_prob: float = 0.0,
        transformation_type: str = 'standardize',
        shuffle_data: bool = True,
        target_feature: Optional[int] = None,
        random_seed: Optional[int] = None,
        negative_one_one_scaling: bool = False,
        remove_outliers: bool = False,
        outlier_quantile: float = 0.95,
        yeo_johnson_grid: bool = True,
    ):
        self.max_num_samples = max_num_samples
        self.max_num_features = max_num_features
        self.train_fraction = train_fraction
        self.dropout_prob = dropout_prob
        self.transformation_type = transformation_type
        self.shuffle_data = shuffle_data
        self.target_feature = target_feature
        self.random_seed = random_seed
        self.negative_one_one_scaling = negative_one_one_scaling
        self.remove_outliers = remove_outliers
        self.outlier_quantile = outlier_quantile
        self.yeo_johnson_grid = yeo_johnson_grid  # currently unused but placeholder if customizing

        if not (0.0 < train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0,1)")
        if transformation_type not in {"standardize", "yeo_johnson"}:
            raise ValueError("transformation_type must be 'standardize' or 'yeo_johnson'")

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
        data_tensor = torch.cat([dataset[i] for i in feature_indices], dim=1)  # (N,F)

        # Select target feature
        target_feat = self._select_target_feature(feature_indices)
        target_col = feature_indices.index(target_feat)
        target_values = data_tensor[:, target_col]

        # Build list of candidate feature columns excluding target
        remaining_cols = [i for i in range(len(feature_indices)) if i != target_col]

        # Apply feature hiding (dropout) on remaining columns
        kept_cols, dropped_original_indices = self._apply_feature_dropout(remaining_cols, feature_indices)

        if len(kept_cols) == 0:
            # Should never happen; safeguard
            kept_cols = [remaining_cols[0]]

        # Truncate to max_num_features (non-padded) if needed
        kept_cols = kept_cols[: self.max_num_features]

        X_no_target = data_tensor[:, kept_cols]  # (N, F_kept)
        Y_tensor = target_values.unsqueeze(-1)  # (N,1)

        # Prepare shapes for Preprocessor: we want [B,N,F] and Y [B,N]
        X_batch = X_no_target.unsqueeze(0)  # [1,N,F_kept]
        Y_batch = Y_tensor.squeeze(-1).unsqueeze(0)  # [1,N]

        # Derive train/test counts
        n_total = X_batch.shape[1]
        n_train = int(n_total * self.train_fraction)
        n_train = max(1, min(n_train, n_total - 1))  # ensure at least 1 test sample
        n_test = n_total - n_train

        max_n_train = int(self.max_num_samples * self.train_fraction)
        max_n_train = max(1, max_n_train)
        max_n_test = self.max_num_samples - max_n_train
        max_n_test = max(1, max_n_test)

        # Setup Preprocessor flags based on transformation_type
        use_yj = self.transformation_type == 'yeo_johnson'
        use_std = True  # both original modes end with standardization

        preproc = Preprocessor(
            n_features=X_no_target.shape[1],
            max_n_features=self.max_num_features,
            n_train_samples=n_train,
            max_n_train_samples=max_n_train,
            n_test_samples=n_test,
            max_n_test_samples=max_n_test,
            negative_one_one_scaling=self.negative_one_one_scaling,
            standardize=use_std,
            yeo_johnson=use_yj,
            remove_outliers=self.remove_outliers,
            outlier_quantile=self.outlier_quantile,
            shuffle_samples=self.shuffle_data,
            shuffle_features=self.shuffle_data,
        )

        processed = preproc.process(X_batch, Y_batch)
        if processed is None:
            raise RuntimeError("Preprocessor returned None (insufficient samples or features).")

        X_train_b, X_test_b, Y_train_b, Y_test_b = processed  # each with batch dim

        # Strip batch dimension and align shapes with legacy API (Y has last dim=1)
        X_train = X_train_b[0]
        X_test = X_test_b[0]
        Y_train = Y_train_b[0].unsqueeze(-1)
        Y_test = Y_test_b[0].unsqueeze(-1)

        output = {
            'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
        }

        metadata = {
            'target_feature': target_feat,
            'target_col_original': target_col,
            'original_feature_indices': feature_indices,
            'kept_feature_original_indices': [feature_indices[i] for i in kept_cols],
            'dropped_feature_original_indices': dropped_original_indices,
            'original_num_features': original_num_features,
            'original_num_samples': original_num_samples,
            'train_fraction': self.train_fraction,
            'n_train_requested': n_train,
            'n_test_requested': n_test,
            'max_num_samples': self.max_num_samples,
            'max_num_features': self.max_num_features,
            'transformation_type': self.transformation_type,
            'negative_one_one_scaling': self.negative_one_one_scaling,
            'remove_outliers': self.remove_outliers,
            'outlier_quantile': self.outlier_quantile,
            'dropout_prob': self.dropout_prob,
            'shuffle_data': self.shuffle_data,
            'random_seed': self.random_seed,
            'preprocessor_padding': {
                'X_train_shape': tuple(X_train.shape),
                'X_test_shape': tuple(X_test.shape),
            },
        }

        return output, metadata

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

    def _select_target_feature(self, feature_indices: list) -> int:
        if self.target_feature is not None:
            if self.target_feature not in feature_indices:
                raise ValueError("Provided target_feature not in dataset")
            return self.target_feature
        return random.choice(feature_indices)

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