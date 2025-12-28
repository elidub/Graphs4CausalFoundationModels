
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
      - Optional test-time feature masking to simulate distribution shift.

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
    intervened_feature : Optional[int]
        Index of the feature that was intervened upon. This feature is never
        masked in the test set.
    random_seed : Optional[int]
        Reproducibility for target selection, dropout & shuffling.
    test_feature_mask_fraction : float, default 0.0
        Fraction of non-zero features to mask (set to zero) in the test set only.
        This creates a distribution shift where test has fewer features than train.
        If set to 0.5 and there are 6 non-zero features in test, 3 will be randomly
        selected and zeroed out. The target and intervened features are never masked.
        Must be in [0, 1). Default 0.0 means no masking.
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
        intervened_feature: Optional[int] = None,
        random_seed: Optional[int] = None,
        test_feature_mask_fraction: float = 0.0,
    # Legacy combined flags (kept for backward-compat)
    negative_one_one_scaling: bool = True,
    standardize: bool = True,
    # New split flags (override legacy when provided)
    feature_standardize: Optional[bool] = None,
    feature_negative_one_one_scaling: Optional[bool] = None,
    target_negative_one_one_scaling: Optional[bool] = None,
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
        self.intervened_feature = intervened_feature
        self.random_seed = random_seed
        self.test_feature_mask_fraction = test_feature_mask_fraction

        # Store Preprocessor-compatible config
        self.n_features = n_features
        self.max_n_features = max_n_features
        self.n_train_samples = n_train_samples
        self.max_n_train_samples = max_n_train_samples
        self.n_test_samples = n_test_samples
        self.max_n_test_samples = max_n_test_samples
        # Store new split flags with sensible defaults
        # Default: features standardized, target scaled to [-1,1]
        self.feature_standardize = feature_standardize if feature_standardize is not None else standardize
        self.feature_negative_one_one_scaling = (
            feature_negative_one_one_scaling if feature_negative_one_one_scaling is not None else False
        )
        self.target_negative_one_one_scaling = (
            target_negative_one_one_scaling if target_negative_one_one_scaling is not None else True
        )
        # Keep legacy values for backward compatibility
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
        
        # Internal state flags
        self.has_dummy_feature = False  # Set to True when dummy feature is added

        # Basic validations mirroring Preprocessor expectations
        assert 0 < outlier_quantile <= 1.0, "outlier_quantile must be in (0,1]"
        assert n_features <= max_n_features, "n_features <= max_n_features"
        assert n_train_samples <= max_n_train_samples, "n_train_samples <= max_n_train_samples"
        assert n_test_samples <= max_n_test_samples, "n_test_samples <= max_n_test_samples"
        assert 0.0 <= test_feature_mask_fraction <= 1.0, "test_feature_mask_fraction must be in [0,1]"

    # ------------------------------------------------------------------
    def process(
        self,
        dataset: Dict[int, torch.Tensor],
        mode: str = 'fast'
    ) -> Tuple[torch.Tensor, ...]:
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        if mode not in {"fast", "safe"}:
            raise ValueError("mode must be 'fast' or 'safe'")

        self._validate(dataset, mode)

        feature_indices = sorted(dataset.keys())
        if self.intervened_feature is not None:
            if self.intervened_feature not in feature_indices:
                raise ValueError("intervened_feature not present in dataset")
        original_num_features = len(feature_indices)
        original_num_samples = next(iter(dataset.values())).shape[0]

        # Build data tensor [N,F_total]
        data_tensor = torch.cat([dataset[i] for i in feature_indices], dim=1)

        # Target selection (exclude features with zero variance)
        target_feat = self._select_target_feature(feature_indices, data_tensor)
        self.selected_target_feature = target_feat  # Store for later access
        target_col = feature_indices.index(target_feat)
        target_values = data_tensor[:, target_col]

        # Candidate (non-target) columns
        remaining_cols = [i for i in range(len(feature_indices)) if i != target_col]
        # Exclude intervened feature from X if present
        intervened_col = None
        intervened_tensor = None
        if self.intervened_feature is not None:
            intervened_col = feature_indices.index(self.intervened_feature)
            remaining_cols = [c for c in remaining_cols if c != intervened_col]
            intervened_tensor = dataset[self.intervened_feature]  # (N,1) raw (untransformed)

        # Feature dropout
        kept_cols, dropped_original_indices = self._apply_feature_dropout(remaining_cols, feature_indices)
        
        # Enforce requested n_features (truncate if surplus)
        kept_cols = kept_cols[: self.n_features]
        
        # Store the feature indices (node names) that were kept for later access
        # This is the PRE-SHUFFLE order
        self.kept_feature_indices = [feature_indices[col] for col in kept_cols]
        # We'll update this to the POST-SHUFFLE order after preprocessing

        X_no_target = data_tensor[:, kept_cols]  # (N, n_features)
        Y_tensor = target_values.unsqueeze(-1)

        # Handle edge case: no features (only T and Y)
        # Add a dummy zero feature to avoid empty X
        actual_n_features = self.n_features
        if X_no_target.shape[1] == 0:
            X_no_target = torch.zeros(X_no_target.shape[0], 1, dtype=X_no_target.dtype, device=X_no_target.device)
            self.has_dummy_feature = True
            actual_n_features = 1  # Update to reflect the dummy feature
        else:
            self.has_dummy_feature = False

        # Batch dimension for Preprocessor
        X_batch = X_no_target.unsqueeze(0)  # [1,N,F]
        Y_batch = Y_tensor.squeeze(-1).unsqueeze(0)  # [1,N]
        
        preproc = Preprocessor(
            n_features=actual_n_features,  # Use actual_n_features which accounts for dummy
            max_n_features=self.max_n_features,
            n_train_samples=self.n_train_samples,
            max_n_train_samples=self.max_n_train_samples,
            n_test_samples=self.n_test_samples,
            max_n_test_samples=self.max_n_test_samples,
            # Use split controls: features standardized; target scaled to [-1,1]
            feature_negative_one_one_scaling=self.feature_negative_one_one_scaling,
            feature_standardize=self.feature_standardize,
            yeo_johnson=self.yeo_johnson,
            remove_outliers=self.remove_outliers,
            outlier_quantile=self.outlier_quantile,
            shuffle_samples=self.shuffle_samples,
            shuffle_features=self.shuffle_features,
            y_clip_quantile=self.y_clip_quantile,
            target_negative_one_one_scaling=self.target_negative_one_one_scaling,
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
        
        # Apply feature permutation to kept_feature_indices if features were shuffled
        if preproc.feature_permutation is not None:
            # The permutation tells us: new_position[i] came from old_position[perm[i]]
            # We need to reorder kept_feature_indices according to this permutation
            perm = preproc.feature_permutation.cpu().tolist()
            # Only apply permutation to the actual features (not dummy or padding)
            num_real_features = len(self.kept_feature_indices)
            if len(perm) >= num_real_features:
                # Reorder kept_feature_indices according to the permutation
                self.kept_feature_indices = [self.kept_feature_indices[perm[i]] for i in range(num_real_features)]

        if intervened_tensor is None:
            return X_train, Y_train, X_test, Y_test
        else:
            # Split intervened tensor into train/test (preserve ordering)
            T_train = intervened_tensor[: self.n_train_samples]
            T_test = intervened_tensor[self.n_train_samples: self.n_train_samples + self.n_test_samples]
            return X_train, T_train, Y_train, X_test, T_test, Y_test

    # ------------------------------------------------------------------
    def process_from_splits(
        self,
        train_dataset: Dict[int, torch.Tensor],
        test_dataset: Dict[int, torch.Tensor],
        *,
        mode: str = 'fast'
    ) -> Tuple[torch.Tensor, ...]:
        """Process already-split train and test datasets without shuffling between them.

        This method preserves the original ordering inside the provided train and
        test sets (i.e., no cross-set shuffling). It temporarily forces
        ``shuffle_samples = False`` during preprocessing and concatenates the
        datasets (train first, then test) so the underlying `Preprocessor`
        logic can operate unchanged.

        Assumptions
        -----------
        - `n_train_samples` and `n_test_samples` provided at construction match
          the number of rows in `train_dataset` and `test_dataset` respectively.
        - Feature indices (keys) are identical between train and test.
        - Each tensor has shape (N, 1).

        Parameters
        ----------
        train_dataset : Dict[int, torch.Tensor]
            Mapping feature index -> column tensor for training set.
        test_dataset : Dict[int, torch.Tensor]
            Mapping feature index -> column tensor for test set.
        mode : {'fast','safe'}
            Validation strictness (mirrors `process`).

        Returns
        -------
        X_train, Y_train, X_test, Y_test : torch.Tensor
            Processed tensors with shapes matching the configuration
            (after padding / transforms). Ordering within each split preserved.

        Notes
        -----
        - Target feature selection & dropout are applied on the *combined* data
          for simplicity. If you need target determined solely from train set,
          adapt by computing variance only on train columns before concatenation.
        - Cached random seed logic is respected (if set).
        - The object's `shuffle_samples` flag is restored afterward.
        """
        if set(train_dataset.keys()) != set(test_dataset.keys()):
            raise ValueError("Train and test must have identical feature index sets.")
        if not train_dataset or not test_dataset:
            raise ValueError("Empty train or test dataset.")

        # Validate shapes & counts separately (reuse _validate)
        self._validate(train_dataset, mode)
        self._validate(test_dataset, mode)

        n_train = next(iter(train_dataset.values())).shape[0]
        n_test = next(iter(test_dataset.values())).shape[0]
        if n_train != self.n_train_samples:
            raise ValueError(f"Provided train rows {n_train} != configured n_train_samples {self.n_train_samples}.")
        if n_test != self.n_test_samples:
            raise ValueError(f"Provided test rows {n_test} != configured n_test_samples {self.n_test_samples}.")

        # Concatenate train then test for each feature (including intervened feature)
        merged: Dict[int, torch.Tensor] = {}
        if self.intervened_feature is not None:
            if self.intervened_feature not in train_dataset:
                raise ValueError("intervened_feature missing from train_dataset")
            if self.intervened_feature not in test_dataset:
                raise ValueError("intervened_feature missing from test_dataset")
        for k in train_dataset.keys():
            merged[k] = torch.cat([train_dataset[k], test_dataset[k]], dim=0)

        # Temporarily disable sample shuffling (preserve order)
        original_shuffle = self.shuffle_samples
        self.shuffle_samples = False
        try:
            result = self.process(merged, mode=mode)
            if len(result) == 4:
                X_train, Y_train, X_test, Y_test = result
                T_train, T_test = None, None
            elif len(result) == 6:
                X_train, T_train, Y_train, X_test, T_test, Y_test = result
            else:
                raise ValueError(f"Unexpected process output length: {len(result)}")
        finally:
            # Restore original flag
            self.shuffle_samples = original_shuffle

        # Sanity: ensure split alignment preserved (optional minimal check)
        if X_train.shape[0] != self.n_train_samples:
            raise RuntimeError("Post-processing train sample count mismatch.")
        if X_test.shape[0] != self.n_test_samples:
            raise RuntimeError("Post-processing test sample count mismatch.")

        # Apply test feature masking if requested
        if self.test_feature_mask_fraction > 0.0:
            X_test = self._apply_test_feature_masking(X_test)

        if self.intervened_feature is None:
            return X_train, Y_train, X_test, Y_test
        else:
            # Use T values extracted from process (already split correctly)
            return X_train, T_train, Y_train, X_test, T_test, Y_test

    # ------------------------------------------------------------------
    def process_from_splits_separate(
        self,
        train_dataset: Dict[int, torch.Tensor],
        test_dataset: Dict[int, torch.Tensor],
        *,
        mode: str = 'fast'
    ) -> Tuple[torch.Tensor, ...]:
        """Process train and test datasets completely independently with NO data leakage.
        
        This method ensures complete separation between train and test preprocessing by:
        - Processing train data separately to get statistics (mean, std, quantiles, etc.)
        - Processing test data separately with its own statistics
        - Using the SAME target feature and feature selection for both
        - Ensuring NO information from test leaks into train preprocessing
        
        This is critical for interventional datasets where test has a different distribution
        due to interventions - concatenating them would corrupt both train and test statistics.
        
        Key Differences from process_from_splits
        -----------------------------------------
        - process_from_splits: Concatenates train+test, computes shared statistics, then splits
        - process_from_splits_separate: Processes train and test INDEPENDENTLY with separate statistics
        
        Parameters
        ----------
        train_dataset : Dict[int, torch.Tensor]
            Mapping feature index -> column tensor for training set.
        test_dataset : Dict[int, torch.Tensor]
            Mapping feature index -> column tensor for test set.
        mode : {'fast','safe'}
            Validation strictness.
            
        Returns
        -------
        X_train, Y_train, X_test, Y_test : torch.Tensor
            OR
        X_train, T_train, Y_train, X_test, T_test, Y_test : torch.Tensor
            Processed tensors. Train and test have independent preprocessing statistics.
            
        Notes
        -----
        - Target feature is selected based on TRAIN variance only
        - Feature dropout uses the SAME random selections for both (via random seed)
        - Each dataset gets its own standardization statistics
        - Sample shuffling is disabled to preserve ordering
        - The SAME features are kept for both train and test (consistent structure)
        """
        if set(train_dataset.keys()) != set(test_dataset.keys()):
            raise ValueError("Train and test must have identical feature index sets.")
        if not train_dataset or not test_dataset:
            raise ValueError("Empty train or test dataset.")

        # Validate both datasets
        self._validate(train_dataset, mode)
        self._validate(test_dataset, mode)

        n_train = next(iter(train_dataset.values())).shape[0]
        n_test = next(iter(test_dataset.values())).shape[0]
        if n_train != self.n_train_samples:
            raise ValueError(f"Provided train rows {n_train} != configured n_train_samples {self.n_train_samples}.")
        if n_test != self.n_test_samples:
            raise ValueError(f"Provided test rows {n_test} != configured n_test_samples {self.n_test_samples}.")

        # Save original settings that we'll temporarily modify
        original_shuffle = self.shuffle_samples
        original_target = self.target_feature
        original_seed = self.random_seed
        original_n_train = self.n_train_samples
        original_n_test = self.n_test_samples
        original_max_n_train = self.max_n_train_samples
        original_max_n_test = self.max_n_test_samples
        
        # Set random seed for reproducible target selection and dropout
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
        
        # Disable sample shuffling to preserve ordering
        self.shuffle_samples = False
        
        try:
            # Step 1: Process TRAIN data independently
            # Temporarily configure to treat entire train dataset as "train+test" 
            # so process() doesn't try to split it further
            self.n_train_samples = n_train
            self.n_test_samples = 0  # No split - take all as train
            self.max_n_train_samples = n_train
            self.max_n_test_samples = n_train  # Same as train for consistency
            
            train_result = self.process(train_dataset, mode=mode)
            
            # Extract the selected target and kept features from train processing
            selected_target = self.selected_target_feature
            kept_features = self.kept_feature_indices if hasattr(self, 'kept_feature_indices') else None
            has_dummy = self.has_dummy_feature
            
            # Extract train data (it's in the "train" portion of the result)
            if len(train_result) == 4:
                X_train, Y_train, _, _ = train_result
                T_train = None
                has_treatment = False
            elif len(train_result) == 6:
                X_train, T_train_raw, Y_train, _, _, _ = train_result
                has_treatment = True
            else:
                raise ValueError(f"Unexpected train process output length: {len(train_result)}")
            
            # Trim to actual train samples (process() may have padded)
            X_train = X_train[:original_n_train]
            Y_train = Y_train[:original_n_train]
            if has_treatment:
                # T from process() is RAW (unpreprocessed) - we need to preprocess it
                T_train_raw = T_train_raw[:original_n_train]
            
            # Step 2: Process TEST data independently with SAME target and feature selection
            # Fix the target feature to match train
            self.target_feature = selected_target
            
            # Reset random seed to get the SAME dropout pattern
            if self.random_seed is not None:
                torch.manual_seed(self.random_seed)
                np.random.seed(self.random_seed)
                random.seed(self.random_seed)
            
            # Reconfigure for test dataset
            self.n_train_samples = n_test
            self.n_test_samples = 0  # No split - take all as train
            self.max_n_train_samples = n_test
            self.max_n_test_samples = n_test
            
            test_result = self.process(test_dataset, mode=mode)
            
            # Verify both have same structure (with/without intervention)
            if len(test_result) != len(train_result):
                raise RuntimeError(
                    f"Train and test processing returned different structures: "
                    f"{len(train_result)} vs {len(test_result)}"
                )
            
            # Extract test data (it's in the "train" portion since we set n_test_samples=0)
            if len(test_result) == 4:
                X_test, Y_test, _, _ = test_result
                T_test = None
            elif len(test_result) == 6:
                X_test, T_test_raw, Y_test, _, _, _ = test_result
            else:
                raise ValueError(f"Unexpected test process output length: {len(test_result)}")
            
            # Trim to actual test samples
            X_test = X_test[:original_n_test]
            Y_test = Y_test[:original_n_test]
            if has_treatment:
                # T from process() is RAW (unpreprocessed) - we need to preprocess it
                T_test_raw = T_test_raw[:original_n_test]
            
            # Step 3: Preprocess treatment variable separately with INDEPENDENT statistics
            if has_treatment:
                from .Preprocessor import Preprocessor as TPreprocessor
                
                # Process train T
                t_preproc_train = TPreprocessor(
                    n_features=1,  # T is a single feature
                    max_n_features=1,
                    n_train_samples=original_n_train,
                    max_n_train_samples=original_n_train,
                    n_test_samples=0,
                    max_n_test_samples=original_n_train,
                    feature_negative_one_one_scaling=self.feature_negative_one_one_scaling,
                    feature_standardize=self.feature_standardize,
                    yeo_johnson=self.yeo_johnson,
                    remove_outliers=self.remove_outliers,
                    outlier_quantile=self.outlier_quantile,
                    shuffle_samples=False,  # Don't shuffle - preserve order
                    shuffle_features=False,  # Only 1 feature, no need
                    y_clip_quantile=None,  # Not a target
                    target_negative_one_one_scaling=False,  # T is not a target
                    eps=self.eps,
                    device=self.device,
                    dtype=self.dtype,
                )
                
                # Reshape for batch processing: [N,1] -> [1,N,1]
                T_train_batch = T_train_raw.unsqueeze(0)  # [1, N_train, 1]
                
                # Create dummy Y (not used for T preprocessing)
                Y_train_dummy = torch.zeros(1, original_n_train, device=T_train_raw.device, dtype=T_train_raw.dtype)
                
                # Process train T
                t_train_processed = t_preproc_train.process(T_train_batch, Y_train_dummy)
                if t_train_processed is not None:
                    T_train, _, _, _ = t_train_processed
                    T_train = T_train[0]  # Remove batch dimension: [1,N,1] -> [N,1]
                else:
                    T_train = T_train_raw
                
                # Process test T with independent statistics
                t_preproc_test = TPreprocessor(
                    n_features=1,
                    max_n_features=1,
                    n_train_samples=original_n_test,
                    max_n_train_samples=original_n_test,
                    n_test_samples=0,
                    max_n_test_samples=original_n_test,
                    feature_negative_one_one_scaling=self.feature_negative_one_one_scaling,
                    feature_standardize=self.feature_standardize,
                    yeo_johnson=self.yeo_johnson,
                    remove_outliers=self.remove_outliers,
                    outlier_quantile=self.outlier_quantile,
                    shuffle_samples=False,
                    shuffle_features=False,
                    y_clip_quantile=None,
                    target_negative_one_one_scaling=False,
                    eps=self.eps,
                    device=self.device,
                    dtype=self.dtype,
                )
                
                T_test_batch = T_test_raw.unsqueeze(0)
                Y_test_dummy = torch.zeros(1, original_n_test, device=T_test_raw.device, dtype=T_test_raw.dtype)
                
                t_test_processed = t_preproc_test.process(T_test_batch, Y_test_dummy)
                if t_test_processed is not None:
                    T_test, _, _, _ = t_test_processed
                    T_test = T_test[0]  # Remove batch dimension
                else:
                    T_test = T_test_raw
            
            # Now pad to the ORIGINAL max dimensions
            # Pad train
            if X_train.shape[0] < original_max_n_train:
                pad_rows = original_max_n_train - X_train.shape[0]
                X_train = torch.cat([X_train, torch.zeros(pad_rows, X_train.shape[1], dtype=X_train.dtype, device=X_train.device)], dim=0)
                Y_train = torch.cat([Y_train, torch.zeros(pad_rows, Y_train.shape[1], dtype=Y_train.dtype, device=Y_train.device)], dim=0)
                if has_treatment:
                    T_train = torch.cat([T_train, torch.zeros(pad_rows, T_train.shape[1], dtype=T_train.dtype, device=T_train.device)], dim=0)
            
            # Pad test
            if X_test.shape[0] < original_max_n_test:
                pad_rows = original_max_n_test - X_test.shape[0]
                X_test = torch.cat([X_test, torch.zeros(pad_rows, X_test.shape[1], dtype=X_test.dtype, device=X_test.device)], dim=0)
                Y_test = torch.cat([Y_test, torch.zeros(pad_rows, Y_test.shape[1], dtype=Y_test.dtype, device=Y_test.device)], dim=0)
                if has_treatment:
                    T_test = torch.cat([T_test, torch.zeros(pad_rows, T_test.shape[1], dtype=T_test.dtype, device=T_test.device)], dim=0)
            
            # Combine results
            if has_treatment:
                result = (X_train, T_train, Y_train, X_test, T_test, Y_test)
            else:
                result = (X_train, Y_train, X_test, Y_test)
            
            # Apply test feature masking if requested (only to test set)
            if self.test_feature_mask_fraction > 0.0:
                if has_treatment:
                    X_test = self._apply_test_feature_masking(X_test)
                    result = (X_train, T_train, Y_train, X_test, T_test, Y_test)
                else:
                    X_test = self._apply_test_feature_masking(X_test)
                    result = (X_train, Y_train, X_test, Y_test)
            
            return result
            
        finally:
            # Restore ALL original settings
            self.shuffle_samples = original_shuffle
            self.target_feature = original_target
            self.random_seed = original_seed
            self.n_train_samples = original_n_train
            self.n_test_samples = original_n_test
            self.max_n_train_samples = original_max_n_train
            self.max_n_test_samples = original_max_n_test

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
        eps = 1e-4
        valid_mask = variances > eps
        valid_cols = [i for i, valid in enumerate(valid_mask) if valid]
        valid_features = [feature_indices[i] for i in valid_cols]
        
        # If user specified a target, check it's valid (and not the intervened feature)
        if self.target_feature is not None:
            if self.target_feature not in feature_indices:
                raise ValueError("Provided target_feature not in dataset")
            if self.intervened_feature is not None and self.target_feature == self.intervened_feature:
                raise ValueError("Target feature cannot be the intervened feature.")
            
            # Check if specified target has zero variance
            target_col = feature_indices.index(self.target_feature)
            if not valid_mask[target_col]:
                import warnings
                warnings.warn(
                    f"Provided target_feature {self.target_feature} has zero variance "
                    f"(trivial prediction task) - selecting a different target randomly instead"
                )
                # Don't use this target, fall through to random selection
                self.target_feature = None
            else:
                return self.target_feature
        
        # Random selection with variance-based preference
        # Remove intervened feature from candidate set for random selection
        if self.intervened_feature is not None and self.intervened_feature in valid_features:
            valid_features = [f for f in valid_features if f != self.intervened_feature]
        remaining_candidate_pool = [f for f in feature_indices if f in valid_features] if len(valid_features) > 0 else [f for f in feature_indices if f != self.intervened_feature] if self.intervened_feature is not None else feature_indices

        # Case 1: No features with sufficient variance - just pick randomly (excluding intervened if present)
        if len(valid_features) == 0:
            return random.choice(remaining_candidate_pool)
        
        # Case 2: Features with sufficient variance exist
        # Extract variances for valid features only
        valid_variances = variances[valid_cols]
        
        # Compute 0.8 quantile of variance distribution
        variance_80_quantile = torch.quantile(valid_variances, 0.9).item()
        
        # Find features with variance above 0.8 quantile
        high_variance_mask = valid_variances > variance_80_quantile
        high_variance_cols = [valid_cols[i] for i, is_high in enumerate(high_variance_mask) if is_high]
        high_variance_features = [feature_indices[i] for i in high_variance_cols]
        if self.intervened_feature is not None:
            high_variance_features = [f for f in high_variance_features if f != self.intervened_feature]
        
        # With probability 0.9, select uniformly from high-variance features
        # With probability 0.1, select uniformly from all valid features
        if random.random() >= 1.0 and len(high_variance_features) > 0:
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

    def _apply_test_feature_masking(self, X_test: torch.Tensor) -> torch.Tensor:
        """Randomly mask out (zero out) a fraction of non-zero features in test set.
        
        This creates a distribution shift where the test set has fewer features than 
        the training set, simulating scenarios where some features are unavailable 
        at test time.
        
        The target feature and intervened feature (if present) are never masked.
        
        Parameters
        ----------
        X_test : torch.Tensor
            Test features of shape (n_test_samples, max_n_features)
            
        Returns
        -------
        torch.Tensor
            Test features with some columns masked (set to zero)
            
        Notes
        -----
        - Identifies non-zero columns (features that weren't already padded/dropped)
        - Randomly selects a fraction of these to mask out
        - Never masks the target or intervened feature columns
        - Masking is applied by setting entire columns to zero
        """
        if self.test_feature_mask_fraction <= 0.0:
            return X_test
        
        # Identify non-zero columns (columns that have at least one non-zero value)
        # Use a small epsilon to account for numerical precision
        eps = 1e-8
        column_is_nonzero = (X_test.abs().sum(dim=0) > eps)
        nonzero_cols = torch.where(column_is_nonzero)[0].tolist()
        
        if len(nonzero_cols) == 0:
            # All columns are zero, nothing to mask
            return X_test
        
        # Calculate number of features to mask
        n_to_mask = int(len(nonzero_cols) * self.test_feature_mask_fraction)
        
        if n_to_mask == 0:
            # Fraction too small to mask any features
            return X_test
        
        # Randomly select features to mask
        if self.random_seed is not None:
            # Use a derived seed for test masking to maintain reproducibility
            # but different from dropout seed
            torch.manual_seed(self.random_seed + 999999)
            random.seed(self.random_seed + 999999)
        
        cols_to_mask = random.sample(nonzero_cols, n_to_mask)
        
        # Create a copy to avoid in-place modification
        X_test_masked = X_test.clone()
        
        # Mask the selected columns by setting them to zero
        X_test_masked[:, cols_to_mask] = 0.0
        
        return X_test_masked