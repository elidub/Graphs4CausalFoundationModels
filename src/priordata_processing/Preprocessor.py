from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor


class Preprocessor:
    """
    A module for preprocessing a tabular dataset. The goal is to take in
    X (shape [batch_size, n_samples, n_features]) and Y (shape [batch_size, n_samples])
    and return X_train, X_test, Y_train, Y_test tensors ready for training and evaluation.

    All statistics / transforms (outlier thresholds, Yeo-Johnson lambdas, standardization,
    and min-max for [-1, 1] scaling) are fitted on the *training split only* per batch
    and per feature, then applied to both train and test.
    """

    def __init__(
        self,
        n_features: int,
        max_n_features: int,
        n_train_samples: int,
        max_n_train_samples: int,
        n_test_samples: int,
        max_n_test_samples: int,
        negative_one_one_scaling: bool = True,
        standardize: bool = False,
        yeo_johnson: bool = False,
        remove_outliers: bool = True,
        outlier_quantile: float = 0.95,
        shuffle_samples: bool = True,
        shuffle_features: bool = True,
        eps: float = 1e-8,
        y_clip_quantile: Optional[float] = None,  # if you ever want Y winsorization
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args
        ----
        n_features: Desired number of features in the dataset that are non-zero (i.e. not padding).
                    If there are fewer features available, returns None and prints a warning.
        max_n_features: Maximum number of features to pad to.
        n_train_samples: Desired number of training samples (non-zero). If fewer exist, returns None.
        max_n_train_samples: Maximum number of training samples to pad to.
        n_test_samples: Desired number of test samples (non-zero). If fewer exist, returns None.
        max_n_test_samples: Maximum number of test samples to pad to.
        negative_one_one_scaling: Whether to scale features and targets to [-1, 1] using train stats.
        standardize: Whether to standardize features (zero mean, unit variance) using train stats.
        yeo_johnson: Whether to apply Yeo-Johnson transform to features (fit on train) before standardization.
        remove_outliers: Whether to winsorize features based on train-quantiles.
        outlier_quantile: Upper quantile (q). We winsorize using (1-q, q). Example: 0.95 → clamp to [p5, p95].
        shuffle_samples: Whether to shuffle rows before splitting into train/test.
        shuffle_features: Whether to shuffle columns before selecting n_features.
        eps: Small numerical constant for divisions / logs.
        y_clip_quantile: Optional winsorization for Y based on train quantiles (None = no clipping).
        device/dtype: Optional overrides for output tensors.
        """
        assert 0 < outlier_quantile <= 1.0, "outlier_quantile must be in (0, 1]."
        assert n_features <= max_n_features, "n_features must be <= max_n_features."
        assert n_train_samples <= max_n_train_samples, "n_train_samples must be <= max_n_train_samples."
        assert n_test_samples <= max_n_test_samples, "n_test_samples must be <= max_n_test_samples."

        self.n_features = n_features
        self.max_n_features = max_n_features
        self.n_train = n_train_samples
        self.max_n_train = max_n_train_samples
        self.n_test = n_test_samples
        self.max_n_test = max_n_test_samples

        self.negative_one_one_scaling = negative_one_one_scaling
        self.standardize = standardize
        self.yeo_johnson = yeo_johnson
        self.remove_outliers = remove_outliers
        self.outlier_quantile = outlier_quantile
        self.shuffle_samples = shuffle_samples
        self.shuffle_features = shuffle_features
        self.eps = eps
        self.y_clip_quantile = y_clip_quantile

        self.device = device
        self.dtype = dtype

    # ------------------------ public API ------------------------

    def process(
        self,
        X: Tensor,  # [B, N, F]
        Y: Tensor,  # [B, N]
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        Returns:
            (X_train, X_test, Y_train, Y_test)
            Shapes:
                X_train: [B, max_n_train_samples, max_n_features]
                X_test:  [B, max_n_test_samples,  max_n_features]
                Y_train: [B, max_n_train_samples]
                Y_test:  [B, max_n_test_samples]
        Or None if the dataset does not meet requested sizes.
        """
        self._validate_inputs(X, Y)

        B, N, F = X.shape

        # If insufficient samples or features → return None (as requested)
        if N < (self.n_train + self.n_test):
            print(f"[Preprocessor] Warning: Need at least {self.n_train + self.n_test} samples, but got {N}. Returning None.")
            return None
        if F < self.n_features:
            print(f"[Preprocessor] Warning: Need at least {self.n_features} features, but got {F}. Returning None.")
            return None

        # Optionally shuffle samples and features (same permutation per batch for features)
        X, Y = self._maybe_shuffle(X, Y)

        # Select desired non-padded counts
        X = X[:, :, :self.n_features]
        # Split into train/test by samples
        X_train = X[:, :self.n_train, :]
        X_test = X[:, self.n_train:self.n_train + self.n_test, :]
        Y_train = Y[:, :self.n_train]
        Y_test = Y[:, self.n_train:self.n_train + self.n_test]

        # Fit transforms on train and apply to both
        X_train, X_test = self._fit_apply_feature_pipeline(X_train, X_test)
        Y_train, Y_test = self._fit_apply_target_pipeline(Y_train, Y_test)

        # Pad to maxima
        X_train = self._pad_features_and_samples(X_train, self.max_n_train, self.max_n_features)
        X_test = self._pad_features_and_samples(X_test, self.max_n_test, self.max_n_features)
        Y_train = self._pad_samples(Y_train, self.max_n_train)
        Y_test = self._pad_samples(Y_test, self.max_n_test)

        # Cast/device if requested
        if self.dtype is not None:
            X_train = X_train.to(self.dtype)
            X_test = X_test.to(self.dtype)
            Y_train = Y_train.to(self.dtype)
            Y_test = Y_test.to(self.dtype)
        if self.device is not None:
            X_train = X_train.to(self.device)
            X_test = X_test.to(self.device)
            Y_train = Y_train.to(self.device)
            Y_test = Y_test.to(self.device)

        return X_train, X_test, Y_train, Y_test

    # ------------------------ helpers: validation / shuffling ------------------------

    def _validate_inputs(self, X: Tensor, Y: Tensor) -> None:
        if X.dim() != 3:
            raise ValueError(f"X must have shape [B, N, F], got {tuple(X.shape)}.")
        if Y.dim() != 2:
            raise ValueError(f"Y must have shape [B, N], got {tuple(Y.shape)}.")
        if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
            raise ValueError(f"Batch size / sample count mismatch between X{tuple(X.shape)} and Y{tuple(Y.shape)}.")

    def _maybe_shuffle(self, X: Tensor, Y: Tensor) -> Tuple[Tensor, Tensor]:
        B, N, F = X.shape
        # Shuffle samples (rows)
        if self.shuffle_samples:
            perm_rows = torch.argsort(torch.rand(B, N, device=X.device), dim=1)
            X = torch.gather(X, 1, perm_rows.unsqueeze(-1).expand(B, N, F))
            Y = torch.gather(Y, 1, perm_rows)

        # Shuffle features (columns) — same permutation across batches to keep alignment
        if self.shuffle_features:
            perm_cols = torch.randperm(F, device=X.device)
            X = X[:, :, perm_cols]

        return X, Y

    # ------------------------ helpers: feature pipeline ------------------------

    def _fit_apply_feature_pipeline(self, Xtr: Tensor, Xte: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply in order (when enabled):
          1) Outlier winsorization (fit on Xtr quantiles)
          2) Yeo-Johnson transform (fit lambdas on Xtr)
          3) Standardization (fit mean/std on Xtr)
          4) [-1, 1] scaling (fit min/max on Xtr)
        """
        if self.remove_outliers:
            Xtr, Xte = self._winsorize_train_test(Xtr, Xte, q=self.outlier_quantile)

        if self.yeo_johnson:
            lambdas = self._fit_yeo_johnson_lambdas(Xtr)          # [B, F]
            Xtr = self._apply_yeo_johnson(Xtr, lambdas)
            Xte = self._apply_yeo_johnson(Xte, lambdas)

        if self.standardize:
            mean = Xtr.mean(dim=1, keepdim=True)                  # [B,1,F]
            std = Xtr.std(dim=1, keepdim=True).clamp_min(self.eps)
            Xtr = (Xtr - mean) / std
            Xte = (Xte - mean) / std

        if self.negative_one_one_scaling:
            # Min-max per (B,F) using train, then scale both sets
            xmin = Xtr.amin(dim=1, keepdim=True)
            xmax = Xtr.amax(dim=1, keepdim=True)
            rng = (xmax - xmin).clamp_min(self.eps)
            Xtr = 2.0 * (Xtr - xmin) / rng - 1.0
            Xte = 2.0 * (Xte - xmin) / rng - 1.0

        return Xtr, Xte

    def _winsorize_train_test(self, Xtr: Tensor, Xte: Tensor, q: float) -> Tuple[Tensor, Tensor]:
        """
        Winsorize features using train quantiles.
        Lower = (1-q), Upper = q. Example: q=0.95 → clamp to [p5, p95].
        Quantiles are computed per (batch, feature) across the sample dimension.
        """
        q = float(q)
        if not (0.5 < q <= 1.0):
            raise ValueError("outlier_quantile should be in (0.5, 1.0].")
        lower_q = 1.0 - q
        # torch.quantile supports q as tensor
        qs = torch.tensor([lower_q, q], device=Xtr.device, dtype=Xtr.dtype)
        # Compute per (B,F)
        # shape: [B, 2, F]
        Q = torch.quantile(Xtr.transpose(1, 2), qs, dim=-1, keepdim=False).transpose(0, 1)
        lo = Q[:, 0, :].unsqueeze(1)  # [B,1,F]
        hi = Q[:, 1, :].unsqueeze(1)  # [B,1,F]
        Xtr = Xtr.clamp(min=lo, max=hi)
        Xte = Xte.clamp(min=lo, max=hi)
        return Xtr, Xte

    # ------------------------ helpers: Yeo-Johnson ------------------------

    @staticmethod
    def _yeo_johnson_transform(x: Tensor, lam: Tensor, eps: float) -> Tensor:
        """
        Yeo-Johnson transform for any real x.
        lam shape: broadcastable to x (e.g., [B,1,F]).
        """
        # piecewise
        pos = x >= 0
        lam_near0 = torch.abs(lam) < 1e-6

        # For x >= 0:
        # lam != 0: ((x + 1)^lam - 1) / lam
        # lam == 0: log(x + 1)
        out_pos = torch.where(
            lam_near0, torch.log1p(x.clamp_min(0.0) + 0.0), ((x + 1.0).clamp_min(eps) ** lam - 1.0) / (lam + 0.0)
        )

        # For x < 0:
        # lam != 2: - ((-x + 1)^(2 - lam) - 1) / (2 - lam)
        # lam == 2: -log(-x + 1)
        two_minus_lam = 2.0 - lam
        near2 = torch.abs(two_minus_lam) < 1e-6
        xm = (-x).clamp_min(0.0) + 1.0
        out_neg = torch.where(
            near2, -torch.log(xm.clamp_min(eps)), -((xm.clamp_min(eps) ** two_minus_lam - 1.0) / two_minus_lam)
        )

        return torch.where(pos, out_pos, out_neg)

    def _fit_yeo_johnson_lambdas(self, Xtr: Tensor) -> Tensor:
        """
        Fit per-batch, per-feature lambda by grid-search MLE under normality:
        maximizes Gaussian log-likelihood of transformed data (up to constants)
        using the Jacobian term of YJ (sum log|dT/dx|). This is a pragmatic, stable approach.

        Returns:
            lambdas: [B, F]
        """
        B, N, F = Xtr.shape
        # Lambda grid
        grid = torch.linspace(-2.0, 2.0, steps=41, device=Xtr.device, dtype=Xtr.dtype)  # 0.1 step
        # Prepare broadcast shapes
        x = Xtr.unsqueeze(2)  # [B, N, 1, F]
        lam = grid.view(1, 1, -1, 1)  # [1,1,L,1] -> broadcast to [B,N,L,F]

        # Transform
        xt = self._yeo_johnson_transform(x, lam, self.eps)  # [B, N, L, F]

        # Gaussian MLE log-likelihood (per (B,L,F)): -N/2 * log(var) + Jacobian term
        # Compute mean/var across N
        mean = xt.mean(dim=1, keepdim=True)
        var = xt.var(dim=1, unbiased=False, keepdim=True).clamp_min(self.eps)
        ll_gauss = -0.5 * (xt - mean).pow(2).sum(dim=1) / var  # [B, L, F] up to constants
        ll_gauss += -0.5 * torch.log(var) * xt.shape[1]        # add -N/2 log(var)

        # Add Jacobian log|dT/dx|
        # For Yeo-Johnson, the derivative:
        # x>=0: (x+1)^(lam-1)
        # x<0:  (1-x)^(1-lam)
        pos = Xtr >= 0
        jac_pos = ((Xtr + 1.0).clamp_min(self.eps)) ** (lam - 1.0)  # [B,N,L,F]
        jac_neg = ((1.0 - Xtr).clamp_min(self.eps)) ** (1.0 - lam)
        log_jac = torch.where(pos.unsqueeze(2), torch.log(jac_pos.clamp_min(self.eps)), torch.log(jac_neg.clamp_min(self.eps)))
        ll = ll_gauss + log_jac.sum(dim=1)  # [B, L, F]

        # Choose best lambda per (B,F)
        idx = torch.argmax(ll, dim=1)  # [B, F] index into grid
        lambdas = grid[idx]            # [B, F]
        return lambdas

    def _apply_yeo_johnson(self, X: Tensor, lambdas: Tensor) -> Tensor:
        # reshape lambdas to [B,1,F] for broadcasting across samples
        lam = lambdas.unsqueeze(1)
        return self._yeo_johnson_transform(X, lam, self.eps)

    # ------------------------ helpers: target pipeline ------------------------

    def _fit_apply_target_pipeline(self, Ytr: Tensor, Yte: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Currently only supports optional [-1,1] min-max scaling for Y (fit on train).
        Optionally winsorize Y if y_clip_quantile is provided.
        """
        if self.y_clip_quantile is not None:
            q = float(self.y_clip_quantile)
            if not (0.5 < q <= 1.0):
                raise ValueError("y_clip_quantile should be in (0.5, 1.0].")
            qs = torch.tensor([1.0 - q, q], device=Ytr.device, dtype=Ytr.dtype)
            Q = torch.quantile(Ytr, qs, dim=1, keepdim=True)  # [B,2,1] effectively
            lo = Q[:, 0:1, :]
            hi = Q[:, 1:2, :]
            Ytr = Ytr.clamp(min=lo.squeeze(-1), max=hi.squeeze(-1))
            Yte = Yte.clamp(min=lo.squeeze(-1), max=hi.squeeze(-1))

        if self.negative_one_one_scaling:
            ymin = Ytr.amin(dim=1, keepdim=True)
            ymax = Ytr.amax(dim=1, keepdim=True)
            rng = (ymax - ymin).clamp_min(self.eps)
            Ytr = 2.0 * (Ytr - ymin) / rng - 1.0
            Yte = 2.0 * (Yte - ymin) / rng - 1.0

        return Ytr, Yte

    # ------------------------ helpers: padding ------------------------

    def _pad_features_and_samples(self, X: Tensor, max_ns: int, max_nf: int) -> Tensor:
        """
        Pad X [B, n_samples, n_features] to [B, max_ns, max_nf] with zeros.
        """
        B, Ns, Nf = X.shape
        if Ns > max_ns or Nf > max_nf:
            # Should not happen if caller respects sizes, but guard anyway:
            X = X[:, :min(Ns, max_ns), :min(Nf, max_nf)]
            Ns, Nf = X.shape[1], X.shape[2]

        pad_s = max_ns - Ns
        pad_f = max_nf - Nf
        if pad_s == 0 and pad_f == 0:
            return X
        pad = (0, pad_f, 0, pad_s)  # (last_dim_left, last_dim_right, second_last_left, second_last_right)
        return torch.nn.functional.pad(X, pad, mode="constant", value=0.0)

    def _pad_samples(self, Y: Tensor, max_ns: int) -> Tensor:
        """
        Pad Y [B, n_samples] to [B, max_ns] with zeros.
        """
        B, Ns = Y.shape
        if Ns > max_ns:
            Y = Y[:, :max_ns]
            Ns = max_ns
        pad_s = max_ns - Ns
        if pad_s == 0:
            return Y
        pad = (0, pad_s)
        return torch.nn.functional.pad(Y, pad, mode="constant", value=0.0)
