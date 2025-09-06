from __future__ import annotations
from typing import Iterable, Optional, Tuple
from abc import ABC, abstractmethod
import torch
from torch import Tensor
import math


class PosteriorPredictive(ABC):
    @abstractmethod
    def average_log_prob(self, pred: Tensor, y: Tensor) -> Tensor:
        pass

    def mode(self, pred: Tensor) -> Tensor:
        raise NotImplementedError

    def mean(self, pred: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, pred: Tensor, num_samples: int) -> Tensor:
        raise NotImplementedError


class BarDistribution(PosteriorPredictive):
    """
    Piecewise-uniform bars (fixed edges from data) + Gaussian half-tails at the ends.

    Model parameters per point (num_params = K + 4):
      - logits for mixture over [LeftTail, Bar_0..Bar_{K-1}, RightTail]  (length K+2)
      - tail scale multipliers raw: [s_L_raw, s_R_raw]                   (length 2)

    Tails:
      Left  (y < edge_left):  f(y) = (2 p_L) * N(y; edge_left, s_L)
      Right (y ≥ edge_right): f(y) = (2 p_R) * N(y; edge_right, s_R)
    """

    def __init__(
        self,
        num_bars: int = 11,
        min_width: float = 1e-6,
        scale_floor: float = 1e-6,   # ensures positive tail scales after softplus
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_fit_items: Optional[int] = None,
        log_prob_clip_min: float = -50.0,  # Minimum log probability to prevent -inf
        log_prob_clip_max: float = 50.0,   # Maximum log probability to prevent overflow
    ):
        if num_bars < 1:
            raise ValueError("num_bars must be >= 1")
        self.num_bars = int(num_bars)
        self.min_width = float(min_width)
        self.scale_floor = float(scale_floor)
        self.max_fit_items = max_fit_items
        self.log_prob_clip_min = float(log_prob_clip_min)
        self.log_prob_clip_max = float(log_prob_clip_max)

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        # Learned geometry (via .fit)
        self.centers: Optional[Tensor] = None   # (K,)
        self.edges: Optional[Tensor] = None     # (K+1,)
        self.widths: Optional[Tensor] = None    # (K,)
        self.base_s_left: Optional[Tensor] = None
        self.base_s_right: Optional[Tensor] = None

        # constants
        self._log_floor = torch.tensor(1e-300, device=self.device, dtype=self.dtype)
        self._norm_const = torch.tensor(1.0 / math.sqrt(2.0 * math.pi), device=self.device, dtype=self.dtype)

    # ------------------------- Fit bar locations -------------------------

    @torch.no_grad()
    def fit(self, dataloader: Iterable[Tuple[Tensor, Tensor, Tensor, Tensor]]) -> "BarDistribution":
        ys = []
        total = 0
        for batch in dataloader:
            if len(batch) != 4:
                raise ValueError("Each dataloader item must be (X_train, y_train, X_test, y_test).")
            _, y_tr, _, y_te = batch
            if y_tr.ndim != 2 or y_te.ndim != 2:
                raise ValueError("y_train and y_test must be (B, N/M).")
            y_tr_flat = y_tr.reshape(-1)
            y_te_flat = y_te.reshape(-1)

            if self.max_fit_items is not None:
                remaining = max(0, self.max_fit_items - total)
                if remaining <= 0:
                    break
                need_tr = min(remaining // 2, y_tr_flat.numel())
                need_te = min(remaining - need_tr, y_te_flat.numel())
                if need_tr > 0:
                    idx = torch.randint(0, y_tr_flat.numel(), (need_tr,), device=y_tr_flat.device)
                    ys.append(y_tr_flat[idx].cpu())
                    total += need_tr
                if need_te > 0:
                    idx = torch.randint(0, y_te_flat.numel(), (need_te,), device=y_te_flat.device)
                    ys.append(y_te_flat[idx].cpu())
                    total += need_te
            else:
                ys.append(y_tr_flat.cpu())
                ys.append(y_te_flat.cpu())

        if not ys:
            raise ValueError("No y data collected for fit().")

        y_all = torch.cat(ys, dim=0).to(torch.float64)
        y_min, y_max = torch.min(y_all), torch.max(y_all)
        if not (torch.isfinite(y_min) and torch.isfinite(y_max)):
            raise ValueError("Non-finite y encountered during fit().")

        K = self.num_bars
        probs = torch.linspace(0.0, 1.0, steps=K + 1, dtype=torch.float64)
        mids = (probs[:-1] + probs[1:]) * 0.5
        try:
            centers = torch.quantile(y_all, mids, method="linear")
        except TypeError:
            centers = torch.quantile(y_all, mids, interpolation="linear")

        # Enforce strict increasing centers (tiny jitter if needed)
        scale = torch.maximum(y_max - y_min, torch.tensor(1.0, dtype=torch.float64))
        eps = (1e-9 * float(K)) * scale
        for i in range(K - 1):
            if centers[i + 1] - centers[i] <= 0:
                centers[i + 1] = centers[i] + eps

        edges = torch.empty(K + 1, dtype=torch.float64)
        if K == 1:
            q25, q75 = torch.quantile(y_all, torch.tensor([0.25, 0.75], dtype=torch.float64))
            width = float(max(q75 - q25, 1e-6))
            edges[0] = centers[0] - 0.5 * width
            edges[1] = centers[0] + 0.5 * width
        else:
            edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
            edges[0] = centers[0] - 0.5 * float(centers[1] - centers[0])
            edges[-1] = centers[-1] + 0.5 * float(centers[-1] - centers[-2])

        widths = torch.diff(edges).clamp(min=self.min_width)

        # Tail base scales from near-edge widths
        base_s_left = float(widths[0])
        base_s_right = float(widths[-1])

        self.centers = centers.to(self.device, self.dtype)
        self.edges = edges.to(self.device, self.dtype)
        self.widths = widths.to(self.device, self.dtype)
        self.base_s_left = torch.tensor(base_s_left, device=self.device, dtype=self.dtype)
        self.base_s_right = torch.tensor(base_s_right, device=self.device, dtype=self.dtype)
        return self

    # ------------------------------ Interface ------------------------------

    @torch.inference_mode()
    def average_log_prob(self, pred: Tensor, y: Tensor) -> Tensor:
        """
        pred: (B, M, K+4)
        y:    (B, M)
        return: (B,)
        """
        self._check_ready()
        B, M = y.shape
        K = self.num_bars
        self._validate_pred(pred, (B, M, K + 4))

        pdf = self._pdf_from_pred(pred, y)  # (B,M)
        logpdf = torch.log(pdf)  # pdf is already clipped in _pdf_from_pred
        logpdf = torch.clamp(logpdf, min=self.log_prob_clip_min, max=self.log_prob_clip_max)
        return logpdf.mean(dim=1)

    @torch.inference_mode()
    def mode(self, pred: Tensor) -> Tensor:
        """
        Argmax among: left edge (tail), the densest bar, right edge (tail).
        """
        self._check_ready()
        B, M, _ = pred.shape
        K = self.num_bars
        self._validate_pred(pred, (B, M, K + 4))

        w_logits, sL_raw, sR_raw = self._unpack(pred)
        probs = torch.softmax(w_logits, dim=-1)  # (B,M,K+2)
        pL = probs[..., 0]
        pBars = probs[..., 1:-1]                 # (B,M,K)
        pR = probs[..., -1]

        sL = self.base_s_left * (torch.nn.functional.softplus(sL_raw) + self.scale_floor)
        sR = self.base_s_right * (torch.nn.functional.softplus(sR_raw) + self.scale_floor)

        # Tail densities at edges (Gaussian peak at mean=edge):
        # 2 * pTail * N(edge; edge, s) = 2*pTail * (1/(sqrt(2π)*s))
        dens_left_edge = (2.0 * pL) * (self._norm_const / sL)
        dens_right_edge = (2.0 * pR) * (self._norm_const / sR)

        # Max bar density
        bar_densities = pBars / self.widths.view(1, 1, K)
        max_bar_dens, max_idx = torch.max(bar_densities, dim=-1)

        candidates = torch.stack([dens_left_edge, max_bar_dens, dens_right_edge], dim=-1)
        arg = candidates.argmax(dim=-1)

        mode_vals = torch.empty(B, M, device=self.device, dtype=self.dtype)
        edge_left = self.edges[0]
        edge_right = self.edges[-1]
        mode_vals[arg == 0] = edge_left
        mode_vals[arg == 2] = edge_right

        centers = self.centers.view(1, 1, K).expand(B, M, K)
        mode_bar = torch.gather(centers, dim=-1, index=max_idx.unsqueeze(-1)).squeeze(-1)
        mode_vals[arg == 1] = mode_bar[arg == 1]
        return mode_vals

    @torch.inference_mode()
    def mean(self, pred: Tensor) -> Tensor:
        """
        Mean is always finite with Gaussian tails.

        Bars: mean at midpoints weighted by bar probs.
        Left half-Gaussian with mean at edge: E[y | left] = edge_left - sL * sqrt(2/π)
        Right half-Gaussian with mean at edge: E[y | right] = edge_right + sR * sqrt(2/π)
        """
        self._check_ready()
        B, M, _ = pred.shape
        K = self.num_bars
        self._validate_pred(pred, (B, M, K + 4))

        w_logits, sL_raw, sR_raw = self._unpack(pred)
        probs = torch.softmax(w_logits, dim=-1)
        pL = probs[..., 0]
        pBars = probs[..., 1:-1]  # (B,M,K)
        pR = probs[..., -1]

        sL = self.base_s_left * (torch.nn.functional.softplus(sL_raw) + self.scale_floor)
        sR = self.base_s_right * (torch.nn.functional.softplus(sR_raw) + self.scale_floor)

        mids = 0.5 * (self.edges[:-1] + self.edges[1:])       # (K,)
        bar_mean = (pBars * mids.view(1, 1, K)).sum(dim=-1)   # (B,M)

        c = math.sqrt(2.0 / math.pi)
        edge_left = self.edges[0]
        edge_right = self.edges[-1]
        E_left = edge_left - c * sL
        E_right = edge_right + c * sR

        return pL * E_left + pR * E_right + bar_mean  # (B,M)

    @torch.inference_mode()
    def sample(self, pred: Tensor, num_samples: int) -> Tensor:
        """
        Returns: (B, S, M)
        """
        self._check_ready()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        B, M, _ = pred.shape
        K = self.num_bars
        self._validate_pred(pred, (B, M, K + 4))

        w_logits, sL_raw, sR_raw = self._unpack(pred)
        probs = torch.softmax(w_logits, dim=-1)  # (B,M,K+2)

        sL = self.base_s_left * (torch.nn.functional.softplus(sL_raw) + self.scale_floor)
        sR = self.base_s_right * (torch.nn.functional.softplus(sR_raw) + self.scale_floor)

        # Sample mixture component via inverse CDF on cumulative probs
        cum = torch.cumsum(probs, dim=-1)  # (B,M,K+2)
        u = torch.rand(B, num_samples, M, device=self.device, dtype=self.dtype)
        
        # Vectorized sampling: flatten and use bucketize for each (B,M) combination
        cum_flat = cum.view(-1, cum.shape[-1])  # (B*M, K+2)
        u_flat = u.view(-1)  # (B*S*M,)
        
        # For each u value, find which cumulative bin it falls into
        comp_flat = torch.zeros_like(u_flat, dtype=torch.long)
        for i in range(cum_flat.shape[0]):
            # Find indices in u_flat that correspond to this (B,M) position
            start_idx = i * num_samples
            end_idx = (i + 1) * num_samples
            if start_idx < u_flat.shape[0]:
                end_idx = min(end_idx, u_flat.shape[0])
                u_subset = u_flat[start_idx:end_idx]
                comp_flat[start_idx:end_idx] = torch.bucketize(u_subset, cum_flat[i], right=False)
        
        comp = comp_flat.view(B, num_samples, M)  # (B,S,M)

        out = torch.empty(B, num_samples, M, device=self.device, dtype=self.dtype)

        # Gaussian helpers
        def half_normal(scale: Tensor, size_mask):
            # |N(0, scale^2)|
            z = torch.randn(size_mask.sum().item(), device=self.device, dtype=self.dtype)
            return torch.abs(z) * scale[size_mask]

        # Left tail: y = edge_left - |N(0, sL^2)|
        maskL = comp == 0
        if maskL.any():
            out[maskL] = self.edges[0] - half_normal(sL.unsqueeze(1).expand(B, num_samples, M), maskL)

        # Bars: uniform on [edge_k, edge_{k+1}]
        edges = self.edges
        if K > 0:
            for k in range(K):
                maskK = comp == (k + 1)
                if maskK.any():
                    a = edges[k]
                    b = edges[k + 1]
                    uK = torch.rand(maskK.sum().item(), device=self.device, dtype=self.dtype)
                    out[maskK] = a + (b - a) * uK

        # Right tail: y = edge_right + |N(0, sR^2)|
        maskR = comp == (K + 1)
        if maskR.any():
            out[maskR] = self.edges[-1] + half_normal(sR.unsqueeze(1).expand(B, num_samples, M), maskR)

        return out

    # ------------------------------ Internals ------------------------------

    def _check_ready(self):
        if any(v is None for v in [self.centers, self.edges, self.widths, self.base_s_left, self.base_s_right]):
            raise RuntimeError("BarDistribution not fitted. Call .fit(dataloader) first.")

    @staticmethod
    def _validate_pred(pred: Tensor, expected_shape: Tuple[int, int, int]):
        if pred.ndim != 3:
            raise ValueError(f"pred must be rank-3 (B,M,P), got {tuple(pred.shape)}")
        b, m, p = pred.shape
        eb, em, ep = expected_shape
        if (b, m, p) != (eb, em, ep):
            raise ValueError(f"pred has shape {(b,m,p)} but expected {(eb,em,ep)} (with P=K+4).")

    def _unpack(self, pred: Tensor):
        K = self.num_bars
        w_logits = pred[..., : K + 2]   # (B,M,K+2)
        sL_raw = pred[..., K + 2]       # (B,M)
        sR_raw = pred[..., K + 3]       # (B,M)
        return w_logits, sL_raw, sR_raw

    def _pdf_from_pred(self, pred: Tensor, y: Tensor) -> Tensor:
        """
        Vectorized pdf(y) with mixture params from pred.
        y: (B,M)  -> returns (B,M)
        """
        B, M = y.shape
        w_logits, sL_raw, sR_raw = self._unpack(pred)

        probs = torch.softmax(w_logits, dim=-1)  # (B,M,K+2)
        pL = probs[..., 0]
        pBars = probs[..., 1:-1]                 # (B,M,K)
        pR = probs[..., -1]

        sL = self.base_s_left * (torch.nn.functional.softplus(sL_raw) + self.scale_floor)
        sR = self.base_s_right * (torch.nn.functional.softplus(sR_raw) + self.scale_floor)

        pdf = torch.empty_like(y, dtype=self.dtype, device=self.device)
        left_mask = y < self.edges[0]
        right_mask = y >= self.edges[-1]
        mid_mask = ~(left_mask | right_mask)

        # Left half-Gaussian
        if left_mask.any():
            z = (y[left_mask] - self.edges[0]) / sL[left_mask]  # negative values
            gauss = self._norm_const * torch.exp(-0.5 * z * z) / sL[left_mask]
            pdf[left_mask] = (2.0 * pL[left_mask]) * gauss

        # Right half-Gaussian
        if right_mask.any():
            z = (y[right_mask] - self.edges[-1]) / sR[right_mask]  # nonnegative
            gauss = self._norm_const * torch.exp(-0.5 * z * z) / sR[right_mask]
            pdf[right_mask] = (2.0 * pR[right_mask]) * gauss

        # Bars
        if mid_mask.any():
            internal = self.edges[1:-1]  # (K-1,)
            k = torch.bucketize(y[mid_mask], internal, right=False)  # (num_mid,)
            widths_k = self.widths[k]
            pBars_all = pBars[mid_mask]                               # (num_mid, K)
            dens = pBars_all.gather(dim=-1, index=k.view(-1, 1)).squeeze(-1) / widths_k
            pdf[mid_mask] = dens

        return torch.clamp(pdf, min=torch.exp(torch.tensor(self.log_prob_clip_min, device=self.device, dtype=self.dtype)))

    # ------------------------- Convenience -------------------------

    @property
    def num_params(self) -> int:
        """Number of parameters your model must output per test point."""
        return self.num_bars + 4
