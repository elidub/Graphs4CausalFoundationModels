from __future__ import annotations
from typing import Iterable, Optional, Tuple
from abc import ABC, abstractmethod
import torch
from torch import Tensor
import math

# Robust import: try relative to package, then absolute, then with src prefix
try:
    from .PosteriorPredictive import PosteriorPredictive
except Exception:
    try:
        from Losses.PosteriorPredictive import PosteriorPredictive
    except Exception:
        from src.Losses.PosteriorPredictive import PosteriorPredictive


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
        log_prob_clip_min: float = -50.0,  # Minimum log probability to prevent -inf in training loops
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

        # constants (stored with correct dtype/device)
        self._LOG_TWO = torch.tensor(math.log(2.0), device=self.device, dtype=self.dtype)
        self._LOG_SQRT_2PI = torch.tensor(0.5 * math.log(2.0 * math.pi), device=self.device, dtype=self.dtype)
        # pragmatic ceiling to avoid ridiculous scales; doesn’t change API
        self._SCALE_CEIL = torch.tensor(1e6, device=self.device, dtype=self.dtype)
        # guard tiny widths
        self._MIN_WIDTH = torch.tensor(self.min_width, device=self.device, dtype=self.dtype)

    # ------------------------- Fit bar locations -------------------------

    @torch.no_grad()
    def fit(self, dataloader: Iterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
            max_batches: Optional[int] = None) -> "BarDistribution":
        ys = []
        total = 0
        batch_count = 0

        for batch in dataloader:
            if max_batches is not None and batch_count >= max_batches:
                break
            if len(batch) != 4:
                raise ValueError("Each dataloader item must be (X_train, y_train, X_test, y_test).")
            _, y_tr, _, y_te = batch
            y_tr = y_tr.squeeze(-1)
            y_te = y_te.squeeze(-1)

            if y_tr.ndim != 2 or y_te.ndim != 2:
                raise ValueError("y_train and y_test must be (B, N/M).")

            y_tr_flat = y_tr.reshape(-1).detach()
            y_te_flat = y_te.reshape(-1).detach()

            if self.max_fit_items is not None:
                remaining = max(0, self.max_fit_items - total)
                if remaining <= 0:
                    break
                need_tr = min(remaining // 2, y_tr_flat.numel())
                need_te = min(remaining - need_tr, y_te_flat.numel())
                if need_tr > 0:
                    idx = torch.randint(0, y_tr_flat.numel(), (need_tr,), device=y_tr_flat.device)
                    ys.append(y_tr_flat.index_select(0, idx).cpu())
                    total += need_tr
                if need_te > 0:
                    idx = torch.randint(0, y_te_flat.numel(), (need_te,), device=y_te_flat.device)
                    ys.append(y_te_flat.index_select(0, idx).cpu())
                    total += need_te
            else:
                ys.append(y_tr_flat.cpu())
                ys.append(y_te_flat.cpu())

            batch_count += 1

        if not ys:
            raise ValueError("No y data collected for fit().")

        y_all = torch.cat(ys, dim=0).to(torch.float64)
        y_all = y_all[torch.isfinite(y_all)]
        if y_all.numel() == 0:
            raise ValueError("All y values were non-finite in fit().")

        y_min, y_max = torch.min(y_all), torch.max(y_all)

        K = self.num_bars
        probs = torch.linspace(0.0, 1.0, steps=K + 1, dtype=torch.float64)
        mids = (probs[:-1] + probs[1:]) * 0.5
        try:
            centers = torch.quantile(y_all, mids, method="linear")
        except TypeError:
            centers = torch.quantile(y_all, mids, interpolation="linear")

        # Enforce strict increasing centers (tiny jitter if needed)
        scale = torch.clamp(y_max - y_min, min=torch.tensor(1.0, dtype=torch.float64))
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
            edges[0] = centers[0] - 0.5 * float(max(centers[1] - centers[0], 1e-6))
            edges[-1] = centers[-1] + 0.5 * float(max(centers[-1] - centers[-2], 1e-6))

        widths = torch.diff(edges)
        widths = torch.clamp(widths, min=self.min_width)

        # Tail base scales from near-edge widths (clamped)
        base_s_left = float(max(widths[0].item(), self.min_width))
        base_s_right = float(max(widths[-1].item(), self.min_width))

        self.centers = centers.to(self.device, self.dtype)
        self.edges = edges.to(self.device, self.dtype)
        self.widths = widths.to(self.device, self.dtype)
        self.base_s_left = torch.tensor(base_s_left, device=self.device, dtype=self.dtype)
        self.base_s_right = torch.tensor(base_s_right, device=self.device, dtype=self.dtype)
        return self

    # ------------------------------ Interface ------------------------------

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

        logpdf = self._logpdf_from_pred(pred, y)  # (B,M)
        # Bound extreme tails to keep training stable
        logpdf = torch.clamp(logpdf, min=self.log_prob_clip_min, max=self.log_prob_clip_max)
        return logpdf.mean(dim=1)

    def mode(self, pred: Tensor) -> Tensor:
        """
        Argmax among: left edge (tail), the densest bar, right edge (tail).
        """
        self._check_ready()
        B, M, _ = pred.shape
        K = self.num_bars
        self._validate_pred(pred, (B, M, K + 4))

        w_logits, sL_raw, sR_raw = self._unpack(pred)
        log_probs = torch.log_softmax(w_logits, dim=-1)  # (B,M,K+2)

        # Scales (safe)
        sL = self._safe_scale(self.base_s_left, sL_raw)
        sR = self._safe_scale(self.base_s_right, sR_raw)

        # log density at edges for tails: log(2 pTail) + logN(edge; edge, s) = log 2 + log pTail - log s - 0.5 log(2π)
        log_dens_left_edge = self._LOG_TWO + log_probs[..., 0] - torch.log(sL) - self._LOG_SQRT_2PI
        log_dens_right_edge = self._LOG_TWO + log_probs[..., -1] - torch.log(sR) - self._LOG_SQRT_2PI

        # Bar density (uniform): log p_k - log width_k
        log_bar_densities = log_probs[..., 1:-1] - torch.log(self.widths.view(1, 1, K))
        max_log_bar_dens, max_idx = torch.max(log_bar_densities, dim=-1)

        # Compare three candidates in log-space
        candidates = torch.stack([log_dens_left_edge, max_log_bar_dens, log_dens_right_edge], dim=-1)
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
        # Convert to probs carefully; clamp small to prevent all-zero after extreme logits
        log_probs = torch.log_softmax(w_logits, dim=-1)
        probs = torch.exp(torch.clamp(log_probs, min=self.log_prob_clip_min))  # (B,M,K+2)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        pL = probs[..., 0]
        pBars = probs[..., 1:-1]  # (B,M,K)
        pR = probs[..., -1]

        sL = self._safe_scale(self.base_s_left, sL_raw)
        sR = self._safe_scale(self.base_s_right, sR_raw)

        mids = 0.5 * (self.edges[:-1] + self.edges[1:])       # (K,)
        bar_mean = (pBars * mids.view(1, 1, K)).sum(dim=-1)   # (B,M)

        c = math.sqrt(2.0 / math.pi)
        edge_left = self.edges[0]
        edge_right = self.edges[-1]
        E_left = edge_left - c * sL
        E_right = edge_right + c * sR

        return pL * E_left + pR * E_right + bar_mean  # (B,M)

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
        # Use softmax here (sampling needs probs); stay stable by subtracting max
        probs = torch.softmax(w_logits, dim=-1)  # (B,M,K+2)

        sL = self._safe_scale(self.base_s_left, sL_raw)
        sR = self._safe_scale(self.base_s_right, sR_raw)

        # Cumulative probabilities
        cum = torch.cumsum(probs, dim=-1)  # (B,M,K+2)
        # ensure last exactly 1 to avoid out-of-range indices
        cum[..., -1] = 1.0

        # Vectorized sampling with searchsorted
        u = torch.rand(B, num_samples, M, device=self.device, dtype=self.dtype)  # (B,S,M)
        cum_flat = cum.reshape(B * M, K + 2)
        u_flat = u.permute(0, 2, 1).reshape(B * M, num_samples)                  # (B*M,S)
        comp_flat = torch.searchsorted(cum_flat, u_flat, right=False)            # (B*M,S), long
        comp = comp_flat.reshape(B, M, num_samples).permute(0, 2, 1).contiguous()  # (B,S,M)

        out = torch.empty(B, num_samples, M, device=self.device, dtype=self.dtype)

        # Gaussian helpers: |N(0, s^2)|
        def half_normal(x_scale: Tensor, mask: Tensor) -> Tensor:
            # x_scale shape: (B,S,M)
            n = int(mask.sum().item())
            if n == 0:
                return torch.empty(0, device=self.device, dtype=self.dtype)
            z = torch.randn(n, device=self.device, dtype=self.dtype)
            scales = x_scale[mask]
            return torch.abs(z) * scales

        # Broadcast scales to (B,S,M)
        sL_b = sL.unsqueeze(1).expand(B, num_samples, M)
        sR_b = sR.unsqueeze(1).expand(B, num_samples, M)

        # Left tail: y = edge_left - |N(0, sL^2)|
        maskL = comp == 0
        if maskL.any():
            out[maskL] = self.edges[0] - half_normal(sL_b, maskL)

        # Bars: uniform on [edge_k, edge_{k+1}]
        if K > 0:
            edges = self.edges  # (K+1,)
            # For bar components, comp in [1..K]
            bar_mask_any = (comp >= 1) & (comp <= K)
            if bar_mask_any.any():
                # compute k index for bars
                k_idx = torch.clamp(comp - 1, min=0, max=K - 1)
                a = edges[k_idx]               # (B,S,M)
                b = edges[torch.clamp(k_idx + 1, max=K)]  # safe next edge
                uK = torch.rand_like(out)
                out[bar_mask_any] = (a + (b - a) * uK)[bar_mask_any]

        # Right tail: y = edge_right + |N(0, sR^2)|
        maskR = comp == (K + 1)
        if maskR.any():
            out[maskR] = self.edges[-1] + half_normal(sR_b, maskR)

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

    def _safe_scale(self, base: Tensor, raw: Tensor) -> Tensor:
        """
        Map raw -> positive scale with floors/ceilings to avoid 0, inf, NaN.
        """
        s = torch.nn.functional.softplus(raw) + self.scale_floor
        s = base * s
        # clamp to reasonable numeric bounds
        s = torch.clamp(s, min=torch.finfo(self.dtype).tiny, max=self._SCALE_CEIL)
        return s

    def _logpdf_from_pred(self, pred: Tensor, y: Tensor) -> Tensor:
        """
        Vectorized log pdf(y) with mixture params from pred.
        y: (B,M)  -> returns (B,M) in log-space
        """
        B, M = y.shape
        K = self.num_bars
        w_logits, sL_raw, sR_raw = self._unpack(pred)

        # Mixture weights in log-space
        log_w = torch.log_softmax(w_logits, dim=-1)  # (B,M,K+2)

        # Scales (safe)
        sL = self._safe_scale(self.base_s_left, sL_raw)   # (B,M)
        sR = self._safe_scale(self.base_s_right, sR_raw)  # (B,M)

        # Masks for support pieces
        left_mask = y < self.edges[0]
        right_mask = y >= self.edges[-1]
        mid_mask = ~(left_mask | right_mask)

        # Output container
        logpdf = torch.full((B, M), -float("inf"), device=self.device, dtype=self.dtype)

        # ---- Left tail log-density (half-normal centered at edge_left) ----
        if left_mask.any():
            z = (y[left_mask] - self.edges[0]) / sL[left_mask]  # negative values allowed
            # log N(y; edge, s) = -0.5 z^2 - log s - 0.5 log(2π)
            log_gauss = -0.5 * (z * z) - torch.log(sL[left_mask]) - self._LOG_SQRT_2PI
            # include factor 2 and mixture weight
            logpdf[left_mask] = self._LOG_TWO + log_w[..., 0][left_mask] + log_gauss

        # ---- Right tail log-density (half-normal centered at edge_right) ----
        if right_mask.any():
            z = (y[right_mask] - self.edges[-1]) / sR[right_mask]  # nonnegative
            log_gauss = -0.5 * (z * z) - torch.log(sR[right_mask]) - self._LOG_SQRT_2PI
            logpdf[right_mask] = self._LOG_TWO + log_w[..., -1][right_mask] + log_gauss

        # ---- Bars (uniform) between edges ----
        if mid_mask.any():
            # Identify the bar index for each mid point
            internal = self.edges[1:-1]  # (K-1,)
            k = torch.bucketize(y[mid_mask], internal, right=False)  # in [0..K-1]
            # log density = log p_k - log width_k
            log_pbars = log_w[..., 1:-1][mid_mask]  # (Nmid, K)
            # select the active bar weight per point
            log_p_k = log_pbars.gather(dim=-1, index=k.view(-1, 1)).squeeze(-1)  # (Nmid,)
            log_width_k = torch.log(self.widths[k])  # (Nmid,)
            logpdf[mid_mask] = log_p_k - log_width_k

        # final numeric clipping
        return torch.clamp(logpdf, min=self.log_prob_clip_min, max=self.log_prob_clip_max)

    # ------------------------- Convenience -------------------------

    @property
    def num_params(self) -> int:
        """Number of parameters your model must output per test point."""
        return self.num_bars + 4