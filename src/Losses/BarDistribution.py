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
      - logits over [LeftTail, Bar_0..Bar_{K-1}, RightTail]   (length K+2)
      - tail scale raw params: [s_L_raw, s_R_raw]             (length 2)

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
        log_prob_clip_min: float = -50.0,
        log_prob_clip_max: float = 50.0,
        use_simple_equidistant_fit: bool = True,  # Use equidistant bars instead of quantile-based
    ):
        if num_bars < 1:
            raise ValueError("num_bars must be >= 1")
        self.num_bars = int(num_bars)
        self.min_width = float(min_width)
        self.scale_floor = float(scale_floor)
        self.max_fit_items = max_fit_items
        self.log_prob_clip_min = float(log_prob_clip_min)
        self.log_prob_clip_max = float(log_prob_clip_max)
        self.use_simple_equidistant_fit = bool(use_simple_equidistant_fit)

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        # Learned geometry (via .fit)
        self.centers: Optional[Tensor] = None   # (K,)
        self.edges: Optional[Tensor] = None     # (K+1,)
        self.widths: Optional[Tensor] = None    # (K,)
        self.base_s_left: Optional[Tensor] = None
        self.base_s_right: Optional[Tensor] = None

        # Will be (re)built for the right dtype/device later as needed
        self._norm_const: Optional[Tensor] = None
        self._log_norm_const: Optional[Tensor] = None
        self._tiny: Optional[Tensor] = None

    # ------------------------- Helpers -------------------------

    def _ensure_consts(self, device: torch.device, dtype: torch.dtype):
        # dtype-aware tiny to avoid log(0), underflow, etc.
        finfo = torch.finfo(dtype)
        tiny_val = max(1e-38, float(finfo.tiny))  # keep >= 1e-38 for fp32; for fp16 finfo.tiny is used
        self._tiny = torch.tensor(tiny_val, device=device, dtype=dtype)

        norm_const = 1.0 / math.sqrt(2.0 * math.pi)
        self._norm_const = torch.tensor(norm_const, device=device, dtype=dtype)
        self._log_norm_const = torch.tensor(math.log(norm_const), device=device, dtype=dtype)

    def _safe_softplus(self, x: Tensor) -> Tensor:
        # More numerically steady softplus for large negative x (avoid denorms)
        # torch.nn.functional.softplus has beta and threshold knobs; use a mild threshold.
        return torch.nn.functional.softplus(x, beta=1.0, threshold=20.0)

    def _safe_scale(self, base: Tensor, raw: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
        # s = base * (softplus(raw) + floor); clamp to dtype eps to avoid 0
        sp = self._safe_softplus(raw)
        out = base * (sp + self.scale_floor)
        eps = torch.finfo(dtype).eps
        return torch.clamp(out, min=eps)

    def _safe_log(self, x: Tensor) -> Tensor:
        # Avoid -inf by clamping with dtype-aware tiny
        return torch.log(torch.clamp(x, min=self._tiny))

    def _adopt_pred_ctx(self, pred: Tensor) -> Tuple[torch.device, torch.dtype]:
        # Everything follows pred's device/dtype if possible (backward compatible with constructor defaults)
        device = pred.device if pred.is_cuda or pred.device != torch.device("cpu") else self.device
        dtype = pred.dtype if pred.dtype is not None else self.dtype
        self._ensure_consts(device, dtype)
        return device, dtype

    # ------------------------- Fit bar locations -------------------------

    @torch.no_grad()
    def fit(self, dataloader: Iterable[Tuple[Tensor, Tensor, Tensor, Tensor]],
            max_batches: Optional[int] = None) -> "BarDistribution":
        """
        Fit the BarDistribution to data from a dataloader.

        Args:
            dataloader: DataLoader yielding (X_train, y_train, X_test, y_test) tuples
            max_batches: Maximum number of batches to use for fitting. If None, uses all batches.

        Returns:
            self for method chaining
        """
        ys = []
        total = 0
        batch_count = 0

        for batch in dataloader:
            if max_batches is not None and batch_count >= max_batches:
                break
            # Accept classic 4-tuple, interventional 6-tuple, or curriculum 6-tuple
            if isinstance(batch, (list, tuple)):
                if len(batch) == 4:
                    # Observational format: (X_train, y_train, X_test, y_test)
                    _, y_tr, _, y_te = batch
                elif len(batch) == 6:
                    # Check if this is interventional or curriculum format
                    # Interventional: (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv)
                    # Curriculum: (X_train, y_train, X_test, y_test, t, alpha)
                    # Distinguish by shape: T_obs/T_intv are 3D, t/alpha are scalars
                    if batch[1].dim() >= 2:
                        # Interventional format
                        _, _, y_tr, _, _, y_te = batch
                    else:
                        # Curriculum format (legacy)
                        _, y_tr, _, y_te, _t, _alpha = batch
                else:
                    raise ValueError(
                        f"Each dataloader item must be (X_train, y_train, X_test, y_test) or 6-element format; got length {len(batch)}"
                    )
            else:
                raise ValueError("Dataloader batch must be a tuple/list of length 4 or 6.")
            
            # Handle both (B, N) and (B, N, 1) shapes
            if y_tr.ndim == 3 and y_tr.shape[-1] == 1:
                y_tr = y_tr.squeeze(-1)
            if y_te.ndim == 3 and y_te.shape[-1] == 1:
                y_te = y_te.squeeze(-1)
            
            if y_tr.ndim != 2 or y_te.ndim != 2:
                raise ValueError(f"y_train and y_test must be (B, N/M) or (B, N/M, 1), got shapes {y_tr.shape} and {y_te.shape}.")

            y_tr_flat = y_tr.reshape(-1)
            y_te_flat = y_te.reshape(-1)

            # Filter non-finite early for robustness
            y_tr_flat = y_tr_flat[torch.isfinite(y_tr_flat)]
            y_te_flat = y_te_flat[torch.isfinite(y_te_flat)]

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
                if y_tr_flat.numel() > 0:
                    ys.append(y_tr_flat.cpu())
                if y_te_flat.numel() > 0:
                    ys.append(y_te_flat.cpu())

            batch_count += 1

        if not ys:
            raise ValueError("No y data collected for fit().")

        # Use double precision for the geometry, then cast down
        y_all = torch.cat(ys, dim=0).to(torch.float64)
        y_all = y_all[torch.isfinite(y_all)]
        if y_all.numel() == 0:
            raise ValueError("All y are non-finite in fit().")
        
        # Print number of datapoints used for fitting
        print(f"[BarDistribution] Fitting with {y_all.numel()} datapoints from {batch_count} batches")

        y_min, y_max = torch.min(y_all), torch.max(y_all)
        if not (torch.isfinite(y_min) and torch.isfinite(y_max)):
            raise ValueError("Non-finite y encountered during fit().")

        # Handle near-constant data robustly
        span = (y_max - y_min).item()
        if span <= 0 or not math.isfinite(span):
            # Make a tiny artificial span to define bars
            span = 1.0
            y_min = y_min - 0.5
            y_max = y_min + span

        K = self.num_bars

        if self.use_simple_equidistant_fit:
            # Simple equidistant fitting: evenly spaced bars across the data range
            print(f"[BarDistribution] Using simple equidistant fitting method")
            
            # Create K+1 evenly spaced edges from y_min to y_max
            edges = torch.linspace(float(y_min), float(y_max), steps=K + 1, dtype=torch.float64)
            
            # Centers are midpoints of edges
            centers = 0.5 * (edges[:-1] + edges[1:])
            
            # All widths are equal
            widths = torch.diff(edges)
            # Ensure minimum width
            widths = torch.clamp(widths, min=self.min_width)
            # Reconstruct edges from clamped widths to guarantee consistency
            edges = torch.cat([edges[:1], edges[:1] + torch.cumsum(widths, dim=0)], dim=0)
        else:
            # Original quantile-based centers; robust monotonic enforcement
            print(f"[BarDistribution] Using quantile-based fitting method")
            probs = torch.linspace(0.0, 1.0, steps=K + 1, dtype=torch.float64)
            mids = (probs[:-1] + probs[1:]) * 0.5
            try:
                centers = torch.quantile(y_all, mids, method="linear")
            except TypeError:
                centers = torch.quantile(y_all, mids, interpolation="linear")

            # Strictly increasing centers (epsilon grows with K and span)
            eps = max(1e-12, 1e-9 * K) * float(y_max - y_min)
            eps = torch.tensor(eps, dtype=torch.float64)
            for i in range(K - 1):
                if centers[i + 1] <= centers[i]:
                    centers[i + 1] = centers[i] + eps

            edges = torch.empty(K + 1, dtype=torch.float64)
            if K == 1:
                # Use IQR for width, fall back to small width if degenerate
                q25, q75 = torch.quantile(y_all, torch.tensor([0.25, 0.75], dtype=torch.float64))
                width = float(max(q75 - q25, self.min_width))
                edges[0] = centers[0] - 0.5 * width
                edges[1] = centers[0] + 0.5 * width
            else:
                edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
                # Extrapolate outer edges
                edges[0] = centers[0] - 0.5 * float(centers[1] - centers[0])
                edges[-1] = centers[-1] + 0.5 * float(centers[-1] - centers[-2])

            # Ensure strict monotonic edges and minimum widths
            edges = torch.clip(edges, min=float(y_min) - 10.0 * span, max=float(y_max) + 10.0 * span)
            widths = torch.diff(edges)
            widths = torch.clamp(widths, min=self.min_width)
            # Reconstruct edges from clamped widths to guarantee increasing edges
            edges = torch.cat([edges[:1], edges[:1] + torch.cumsum(widths, dim=0)], dim=0)

        # Tail base scales ~ near-edge widths (clamped)
        base_s_left = float(max(widths[0].item(), self.min_width))
        base_s_right = float(max(widths[-1].item(), self.min_width))

        # Cast to target dtype/device
        self.centers = centers.to(self.device, self.dtype)
        self.edges = edges.to(self.device, self.dtype)
        self.widths = widths.to(self.device, self.dtype)
        self.base_s_left = torch.tensor(base_s_left, device=self.device, dtype=self.dtype)
        self.base_s_right = torch.tensor(base_s_right, device=self.device, dtype=self.dtype)
        # Refresh constants
        self._ensure_consts(self.device, self.dtype)
        
        # Compute loss with constant (uniform) prediction on fitting data
        self._compute_constant_prediction_loss(y_all)
        
        return self

    def _compute_constant_prediction_loss(self, y_all: Tensor) -> None:
        """
        Compute and print the negative log-likelihood on fitting data using a 
        constant (uniform) prediction across all bars.
        
        This creates a "baseline" prediction where:
        - All bar logits are equal (uniform probability across bars and tails)
        - Tail scales use the base scales without modification
        
        Processes data in batches and averages the results.
        
        Args:
            y_all: All y values used for fitting (already filtered for finite values)
        """
        K = self.num_bars
        n_samples = y_all.shape[0]
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        log_probs = []
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            y_chunk = y_all[i:end_idx]
            chunk_size = y_chunk.shape[0]
            
            # Create constant prediction for this batch
            # Shape: (1, chunk_size, K+4)
            # First K+2 are logits (will be uniformly distributed after softmax)
            # Last 2 are tail scale raw params (0 means use base scales)
            const_pred = torch.zeros(1, chunk_size, K + 4, device=self.device, dtype=self.dtype)
            
            # Move y_chunk to correct device/dtype
            y_batch = y_chunk.to(device=self.device, dtype=self.dtype).unsqueeze(0)  # (1, chunk_size)
            
            # Compute average log probability for this batch
            batch_avg_log_prob = self.average_log_prob(const_pred, y_batch)  # (1,)
            log_probs.append(batch_avg_log_prob.item())
        
        # Average across all batches
        avg_log_prob = sum(log_probs) / len(log_probs)
        loss = -avg_log_prob
        
        print(f"\n[BarDistribution] Constant prediction baseline on fitting data:")
        print(f"   Number of samples: {n_samples}")
        print(f"   Number of batches: {len(log_probs)}")
        print(f"   Negative log-likelihood (loss): {loss:.6f}")
        print(f"   Average log-probability: {avg_log_prob:.6f}")

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

        device, dtype = self._adopt_pred_ctx(pred)
        y = y.to(device=device, dtype=dtype)

        pdf = self._pdf_from_pred(pred, y)  # (B,M), safely clipped inside
        logpdf = self._safe_log(pdf)
        logpdf = torch.clamp(logpdf, min=self.log_prob_clip_min, max=self.log_prob_clip_max).to(dtype=logpdf.dtype)
        return logpdf.mean(dim=1)

    def mode(self, pred: Tensor) -> Tensor:
        """
        Argmax among: left edge (tail), the densest bar, right edge (tail).
        """
        self._check_ready()
        B, M, _ = pred.shape
        K = self.num_bars
        self._validate_pred(pred, (B, M, K + 4))

        device, dtype = self._adopt_pred_ctx(pred)

        w_logits, sL_raw, sR_raw = self._unpack(pred)
        probs = torch.softmax(w_logits, dim=-1)  # (B,M,K+2)
        pL = probs[..., 0]
        pBars = probs[..., 1:-1]                 # (B,M,K)
        pR = probs[..., -1]

        sL = self._safe_scale(self.base_s_left.to(device, dtype), sL_raw, device, dtype)
        sR = self._safe_scale(self.base_s_right.to(device, dtype), sR_raw, device, dtype)

        # Tail densities at edges: 2 pTail * N(edge; edge, s) = 2 pTail * (norm_const / s)
        dens_left_edge = (2.0 * pL) * (self._norm_const / sL)
        dens_right_edge = (2.0 * pR) * (self._norm_const / sR)

        # Max bar density (constant per bar interval)
        widths = self.widths.view(1, 1, K).to(device=device, dtype=dtype)
        bar_densities = pBars / widths
        max_bar_dens, max_idx = torch.max(bar_densities, dim=-1)

        candidates = torch.stack([dens_left_edge, max_bar_dens, dens_right_edge], dim=-1)
        arg = candidates.argmax(dim=-1)

        mode_vals = torch.empty(B, M, device=device, dtype=dtype)
        edge_left = self.edges[0].to(device=device, dtype=dtype)
        edge_right = self.edges[-1].to(device=device, dtype=dtype)
        mode_vals[arg == 0] = edge_left
        mode_vals[arg == 2] = edge_right

        centers = self.centers.view(1, 1, K).to(device=device, dtype=dtype).expand(B, M, K)
        mode_bar = torch.gather(centers, dim=-1, index=max_idx.unsqueeze(-1)).squeeze(-1)
        mode_vals[arg == 1] = mode_bar[arg == 1]
        return mode_vals

    def mean(self, pred: Tensor) -> Tensor:
        """
        Mean is always finite with Gaussian tails.

        Bars: mean at midpoints weighted by bar probs.
        Left half-Gaussian:  E[y | left]  = edge_left  - sL * sqrt(2/π)
        Right half-Gaussian: E[y | right] = edge_right + sR * sqrt(2/π)
        """
        self._check_ready()
        B, M, _ = pred.shape
        K = self.num_bars
        self._validate_pred(pred, (B, M, K + 4))

        device, dtype = self._adopt_pred_ctx(pred)

        w_logits, sL_raw, sR_raw = self._unpack(pred)
        probs = torch.softmax(w_logits, dim=-1)
        pL = probs[..., 0]
        pBars = probs[..., 1:-1]  # (B,M,K)
        pR = probs[..., -1]

        sL = self._safe_scale(self.base_s_left.to(device, dtype), sL_raw, device, dtype)
        sR = self._safe_scale(self.base_s_right.to(device, dtype), sR_raw, device, dtype)

        mids = 0.5 * (self.edges[:-1] + self.edges[1:])       # (K,)
        mids = mids.to(device=device, dtype=dtype)
        bar_mean = torch.sum(pBars * mids.view(1, 1, K), dim=-1)  # (B,M)

        c = math.sqrt(2.0 / math.pi)
        c = torch.tensor(c, device=device, dtype=dtype)
        edge_left = self.edges[0].to(device=device, dtype=dtype)
        edge_right = self.edges[-1].to(device=device, dtype=dtype)
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

        device, dtype = self._adopt_pred_ctx(pred)

        w_logits, sL_raw, sR_raw = self._unpack(pred)
        probs = torch.softmax(w_logits, dim=-1).to(device=device, dtype=dtype)  # (B,M,K+2)

        sL = self._safe_scale(self.base_s_left.to(device, dtype), sL_raw, device, dtype)  # (B,M)
        sR = self._safe_scale(self.base_s_right.to(device, dtype), sR_raw, device, dtype)  # (B,M)

        # --- Sample mixture component robustly (vectorized) ---
        # Flatten (B,M,K+2) -> (BM, K+2) and sample S per row with replacement
        comps = torch.multinomial(
            probs.view(-1, K + 2),
            num_samples=num_samples,
            replacement=True
        )  # (BM, S), values in [0, K+1]
        comps = comps.view(B, M, num_samples).permute(0, 2, 1).contiguous()  # (B, S, M)

        # Prepare output
        out = torch.empty(B, num_samples, M, device=device, dtype=dtype)

        # Broadcast scales to (B,S,M) for masked sampling
        sL_b = sL.unsqueeze(1).expand(B, num_samples, M)
        sR_b = sR.unsqueeze(1).expand(B, num_samples, M)

        # Masks
        maskL = comps.eq(0)
        maskR = comps.eq(K + 1)

        # Half-normal sampler: |N(0, s^2)|
        def sample_half_normal(scale_vals: Tensor, n: int) -> Tensor:
            z = torch.randn(n, device=device, dtype=dtype)
            return z.abs() * scale_vals

        # Left tail: y = edge_left - |N(0, sL^2)|
        if maskL.any():
            nL = int(maskL.sum().item())
            noise = sample_half_normal(sL_b[maskL], nL)
            out[maskL] = self.edges[0].to(device, dtype) - noise

        # Bars: uniform on [edge_k, edge_{k+1}) for k in [0, K-1]
        if K > 0:
            edges = self.edges.to(device=device, dtype=dtype)
            # For speed, we sample all uniforms, then pick (a,b) per mask
            for k in range(K):
                maskK = comps.eq(k + 1)
                if maskK.any():
                    nK = int(maskK.sum().item())
                    u = torch.rand(nK, device=device, dtype=dtype)
                    a = edges[k]
                    b = edges[k + 1]
                    out[maskK] = a + (b - a) * u

        # Right tail: y = edge_right + |N(0, sR^2)|
        if maskR.any():
            nR = int(maskR.sum().item())
            noise = sample_half_normal(sR_b[maskR], nR)
            out[maskR] = self.edges[-1].to(device, dtype) + noise

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

    def _unpack(self, pred: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        K = self.num_bars
        # Preserve pred's device/dtype
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
        device, dtype = self._adopt_pred_ctx(pred)

        w_logits, sL_raw, sR_raw = self._unpack(pred)
        probs = torch.softmax(w_logits, dim=-1)  # (B,M,K+2)
        pL = probs[..., 0]
        pBars = probs[..., 1:-1]                 # (B,M,K)
        pR = probs[..., -1]

        sL = self._safe_scale(self.base_s_left.to(device, dtype), sL_raw, device, dtype)
        sR = self._safe_scale(self.base_s_right.to(device, dtype), sR_raw, device, dtype)

        pdf = torch.empty_like(y, dtype=dtype, device=device)

        edges = self.edges.to(device=device, dtype=dtype)
        widths = self.widths.to(device=device, dtype=dtype)

        # Regions
        left_mask = y < edges[0]
        right_mask = y >= edges[-1]
        mid_mask = ~(left_mask | right_mask)

        # Left half-Gaussian: f(y) = 2 pL * N(y; edge_left, sL)
        if left_mask.any():
            yL = y[left_mask]
            sL_sel = sL[left_mask]
            z = (yL - edges[0]) / sL_sel  # negative values
            # log N = log_norm_const - log s - 0.5 z^2
            log_gauss = self._log_norm_const - torch.log(sL_sel) - 0.5 * z * z
            log_pdf = torch.log(torch.tensor(2.0, device=device, dtype=dtype)) + torch.log(pL[left_mask]) + log_gauss
            clamped_log = torch.clamp(log_pdf, min=self.log_prob_clip_min, max=self.log_prob_clip_max)
            pdf[left_mask] = torch.exp(clamped_log).to(dtype=pdf.dtype)

        # Right half-Gaussian
        if right_mask.any():
            yR = y[right_mask]
            sR_sel = sR[right_mask]
            z = (yR - edges[-1]) / sR_sel  # nonnegative
            log_gauss = self._log_norm_const - torch.log(sR_sel) - 0.5 * z * z
            log_pdf = torch.log(torch.tensor(2.0, device=device, dtype=dtype)) + torch.log(pR[right_mask]) + log_gauss
            clamped_log = torch.clamp(log_pdf, min=self.log_prob_clip_min, max=self.log_prob_clip_max)
            pdf[right_mask] = torch.exp(clamped_log).to(dtype=pdf.dtype)

        # Bars (constant density within bar): f(y) = p_k / width_k
        if mid_mask.any():
            # internal edges are edges[1:-1]
            internal = edges[1:-1]  # (K-1,)
            # bucketize returns k in [0, K-1] mapping to intervals: [edge_k, edge_{k+1})
            k = torch.bucketize(y[mid_mask], internal, right=False)
            widths_k = widths[k]  # (num_mid,)
            pBars_all = pBars[mid_mask]  # (num_mid, K)
            dens = pBars_all.gather(dim=-1, index=k.view(-1, 1)).squeeze(-1) / widths_k
            # Directly clamp density to avoid zeros/denorms
            # Ensure result matches pdf dtype
            clamped = torch.clamp(dens, min=self._tiny)
            pdf[mid_mask] = clamped.to(dtype=pdf.dtype)

        # Final clamp for safety (to avoid log(0) upstream)
        return torch.clamp(pdf, min=self._tiny).to(dtype=pdf.dtype)

    # ------------------------- Convenience -------------------------

    @property
    def num_params(self) -> int:
        """Number of parameters your model must output per test point."""
        return self.num_bars + 4