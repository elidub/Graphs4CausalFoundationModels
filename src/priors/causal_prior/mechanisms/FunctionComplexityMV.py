from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import math, zlib
import numpy as np
import torch
from torch import Tensor
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ----------------------- Reports -----------------------

@dataclass
class ComplexityReport:
    # compression (bits/sample)
    bps: float
    bps_delta: float
    # spectrum (grid mode exact, MC mode surrogate)
    spectral_entropy: float
    highfreq_energy_frac: float
    # geometry
    dirichlet_energy: float         # ∫ ||∇f||^2  (discrete sum/mean proxy)
    grad_tv: float                  # total variation of gradient (grid) / surrogate
    lipschitz_max: float            # max |Δf| / ||Δx||
    def to_dict(self) -> Dict[str, float]: return asdict(self)

@dataclass
class RelativeComplexity:
    baseline: str
    absolute: ComplexityReport
    ratios: Dict[str, float]


# ----------------------- Helper modules -----------------------

class _ToModule(nn.Module):
    def __init__(self, f: Callable[[Tensor], Tensor]): super().__init__(); self.f = f
    def forward(self, x: Tensor) -> Tensor: return self.f(x)

def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float("nan") if abs(b) < eps else float(a / b)


# ----------------------- Multivariate complexity -----------------------

class FunctionComplexityMV:
    """
    Complexity metrics for multivariate scalar functions f: R^D -> R.

    Two sampling modes:
      • GRID mode:   provide grid_shape=(n1,...,nD) to sample on a regular grid within x_bounds.
                     Pros: true n-D FFT, clean finite differences.
      • MC mode:     provide n_samples; inputs are i.i.d. (Normal or Uniform per dim). Surrogate spectrum
                     via 1-D PCA ordering; k-NN LS gradient estimates.

    Metrics (all based on normalized outputs y = (f - mean) / std):
      - bps, bps_delta: zlib bits per sample (raw & delta-encoded along traversal)
      - spectral_entropy, highfreq_energy_frac
      - dirichlet_energy ≈ mean(||∇f||^2)
      - grad_tv: TV of gradient field (grid) or surrogate in MC
      - lipschitz_max = max |Δf| / ||Δx||

    Args
    ----
    x_bounds : Sequence[Tuple[float,float]]
        Bounds per dimension [(low1, high1), ..., (lowD, highD)].
    grid_shape : Optional[Tuple[int,...]]
        Grid sizes (D-tuple). If provided → GRID mode.
    n_samples : Optional[int]
        Number of random samples (MC mode) if grid_shape is None.
    input_dist : {"uniform","normal"}
        MC mode only: per-dimension distribution within bounds.
    device, dtype : torch settings.
    """

    def __init__(
        self,
        x_bounds: Sequence[Tuple[float, float]],
        grid_shape: Optional[Tuple[int, ...]] = None,
        n_samples: Optional[int] = None,
        input_dist: str = "uniform",
        highfreq_cutoff_frac: float = 0.25,
        quant_bits: int = 12,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.x_bounds = list(map(tuple, x_bounds))
        self.D = len(self.x_bounds)
        self.grid_shape = tuple(grid_shape) if grid_shape is not None else None
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.input_dist = input_dist
        self.device = torch.device(device)
        self.dtype = dtype
        self.gen = generator
        self.hicut = float(highfreq_cutoff_frac)
        self.quant_bits = int(quant_bits)

        if self.grid_shape is None and self.n_samples is None:
            raise ValueError("Provide either grid_shape (GRID mode) or n_samples (MC mode).")

    # --------------- Public API ---------------

    @torch.no_grad()
    def evaluate(self, f: Union[nn.Module, Callable[[Tensor], Tensor]]) -> ComplexityReport:
        if self.grid_shape is not None:
            X, y = self._sample_grid(f)         # X: (..., D), y: (...)
            y_n = self._normalize(y)
            bps, bps_delta = self._compression_grid(y_n)
            sent, hff = self._spectrum_grid(y_n)
            dE, tvg, Lmax = self._geometry_grid(X, y)
            return ComplexityReport(bps, bps_delta, sent, hff, dE, tvg, Lmax)
        else:
            X, y = self._sample_mc(f)           # X: (N, D), y: (N,)
            y_n = self._normalize(y)
            order = self._pca_order(X)          # permutation for 1-D traversal
            bps, bps_delta = self._compression_seq(y_n[order])
            sent, hff = self._spectrum_seq(y_n[order])
            dE, tvg, Lmax = self._geometry_mc(X, y)
            return ComplexityReport(bps, bps_delta, sent, hff, dE, tvg, Lmax)

    @torch.no_grad()
    def evaluate_relative(
        self,
        f: Union[nn.Module, Callable[[Tensor], Tensor]],
        baseline: Union[str, nn.Module, Callable[[Tensor], Tensor]] = "relu",
    ) -> RelativeComplexity:
        rep_abs = self.evaluate(f)
        name, base = self._make_baseline(baseline)
        rep_base = self.evaluate(base)
        ratios = {k: _safe_div(getattr(rep_abs, k), getattr(rep_base, k)) for k in rep_abs.to_dict().keys()}
        return RelativeComplexity(name, rep_abs, ratios)

    # --------------- Baselines ---------------

    def _make_baseline(self, baseline) -> Tuple[str, nn.Module]:
        if isinstance(baseline, str):
            s = baseline.lower()
            if s == "relu":     return "relu", _ToModule(lambda x: torch.relu(x[..., 0]).unsqueeze(-1) if x.dim()>1 else torch.relu(x))
            if s in ("identity","id"):
                return "identity", _ToModule(lambda x: x[..., 0].unsqueeze(-1) if x.dim()>1 else x)
            if s in ("square","x2","x^2"):
                return "square", _ToModule(lambda x: (x[..., 0]**2).unsqueeze(-1) if x.dim()>1 else x**2)
            raise ValueError("baseline must be 'relu','identity','square' or a module/callable.")
        if isinstance(baseline, nn.Module):  # assume it maps (B,D) or (...,D) -> scalar
            return baseline.__class__.__name__, baseline
        return getattr(baseline, "__name__", "callable_baseline"), _ToModule(baseline)

    # --------------- Sampling ---------------

    def _linspaces(self) -> List[Tensor]:
        ls = []
        for (lo, hi), n in zip(self.x_bounds, self.grid_shape):
            ls.append(torch.linspace(lo, hi, n, device=self.device, dtype=self.dtype))
        return ls

    def _meshgrid(self, ls: List[Tensor]) -> Tensor:
        grids = torch.meshgrid(*ls, indexing="ij")  # D tensors each of shape grid_shape
        X = torch.stack(grids, dim=-1)              # (..., D)
        return X

    def _sample_grid(self, f) -> Tuple[Tensor, Tensor]:
        ls = self._linspaces()
        X = self._meshgrid(ls)  # (..., D)
        y = f(X.reshape(-1, self.D)).reshape(self.grid_shape)  # (...)
        if y.dim() > len(self.grid_shape):  # if extra dim, squeeze
            y = y.squeeze(-1)
        return X, y

    def _sample_mc(self, f) -> Tuple[Tensor, Tensor]:
        N = self.n_samples
        lows = torch.tensor([a for a, _ in self.x_bounds], device=self.device, dtype=self.dtype)
        highs = torch.tensor([b for _, b in self.x_bounds], device=self.device, dtype=self.dtype)
        if self.input_dist == "uniform":
            U = torch.rand((N, self.D), generator=self.gen, device=self.device, dtype=self.dtype)
            X = lows + (highs - lows) * U
        elif self.input_dist == "normal":
            mu = (lows + highs) * 0.5
            sig = (highs - lows) / 6.0
            X = mu + sig * torch.randn((N, self.D), generator=self.gen, device=self.device, dtype=self.dtype)
        else:
            raise ValueError("input_dist must be 'uniform' or 'normal'")
        y = f(X)
        if y.dim() > 1:
            y = y.squeeze(-1)
        return X, y

    # --------------- Normalization & quantization ---------------

    def _normalize(self, y: Tensor) -> Tensor:
        y = y - y.mean()
        s = y.std(unbiased=False)
        return (y / (s + 1e-12)) if float(s) > 0 else y

    def _quantize(self, y: Tensor) -> np.ndarray:
        # robust: median & IQR to [0,1], then to uint16
        y_np = y.detach().cpu().numpy().astype(np.float64)
        med = np.median(y_np)
        iqr = np.subtract(*np.percentile(y_np, [75, 25]))
        scale = (iqr if iqr > 1e-8 else y_np.std() + 1e-8)
        z = 0.5 + 0.25 * (y_np - med) / scale
        z = np.clip(z, 0.0, 1.0)
        levels = (1 << self.quant_bits) - 1
        return (z * levels + 0.5).astype(np.uint16)

    # --------------- Compression ---------------

    def _compression_grid(self, y_norm: Tensor) -> Tuple[float, float]:
        q = self._quantize(y_norm.reshape(-1))
        raw = q.tobytes()
        comp = zlib.compress(raw, level=9)
        bps = (len(comp) * 8) / float(q.size)
        dq = np.diff(q.astype(np.int32), prepend=int(q[0]))
        comp_d = zlib.compress(dq.tobytes(), level=9)
        bps_d = (len(comp_d) * 8) / float(q.size)
        return bps, bps_d

    def _compression_seq(self, y_seq: Tensor) -> Tuple[float, float]:
        q = self._quantize(y_seq.reshape(-1))
        raw = q.tobytes()
        comp = zlib.compress(raw, level=9)
        bps = (len(comp) * 8) / float(q.size)
        dq = np.diff(q.astype(np.int32), prepend=int(q[0]))
        comp_d = zlib.compress(dq.tobytes(), level=9)
        bps_d = (len(comp_d) * 8) / float(q.size)
        return bps, bps_d

    # --------------- Spectrum ---------------

    def _spectrum_grid(self, y_norm: Tensor) -> Tuple[float, float]:
        # n-D window (product of Hanns per axis)
        shape = y_norm.shape
        win_axes = [torch.hann_window(n, device=self.device, dtype=self.dtype) for n in shape]
        W = 1.0
        for ax, n in enumerate(shape):
            W = W * win_axes[ax].reshape([n if i == ax else 1 for i in range(len(shape))])
        Y = tfft.fftn((y_norm - y_norm.mean()) * W)
        power = (Y.real**2 + Y.imag**2)
        psd = power / (power.sum() + 1e-20)

        # spectral entropy
        sent = float(-(psd * (psd + 1e-20).log()).sum().item())

        # high-frequency fraction: radial cutoff in frequency index space
        freq_coords = [torch.fft.fftfreq(n, d=1.0, device=self.device) for n in shape]
        grids = torch.meshgrid(*freq_coords, indexing="ij")
        r = torch.sqrt(sum(g**2 for g in grids))
        r_flat = r.reshape(-1)
        psd_flat = psd.reshape(-1)
        thr = torch.quantile(r_flat, torch.tensor(1.0 - self.hicut, device=self.device))
        hf_frac = float(psd_flat[r_flat >= thr].sum().item())
        return sent, hf_frac

    def _spectrum_seq(self, y_seq: Tensor) -> Tuple[float, float]:
        # 1-D surrogate FFT on traversal sequence
        n = y_seq.numel()
        w = torch.hann_window(n, device=self.device, dtype=self.dtype)
        Y = tfft.rfft((y_seq - y_seq.mean()) * w)
        power = (Y.real**2 + Y.imag**2)
        psd = power / (power.sum() + 1e-20)
        sent = float(-(psd * (psd + 1e-20).log()).sum().item())
        k_cut = int((1.0 - self.hicut) * psd.numel())
        k_cut = max(1, min(k_cut, psd.numel()-1))
        hf_frac = float(psd[k_cut:].sum().item())
        return sent, hf_frac

    # --------------- Geometry ---------------

    def _geometry_grid(self, X: Tensor, y: Tensor) -> Tuple[float, float, float]:
        # X: (..., D), y: (...)
        diffs = []
        Lmax = 0.0
        # per-axis forward differences
        for ax in range(self.D):
            slice1 = [slice(None)] * self.D
            slice2 = [slice(None)] * self.D
            slice1[ax] = slice(1, None)
            slice2[ax] = slice(None, -1)
            y1 = y[tuple(slice1)]
            y2 = y[tuple(slice2)]
            # Δf and Δx along axis
            delta_f = y1 - y2
            x1 = X[tuple(slice1 + [slice(None)])]
            x2 = X[tuple(slice2 + [slice(None)])]
            delta_x = (x1[..., ax] - x2[..., ax]).abs() + 1e-12
            slope = delta_f / delta_x
            Lmax = max(Lmax, float(slope.abs().max().item()))
            diffs.append(slope)

        # gradient magnitude (coordinate finite difference as proxy)
        # align shapes by padding last element to original size
        grads_sq = 0.0
        for ax, g in enumerate(diffs):
            pad = [0, 0] * (g.dim())  # no-op pad; we’ll upsample to match original by last-replication
            # match size by simple replicate of last along axis
            g_full = torch.cat([g, g[(slice(None),)*ax + (-1,)] .unsqueeze(ax)], dim=ax)
            grads_sq = grads_sq + g_full**2
        dirichlet_energy = float(grads_sq.mean().item())

        # TV of gradient: sum of |∇f(x+e_i) - ∇f(x)| along axes (proxy)
        tv = 0.0
        for ax, g in enumerate(diffs):
            # difference along another axis (cyclic) to measure variation
            ax2 = (ax + 1) % self.D
            # take forward difference of slope field along ax2
            slice1 = [slice(None)] * g.dim()
            slice2 = [slice(None)] * g.dim()
            slice1[ax2] = slice(1, None)
            slice2[ax2] = slice(None, -1)
            dg = g[tuple(slice1)] - g[tuple(slice2)]
            tv += dg.abs().mean()
        grad_tv = float(tv.item())
        return dirichlet_energy, grad_tv, Lmax

    def _geometry_mc(self, X: Tensor, y: Tensor, k: int = 16) -> Tuple[float, float, float]:
        # k-NN LS gradient estimation per point (small proxy; O(N^2) if naive)
        # For efficiency, subsample centers if N is large.
        N = X.shape[0]
        max_centers = min(4096, N)
        if N > max_centers:
            idx_cent = torch.randperm(N, generator=self.gen, device=self.device)[:max_centers]
            Xc = X[idx_cent]; yc = y[idx_cent]
        else:
            Xc, yc = X, y

        # distances to choose neighbors
        D2 = torch.cdist(Xc, X, p=2)  # (Nc, N)
        nn_idx = torch.topk(D2, k=min(k, N), largest=False).indices  # (Nc, k)

        grads = []
        Lmax = 0.0
        for i in range(nn_idx.shape[0]):
            nbr = nn_idx[i]                      # (k,)
            Xi = X[nbr]                          # (k, D)
            yi = y[nbr]                          # (k,)
            x0 = Xi.mean(dim=0, keepdim=True)
            y0 = yi.mean()
            A = Xi - x0                           # (k, D)
            b = (yi - y0).unsqueeze(-1)          # (k, 1)
            # ridge LS for stability
            AtA = A.T @ A + 1e-6 * torch.eye(self.D, device=self.device, dtype=self.dtype)
            g = torch.linalg.solve(AtA, A.T @ b).squeeze(-1)  # (D,)
            grads.append(g)
            # Lipschitz proxy from neighbors
            df = (yi - y0).abs()
            dx = torch.norm(Xi - x0, dim=1) + 1e-12
            Lmax = max(Lmax, float((df / dx).max().item()))
        G = torch.stack(grads, dim=0) if grads else torch.zeros((1, self.D), device=self.device, dtype=self.dtype)
        dirichlet_energy = float((G.pow(2).sum(dim=1)).mean().item())

        # TV surrogate: variation of gradient vectors across centers
        if G.shape[0] > 1:
            # kNN among centers
            Dc = torch.cdist(Xc[:G.shape[0]], Xc[:G.shape[0]], p=2)
            K = min(8, G.shape[0]-1)
            idx_nb = torch.topk(Dc, k=K+1, largest=False).indices[:, 1:]  # exclude self
            diffs = []
            for i in range(G.shape[0]):
                diffs.append((G[idx_nb[i]] - G[i]).norm(dim=1).mean())
            grad_tv = float(torch.stack(diffs).mean().item())
        else:
            grad_tv = 0.0
        return dirichlet_energy, grad_tv, Lmax

    # --------------- PCA ordering for MC compression/spectrum ---------------

    def _pca_order(self, X: Tensor) -> Tensor:
        Xc = X - X.mean(dim=0, keepdim=True)
        C = (Xc.T @ Xc) / Xc.shape[0]
        # top eigenvector via power iteration
        v = torch.randn((self.D, 1), device=self.device, dtype=self.dtype)
        for _ in range(16):
            v = C @ v
            v = v / (v.norm() + 1e-12)
        scores = (Xc @ v).squeeze(-1)
        return torch.argsort(scores)


# ----------------------- Batched analyzer -----------------------

class ComplexityBatchAnalyzerMV:
    """
    Batch evaluation and plotting helper for multivariate functions.
    """

    def __init__(self, fc: FunctionComplexityMV) -> None:
        self.fc = fc

    @torch.no_grad()
    def analyze(
        self,
        functions: Sequence[Union[nn.Module, Callable[[Tensor], Tensor]]],
        names: Optional[Sequence[str]] = None,
        baselines: Sequence[Union[str, nn.Module, Callable[[Tensor], Tensor]]] = ("relu", "identity"),
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[str]]:
        """
        Returns:
          - abs_mat: (N_funcs, N_metrics) absolute metrics
          - rel_dict: mapping baseline_name -> (N_funcs, N_metrics) ratios
          - metric_names: list of metric field names in order
        """
        if names is None:
            names = [getattr(f, "__name__", f.__class__.__name__) for f in functions]
        reports = [self.fc.evaluate(fn) for fn in tqdm(functions, desc="Evaluating functions")]
        metric_names = list(reports[0].to_dict().keys()) if reports else []
        abs_mat = torch.tensor([[getattr(r, m) for m in metric_names] for r in reports], dtype=torch.float32)

        rel_dict: Dict[str, torch.Tensor] = {}
        for b in tqdm(baselines, desc="Computing relative metrics"):
            bname, _ = self.fc._make_baseline(b)
            brep = self.fc.evaluate_relative(functions[0], baseline=b).absolute  # compute baseline metrics
            bvec = torch.tensor([getattr(brep, m) for m in metric_names], dtype=torch.float32)
            # recompute baseline directly (not via first function!)
            bvec = torch.tensor([getattr(self.fc.evaluate(self.fc._make_baseline(b)[1]), m) for m in metric_names], dtype=torch.float32)
            denom = torch.where(bvec.abs() < 1e-12, torch.nan, bvec)
            rel = abs_mat / denom
            rel_dict[bname] = rel
        return abs_mat, rel_dict, metric_names

    def plot_histograms(
        self,
        matrix: torch.Tensor,
        metric_names: Sequence[str],
        *,
        cols: int = 3,
        bins: int = 40,
        title: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        m = len(metric_names)
        rows = int(math.ceil(m / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14, 9))
        axes = np.array(axes).reshape(-1)
        for ax, j in zip(axes, range(m)):
            vals = matrix[:, j].detach().cpu().numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                ax.set_visible(False); continue
            ax.hist(vals, bins=bins)
            mu, med = np.mean(vals), np.median(vals)
            q1, q3 = np.percentile(vals, [25, 75])
            vmin, vmax = vals.min(), vals.max()
            for v, ls, lab in [(mu, "--", f"mean={mu:.3g}"), (med, "-.", f"median={med:.3g}"),
                               (q1, ":", f"Q1={q1:.3g}"), (q3, ":", f"Q3={q3:.3g}"),
                               (vmin, "-", f"min={vmin:.3g}"), (vmax, "-", f"max={vmax:.3g}")]:
                ax.axvline(v, linestyle=ls, linewidth=1, label=lab)
            ax.set_title(metric_names[j])
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8, loc="lower left")
            # stats box
            txt = f"n={vals.size}\nmean={mu:.4g}\nmedian={med:.4g}\nQ1={q1:.4g}  Q3={q3:.4g}\nmin={vmin:.4g}  max={vmax:.4g}"
            ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        for k in range(m, len(axes)):
            axes[k].set_visible(False)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    def analyze_and_plot(
        self,
        functions: Sequence[Union[nn.Module, Callable[[Tensor], Tensor]]],
        names: Optional[Sequence[str]] = None,
        bins: int = 40,
    ):
        abs_mat, rel_dict, metric_names = self.analyze(functions, names)
        self.plot_histograms(abs_mat, metric_names, bins=bins, title="Absolute metrics (multivariate)")
        for bname, mat in rel_dict.items():
            self.plot_histograms(mat, metric_names, bins=bins, title=f"Ratios vs {bname} (multivariate)")
        return abs_mat, rel_dict, metric_names


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from priors.causal_prior.mechanisms.SampleMLPMechanism import SampleMLPMechanism  # Assuming this is defined elsewhere

    # ---------------- Config ----------------
    D = 1                     # input_dim of each mechanism (parents features)
    NUM_MECHS = 100              # how many random mechanisms to compare
    MAX_HIDDEN = 0
    MIN_HIDDEN = 0
    HIDDEN_DIM = 1
    NONLINS = "sophisticated_sampling_1"
    ACT_MODE = "post"           # 'pre' or 'post'
    DEVICE = "cpu"

    # Complexity sampling config (MC mode, good for D>3)
    X_BOUNDS = [(-3.0, 3.0)] * D
    N_SAMPLES = 20_000          # MC samples for the analyzer
    INPUT_DIST = "normal"       # 'uniform' or 'normal'
    BINS = 40

    # ------------- Wrapper: make SampledMechanism look like f: R^D -> R -------------


    # ------------- Build a bunch of mechanisms with fixed hyperparams -------------
    mechs = []
    names = []
    for i in tqdm(range(NUM_MECHS), desc="Building mechanisms"):
        gen = torch.Generator(device=DEVICE).manual_seed(10_000 + i)
        mech = SampleMLPMechanism(
            input_dim=D,
            node_shape=(),                  # scalar node
            nonlins=NONLINS,
            max_hidden_layers=MAX_HIDDEN,
            min_hidden_layers=MIN_HIDDEN,
            hidden_dim=HIDDEN_DIM,
            activation_mode=ACT_MODE,
            generator=gen,
            name=f"mech_{i}"
        ).to(DEVICE)
        mechs.append(mech)
        names.append(f"SM_{i}")

    # ------------- Analyze & plot -------------
    # Import the MV analyzer classes you built earlier:
    # from complexity_mv import FunctionComplexityMV, ComplexityBatchAnalyzerMV

    fc = FunctionComplexityMV(
        x_bounds=X_BOUNDS,
        grid_shape=None,              # MC mode (use n_samples)
        n_samples=N_SAMPLES,
        input_dist=INPUT_DIST,
        device=DEVICE,
        dtype=torch.float32,
    )
    analyzer = ComplexityBatchAnalyzerMV(fc)

    # This will:
    #  - compute absolute metrics for each mechanism
    #  - compute ratios vs baselines ('relu', 'identity') on the same inputs
    #  - plot histograms with mean/median/quartiles/min/max annotated
    abs_mat, rel_dict, metric_names = analyzer.analyze_and_plot(
        mechs,
        names=names,
        bins=BINS,
    )

    # Optional: peek at the first few rows numerically
    import numpy as np
    print("\nMetrics:", metric_names)
    print("Absolute (first 3 mechanisms):")
    print(np.round(abs_mat[:3].numpy(), 4))

    for bname, mat in rel_dict.items():
        print(f"\nRatios vs {bname} (first 3 mechanisms):")
        print(np.round(mat[:3].numpy(), 4))
