from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple, List, Union, Sequence
import math, zlib
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class ComplexityReport:
    # compression
    bytes_zlib: int
    bps: float                  # bits per sample (compressed)
    bps_delta: float            # bits per sample after delta-encoding
    # spectrum
    spectral_entropy: float
    highfreq_energy_frac: float
    # geometry
    curvature_energy: float
    tv_slope: float
    lipschitz_max: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RelativeComplexity:
    baseline: str
    absolute: ComplexityReport
    ratios: Dict[str, float]    # key -> (sample / baseline), safe-div


class FunctionComplexity:
    """
    Complexity proxies for scalar activation f(x):
      - Compression-based (zlib) on quantized samples: bps, bps_delta
      - Spectral entropy & high-frequency energy fraction
      - Curvature energy, total variation of slope, max slope (Lipschitz proxy)

    Use:
      fc = FunctionComplexity()
      rep_abs = fc.evaluate(f)
      rep_rel = fc.evaluate_relative(f, baseline='relu')  # or 'id', 'square', or an nn.Module
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-5.0, 5.0),
        n_points: int = 2048,
        quant_bits: int = 12,
        highfreq_cutoff_frac: float = 0.25,   # top 25% frequencies as "high"
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.x_range = x_range
        self.n = int(n_points)
        self.quant_bits = int(quant_bits)
        self.hicut = float(highfreq_cutoff_frac)
        self.device = torch.device(device)
        self.dtype = dtype
        self.gen = generator

    # ---------------- Public API ----------------

    @torch.no_grad()
    def evaluate(self, f: nn.Module | Callable[[Tensor], Tensor]) -> ComplexityReport:
        x, y_norm = self._sample_and_normalize(f)
        bps, bps_delta, raw_bytes = self._compression_bits(y_norm)
        sentropy, hf_frac = self._spectral_metrics(y_norm)
        curv_energy, tv_slope, Lmax = self._geometry_metrics(x, y_norm)
        return ComplexityReport(
            bytes_zlib=raw_bytes,
            bps=bps,
            bps_delta=bps_delta,
            spectral_entropy=sentropy,
            highfreq_energy_frac=hf_frac,
            curvature_energy=curv_energy,
            tv_slope=tv_slope,
            lipschitz_max=Lmax,
        )

    @torch.no_grad()
    def evaluate_relative(
        self,
        f: nn.Module | Callable[[Tensor], Tensor],
        baseline: Union[str, nn.Module, Callable[[Tensor], Tensor]] = "relu",
    ) -> RelativeComplexity:
        # absolute for sample
        rep_abs = self.evaluate(f)

        # baseline module
        base_name, base_mod = self._make_baseline(baseline)
        rep_base = self.evaluate(base_mod)

        ratios = {
            k: self._safe_div(getattr(rep_abs, k), getattr(rep_base, k))
            for k in rep_abs.to_dict().keys()
            if isinstance(getattr(rep_abs, k), (int, float)) and isinstance(getattr(rep_base, k), (int, float))
        }

        return RelativeComplexity(baseline=base_name, absolute=rep_abs, ratios=ratios)

    # ---------------- Internals ----------------

    def _make_baseline(
        self, baseline: Union[str, nn.Module, Callable[[Tensor], Tensor]]
    ) -> Tuple[str, nn.Module]:
        if isinstance(baseline, str):
            name = baseline.lower()
            if name == "relu":
                return "relu", nn.ReLU()
            if name in ("id", "identity"):
                return "identity", _ToModule(lambda x: x)
            if name in ("square", "x2", "x^2"):
                return "square", _ToModule(lambda x: x * x)
            raise ValueError(f"Unknown baseline '{baseline}'. Use 'relu', 'identity', or 'square'.")
        # callable/module path
        if isinstance(baseline, nn.Module):
            return baseline.__class__.__name__, baseline
        # wrap callable
        return getattr(baseline, "__name__", "callable_baseline"), _ToModule(baseline)

    @torch.no_grad()
    def _sample_and_normalize(self, f: nn.Module | Callable[[Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
        x = torch.linspace(self.x_range[0], self.x_range[1], self.n, device=self.device, dtype=self.dtype)
        y = f(x.view(1, -1)).squeeze(0).to(self.device, self.dtype)  # shape (n,)
        y = y - y.mean()
        y_std = y.std(unbiased=False)
        if float(y_std) > 0:
            y = y / (y_std + 1e-12)
        return x, y

    def _quantize(self, y: Tensor) -> np.ndarray:
        y_np = y.detach().cpu().numpy().astype(np.float64)
        med = np.median(y_np)
        iqr = np.subtract(*np.percentile(y_np, [75, 25]))
        scale = (iqr if iqr > 1e-8 else y_np.std() + 1e-8)
        z = 0.5 + 0.25 * (y_np - med) / scale
        z = np.clip(z, 0.0, 1.0)
        levels = (1 << self.quant_bits) - 1
        return (z * levels + 0.5).astype(np.uint16)

    def _compression_bits(self, y: Tensor) -> Tuple[float, float, int]:
        q = self._quantize(y)
        raw = q.tobytes()
        comp = zlib.compress(raw, level=9)
        bps = (len(comp) * 8) / float(self.n)

        dq = np.diff(q.astype(np.int32), prepend=int(q[0]))
        comp_d = zlib.compress(dq.tobytes(), level=9)
        bps_d = (len(comp_d) * 8) / float(self.n)
        return bps, bps_d, len(comp)

    def _spectral_metrics(self, y: Tensor) -> Tuple[float, float]:
        n = y.numel()
        w = torch.hann_window(n, device=y.device, dtype=y.dtype)
        Y = tfft.rfft((y - y.mean()) * w)
        power = (Y.real**2 + Y.imag**2)
        psd = power / torch.sum(power + 1e-20)

        sentropy = float(-(psd * (psd + 1e-20).log()).sum().item())

        k_cut = int((1.0 - self.hicut) * psd.numel())
        k_cut = max(1, min(k_cut, psd.numel() - 1))
        hf_frac = float(psd[k_cut:].sum().item())
        return sentropy, hf_frac

    def _geometry_metrics(self, x: Tensor, y: Tensor) -> Tuple[float, float, float]:
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        slope = dy / (dx + 1e-12)
        Lmax = float(slope.abs().max().item())
        tv_slope = float((slope[1:] - slope[:-1]).abs().sum().item())
        d2 = slope[1:] - slope[:-1]
        dx_mid = (dx[1:] + dx[:-1]) * 0.5
        curv_energy = float((d2**2 * dx_mid).sum().item())
        return curv_energy, tv_slope, Lmax

    def _safe_div(self, num: float, den: float, eps: float = 1e-12) -> float:
        return float("nan") if abs(den) < eps else float(num / den)


class _ToModule(nn.Module):
    def __init__(self, f: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.f = f
    def forward(self, x: Tensor) -> Tensor:
        return self.f(x)


# assumes FunctionComplexity, ComplexityReport, RelativeComplexity are already defined (from the previous message)
# from your_module import FunctionComplexity

@dataclass
class FnSpec:
    fn: Union[nn.Module, callable]
    name: str

class ComplexityBatchAnalyzer:
    """
    Batch-evaluate complexity metrics for many functions and visualize distributions.
    """

    def __init__(
        self,
        fc: FunctionComplexity,
        *,
        baseline_relu: Union[str, nn.Module] = "relu",
        baseline_id: Union[str, nn.Module] = "identity",
    ) -> None:
        self.fc = fc
        self.baseline_relu = baseline_relu
        self.baseline_id = baseline_id

    @torch.no_grad()
    def analyze(
        self,
        functions: Sequence[Union[nn.Module, callable]],
        names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute absolute metrics and ratios vs ReLU and Identity for each function.
        Returns a long DataFrame (one row per function).
        """
        if names is None:
            names = [getattr(f, "__name__", f.__class__.__name__) for f in functions]
        if len(names) != len(functions):
            raise ValueError("names and functions must have same length.")

        rows: List[Dict[str, float]] = []

        # evaluate baselines once for reproducibility/fairness (same fc settings)
        base_relu_name, base_relu_mod = self.fc._make_baseline(self.baseline_relu)
        base_id_name, base_id_mod     = self.fc._make_baseline(self.baseline_id)
        base_relu = self.fc.evaluate(base_relu_mod)
        base_id   = self.fc.evaluate(base_id_mod)

        def safe_div(num: float, den: float, eps: float = 1e-12) -> float:
            return float("nan") if abs(den) < eps else float(num / den)

        for fn, nm in tqdm(zip(functions, names), desc="Analyzing functions", total=len(functions)):
            rep = self.fc.evaluate(fn)  # absolute
            # ratios vs relu
            rat_relu = {
                k: safe_div(getattr(rep, k), getattr(base_relu, k))
                for k in rep.to_dict().keys()
            }
            # ratios vs identity
            rat_id = {
                k: safe_div(getattr(rep, k), getattr(base_id, k))
                for k in rep.to_dict().keys()
            }
            row = {"name": nm}
            # flatten: abs.*, rel_relu.*, rel_id.*
            for k, v in rep.to_dict().items():
                row[f"{k}__abs"] = v
                row[f"{k}__rel_relu"] = rat_relu[k]
                row[f"{k}__rel_id"] = rat_id[k]
            rows.append(row)

        df = pd.DataFrame.from_records(rows)
        return df

    def plot_histograms(
        self,
        df: pd.DataFrame,
        metrics: Optional[Sequence[str]] = None,
        view: str = "abs",
        cols: int = 3,
        bins: int = 40,
        figsize: Tuple[int, int] = (14, 9),
        suptitle: Optional[str] = None,
        show_legend: bool = True,
    ) -> None:
        """
        Plot histograms for chosen metrics.
        `view` in {"abs", "rel_relu", "rel_id"} selects which columns to draw.

        metrics default: all base metric names present in the DataFrame.
        """
        if view not in {"abs", "rel_relu", "rel_id"}:
            raise ValueError("view must be one of {'abs','rel_relu','rel_id'}")

        # infer base metric names from df columns
        all_base = sorted(
            {c.split("__")[0] for c in df.columns if c.endswith("__abs")}
        )
        if metrics is None:
            metrics = all_base

        m = len(metrics)
        rows = int(math.ceil(m / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        for ax, metric in zip(axes, metrics):
            col = f"{metric}__{view}"
            if col not in df.columns:
                ax.set_visible(False)
                continue

            vals = df[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                ax.set_visible(False)
                continue

            ax.hist(vals, bins=bins)

            # compute stats
            mean = np.mean(vals)
            median = np.median(vals)
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            vmin = np.min(vals)
            vmax = np.max(vals)

            # vertical markers
            ax.axvline(mean,   linestyle="--", linewidth=1, label=f"mean={mean:.3g}")
            ax.axvline(median, linestyle="-.", linewidth=1, label=f"median={median:.3g}")
            ax.axvline(q1,     linestyle=":",  linewidth=1, label=f"Q1={q1:.3g}")
            ax.axvline(q3,     linestyle=":",  linewidth=1, label=f"Q3={q3:.3g}")
            ax.axvline(vmin,   linestyle="-",  linewidth=0.8, label=f"min={vmin:.3g}")
            ax.axvline(vmax,   linestyle="-",  linewidth=0.8, label=f"max={vmax:.3g}")

            ax.set_title(f"{metric} ({view})")
            ax.grid(True, alpha=0.3)

            # small stats box
            text = (
                f"n={vals.size}\n"
                f"mean={mean:.4g}\n"
                f"median={median:.4g}\n"
                f"Q1={q1:.4g}  Q3={q3:.4g}\n"
                f"min={vmin:.4g}  max={vmax:.4g}"
            )
            ax.text(
                0.98, 0.98, text,
                transform=ax.transAxes,
                ha="right", va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

            if show_legend:
                # keep this small to avoid clutter
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=8, loc="lower left")

        # hide extra axes
        for j in range(len(metrics), len(axes)):
            axes[j].set_visible(False)

        if suptitle:
            fig.suptitle(suptitle)
        fig.tight_layout()
        plt.show()

    def analyze_and_plot(
        self,
        functions: Sequence[Union[nn.Module, callable]],
        names: Optional[Sequence[str]] = None,
        *,
        metrics_abs: Optional[Sequence[str]] = None,
        metrics_rel_relu: Optional[Sequence[str]] = None,
        metrics_rel_id: Optional[Sequence[str]] = None,
        bins: int = 40,
    ) -> pd.DataFrame:
        """
        Convenience: analyze -> plot absolute, rel_relu, rel_id histograms.
        Returns the DataFrame.
        """
        df = self.analyze(functions, names)

        # Default metric lists = all metrics found in df
        base_metrics = sorted({c.split("__")[0] for c in df.columns if c.endswith("__abs")})

        self.plot_histograms(
            df, metrics=metrics_abs or base_metrics, view="abs",
            suptitle="Complexity metrics — absolute", bins=bins
        )
        self.plot_histograms(
            df, metrics=metrics_rel_relu or base_metrics, view="rel_relu",
            suptitle="Complexity metrics — ratios vs ReLU", bins=bins
        )
        self.plot_histograms(
            df, metrics=metrics_rel_id or base_metrics, view="rel_id",
            suptitle="Complexity metrics — ratios vs Identity", bins=bins
        )
        return df


if __name__ == "__main__":
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from priors.causal_prior.mechanisms.SampleMLPMechanism import RandomActivation

    # Your random activation

    # Build a few functions to compare
    g = torch.Generator().manual_seed(42)
    funcs = [
        # several random activations:
        *(RandomActivation("tabicl", generator=g) for _ in range(10_000)),
    ]

    names = [f"RandAct{i}" for i in range(10_000)]

    fc = FunctionComplexity(x_range=(-2,2), n_points=4096)
    an = ComplexityBatchAnalyzer(fc)

    df = an.analyze_and_plot(funcs, names, bins=50)

    # Also investigate summarized df
    print(df.describe())
