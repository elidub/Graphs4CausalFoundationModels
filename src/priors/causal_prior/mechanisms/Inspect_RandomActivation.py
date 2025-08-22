from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal, Sequence, Dict
import math
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from priors.causal_prior.mechanisms.SampleMLPMechanism import RandomActivation

@dataclass
class InputSpec:
    kind: Literal["normal", "uniform"] = "normal"
    mean: float = 0.0
    std: float = 1.0
    low: float = -1.0
    high: float = 1.0


class ActivationInspector:
    def __init__(
        self,
        nonlins: str = "mixed",
        clamp: Tuple[float, float] = (-1000.0, 1000.0),
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ) -> None:
        self.nonlins = nonlins
        self.clamp = clamp
        self.device = torch.device(device)
        self.dtype = dtype
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(seed)

    # ---------- (a) plot sampled activations ----------
    @torch.no_grad()
    def plot_sampled_curves(
        self,
        num_funcs: int = 12,
        x_range: Tuple[float, float] = (-5.0, 5.0),
        n_points: int = 400,
        title: Optional[str] = None,
        cols: int = 4,
        figsize: Tuple[int, int] = (12, 8),
        share_axes: bool = True,
    ) -> List[torch.nn.Module]:
        xs = torch.linspace(x_range[0], x_range[1], n_points, device=self.device, dtype=self.dtype).view(1, -1)
        acts: List[torch.nn.Module] = [
            RandomActivation(self.nonlins, clamp=self.clamp, generator=self.gen, device=self.device)
            for _ in range(num_funcs)
        ]

        rows = int(math.ceil(num_funcs / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=share_axes, sharey=share_axes)
        axes = np.array(axes).reshape(-1)
        for i, (ax, act) in enumerate(zip(axes, acts)):
            ys = act(xs).squeeze(0).detach().cpu().numpy()
            ax.plot(xs.squeeze(0).cpu().numpy(), ys)
            ax.set_title(f"sample {i+1}")
            ax.grid(True, alpha=0.3)

        for j in range(len(acts), len(axes)):
            axes[j].set_visible(False)

        if title:
            fig.suptitle(title)
        fig.tight_layout()
        plt.show()
        return acts

    # ---------- (b) stats (ratios except clipped/large) ----------
    @torch.no_grad()
    def stats_over_inputs(
        self,
        n_funcs: int = 200,
        n_inputs: int = 50_000,
        input_spec: InputSpec = InputSpec("normal", 0.0, 1.0),
        per_func_clip_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for k in range(n_funcs):
            act = RandomActivation(self.nonlins, clamp=self.clamp, generator=self.gen, device=self.device)
            x = self._sample_inputs(n_inputs, input_spec)          # (n,)
            y = act(x.unsqueeze(0)).squeeze(0)                     # (n,)
            stats = self._summarize_xy(x, y, clip_thr=per_func_clip_threshold)
            stats["func_id"] = k
            rows.append(stats)
        df = pd.DataFrame.from_records(rows)
        order = [
            "func_id",
            "mean_ratio", "std_ratio", "var_ratio",
            "skew_ratio", "kurt_excess_ratio",
            "pearson_r", "spearman_like",
            "avg_abs_slope", "max_abs_slope",
            "frac_clipped_x", "frac_clipped_y",
            "frac_large_x", "frac_large_y",
        ]
        df = df[[c for c in order if c in df.columns]]
        return df

    @torch.no_grad()
    def plot_stat_histograms(
        self,
        df: pd.DataFrame,
        columns: Optional[Sequence[str]] = None,
        bins: int = 40,
        figsize: Tuple[int, int] = (12, 8),
        cols: int = 3,
        title: Optional[str] = None,
    ) -> None:
        if columns is None:
            columns = [
                "mean_ratio", "std_ratio", "var_ratio",
                "skew_ratio", "kurt_excess_ratio",
                "pearson_r", "avg_abs_slope", "max_abs_slope",
                "frac_clipped_y", "frac_large_y",  # often the most interesting
            ]
        m = len(columns)
        rows = int(math.ceil(m / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, columns):
            if col not in df.columns:
                ax.set_visible(False); continue
            vals = df[col].to_numpy()
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=bins)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    # ---------- internals ----------
    def _sample_inputs(self, n: int, spec: InputSpec) -> Tensor:
        if spec.kind == "normal":
            return spec.mean + spec.std * torch.randn((n,), generator=self.gen, device=self.device, dtype=self.dtype)
        if spec.kind == "uniform":
            return torch.empty((n,), device=self.device, dtype=self.dtype).uniform_(spec.low, spec.high, generator=self.gen)
        raise ValueError(f"Unknown input kind: {spec.kind}")

    def _safe_div(self, num: float, den: float, eps: float = 1e-12) -> float:
        return float("nan") if abs(den) < eps else float(num / den)

    def _skew_kurt(self, v: Tensor) -> Tuple[float, float]:
        m = v.mean()
        c = v - m
        var = (c**2).mean()
        if var <= 0:
            return float("nan"), float("nan")
        std = torch.sqrt(var)
        skew = (c**3).mean() / (std**3)
        kurt = (c**4).mean() / (var**2) - 3.0
        return float(skew.item()), float(kurt.item())

    def _summarize_xy(self, x: Tensor, y: Tensor, clip_thr: Optional[float]) -> Dict[str, float]:
        # base moments
        xm = float(x.mean().item())
        ym = float(y.mean().item())
        xs = float(x.std(unbiased=False).item())
        ys = float(y.std(unbiased=False).item())
        x_skew, x_kurt = self._skew_kurt(x)
        y_skew, y_kurt = self._skew_kurt(y)

        # ratios where sensible
        mean_ratio = self._safe_div(ym, xm)
        std_ratio  = self._safe_div(ys, xs)
        var_ratio  = self._safe_div(ys**2, xs**2)
        skew_ratio = self._safe_div(y_skew, x_skew) if math.isfinite(x_skew) else float("nan")
        kurt_ratio = self._safe_div(y_kurt, x_kurt) if math.isfinite(x_kurt) else float("nan")

        # clipping & large fractions — NOT ratios
        if self.clamp is not None:
            lo, hi = self.clamp
            frac_clipped_x = float(((x <= lo) | (x >= hi)).float().mean().item())
            frac_clipped_y = float(((y <= lo) | (y >= hi)).float().mean().item())
        else:
            frac_clipped_x = float("nan")
            frac_clipped_y = float("nan")

        if clip_thr is None:
            thr = 0.9 * (max(abs(self.clamp[0]), abs(self.clamp[1])) if self.clamp is not None else 1e6)
        else:
            thr = float(clip_thr)
        frac_large_x = float((x.abs() >= thr).float().mean().item())
        frac_large_y = float((y.abs() >= thr).float().mean().item())

        # correlations (not ratios)
        cov = float(((x - xm) * (y - ym)).mean().item())
        denom = xs * ys
        pearson_r = cov / denom if denom > 0 else float("nan")

        # rank correlation
        def _rank(t: Tensor) -> Tensor:
            s = torch.argsort(t)
            r = torch.empty_like(s, dtype=torch.float32)
            r[s] = torch.arange(1, t.numel() + 1, device=t.device, dtype=torch.float32)
            return r
        rx, ry = _rank(x), _rank(y)
        rxm = float(rx.mean().item()); rym = float(ry.mean().item())
        rxs = float(rx.std(unbiased=False).item()); rys = float(ry.std(unbiased=False).item())
        spearman_like = float((((rx - rxm) * (ry - rym)).mean() / (rxs * rys + 1e-12)).item())

        # slope proxies
        xsrt, idx = torch.sort(x)
        ysrt = y[idx]
        dx = xsrt[1:] - xsrt[:-1]
        dy = ysrt[1:] - ysrt[:-1]
        valid = dx.abs() > 0
        slope = (dy[valid] / dx[valid]).abs()
        avg_abs_slope = float(slope.mean().item()) if slope.numel() else float("nan")
        max_abs_slope = float(slope.max().item()) if slope.numel() else float("nan")

        return {
            "mean_ratio": mean_ratio,
            "std_ratio": std_ratio,
            "var_ratio": var_ratio,
            "skew_ratio": skew_ratio,
            "kurt_excess_ratio": kurt_ratio,
            "pearson_r": float(pearson_r),
            "spearman_like": float(spearman_like),
            "avg_abs_slope": avg_abs_slope,
            "max_abs_slope": max_abs_slope,
            "frac_clipped_x": frac_clipped_x,
            "frac_clipped_y": frac_clipped_y,
            "frac_large_x": frac_large_x,
            "frac_large_y": frac_large_y,
        }

    # ---------- one-shot ----------
    @torch.no_grad()
    def inspect(
        self,
        *,
        plot_funcs: int = 12,
        x_range: Tuple[float, float] = (-5, 5),
        n_grid: int = 400,
        n_funcs_stats: int = 200,
        n_inputs_per_func: int = 50_000,
        input_spec: InputSpec = InputSpec("normal", 0.0, 1.0),
        hist_columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        self.plot_sampled_curves(
            num_funcs=plot_funcs,
            x_range=x_range,
            n_points=n_grid,
            title=f"RandomActivation samples — nonlins='{self.nonlins}'",
        )
        df = self.stats_over_inputs(
            n_funcs=n_funcs_stats,
            n_inputs=n_inputs_per_func,
            input_spec=input_spec,
        )
        self.plot_stat_histograms(
            df,
            columns=hist_columns or [
                "mean_ratio","std_ratio","var_ratio",
                "skew_ratio","kurt_excess_ratio",
                "pearson_r","avg_abs_slope","max_abs_slope",
                "frac_clipped_y","frac_large_y",
            ],
            title="Activation statistics (ratios except clipped/large)",
        )
        return df

if __name__ == "__main__":
    # inspector for your 'sophisticated_sampling_1' family
    insp = ActivationInspector(nonlins="tabicl", seed=123)
    df = insp.inspect(
        plot_funcs=12,
        x_range=(-2, 2),
        n_grid=500,
        n_funcs_stats=150,
        n_inputs_per_func=20000,
        input_spec=InputSpec(kind="normal", mean=0.0, std=1.0),
    )
    print(df.describe().T)
