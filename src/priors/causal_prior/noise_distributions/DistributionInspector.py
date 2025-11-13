from __future__ import annotations

import sys
from pathlib import Path
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

# Add src to path for imports when run directly
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from priors.causal_prior.noise_distributions.MixedDist_RandomSTD import MixedDistRandomStd
from priors.causal_prior.noise_distributions.Sample_STD import GammaMeanStd

@dataclass
class IIDReport:
    n: int
    acf_lags: List[int]
    acf_values: List[float]
    acf_sig_lags: List[int]
    runs_z: float
    runs_p_two_sided: float
    iid_pass: bool


@dataclass
class TimingReport:
    n_total: int
    repeats: int
    per_sample_mean_s: float
    per_sample_ci95: Tuple[float, float]
    total_time_est_s: float
    total_time_ci95_s: Tuple[float, float]
    total_time_measured_s: float


class DistributionInspector:
    """
    Inspect samples from a scalar distribution:
      - sample many values
      - measure draw times with repeats, report mean and 95% bootstrap CI
      - plot histogram
      - run IID checks (ACF bands + runs test)
    """

    def __init__(self, dist, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        self.dist = dist
        self.device = torch.device(device)
        self.dtype = dtype
        # try to move distribution if it exposes .to
        if hasattr(self.dist, "to"):
            self.dist = self.dist.to(self.device, self.dtype)

    # ------------------------ public API ------------------------

    def inspect(
        self,
        n: int,
        *,
        repeats: int = 20,
        bins: int = 60,
        iid_max_lag: int = 20,
        iid_bonferroni: bool = True,
        bootstrap_iters: int = 2000,
        rng_seed: Optional[int] = 0,
        show: bool = True,
        savefig: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Main entry point.

        Returns a dict with:
          - 'timing': TimingReport
          - 'iid': IIDReport
          - 'samples': np.ndarray (the drawn values; beware of memory if n is huge)
        """
        # -------- timing (with repeats) --------
        timing = self._benchmark(n, repeats=repeats, B=bootstrap_iters, seed=rng_seed)

        # -------- draw a single batch for stats/plots --------
        x = self._draw(n)  # Tensor on device
        x_np = x.detach().cpu().numpy()

        # -------- iid checks --------
        iid = self._iid_checks(
            x_np,
            max_lag=iid_max_lag,
            bonferroni=iid_bonferroni,
        )

        # -------- plot --------
        if show or savefig:
            self._plot_hist(
                x_np,
                bins=bins,
                title=title or "Distribution samples",
                subtitle=self._fmt_timing_line(timing),
                savefig=savefig,
                show=show,
            )

        print("""Stats: mean={:.3f}, std={:.3f}, min={:.3f}, max={:.3f}, median={:.3f}""".format(
            x_np.mean(), x_np.std(ddof=1), x_np.min(), x_np.max(), np.median(x_np)
        ))

        return {"timing": timing, "iid": iid, "samples": x_np}

    # ------------------------ timing ------------------------

    def _benchmark(
        self,
        n_total: int,
        *,
        repeats: int,
        B: int,
        seed: Optional[int],
    ) -> TimingReport:
        """
        Split n_total into (approximately) equal chunks and time each chunk.
        Bootstrap mean per-sample time and scale to n_total for CI on total time.
        """
        if repeats <= 0:
            repeats = 1

        # chunk sizes that sum to n_total
        base = n_total // repeats
        sizes = [base] * repeats
        for i in range(n_total - base * repeats):
            sizes[i] += 1

        times: List[float] = []
        total_measured = 0.0

        # warm-up (tiny) to stabilize first-call overhead
        _ = self._draw(8)

        for m in sizes:
            t0 = time.perf_counter()
            _ = self._draw(m)
            dt = time.perf_counter() - t0
            total_measured += dt
            # record per-sample time for this chunk
            times.append(dt / max(m, 1))

        times_np = np.asarray(times, dtype=np.float64)
        mean_per = float(times_np.mean())

        # bootstrap mean per-sample time
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed)
        if B <= 0:
            lo_per = hi_per = mean_per
        else:
            idx = rng.integers(0, len(times_np), size=(B, len(times_np)))
            boot_means = times_np[idx].mean(axis=1)
            lo_per, hi_per = np.percentile(boot_means, [2.5, 97.5])

        # scale to n_total
        total_est = mean_per * n_total
        total_ci = (float(lo_per) * n_total, float(hi_per) * n_total)

        return TimingReport(
            n_total=n_total,
            repeats=repeats,
            per_sample_mean_s=mean_per,
            per_sample_ci95=(float(lo_per), float(hi_per)),
            total_time_est_s=float(total_est),
            total_time_ci95_s=(float(total_ci[0]), float(total_ci[1])),
            total_time_measured_s=float(total_measured),
        )

    def _draw(self, m: int) -> Tensor:
        # Prefer sample_n if available; else sample_shape
        if hasattr(self.dist, "sample_n"):
            return self.dist.sample_n(int(m))
        elif hasattr(self.dist, "sample_shape"):
            return self.dist.sample_shape((int(m),))
        else:
            # very slow fallback
            vals = [self.dist.sample_one() for _ in range(m)]
            return torch.stack(vals).to(self.device, self.dtype)

    # ------------------------ iid checks ------------------------

    def _iid_checks(self, x: np.ndarray, max_lag: int, bonferroni: bool) -> IIDReport:
        n = int(x.size)
        if n < 3:
            return IIDReport(n=n, acf_lags=[], acf_values=[], acf_sig_lags=[], runs_z=float("nan"), runs_p_two_sided=float("nan"), iid_pass=False)

        # standardize
        x_centered = x - x.mean()
        var = float(x_centered.var(ddof=0))
        if var == 0.0:
            # constant sequence fails IID
            return IIDReport(n=n, acf_lags=list(range(1, max_lag + 1)), acf_values=[0.0]*max_lag,
                             acf_sig_lags=list(range(1, max_lag + 1)), runs_z=float("nan"), runs_p_two_sided=float("nan"), iid_pass=False)

        # Autocorrelation up to max_lag
        acf_vals = []
        for k in range(1, max_lag + 1):
            num = np.dot(x_centered[:-k], x_centered[k:])
            den = (n - 0) * var
            rho = float(num / den)
            acf_vals.append(rho)

        # significance bands ±1.96/sqrt(n)
        band = 1.96 / math.sqrt(n)
        sig_lags = [k for k, rho in enumerate(acf_vals, start=1) if abs(rho) > (band / (max_lag if bonferroni else 1.0))]
        # Note: crude Bonferroni by dividing band; alternatively compare p-values to 0.05/max_lag.

        # Runs test about the median
        runs_z, runs_p = self._runs_test_about_median(x)

        iid_pass = (len(sig_lags) == 0) and (abs(runs_z) <= 1.96)

        return IIDReport(
            n=n,
            acf_lags=list(range(1, max_lag + 1)),
            acf_values=acf_vals,
            acf_sig_lags=sig_lags,
            runs_z=runs_z,
            runs_p_two_sided=runs_p,
            iid_pass=iid_pass,
        )

    def _runs_test_about_median(self, x: np.ndarray) -> Tuple[float, float]:
        """
        Wald–Wolfowitz runs test on signs about the median.
        Returns (z, two-sided p-value) using normal approximation.
        """
        n = x.size
        med = np.median(x)
        # Convert to + / - ignoring values exactly equal to the median by jittering them very slightly
        signs = np.sign(x - med)
        zero_mask = (signs == 0)
        if zero_mask.any():
            # add tiny noise to break ties near the median
            eps = np.finfo(float).eps
            signs[zero_mask] = np.sign((x[zero_mask] - med) + np.random.uniform(-eps, eps, size=zero_mask.sum()))
        pos = np.sum(signs > 0)
        neg = np.sum(signs < 0)
        if pos == 0 or neg == 0:
            # all on one side -> minimum runs, very non-IID
            return float("inf"), 0.0

        # count runs
        runs = 1 + np.sum(signs[1:] != signs[:-1])

        mu = 1 + (2 * pos * neg) / (pos + neg)
        sigma2 = (2 * pos * neg * (2 * pos * neg - pos - neg)) / (((pos + neg) ** 2) * (pos + neg - 1))
        sigma = math.sqrt(max(sigma2, 1e-12))

        z = (runs - mu) / sigma
        # two-sided p via normal tail; p = 2 * (1 - Phi(|z|)) = erfc(|z|/sqrt(2))
        p_two = math.erfc(abs(z) / math.sqrt(2.0))
        return float(z), float(p_two)

    # ------------------------ plotting ------------------------

    def _fmt_timing_line(self, t: TimingReport) -> str:
        return (
            f"Time (n={t.n_total}, repeats={t.repeats}) — "
            f"per-sample: {t.per_sample_mean_s*1e6:.2f} µs "
            f"[{t.per_sample_ci95[0]*1e6:.2f}, {t.per_sample_ci95[1]*1e6:.2f}] — "
            f"total est: {t.total_time_est_s:.3f}s "
            f"[{t.total_time_ci95_s[0]:.3f}, {t.total_time_ci95_s[1]:.3f}] — "
            f"measured total: {t.total_time_measured_s:.3f}s"
        )

    def _plot_hist(
        self,
        x: np.ndarray,
        *,
        bins: int,
        title: str,
        subtitle: Optional[str],
        savefig: Optional[str],
        show: bool,
    ) -> None:
        plt.figure(figsize=(8, 5))
        plt.hist(x, bins=bins)
        plt.title(title)
        if subtitle:
            plt.suptitle(subtitle, y=0.98, fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    def _compute_stats(self, x: np.ndarray) -> Dict[str, float]:
        """
        Compute basic statistics of the samples.
        Returns a dict with mean, std, min, max, median.
        """
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)),  # sample std
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "median": float(np.median(x)),
        }

# Suppose you already created a MixedDistRandomStd or MixedDist instance:
mix = MixedDistRandomStd(std_dist=GammaMeanStd(mean=0.5, std=0.2))
# or
#   mix = MixedDist(std=0.5)

inspector = DistributionInspector(mix)
report = inspector.inspect(
    n=200_000,
    repeats=25,            # split the timing into 25 chunks for a CI
    bins=80,
    iid_max_lag=20,
    iid_bonferroni=True,
    bootstrap_iters=2000,
    rng_seed=0,
    show=True,
    savefig=None,
    title="MixedDist (sampled-STD) histogram"
)

print(report["timing"])
print(report["iid"])
