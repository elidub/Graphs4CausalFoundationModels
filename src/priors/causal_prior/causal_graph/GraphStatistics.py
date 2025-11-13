from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List

import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from GraphSampler import GraphSampler
from CausalDAG import CausalDAG  # keep if you use it elsewhere; not needed here directly


class GraphStatistics:
    """
    Sample many random DAGs (via GraphSampler) and compute descriptive statistics.
    Also tracks sampling speed and provides a bootstrap CI for the time to sample 1000 DAGs.
    """

    def __init__(self, sampler: GraphSampler) -> None:
        self.sampler = sampler

    # ---------- public API ----------
    def sample_and_collect(
        self,
        num_samples: int,
        num_nodes: int,
        p: float,
        include_graph: bool = False,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with one row per sampled graph. Includes 'sample_time_s'.
        """
        records: List[Dict[str, Any]] = []
        max_edges = num_nodes * (num_nodes - 1) // 2

        for _ in range(num_samples):
            t0 = time.perf_counter()
            G = self.sampler.sample_dag(num_nodes, p, return_perm=False)
            dt = time.perf_counter() - t0

            rec = self._compute_stats(G, num_nodes, max_edges)
            rec["sample_time_s"] = float(dt)
            if include_graph:
                rec["graph"] = G
            records.append(rec)

        return pd.DataFrame.from_records(records)

    def plot_histograms(
        self,
        stats_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        bins: int = 30,
        figsize: Tuple[int, int] = (12, 10),
        suptitle: Optional[str] = None,
    ):
        """
        Plot histograms for selected numeric columns. Returns (fig, axes).
        """
        if columns is None:
            columns = [c for c in stats_df.columns if np.issubdtype(stats_df[c].dtype, np.number)]

        n_cols = 3
        n_rows = int(np.ceil(len(columns) / n_cols)) if columns else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, columns):
            vals = stats_df[col].values
            vals = vals[~np.isnan(vals)]  # drop NaNs if any
            ax.hist(vals, bins=bins)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)

        # hide any extra axes
        for k in range(len(columns), len(axes)):
            axes[k].set_visible(False)

        if suptitle:
            fig.suptitle(suptitle)
        fig.tight_layout()
        plt.show(block=False)  # keep figure open for annotation by caller
        return fig, axes

    def plot_boxplots(
        self,
        stats_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
    ) -> None:
        if columns is None:
            columns = ["num_edges", "density", "sparsity", "longest_path", "max_in_degree", "max_out_degree"]

        data = [stats_df[c].values for c in columns if c in stats_df.columns]
        labels = [c for c in columns if c in stats_df.columns]

        plt.figure(figsize=figsize)
        plt.boxplot(data, labels=labels, showfliers=False)
        if title:
            plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_corr_heatmap(
        self,
        stats_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (7, 6),
        title: str = "Correlation (Pearson)",
        vmin: float = -1.0,
        vmax: float = 1.0,
    ) -> None:
        if columns is None:
            columns = [c for c in stats_df.columns if np.issubdtype(stats_df[c].dtype, np.number)]
        C = stats_df[columns].corr(method="pearson").values

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(C, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(columns))); ax.set_yticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right")
        ax.set_yticklabels(columns)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_all(
        self,
        num_samples: int,
        num_nodes: int,
        p: float,
        bins: int = 30,
        show_corr: bool = False,
        savepath: Optional[str] = None,
        suptitle: Optional[str] = None,
        bootstrap_iters: int = 2000,
        rng_seed: Optional[int] = 0,
    ) -> pd.DataFrame:
        """
        One-shot: sample graphs, compute stats, and plot everything.
        Also annotates the figure with a bootstrap 95% CI estimate for time to sample 1000 DAGs.
        """
        df = self.sample_and_collect(num_samples, num_nodes, p)

        fig, _ = self.plot_histograms(
            df,
            columns=[c for c in df.columns if np.issubdtype(df[c].dtype, np.number)],
            bins=bins,
            suptitle=suptitle or f"ER-DAG stats (n={num_nodes}, p={p}, k={num_samples})",
        )

        # Bootstrap CI for mean per-graph time, then scale to 1000 DAGs
        est_1000, ci_lo_1000, ci_hi_1000 = self._bootstrap_time_for_1000(df["sample_time_s"].values,
                                                                          B=bootstrap_iters,
                                                                          seed=rng_seed)

        # Annotate figure
        annotation = (
            f"Estimated time for 1000 DAGs: "
            f"{est_1000:.3f} s (95% CI [{ci_lo_1000:.3f}, {ci_hi_1000:.3f}] s)"
        )
        if fig is not None:
            # place annotation centered at bottom margin
            fig.text(0.5, 0.02, annotation, ha="center", va="bottom")
            fig.canvas.draw_idle()
            plt.show()

        # optional save to CSV
        if savepath:
            df.to_csv(savepath, index=False)

        # Also print to console for logs
        print(annotation)

        return df

    # ---------- internals ----------
    def _compute_stats(self, G: nx.DiGraph, n: int, max_edges: int) -> Dict[str, Any]:
        m = G.number_of_edges()
        density = m / max_edges if max_edges > 0 else 0.0
        sparsity = 1.0 - density

        in_degs = np.fromiter((d for _, d in G.in_degree()), dtype=int, count=n) if n else np.array([], dtype=int)
        out_degs = np.fromiter((d for _, d in G.out_degree()), dtype=int, count=n) if n else np.array([], dtype=int)

        max_in = int(in_degs.max()) if in_degs.size else 0
        max_out = int(out_degs.max()) if out_degs.size else 0
        avg_in = float(in_degs.mean()) if in_degs.size else 0.0
        avg_out = float(out_degs.mean()) if out_degs.size else 0.0

        num_sources = int((in_degs == 0).sum()) if in_degs.size else (1 if n == 1 else 0)
        num_sinks = int((out_degs == 0).sum()) if out_degs.size else (1 if n == 1 else 0)

        longest_path = nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) and n > 0 else np.nan

        wcc = list(nx.weakly_connected_components(G))
        num_wcc = len(wcc)
        largest_wcc = max((len(c) for c in wcc), default=0)

        # Optional: transitive reduction / redundancy
        try:
            TR = nx.transitive_reduction(G)
            tr_edges = TR.number_of_edges()
            redundancy = 1.0 - (tr_edges / m) if m > 0 else 0.0
        except nx.NetworkXError:
            tr_edges = np.nan
            redundancy = np.nan

        return {
            "num_nodes": n,
            "num_edges": m,
            "density": density,
            "sparsity": sparsity,
            "avg_in_degree": avg_in,
            "avg_out_degree": avg_out,
            "max_in_degree": max_in,
            "max_out_degree": max_out,
            "num_sources": num_sources,
            "num_sinks": num_sinks,
            "longest_path": int(longest_path) if not np.isnan(longest_path) else np.nan,
            "num_weak_cc": num_wcc,
            "largest_weak_cc": largest_wcc,
            "tr_edges": tr_edges,
            "redundancy": redundancy,
        }

    def _bootstrap_time_for_1000(self, times: np.ndarray, B: int = 2000, seed: Optional[int] = 0):
        """
        Bootstrap the mean per-graph sampling time and scale to 1000 DAGs.
        Returns (estimate_1000, ci_low_1000, ci_high_1000), all in seconds.
        """
        times = np.asarray(times, dtype=float)
        n = times.size
        if n == 0:
            return (np.nan, np.nan, np.nan)

        mean_t = float(times.mean())

        if n == 1 or B <= 0:
            est_1000 = 1000.0 * mean_t
            return (est_1000, est_1000, est_1000)

        rng = np.random.default_rng(seed)
        # shape (B, n) resamples indices
        idx = rng.integers(0, n, size=(B, n))
        boot_means = times[idx].mean(axis=1)
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        return (1000.0 * mean_t, 1000.0 * float(lo), 1000.0 * float(hi))


if __name__ == "__main__":
    # Example usage
    sampler = GraphSampler(seed=42)
    stats = GraphStatistics(sampler)
    df = stats.plot_all(
        num_samples=1000,
        num_nodes=30,
        p=0.2,
        bins=25,
        suptitle="ER-DAG Statistics",
        show_corr=True,
        bootstrap_iters=2000,
        rng_seed=0,
    )
    print(df.head())
