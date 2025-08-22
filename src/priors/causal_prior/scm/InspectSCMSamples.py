from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import time
import importlib.util

import torch
from torch import Tensor
import matplotlib.pyplot as plt

# Import your classes
from priors.causal_prior.scm.SCM import SCM
from priors.causal_prior.causal_graph.CausalDAG import CausalDAG
from priors.causal_prior.noise_distributions.Sample_STD import GammaMeanStd

# soft import seaborn/pandas for pairplot
try:
    import seaborn as sns
    import pandas as pd
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

# check if sklearn is available for mutual information
def _check_sklearn():
    try:
        spec = importlib.util.find_spec("sklearn.feature_selection")
        return spec is not None
    except ImportError:
        return False

_HAS_SKLEARN = _check_sklearn()


# ----------------------------- small utility helpers -----------------------------

def _reduce_per_sample(x: Tensor, reduce: str = "mean") -> Tensor:
    """
    Reduce each sample of a node to a single scalar.
    x: (B, *node_shape) or (B,)
    returns (B,)
    """
    if x.dim() == 1:
        return x
    if reduce == "mean":
        return x.view(x.shape[0], -1).mean(dim=1)
    elif reduce == "sum":
        return x.view(x.shape[0], -1).sum(dim=1)
    elif reduce == "first":
        return x.view(x.shape[0], -1)[:, 0]
    else:
        raise ValueError(f"Unknown reduce='{reduce}'.")

def _to_mpl_array(t: torch.Tensor):
    """Safe conversion for plotting that avoids torch->numpy bridge."""
    return t.detach().cpu().tolist()

def _bootstrap_mean_ci(
    x: Tensor,
    n_boot: int = 2000,
    alpha: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> Tuple[float, float, float]:
    """
    95% bootstrap CI for mean along batch dimension.
    x: (B,) or (B, *tail) -> reduced to (B,) via mean over tail dims.
    Returns: (mean, lo, hi) as floats.
    """
    x = _reduce_per_sample(x, reduce="mean")  # (B,)
    B = x.shape[0]
    if B == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = torch.randint(0, B, (n_boot, B), generator=generator, device=x.device)
    samples = x[idx]                           # (n_boot, B)
    boot_means = samples.mean(dim=1)           # (n_boot,)
    lo = torch.quantile(boot_means, alpha / 2).item()
    hi = torch.quantile(boot_means, 1 - alpha / 2).item()
    return (x.mean().item(), lo, hi)


# ------------------------- the inspector (plots + metrics) -------------------------

@dataclass
class InspectSCMSamples:
    """
    Inspect observational samples drawn from an SCM.

    - Repeatedly samples datasets from the provided SCM.
    - Computes per-node summary stats and 95% bootstrap CIs for the mean.
    - Plots:
        * DAG (always)
        * Big Seaborn Pairplot (pairwise scatter + marginals) over reduced node series
        * 2-D PCA scatter (computed via torch SVD)
    - Benchmarks runtime (safe vs. fast).

    Parameters
    ----------
    scm : SCM
        A fully configured SCM. You can toggle `scm.fast` between safe/fast.
    batch_size : int
        Number of samples per dataset.
    reduce : {"mean","sum","first"}
        How to reduce multi-dimensional node outputs to a scalar per sample
        (used for pairplot, correlations, bootstrap CIs, PCA).
    device : str or torch.device
        Where to collect and operate on tensors for metrics.
    """

    scm: SCM
    batch_size: int
    reduce: str = "mean"
    device: torch.device | str = "cpu"

    # ------------------------------- sampling --------------------------------

    @torch.no_grad()
    def sample_once(self) -> Dict[str, Tensor]:
        """
        One full observational draw:
        uses fixed exogenous/endogenous noise if already sampled,
        otherwise samples them first.
        """
        B = self.batch_size
        if self.scm._fixed_exogenous is None or self.scm._fixed_batch != B:
            self.scm.sample_exogenous(B)
        if self.scm._fixed_endogenous is None or self.scm._fixed_batch != B:
            self.scm.sample_endogenous_noise(B)
        return self.scm.propagate(B)

    @torch.no_grad()
    def sample_many(self, n_datasets: int = 1, clear_between: bool = True) -> List[Dict[str, Tensor]]:
        outs: List[Dict[str, Tensor]] = []
        for _ in range(n_datasets):
            if clear_between:
                self.scm.clear_fixed_noise()
            outs.append(self.sample_once())
        return outs

    # ------------------------------ statistics -------------------------------

    @torch.no_grad()
    def summarize(self, xs: Dict[str, Tensor]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for v, t in xs.items():
            t_cpu = t.detach().to("cpu")
            v1 = _reduce_per_sample(t_cpu, self.reduce)
            stats[v] = {
                "mean": v1.mean().item(),
                "std": v1.std(unbiased=True).item() if v1.numel() > 1 else float("nan"),
                "min": v1.min().item(),
                "max": v1.max().item(),
                "median": v1.median().item(),
            }
        return stats

    @torch.no_grad()
    def bootstrap_mean_cis(
        self,
        xs: Dict[str, Tensor],
        n_boot: int = 2000,
        alpha: float = 0.05,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Tuple[float, float, float]]:
        out: Dict[str, Tuple[float, float, float]] = {}
        for v, t in xs.items():
            out[v] = _bootstrap_mean_ci(
                t.detach().to("cpu"),
                n_boot=n_boot,
                alpha=alpha,
                generator=generator,
            )
        return out

    @torch.no_grad()
    def correlation_matrix(self, xs: Dict[str, Tensor]) -> Tuple[Tensor, List[str]]:
        order = list(self.scm._topo)
        series = []
        for v in order:
            s = _reduce_per_sample(xs[v].detach().to("cpu"), self.reduce)  # (B,)
            series.append(s)
        M = torch.stack(series, dim=0)  # (N, B)
        corr = torch.corrcoef(M)        # (N, N)
        return corr, order

    @torch.no_grad()
    def mutual_information_matrix(self, xs: Dict[str, Tensor]) -> Tuple[Tensor, List[str]]:
        """
        Compute pairwise mutual information matrix between features.
        Returns a symmetric matrix where entry (i,j) is the MI between features i and j.
        """
        if not _HAS_SKLEARN:
            raise RuntimeError(
                "Scikit-learn is required for mutual information. Install with:\n"
                "  pip install scikit-learn"
            )
        
        order = list(self.scm._topo)
        series = []
        for v in order:
            s = _reduce_per_sample(xs[v].detach().to("cpu"), self.reduce)  # (B,)
            series.append(s.numpy())
        
        # Create mutual information matrix
        n_features = len(series)
        mi_matrix = torch.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    # Self-information is entropy, but we'll use a placeholder
                    mi_matrix[i, j] = 1.0
                else:
                    # Compute mutual information
                    from sklearn.feature_selection import mutual_info_regression
                    mi = mutual_info_regression(series[i].reshape(-1, 1), series[j], random_state=42)[0]
                    mi_matrix[i, j] = mi
        
        return mi_matrix, order

    # -------------------------------- plots ----------------------------------

    def plot_dag(self) -> None:
        """Always draw the DAG."""
        if hasattr(self.scm.dag, "draw"):
            self.scm.dag.draw()
        else:
            import networkx as nx
            g = self.scm.dag.g if hasattr(self.scm.dag, "g") else None
            if g is None:
                return
            pos = nx.spring_layout(g, seed=0)
            nx.draw(g, pos, with_labels=True)
            plt.show()

    def _reduced_dataframe(self, xs: Dict[str, Tensor], max_nodes: Optional[int] = None) -> "pd.DataFrame":
        """
        Build a pandas DataFrame with reduced node series (B rows, <=N columns).
        No torch.numpy() used — converted via Python lists.
        """
        if not _HAS_SEABORN:
            raise RuntimeError(
                "Seaborn/pandas are required for the pairplot. Install with:\n"
                "  pip install seaborn pandas  (or conda install -c conda-forge seaborn pandas)"
            )
        names = list(self.scm._topo)
        if max_nodes is not None:
            names = names[:max_nodes]
        data = {}
        for v in names:
            s = _reduce_per_sample(xs[v].detach().to("cpu"), self.reduce)  # (B,)
            data[str(v)] = _to_mpl_array(s)  # list
        return pd.DataFrame(data)

    def plot_pairplot(self, xs: Dict[str, Tensor], max_nodes: Optional[int] = None, bins: int = 30) -> None:
        """
        Big pairwise scatterplot with histograms on the diagonal.
        Implemented via seaborn.PairGrid + matplotlib hist on diag to avoid
        seaborn/pandas version issues seen with pairplot(diag_kind="hist").
        """
        if not _HAS_SEABORN:
            raise RuntimeError(
                "Seaborn/pandas are required for the pairplot. Install with:\n"
                "  pip install seaborn pandas  (or conda install -c conda-forge seaborn pandas)"
            )
        df = self._reduced_dataframe(xs, max_nodes=max_nodes)

        # Use PairGrid explicitly; map diag with plt.hist to dodge the GrouperView.join bug
        g = sns.PairGrid(df, corner=False, diag_sharey=False)
        g.map_lower(sns.scatterplot, s=12, alpha=0.7)
        g.map_upper(sns.scatterplot, s=12, alpha=0.7)
        g.map_diag(plt.hist, bins=bins)
        plt.suptitle("Pairwise scatter (reduced node series)", y=1.02)
        plt.show()

    def plot_pca2d(self, xs: Dict[str, Tensor], max_nodes: Optional[int] = None) -> None:
        """
        2-D PCA scatter over reduced node series (samples as rows, nodes as features).
        Implemented with torch SVD (no sklearn dependency).
        """
        # Build BxN matrix
        names_full = list(self.scm._topo)
        if max_nodes is not None:
            names = names_full[:max_nodes]
        else:
            names = names_full

        cols = []
        for v in names:
            s = _reduce_per_sample(xs[v].detach().to("cpu"), self.reduce)  # (B,)
            cols.append(s)
        X = torch.stack(cols, dim=1)  # (B, N)

        # Center & SVD
        Xc = X - X.mean(dim=0, keepdim=True)
        # (B,N) -> U(B,r), S(r), Vh(N,r)
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        # project to first 2 PCs
        V = Vh.T  # (N, r)
        PCs = Xc @ V[:, :2]  # (B, 2)

        x1 = _to_mpl_array(PCs[:, 0])
        x2 = _to_mpl_array(PCs[:, 1])

        plt.figure(figsize=(5.5, 4.8))
        plt.scatter(x1, x2, s=13, alpha=0.75)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA (2-D) over reduced node series")
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, xs: Dict[str, Tensor], max_nodes: Optional[int] = None) -> None:
        """
        Plot correlation heatmap between features.
        """
        if not _HAS_SEABORN:
            raise RuntimeError(
                "Seaborn is required for heatmaps. Install with:\n"
                "  pip install seaborn"
            )
        
        corr_matrix, node_order = self.correlation_matrix(xs)
        
        # Limit nodes if requested
        if max_nodes is not None:
            corr_matrix = corr_matrix[:max_nodes, :max_nodes]
            node_order = node_order[:max_nodes]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_matrix.numpy(),
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            xticklabels=[str(v) for v in node_order],
            yticklabels=[str(v) for v in node_order],
            fmt='.3f'
        )
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def plot_mutual_information_heatmap(self, xs: Dict[str, Tensor], max_nodes: Optional[int] = None) -> None:
        """
        Plot mutual information heatmap between features.
        """
        if not _HAS_SEABORN:
            raise RuntimeError(
                "Seaborn is required for heatmaps. Install with:\n"
                "  pip install seaborn"
            )
        
        if not _HAS_SKLEARN:
            raise RuntimeError(
                "Scikit-learn is required for mutual information. Install with:\n"
                "  pip install scikit-learn"
            )
        
        mi_matrix, node_order = self.mutual_information_matrix(xs)
        
        # Limit nodes if requested
        if max_nodes is not None:
            mi_matrix = mi_matrix[:max_nodes, :max_nodes]
            node_order = node_order[:max_nodes]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            mi_matrix.numpy(),
            annot=True,
            cmap='viridis',
            square=True,
            xticklabels=[str(v) for v in node_order],
            yticklabels=[str(v) for v in node_order],
            fmt='.3f'
        )
        plt.title("Mutual Information Heatmap")
        plt.tight_layout()
        plt.show()

    # ------------------------------ benchmarking -----------------------------

    @torch.no_grad()
    def benchmark(
        self,
        repeats: int = 20,
        warmup: int = 3,
        test_fast_and_safe: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        B = self.batch_size
        results: Dict[str, Dict[str, float]] = {}

        def _bench_mode(fast_flag: bool) -> Dict[str, float]:
            self.scm.fast = fast_flag
            # Warmup
            for _ in range(max(0, warmup)):
                self.scm.clear_fixed_noise()
                self.scm.sample_exogenous(B)
                self.scm.sample_endogenous_noise(B)
                _ = self.scm.propagate(B)

            exo_times, endo_times, samp_times = [], [], []
            for _ in range(repeats):
                self.scm.clear_fixed_noise()
                t0 = time.perf_counter()
                self.scm.sample_exogenous(B)
                t1 = time.perf_counter()
                self.scm.sample_endogenous_noise(B)
                t2 = time.perf_counter()
                _ = self.scm.propagate(B)
                t3 = time.perf_counter()
                exo_times.append((t1 - t0) * 1000.0)
                endo_times.append((t2 - t1) * 1000.0)
                samp_times.append((t3 - t2) * 1000.0)

            return {
                "exo_ms": float(torch.tensor(exo_times).mean().item()),
                "endo_ms": float(torch.tensor(endo_times).mean().item()),
                "sample_ms": float(torch.tensor(samp_times).mean().item()),
                "total_ms": float(torch.tensor(exo_times).mean().item()
                                  + torch.tensor(endo_times).mean().item()
                                  + torch.tensor(samp_times).mean().item()),
                "exo_times": exo_times,
                "endo_times": endo_times,
                "samp_times": samp_times,
            }

        results["safe"] = _bench_mode(False)
        if test_fast_and_safe:
            results["fast"] = _bench_mode(True)

        return results

    @torch.no_grad()
    def bootstrap_runtime_cis(
        self,
        timings: Dict[str, Dict[str, float]],
        n_boot: int = 2000,
        alpha: float = 0.05,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        """
        Bootstrap confidence intervals for runtime measurements.
        """
        results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        
        for mode, timing_data in timings.items():
            results[mode] = {}
            for timing_type in ["exo_times", "endo_times", "samp_times"]:
                if timing_type in timing_data:
                    times = torch.tensor(timing_data[timing_type])
                    mean, lo, hi = _bootstrap_mean_ci(times, n_boot=n_boot, alpha=alpha, generator=generator)
                    results[mode][timing_type.replace("_times", "_ms")] = (mean, lo, hi)
        
        return results

    # ------------------------------ full report ------------------------------

    @torch.no_grad()
    def full_report(
        self,
        n_boot: int = 2000,
        alpha: float = 0.05,
        do_plots: bool = True,
        pairwise_max_nodes: Optional[int] = None,
        bins: int = 30,
        benchmark_repeats: int = 20,
        benchmark_warmup: int = 3,
    ) -> None:
        """
        Run the whole pipeline once:
          - sample one dataset
          - print per-node stats 
          - ALWAYS draw the DAG
          - draw a Seaborn PairGrid pairplot, a 2-D PCA scatter, and correlation/MI heatmaps (if do_plots)
          - benchmark safe vs. fast modes with bootstrap CIs for runtime
        """
        # Always plot the DAG
        self.plot_dag()

        # Sample one dataset
        self.scm.clear_fixed_noise()
        xs = self.sample_once()

        # Stats
        stats = self.summarize(xs)

        print("\n=== Per-node summary (reduced series) ===")
        for v in self.scm._topo:
            st = stats[v]
            name = str(v)
            print(
                f"{name:>12} | mean={st['mean']:.4f}  "
                f"std={st['std']:.4f}  min={st['min']:.4f}  med={st['median']:.4f}  max={st['max']:.4f}"
            )

        # Plots
        if do_plots:
            if not _HAS_SEABORN:
                raise RuntimeError(
                    "Seaborn/pandas are required for the pairplot. Install with:\n"
                    "  pip install seaborn pandas  (or conda install -c conda-forge seaborn pandas)"
                )
            self.plot_pairplot(xs, max_nodes=pairwise_max_nodes, bins=bins)
            self.plot_pca2d(xs, max_nodes=pairwise_max_nodes)
            self.plot_correlation_heatmap(xs, max_nodes=pairwise_max_nodes)
            
            # Try to plot mutual information heatmap if sklearn is available
            try:
                self.plot_mutual_information_heatmap(xs, max_nodes=pairwise_max_nodes)
            except RuntimeError as e:
                print(f"\nSkipping mutual information heatmap: {e}")

        # Benchmark with bootstrap CIs for runtime
        timings = self.benchmark(repeats=benchmark_repeats, warmup=benchmark_warmup, test_fast_and_safe=True)
        runtime_cis = self.bootstrap_runtime_cis(timings, n_boot=n_boot, alpha=alpha)
        
        print("\n=== Runtime (ms), averaged over repeats with 95% bootstrap CIs ===")
        for mode, tm in timings.items():
            cis = runtime_cis[mode]
            print(f"{mode.upper():>5s} |")
            
            if "exo_ms" in cis:
                mu, lo, hi = cis["exo_ms"]
                print(f"        exo={mu:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")
            
            if "endo_ms" in cis:
                mu, lo, hi = cis["endo_ms"]
                print(f"        endo={mu:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")
            
            if "samp_ms" in cis:
                mu, lo, hi = cis["samp_ms"]
                print(f"        sample={mu:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")
            
            print(f"        total={tm['total_ms']:.3f}")
            print()


# ------------------------------ example usage ------------------------------
if __name__ == "__main__":
    # Example: build SCM with GraphSampler + SampleMechanism + MixedDist, then inspect.
    from priors.causal_prior.causal_graph.GraphSampler import GraphSampler
    from priors.causal_prior.mechanisms.SampleMLPMechanism import SampleMLPMechanism
    from priors.causal_prior.mechanisms.SampleXGBoostMechanism import SampleXGBoostMechanism
    from priors.causal_prior.noise_distributions.MixedDist import MixedDist
    from priors.causal_prior.noise_distributions.MixedDist_RandomSTD import MixedDistRandomStd

    NUM_NODES = 50
    p = 0.4
    SEED = 43
    EXO_STD = 1.0
    ADD_STD = 0.1
    BATCH_SIZE = 256
    RANDOM_ADDITIVE_STD = True  # use MixedDistRandomStd if True, else MixedDist
    XGBOOST_PROB = 0.1  # Probability of using XGBoost mechanism (0.0 = never, 1.0 = always)
    USE_EXOGENOUS_MECHANISM = True  # Use exogenous mechanisms if True

    graph_sampler = GraphSampler(seed=SEED)
    graph = graph_sampler.sample_dag(num_nodes=NUM_NODES, p=p)
    causal_dag = CausalDAG(g=graph, check_acyclic=True)

    # Create a separate generator for mechanism type sampling
    mechanism_type_generator = torch.Generator().manual_seed(SEED + 1000)

    mechanisms = {}
    xgboost_count = 0
    mlp_count = 0
    
    for node in causal_dag.nodes():
        # Sample whether to use XGBoost or MLP based on probability
        use_xgboost = torch.rand(1, generator=mechanism_type_generator).item() < XGBOOST_PROB
        in_dim = max(1, len(causal_dag.parents(node)))  # roots must take scalar exogenous noise
        
        if use_xgboost:
            mechanisms[node] = SampleXGBoostMechanism(
                input_dim=in_dim,
                node_shape=(1,),
                num_hidden_layers=0,
                hidden_dim=0,
                activation_mode="pre",
                n_training_samples=100,
                generator=torch.Generator().manual_seed(SEED),
                name=node,
                add_noise=False  
            )
            xgboost_count += 1
        else:
            mechanisms[node] = SampleMLPMechanism(
                input_dim=in_dim,
                node_shape=(1,),
                nonlins="tabicl",
                num_hidden_layers=0,
                hidden_dim=16,
                activation_mode="pre",
                generator=torch.Generator().manual_seed(SEED),
                name=node,
            )
            mlp_count += 1


    exogenous_variables = causal_dag.exogenous_variables()
    endogenous_variables = causal_dag.endogenous_variables()

    if RANDOM_ADDITIVE_STD is False:
        exo_noise = {var: MixedDist(std=EXO_STD) for var in exogenous_variables}
        endo_noise = {var: MixedDist(std=ADD_STD) for var in endogenous_variables}
    else: 
        exo_std_dist = GammaMeanStd(mean=1.0, std=1.0)
        endo_noise_dist = GammaMeanStd(mean=0.3, std=0.1)
        exo_noise = {var: MixedDistRandomStd(exo_std_dist) for var in exogenous_variables}
        endo_noise = {var: MixedDistRandomStd(endo_noise_dist) for var in endogenous_variables}

    scm = SCM(
        dag=causal_dag,
        mechanisms=mechanisms,
        exogenous_noise=exo_noise,
        endogenous_noise=endo_noise,
        fast=True,
        use_exogenous_mechanisms=USE_EXOGENOUS_MECHANISM
    )

    inspector = InspectSCMSamples(scm=scm, batch_size=BATCH_SIZE, reduce="mean", device="cpu")
    inspector.full_report(
        n_boot=2000,
        alpha=0.05,
        do_plots=True,
        pairwise_max_nodes=10,
        bins=40,
        benchmark_repeats=10,
        benchmark_warmup=2,
    )
