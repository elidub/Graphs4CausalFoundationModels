from __future__ import annotations

import random
from typing import Optional, Tuple, Union, Literal

import numpy as np
import networkx as nx


class GraphSampler:
    """
    Utility class for generating random DAGs (Directed Acyclic Graphs).

    Provides both a legacy (loop-based) implementation and a vectorized version.
    Acyclicity is ensured by sampling edges only from earlier to later nodes in
    a random topological order (random permutation).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        seed : int, optional
            Seed used for reproducibility. Applied to both Python's `random`
            (used in the legacy method) and NumPy's Generator (used in the
            vectorized method).
        """
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            random.seed(seed)

    # -------------------- Vectorized (fast) --------------------
    def sample_ER_DAG(
        self,
        num_nodes: int,
        p: float,
        return_perm: bool = False,
    ) -> Union[nx.DiGraph, Tuple[nx.DiGraph, np.ndarray]]:
        """
        Create a random DAG (vectorized). This is the same as in the old prior, i.e. Erdos-Renyi model.

        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        p : float
            Probability of an edge between any ordered pair (i < j) in a random
            topological order. Must be in [0, 1].
        return_perm : bool, default False
            If True, also return the permutation (topological order) used.

        Returns
        -------
        G : nx.DiGraph
            The generated DAG with nodes labeled 0..num_nodes-1.
        perm : np.ndarray, optional
            The permutation used (only if `return_perm=True`).
        """
        n = int(num_nodes)
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        if n <= 1 or p == 0.0:
            return (G, np.arange(n)) if return_perm else G

        # Random topological order
        perm = self.rng.permutation(n)

        # Strictly upper-triangular Bernoulli mask (acyclic by construction)
        mask = np.triu(self.rng.random((n, n)) < p, k=1)

        # Extract and add edges
        i_idx, j_idx = np.nonzero(mask)
        if i_idx.size:
            src = perm[i_idx]
            dst = perm[j_idx]
            G.add_edges_from(zip(src.tolist(), dst.tolist()))

        return (G, perm) if return_perm else G

    # -------------------- Legacy (loop-based) --------------------
    def sample_ER_DAG_legacy(
        self,
        num_nodes: int,
        p: float,
        return_perm: bool = False,
    ) -> Union[nx.DiGraph, Tuple[nx.DiGraph, np.ndarray]]:
        """
        Create a random DAG. This is the same as in the old prior, i.e. Erdos-Renyi model. (legacy double-loop implementation).

        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        p : float
            Probability of an edge between any ordered pair (i < j) in a random
            topological order. Must be in [0, 1].
        return_perm : bool, default False
            If True, also return the permutation (topological order) used.

        Returns
        -------
        G : nx.DiGraph
            The generated DAG with nodes labeled 0..num_nodes-1.
        perm : np.ndarray, optional
            The permutation used (only if `return_perm=True`).
        """
        n = int(num_nodes)
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")

        G = nx.DiGraph()
        nodes = list(range(n))
        G.add_nodes_from(nodes)

        # Random topological order
        # (use the same RNG family as vectorized for reproducibility if desired)
        perm = self.rng.permutation(nodes)

        # Add edges with probability p, only forward in the topo order
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    source, target = int(perm[i]), int(perm[j])
                    G.add_edge(source, target)

        return (G, perm) if return_perm else G

    # -------------------- Optional convenience wrapper --------------------
    def sample_dag(
        self,
        num_nodes: int,
        p: float,
        method: Literal["ER_vectorized", "ER_legacy"] = "ER_vectorized",
        return_perm: bool = False,
    ) -> Union[nx.DiGraph, Tuple[nx.DiGraph, np.ndarray]]:
        """
        Convenience method to sample a DAG using the specified method.
        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        p : float
            Probability of an edge between any ordered pair (i < j) in a random
            topological order. Must be in [0, 1].
        method : Literal["ER_vectorized", "ER_legacy"], default "ER_vectorized"
            Sampling method to use. "ER_vectorized" uses the vectorized method,
            while "ER_legacy" uses the legacy double-loop implementation.
        return_perm : bool, default False
            If True, also return the permutation (topological order) used.
        Returns
        -------
        G : nx.DiGraph
            The generated DAG with nodes labeled 0..num_nodes-1.
        perm : np.ndarray, optional
            The permutation used (only if `return_perm=True`).
        """
        if method == "ER_vectorized":
            return self.sample_ER_DAG(num_nodes, p, return_perm)
        elif method == "ER_legacy":
            return self.sample_ER_DAG_legacy(num_nodes, p, return_perm)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ER_vectorized' or 'ER_legacy'.")
        
        