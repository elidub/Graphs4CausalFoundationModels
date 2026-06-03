from __future__ import annotations

import numpy as np
from tabicl.prior.mlp_scm import marginalize_graph
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, TYPE_CHECKING
import networkx as nx

from tabicl.prior.reg2cls import Reg2Cls
from gtfm.viz.imshow import imshow

if TYPE_CHECKING:
    from networkx import DiGraph, Graph
    from priors.causal_prior.scm import SCM


# Target-selection rules compared in the thesis target-selection ablation.
# Ordered most-anticausal -> most-causal training distribution.
TARGET_SELECTION_RULES = (
    "shallow_oracle",     # min topological depth (root-leaning)
    "uniform_non_leaf",   # uniform over nodes with >= 1 child
    "uniform",            # uniform over all valid nodes (shipped control)
    "uniform_non_root",   # uniform over nodes with >= 1 parent
    "variance_biased",    # 0.9 from top-variance quantile, 0.1 from full valid pool
    "depth_oracle",       # max topological depth (deepest node)
)


class EmptyEligiblePool(Exception):
    """Raised when a target-selection rule has no eligible node on a sampled SCM.

    Signals to the dataset sampler that the SCM should be rejected and resampled.
    """


class Reg2ClsProcessor:
    """Drop-in replacement for BasicProcessing that uses TabICL's Reg2Cls pipeline.

    Accepts the same constructor signature as BasicProcessing so it can be
    passed as ``processor_class`` to ObservationalDataset without changes
    to the call site. Extra BasicProcessing-specific kwargs are silently ignored.

    After calling ``.process()``, the following attributes are populated:
    - ``selected_target_feature``: the node chosen as the regression target
    - ``kept_feature_indices``: list of feature node names (pre-permutation order)
    - ``adj``: adjacency matrix (max_features+1, max_features+1), permuted and
      padded consistently with X by Reg2Cls (target at last position)

    Note: because Reg2Cls permutes feature columns when ``permute_features=True``,
    the column order of X will not match ``kept_feature_indices`` after ``process()``.
    ``adj`` is always consistent with the final X column order.
    """

    def __init__(
        self,
        n_features: int,
        max_n_features: int,
        n_train_samples: int,
        max_n_train_samples: int,
        n_test_samples: int,
        max_n_test_samples: int,
        seed: Optional[int] = None,
        tabicl_hp: Optional[Dict[str, Any]] = None,
        target_selection_rule: str = "uniform",
        variance_floor: float = 1e-4,
        variance_quantile: float = 0.9,
        variance_bias_prob: float = 0.9,
        **_ignored,
    ):
        # print(f'{max_n_features = }')
        self.n_features = n_features
        self.max_n_features = max_n_features
        self.n_train = n_train_samples
        self.max_train = max_n_train_samples
        self.n_test = n_test_samples
        self.max_test = max_n_test_samples
        self.tabicl_hp = tabicl_hp
        # print(f'{self.tabicl_hp = }')

        if target_selection_rule not in TARGET_SELECTION_RULES:
            raise ValueError(
                f"Unknown target_selection_rule {target_selection_rule!r}; "
                f"expected one of {TARGET_SELECTION_RULES}"
            )
        self.target_selection_rule = target_selection_rule
        self.variance_floor = variance_floor          # nodes below this are never valid targets
        self.variance_quantile = variance_quantile     # top-quantile cutoff for variance_biased
        self.variance_bias_prob = variance_bias_prob   # prob of drawing from the high-variance subset

        # Two independent RNG streams so the only quantity varying across rules is
        # the target draw itself: target selection and feature-subset sampling are decoupled.
        self._gen = torch.Generator()        # target-selection draws
        self._gen_feat = torch.Generator()   # feature-subset draws
        if seed is not None:
            self._gen.manual_seed(seed)
            self._gen_feat.manual_seed(seed + 1_000_003)

        self.selected_target_feature = None
        self.kept_feature_indices = None
        self.adj = None

        # Per-dataset diagnostics populated by _select_target (logged to the dump).
        self.node_variances = None      # list[float], per node, raw SCM emission (pre-standardisation)
        self.node_depths = None         # list[int], per node, topological layer
        self.target_depth = None        # int, depth of the chosen target
        self.target_variance = None     # float, raw variance of the chosen target
        self.eligible_pool_size = None  # int, size of the rule's eligible pool on this SCM
        self.was_rejected = False       # bool, True if the pool was empty (SCM rejected)

    def moralize_and_marginalize(
            self, 
            graph_full: DiGraph,
            nodes_include: list[int],
        ) -> tuple[Graph, Graph]:
            graph_moral = nx.moral_graph(graph_full)

            nodes_exclude = list(graph_full.nodes - set(nodes_include))
            graph_moma = marginalize_graph(graph_moral, nodes_to_eliminate=nodes_exclude)

            return graph_moral, graph_moma

    @staticmethod
    def _topological_depths(graph: "DiGraph", all_nodes: list[int]) -> list[int]:
        """Topological depth (layer) of every node, aligned to ``all_nodes`` order.

        Depth is NOT available on the SCM graph at process time (it is added later
        downstream), so it is computed here directly from the DAG.
        """
        depth: Dict[int, int] = {}
        for layer, nodes in enumerate(nx.topological_generations(graph)):
            for node in nodes:
                depth[node] = layer
        return [depth[n] for n in all_nodes]

    def _choice(self, candidates: list[int]) -> int:
        """Uniform pick from ``candidates`` using the target RNG stream."""
        i = int(torch.randint(len(candidates), (1,), generator=self._gen).item())
        return candidates[i]

    def _select_target(self, scm: SCM, X_all: torch.Tensor, all_nodes: list[int]):
        """Choose the target node per ``self.target_selection_rule``.

        Populates per-node and per-target diagnostics. Returns the chosen node id,
        or ``None`` if the rule's eligible pool is empty (caller should resample).
        """
        g = scm.dag.g

        # Per-node marginal variance on the RAW emission (before Reg2Cls standardises).
        variances = torch.var(X_all, dim=0, unbiased=False)  # (num_nodes,)
        depths = self._topological_depths(g, all_nodes)
        self.node_variances = variances.tolist()
        self.node_depths = depths

        # Variance floor: nodes that are (near-)constant are never valid targets.
        valid = [n for n in all_nodes if variances[n].item() > self.variance_floor]

        rule = self.target_selection_rule
        # Eligible pool = valid nodes that satisfy the rule's hard structural filter.
        if rule == "uniform_non_leaf":
            pool = [n for n in valid if g.out_degree(n) >= 1]
        elif rule == "uniform_non_root":
            pool = [n for n in valid if g.in_degree(n) >= 1]
        else:  # uniform, shallow_oracle, depth_oracle, variance_biased
            pool = list(valid)

        self.eligible_pool_size = len(pool)
        if len(pool) == 0:
            self.was_rejected = True
            return None
        self.was_rejected = False

        # Selection within the eligible pool.
        if rule == "shallow_oracle":
            m = min(depths[n] for n in pool)
            target = self._choice([n for n in pool if depths[n] == m])
        elif rule == "depth_oracle":
            m = max(depths[n] for n in pool)
            target = self._choice([n for n in pool if depths[n] == m])
        elif rule == "variance_biased":
            pool_vars = torch.tensor([variances[n].item() for n in pool])
            cutoff = torch.quantile(pool_vars, self.variance_quantile).item()
            high = [n for n in pool if variances[n].item() > cutoff]
            u = torch.rand(1, generator=self._gen).item()
            target = self._choice(high) if (u < self.variance_bias_prob and len(high) > 0) else self._choice(pool)
        else:  # uniform, uniform_non_leaf, uniform_non_root
            target = self._choice(pool)

        self.target_depth = depths[target]
        self.target_variance = variances[target].item()
        return target

    def process(self, dataset: Dict[Any, torch.Tensor], scm: SCM):
        """Process a raw SCM dataset dict into train/test tensors.

        Args:
            dataset: {node_id: tensor[total_samples, 1]} as returned by ObservationalDataset.
            scm: optional SCM object; if provided, builds the real adjacency matrix
                 (stored in ``self.adj`` after the call). If None, adj is all zeros.

        Returns:
            X_train, Y_train, X_test, Y_test  (same shapes as BasicProcessing.process)
        """
        total = self.n_train + self.n_test
        all_nodes = sorted(dataset.keys())
        assert np.allclose(all_nodes, np.arange(len(all_nodes)))
        # print(f'{all_nodes = }')

        X_all = torch.cat([dataset[n].reshape(total, 1) for n in all_nodes], dim=1)

        assert (self.n_features + 1) <= len(all_nodes), f"{self.n_features = }, {len(all_nodes) = }"
        assert self.n_features <= self.max_n_features, f"{self.n_features = }, {self.max_n_features = }, {len(all_nodes) = }"

        # --- Target selection (configurable rule, own RNG stream) ---
        target = self._select_target(scm, X_all, all_nodes)
        if target is None:
            # No node satisfies the rule's eligibility filter -> reject this SCM.
            raise EmptyEligiblePool(
                f"target_selection_rule={self.target_selection_rule!r} has an empty eligible pool."
            )
        self.selected_target_feature = target

        # --- Feature subset (independent RNG stream so only the target rule varies) ---
        # Decoupled from target selection: features are drawn from the remaining nodes
        # with a separate generator, instead of sharing one permutation with the target.
        remaining = [n for n in all_nodes if n != target]
        feat_perm = torch.randperm(len(remaining), generator=self._gen_feat)
        self.kept_feature_indices = [remaining[i] for i in feat_perm[: self.n_features].tolist()]

        ordered_nodes = self.kept_feature_indices + [self.selected_target_feature]
        nodes_include = ordered_nodes
        graph_moral, graph_moma = self.moralize_and_marginalize(scm.dag.g, nodes_include=nodes_include)
        adj_moma = nx.adjacency_matrix(graph_moma, nodelist = nodes_include).todense()
        adj_moma = torch.from_numpy(adj_moma).float()

        # imshow(adj_moma.numpy(), cmap='Greys', turn_off_axis=False, figsize=2, suptitle = nodes_include)

        y_all = X_all[:, self.selected_target_feature]
        X_all = X_all[:, self.kept_feature_indices]

        # Build adjacency matrix: ordering is [feature_nodes..., target] so target sits
        # at the last position — this matches the convention expected by Reg2Cls.
        num_features = X_all.shape[1]
        # print(f'{ordered_nodes = }')
        # adj = scm.get_adjacency_matrix(node_order=np.arange(len(ordered_nodes)))
        adj_full = scm.get_adjacency_matrix(node_order=np.arange(len(all_nodes))) # this one works when not specifying tick_labels in imshow
        # adj = scm.get_adjacency_matrix(node_order=ordered_nodes)
        # adj = scm.get_adjacency_matrix(node_order=np.arange(len(scm.dag.topo_order())))
        # return X_all, y_all, X_all, y_all, adj

        # Reg2Cls normalises X and y, permutes/pads X, and applies the same permutation
        # to adj — so self.adj is always consistent with the final X column order.
        assert self.tabicl_hp['permute_features'] == False
        hp = {**self.tabicl_hp, "max_features": self.max_n_features}
        X_norm, y_norm, adj_new = Reg2Cls(hp)(X_all, y_all, adj_moma)
        # X_norm, y_norm, adj = X_all, y_all, adj
        assert torch.allclose(adj_new, adj_moma), "Reg2Cls should not permute features when permute_features=False"

        # print(X_norm.shape, self.max_n_features)

        # X_norm = X_norm[:, :self.max_n_features]  # in case Reg2Cls doesn't already truncate to max_features

        pad_train = self.max_train - self.n_train
        pad_test  = self.max_test  - self.n_test

        X_train = F.pad(X_norm[: self.n_train],                               (0, 0, 0, pad_train))
        Y_train = F.pad(y_norm[: self.n_train].unsqueeze(-1),                 (0, 0, 0, pad_train))
        X_test  = F.pad(X_norm[self.n_train : self.n_train + self.n_test],    (0, 0, 0, pad_test))
        Y_test  = F.pad(y_norm[self.n_train : self.n_train + self.n_test].unsqueeze(-1), (0, 0, 0, pad_test))

        return X_train, Y_train, X_test, Y_test, graph_moral, graph_moma, adj_moma
