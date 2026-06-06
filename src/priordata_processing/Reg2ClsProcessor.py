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


# Target-selection rules
# Ordered most-anticausal -> most-causal training distribution
TARGET_SELECTION_RULES = (
    "shallow_oracle",     # min topological depth (root-leaning)
    "uniform_non_leaf",   # uniform over nodes with >= 1 child
    "uniform",            # uniform over all valid nodes (shipped control)
    "uniform_non_root",   # uniform over nodes with >= 1 parent
    "variance_biased",    # 0.9 from top-variance quantile, 0.1 from full valid pool
    "depth_oracle",       # max topological depth (deepest node)
    # --- graded / continuous rules (v2): less degenerate than the single-node oracles ---
    "shallow_band_10",    # uniform over the shallowest band_fraction of valid nodes
    "deep_band_10",       # uniform over the deepest band_fraction of valid nodes
    "mid_band",           # uniform over the middle depth tercile of valid nodes
    "depth_weighted",     # P(node) proportional to softmax(depth / depth_temperature)
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
        band_fraction: float = 0.10,
        depth_temperature: float = 1.0,
        depth_coupling_clean_at: Optional[float] = None,
        feature_selection: str = "random",
        noise_feature_fraction: float = 0.0,
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
        self.band_fraction = band_fraction             # fraction of pool for shallow/deep band rules
        self.depth_temperature = depth_temperature     # temperature for depth_weighted (>0 deep, <0 shallow)
        # Positive-control coupling: when set, label noise decreases with target depth so that
        # Bayes-optimal accuracy rises with depth by construction (None = off, natural prior).
        # flip_prob(depth) = 0.5 * clamp(1 - depth/depth_coupling_clean_at, 0, 1):
        # depth 0 -> 0.5 (labels random, unlearnable); depth >= clean_at -> 0 (clean, learnable).
        self.depth_coupling_clean_at = depth_coupling_clean_at
        # Feature-selection mode:
        #   'random'    : features = random subset of non-target nodes (default, natural prior)
        #   'ancestors' : features = target's ancestors (causal direction), nearest-first,
        #                 padded to n_features with non-descendant distractors. Couples target
        #                 depth to learnability through real causal structure (see _select_features).
        assert feature_selection in ("random", "ancestors"), f"unknown feature_selection {feature_selection!r}"
        self.feature_selection = feature_selection
        # Realism: always reserve this fraction of feature slots for non-descendant
        # distractors (so even deep targets get a realistic signal+noise mix, not 100%
        # clean ancestors). 0.0 = off. Only applies to feature_selection='ancestors'.
        assert 0.0 <= noise_feature_fraction < 1.0, f"noise_feature_fraction must be in [0,1), got {noise_feature_fraction}"
        self.noise_feature_fraction = noise_feature_fraction
        self.n_ancestor_features = None  # diagnostic: how many kept features are true ancestors

        # Two independent sets of random numbers so the only quantity varying across rules is
        # the target draw itself: target selection and feature-subset sampling are decoupled.
        self._gen = torch.Generator()        # target-selection draws
        self._gen_feat = torch.Generator()   # feature-subset draws
        if seed is not None:
            self._gen.manual_seed(seed)
            self._gen_feat.manual_seed(seed + 1000000)

        self.selected_target_feature = None
        self.kept_feature_indices = None
        self.adj = None

        # Per-dataset diagnostics populated by _select_target (logged to the dump).
        self.node_variances = None      # list[float], per node raw SCM variances (before standardisation)
        self.node_depths = None         # list[int], per node topological layer
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
        """
        Topological depth (layer) of every node, aligned to ``all_nodes`` order.
        Computed here directly from the DAG.
        """
        depth: Dict[int, int] = {}
        for layer, nodes in enumerate(nx.topological_generations(graph)):
            for node in nodes:
                depth[node] = layer
        return [depth[n] for n in all_nodes]

    def _choice(self, candidates: list[int]) -> int:
        """Uniform pick from ``candidates`` using the target set of random numbers."""
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
        elif rule == "shallow_band_10":
            # uniform over the shallowest band_fraction of the pool (graded shallow_oracle)
            k = max(1, int(np.ceil(self.band_fraction * len(pool))))
            ordered = sorted(pool, key=lambda n: depths[n])
            target = self._choice(ordered[:k])
        elif rule == "deep_band_10":
            # uniform over the deepest band_fraction of the pool (graded depth_oracle)
            k = max(1, int(np.ceil(self.band_fraction * len(pool))))
            ordered = sorted(pool, key=lambda n: depths[n])
            target = self._choice(ordered[-k:])
        elif rule == "mid_band":
            # uniform over the middle depth tercile of the pool
            ordered = sorted(pool, key=lambda n: depths[n])
            n = len(ordered)
            lo = n // 3
            hi = max(lo + 1, (2 * n) // 3)
            target = self._choice(ordered[lo:hi])
        elif rule == "depth_weighted":
            # continuous: P(node) proportional to softmax(depth / T).
            # T>0 favors deep, T<0 favors shallow, |T|->inf -> uniform.
            d = torch.tensor([float(depths[n]) for n in pool])
            probs = torch.softmax(d / self.depth_temperature, dim=0)
            i = int(torch.multinomial(probs, 1, generator=self._gen).item())
            target = pool[i]
        else:  # uniform, uniform_non_leaf, uniform_non_root
            target = self._choice(pool)

        self.target_depth = depths[target]
        self.target_variance = variances[target].item()
        return target

    def _select_features(self, scm: SCM, target: int, all_nodes: list[int]) -> list[int]:
        """Choose the feature subset per ``self.feature_selection``.

        'random'    : a random subset of the non-target nodes (natural prior).
        'ancestors' : the target's causal ancestors, nearest-first, mixed with
                      non-descendant distractors. ``noise_feature_fraction`` of the slots
                      are always reserved for distractors (realistic signal+noise mix);
                      ancestors fill the rest up to availability. Deep targets get more
                      informative (ancestral) features -> learnable; shallow/root targets
                      get mostly distractors -> hard. Couples depth to learnability through
                      causal structure, at (near-)fixed feature count.
        """
        g = scm.dag.g
        remaining = [n for n in all_nodes if n != target]

        if self.feature_selection == "random":
            feat_perm = torch.randperm(len(remaining), generator=self._gen_feat)
            kept = [remaining[i] for i in feat_perm[: self.n_features].tolist()]
            self.n_ancestor_features = None
            return kept

        # --- ancestors mode ---
        anc = nx.ancestors(g, target)
        desc = nx.descendants(g, target)
        # order ancestors nearest-first (shortest path length TO target, ascending)
        dist_to_target = dict(nx.single_target_shortest_path_length(g, target))  # {source: dist}
        anc_sorted = sorted(anc, key=lambda n: dist_to_target.get(n, 1_000_000))

        # Reserve noise slots so even deep targets keep a realistic signal+noise mix.
        n_noise = round(self.noise_feature_fraction * self.n_features)
        n_informative = self.n_features - n_noise
        kept = anc_sorted[: n_informative]
        self.n_ancestor_features = len(kept)

        if len(kept) < self.n_features:
            # fill the reserved-noise slots (and any unfilled informative slots) with
            # NON-descendant, non-ancestor distractors — uninformative, and never effects,
            # so shallow targets cannot be predicted anticausally from them.
            distractors = [n for n in remaining if n not in anc and n not in desc]
            perm = torch.randperm(len(distractors), generator=self._gen_feat)
            distractors = [distractors[i] for i in perm.tolist()]
            kept = kept + distractors[: self.n_features - len(kept)]
        # if still short (tiny graph dominated by descendants), the task simply has fewer
        # features — acceptable edge case for shallow/root targets.
        return kept

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

        # --- Target selection (configurable rule, own draw of random numsers from generator) ---
        target = self._select_target(scm, X_all, all_nodes)
        if target is None:
            # No node satisfies the rule's eligibility filter -> reject this SCM.
            raise EmptyEligiblePool(
                f"target_selection_rule={self.target_selection_rule!r} has an empty eligible pool."
            )
        self.selected_target_feature = target

        # --- Feature subset (independent set of random numbers so only the target rule varies) ---
        # Decoupled from target selection via a separate generator. 'random' = subset of the
        # remaining nodes; 'ancestors' = causal ancestors padded with non-descendant distractors.
        self.kept_feature_indices = self._select_features(scm, target, all_nodes)

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

        # --- Depth-coupled label noise (positive control; off when depth_coupling_clean_at is None) ---
        # Flip each label with prob that decreases with the target's depth, so deeper targets
        # yield more-learnable tasks. Applied to both train and test rows identically.
        if self.depth_coupling_clean_at:
            d = float(self.target_depth)
            flip_prob = 0.5 * max(0.0, min(1.0, 1.0 - d / self.depth_coupling_clean_at))
            self.label_flip_prob = flip_prob  # diagnostic
            if flip_prob > 0:
                n_classes = int(self.tabicl_hp.get("num_classes", 2))
                flips = torch.rand(y_norm.shape, generator=self._gen_feat) < flip_prob
                if n_classes == 2:
                    y_flipped = 1.0 - y_norm
                else:
                    # random different class for multiclass
                    rand_cls = torch.randint(0, n_classes, y_norm.shape, generator=self._gen_feat).to(y_norm.dtype)
                    y_flipped = torch.where(rand_cls == y_norm, (rand_cls + 1) % n_classes, rand_cls)
                y_norm = torch.where(flips, y_flipped, y_norm)
        else:
            self.label_flip_prob = 0.0

        # print(X_norm.shape, self.max_n_features)

        # X_norm = X_norm[:, :self.max_n_features]  # in case Reg2Cls doesn't already truncate to max_features

        pad_train = self.max_train - self.n_train
        pad_test  = self.max_test  - self.n_test

        X_train = F.pad(X_norm[: self.n_train],                               (0, 0, 0, pad_train))
        Y_train = F.pad(y_norm[: self.n_train].unsqueeze(-1),                 (0, 0, 0, pad_train))
        X_test  = F.pad(X_norm[self.n_train : self.n_train + self.n_test],    (0, 0, 0, pad_test))
        Y_test  = F.pad(y_norm[self.n_train : self.n_train + self.n_test].unsqueeze(-1), (0, 0, 0, pad_test))

        return X_train, Y_train, X_test, Y_test, graph_moral, graph_moma, adj_moma
