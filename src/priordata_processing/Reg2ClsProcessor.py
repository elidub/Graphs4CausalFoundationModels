from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, TYPE_CHECKING

from tabicl.prior.reg2cls import Reg2Cls

if TYPE_CHECKING:
    from networkx import DiGraph
    from priors.causal_prior.scm import SCM

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
        n_features: int,          # unused; feature count driven by SCM node count
        max_n_features: int,
        n_train_samples: int,
        max_n_train_samples: int,
        n_test_samples: int,
        max_n_test_samples: int,
        seed: Optional[int] = None,
        tabicl_hp: Optional[Dict[str, Any]] = None,
        **_ignored,
    ):
        self.max_n_features = max_n_features
        self.n_train = n_train_samples
        self.max_train = max_n_train_samples
        self.n_test = n_test_samples
        self.max_test = max_n_test_samples
        self.tabicl_hp = tabicl_hp
        print(f'{self.tabicl_hp = }')

        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

        self.selected_target_feature = None
        self.kept_feature_indices = None
        self.adj = None

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
        # print(f'{all_nodes = }')

        X_all = torch.cat([dataset[n].reshape(total, 1) for n in all_nodes], dim=1)

        # Randomly select target node
        target_idx = int(torch.randint(len(all_nodes), (1,), generator=self._gen).item())
        self.selected_target_feature = all_nodes[target_idx]
        self.kept_feature_indices = [n for i, n in enumerate(all_nodes) if i != target_idx]

        y_all = X_all[:, target_idx]
        X_all = torch.cat([X_all[:, :target_idx], X_all[:, target_idx + 1:]], dim=1)


        # Build adjacency matrix: ordering is [feature_nodes..., target] so target sits
        # at the last position — this matches the convention expected by Reg2Cls.
        num_features = X_all.shape[1]
        ordered_nodes = self.kept_feature_indices + [self.selected_target_feature]
        # print(f'{ordered_nodes = }')
        # adj = scm.get_adjacency_matrix(node_order=np.arange(len(ordered_nodes)))
        adj = scm.get_adjacency_matrix(node_order=ordered_nodes)
        # adj = scm.get_adjacency_matrix(node_order=np.arange(len(scm.dag.topo_order())))
        # return X_all, y_all, X_all, y_all, adj

        # Reg2Cls normalises X and y, permutes/pads X, and applies the same permutation
        # to adj — so self.adj is always consistent with the final X column order.
        hp = {**self.tabicl_hp, "max_features": self.max_n_features}
        X_norm, y_norm, adj = Reg2Cls(hp)(X_all, y_all, adj)

        pad_train = self.max_train - self.n_train
        pad_test  = self.max_test  - self.n_test

        X_train = F.pad(X_norm[: self.n_train],                               (0, 0, 0, pad_train))
        Y_train = F.pad(y_norm[: self.n_train].unsqueeze(-1),                 (0, 0, 0, pad_train))
        X_test  = F.pad(X_norm[self.n_train : self.n_train + self.n_test],    (0, 0, 0, pad_test))
        Y_test  = F.pad(y_norm[self.n_train : self.n_train + self.n_test].unsqueeze(-1), (0, 0, 0, pad_test))

        return X_train, Y_train, X_test, Y_test, adj
