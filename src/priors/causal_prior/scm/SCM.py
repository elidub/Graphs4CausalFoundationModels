from __future__ import annotations
from typing import Dict, Mapping, Optional, Tuple, List
import torch
from torch import Tensor

from priors.causal_prior.causal_graph.CausalDAG import CausalDAG
from priors.causal_prior.mechanisms.BaseMechanism import BaseMechanism
from priors.causal_prior.noise_distributions.DistributionInterface import Distribution


class SCM:
    """
    Structural Causal Model with vectorized ancestral sampling.

    Exogenous mechanism behavior (controlled by use_exogenous_mechanisms flag)
    -------------------------------------------------------------------------
    - If use_exogenous_mechanisms=False (DEFAULT): 
      Root nodes (no parents) may omit mechanisms entirely; they are treated as pure
      exogenous variables whose values come directly from exogenous noise.
      If a root does have a mechanism, it is ignored for sampling (but we still
      validate `input_dim == 0` in safe mode).
    - If use_exogenous_mechanisms=True:
      All nodes, including roots (exogenous variables), REQUIRE a mechanism.
      For root nodes, the mechanism is a transformation of a scalar exogenous noise:
        x_root = f_root(u_root)
      Concretely, the root mechanism must have input_dim == 1 and receives the
      sampled scalar noise as its input feature (parents tensor of shape (B, 1)).
      No additional `eps` is used for root nodes (eps=None).
    - Non-root nodes must have mechanisms in both modes.

    Noise semantics
    ---------------
    - Exogenous noise: 
      * If use_exogenous_mechanisms=False: defines the values for root nodes directly.
      * If use_exogenous_mechanisms=True: provides ONE scalar per root node and sample;
        this scalar is fed as the sole input feature to the root mechanism.
    - Endogenous noise: passed as `eps` to mechanisms for non-root nodes only.

    Workflow
    --------
    1) scm.sample_exogenous(B)         # samples & fixes exogenous for roots (as ONE flat vector)
    2) scm.sample_endogenous_noise(B)  # samples & fixes eps for non-root nodes (as ONE flat vector)
    3) xs = scm.propagate(B)              # uses the fixed noises

    Fast vs Safe
    ------------
    - fast=True : precomputed plan, no checks/casts, calls mech._forward directly.
    - fast=False: acyclicity + shape & key assertions, uses mech(...) (with checks).

    Parameters
    ----------
    dag : CausalDAG
    mechanisms : Mapping[str, BaseMechanism]
        Mechanisms for (some or all) nodes. Required for all *non-root* nodes.
        For roots: optional if use_exogenous_mechanisms=False, required if True.
    exogenous_noise : Mapping[str, Distribution] | Distribution | None
        For root nodes; behavior depends on use_exogenous_mechanisms flag.
    endogenous_noise : Mapping[str, Distribution] | Distribution | None
        For non-root nodes; passed as `eps` to mechanisms.
    node_shapes : Mapping[str, Tuple[int,...]] | None
        Optional explicit shapes for nodes (especially useful for roots without mechanisms).
        If a node has a mechanism, its shape comes from `mechanism.node_shape`.
        If a root has no mechanism and no `node_shapes[v]`, defaults to scalar ().
    device : torch.device | str
    dtype : torch.dtype
    fast : bool, default False
    use_exogenous_mechanisms : bool, default False
        If True, root nodes require mechanisms with input_dim=1 that transform scalar noise.
        If False, root nodes may omit mechanisms and get values directly from exogenous noise.
    """

    def __init__(
        self,
        dag: CausalDAG,
        mechanisms: Mapping[str, BaseMechanism],
        exogenous_noise: Optional[Mapping[str, Distribution] | Distribution] = None,
        endogenous_noise: Optional[Mapping[str, Distribution] | Distribution] = None,
        node_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        fast: bool = False,
        use_exogenous_mechanisms: bool = False,
    ) -> None:
        self.dag = dag
        self.mechanisms = mechanisms
        self.exogenous_noise = exogenous_noise
        self.endogenous_noise = endogenous_noise
        self.device = torch.device(device)
        self.dtype = dtype
        self.fast = bool(fast)
        self.use_exogenous_mechanisms = bool(use_exogenous_mechanisms)
        self._user_node_shapes = dict(node_shapes) if node_shapes is not None else {}

        # --- Topology & parents
        self._topo: List[str] = self.dag.topo_order()
        self._parents: Dict[str, List[str]] = {v: self.dag.parents(v) for v in self._topo}
        self._is_root: Dict[str, bool] = {v: (len(self._parents[v]) == 0) for v in self._topo}

        # --- Node shapes: from mechanism if present; else from node_shapes; else scalar for roots
        self._node_shape: Dict[str, Tuple[int, ...]] = {}
        for v in self._topo:
            if v in self.mechanisms:
                self._node_shape[v] = tuple(self.mechanisms[v].node_shape)
            elif self._is_root[v]:
                self._node_shape[v] = tuple(self._user_node_shapes.get(v, ()))
            else:
                # non-root without mechanism is invalid; caught in validation (safe mode)
                self._node_shape[v] = tuple(self._user_node_shapes.get(v, ()))  # placeholder

        self._flat_dim: Dict[str, int] = {
            v: (int(torch.tensor(self._node_shape[v]).prod().item()) if self._node_shape[v] else 1)
            for v in self._topo
        }

        # --- Parents feature assembly plan
        self._parent_slices: Dict[str, List[Tuple[str, int, int]]] = {}
        self._parent_total_dim: Dict[str, int] = {}
        for v in self._topo:
            off, slices = 0, []
            for p in self._parents[v]:
                d = self._flat_dim[p]
                slices.append((p, off, off + d))
                off += d
            self._parent_slices[v] = slices
            self._parent_total_dim[v] = off

        # --- Noise slice plans (vectorized)
        roots = [v for v in self._topo if self._is_root[v]]
        self._exo_order: List[str] = roots
        # Endogenous noise only for non-roots
        self._endo_order: List[str] = [v for v in self._topo if not self._is_root[v]]

        self._exo_slices: Dict[str, Tuple[int, int]] = {}
        self._endo_slices: Dict[str, Tuple[int, int]] = {}

        exo_off = 0
        for v in self._exo_order:
            if self.use_exogenous_mechanisms:
                # Scalar exogenous noise per root
                d = 1
            else:
                # Full node dimensionality for direct values
                d = self._flat_dim[v]
            self._exo_slices[v] = (exo_off, exo_off + d)
            exo_off += d
        self._total_exo_dim = exo_off

        endo_off = 0
        for v in self._endo_order:
            d = self._flat_dim[v]
            self._endo_slices[v] = (endo_off, endo_off + d)
            endo_off += d
        self._total_endo_dim = endo_off

        # --- Fixed noise buffers & per-node views
        self._fixed_exogenous_vec: Optional[Tensor] = None
        self._fixed_endogenous_vec: Optional[Tensor] = None
        self._fixed_exogenous: Optional[Dict[str, Tensor]] = None
        self._fixed_endogenous: Optional[Dict[str, Tensor]] = None
        self._fixed_batch: Optional[int] = None

        if not self.fast:
            self._validate_strict()

    # ---------------------- noise preparation API ----------------------

    @torch.no_grad()
    def sample_exogenous(self, num_samples: int) -> Dict[str, Tensor]:
        """Sample & fix exogenous values for roots in ONE flat vector (B, total_exo_dim)."""
        if self._total_exo_dim == 0:
            self._fixed_exogenous_vec = torch.empty(num_samples, 0, device=self.device, dtype=self.dtype)
            self._fixed_exogenous = {}
            self._fixed_batch = num_samples
            return self._fixed_exogenous

        dist = self.exogenous_noise
        if isinstance(dist, Distribution):
            buf = dist.sample_shape((num_samples, self._total_exo_dim))
            if not isinstance(buf, Tensor):
                buf = torch.as_tensor(buf)
            buf = buf.to(device=self.device, dtype=self.dtype)
        else:
            if dist is None:
                raise ValueError("exogenous_noise is None but root nodes exist.")
            buf = torch.empty(num_samples, self._total_exo_dim, device=self.device, dtype=self.dtype)
            for v in self._exo_order:
                if self.use_exogenous_mechanisms:
                    dv = 1
                else:
                    dv = self._flat_dim[v]
                s, e = self._exo_slices[v]
                d_v = dist.get(v, None)
                if d_v is None:
                    raise ValueError(f"No exogenous_noise provided for root '{v}'.")
                x = d_v.sample_shape((num_samples, dv))
                if not isinstance(x, Tensor):
                    x = torch.as_tensor(x)
                buf[:, s:e] = x.to(device=self.device, dtype=self.dtype)

        # Per-root views
        views: Dict[str, Tensor] = {}
        for v in self._exo_order:
            s, e = self._exo_slices[v]
            flat_view = buf[:, s:e]
            if self.use_exogenous_mechanisms:
                # exogenous noise is scalar per root
                views[v] = flat_view.reshape(num_samples)
            else:
                shp = self._node_shape[v]
                node_view = flat_view.reshape(num_samples, *shp) if shp else flat_view.reshape(num_samples)
                views[v] = node_view

        self._fixed_exogenous_vec = buf
        self._fixed_exogenous = views
        self._fixed_batch = num_samples
        return views

    @torch.no_grad()
    def sample_endogenous_noise(self, num_samples: int, *, nodes: Optional[List[str]] = None) -> Dict[str, Tensor]:
        """
        Sample & fix endogenous noise (eps) for *non-root* nodes in one flat vector.
        If `nodes` is provided, resample only those nodes' slices.
        """
        if self._total_endo_dim == 0:
            self._fixed_endogenous_vec = torch.empty(num_samples, 0, device=self.device, dtype=self.dtype)
            self._fixed_endogenous = {}
            self._fixed_batch = num_samples
            return self._fixed_endogenous

        # allocate / reset buffer as needed
        if self._fixed_endogenous_vec is None or self._fixed_batch != num_samples:
            self._fixed_endogenous_vec = torch.empty(num_samples, self._total_endo_dim, device=self.device, dtype=self.dtype)

        target_nodes = nodes if nodes is not None else self._endo_order
        dist = self.endogenous_noise
        if isinstance(dist, Distribution):
            full = dist.sample_shape((num_samples, self._total_endo_dim))
            if not isinstance(full, Tensor):
                full = torch.as_tensor(full)
            full = full.to(device=self.device, dtype=self.dtype)
            if nodes is None:
                self._fixed_endogenous_vec.copy_(full)
            else:
                for v in target_nodes:
                    s, e = self._endo_slices[v]
                    self._fixed_endogenous_vec[:, s:e].copy_(full[:, s:e])
        else:
            if dist is None:
                self._fixed_endogenous_vec.zero_()
            else:
                for v in target_nodes:
                    dv = self._flat_dim[v]
                    s, e = self._endo_slices[v]
                    d_v = dist.get(v, None)
                    if d_v is None:
                        self._fixed_endogenous_vec[:, s:e].zero_()
                    else:
                        e_v = d_v.sample_shape((num_samples, dv))
                        if not isinstance(e_v, Tensor):
                            e_v = torch.as_tensor(e_v)
                        self._fixed_endogenous_vec[:, s:e] = e_v.to(device=self.device, dtype=self.dtype)

        # Per-node views (non-roots only)
        views: Dict[str, Tensor] = {} if (self._fixed_endogenous is None or self._fixed_batch != num_samples) else self._fixed_endogenous
        if views is None:
            views = {}
        for v in self._endo_order:
            s, e = self._endo_slices[v]
            flat_view = self._fixed_endogenous_vec[:, s:e]
            shp = self._node_shape[v]
            node_view = flat_view.reshape(num_samples, *shp) if shp else flat_view.reshape(num_samples)
            views[v] = node_view

        self._fixed_endogenous = views
        self._fixed_batch = num_samples
        return views

    def clear_fixed_noise(self) -> None:
        self._fixed_exogenous_vec = None
        self._fixed_endogenous_vec = None
        self._fixed_exogenous = None
        self._fixed_endogenous = None
        self._fixed_batch = None

    # ---------------------- public sampling API ----------------------

    @torch.no_grad()
    def propagate(
        self,
        num_samples: int,
        *,
        return_endogenous: bool = False,
        return_exogenous: bool = False,
    ):
        if self.fast:
            out = self._sample_fast(num_samples, return_endogenous, return_exogenous)
        else:
            out = self._sample_safe(num_samples, return_endogenous, return_exogenous)
        return out

    # ---------------------- safe mode (checks) ----------------------

    def _validate_strict(self) -> None:
        if not self.dag.is_acyclic():
            raise ValueError("DAG must be acyclic.")

        if self.use_exogenous_mechanisms:
            # All nodes must have mechanisms
            missing = [v for v in self._topo if v not in self.mechanisms]
            if missing:
                raise KeyError(f"Missing mechanisms for nodes: {missing}")

            # Root mechanisms must take exactly one scalar input (the exogenous noise)
            for v in self._topo:
                if self._is_root[v]:
                    if self.mechanisms[v].input_dim != 1:
                        raise ValueError(f"Root node '{v}' must have mechanism.input_dim == 1 (scalar noise input).")
        else:
            # Non-root nodes must have mechanisms
            missing_nonroots = [v for v in self._topo if (not self._is_root[v]) and (v not in self.mechanisms)]
            if missing_nonroots:
                raise KeyError(f"Missing mechanisms for non-root nodes: {missing_nonroots}")

            # Root mechanisms (if provided) must have input_dim == 0
            for v in self._topo:
                if self._is_root[v] and v in self.mechanisms:
                    if self.mechanisms[v].input_dim != 0:
                        raise ValueError(f"Root node '{v}' has mechanism.input_dim != 0.")

        # Mechanism input dims for non-roots must match flattened parent dims
        for v in self._topo:
            if self._is_root[v]:
                continue
            mech = self.mechanisms[v]
            exp_in = self._parent_total_dim[v]
            if mech.input_dim != exp_in:
                raise ValueError(
                    f"Mechanism for '{v}' expects input_dim={mech.input_dim}, "
                    f"but parents flatten to {exp_in}."
                )

        # Ensure shapes are defined for all nodes (roots without mech need user-provided shape or accept scalar)
        if not self.use_exogenous_mechanisms:
            for v in self._topo:
                if v not in self.mechanisms and not self._is_root[v]:
                    raise ValueError(f"Non-root node '{v}' lacks mechanism (required).")

        # Noise maps: keys must be subset of nodes
        for label, noise in (("exogenous_noise", self.exogenous_noise), ("endogenous_noise", self.endogenous_noise)):
            if isinstance(noise, Mapping):
                extra = [k for k in noise.keys() if k not in self._topo]
                if extra:
                    raise KeyError(f"{label} provided for unknown nodes: {extra}")

    def _assert_fixed_noise_safe(self, B: int) -> None:
        # Exogenous must be fixed for roots
        if self._fixed_exogenous is None or self._fixed_exogenous_vec is None:
            raise ValueError("Exogenous noise not fixed. Call `sample_exogenous(num_samples)` first.")
        if self._fixed_batch != B:
            raise ValueError(f"Fixed noise batch size ({self._fixed_batch}) != requested ({B}).")

        for v in self._exo_order:
            x = self._fixed_exogenous[v]
            if self.use_exogenous_mechanisms:
                exp = (B,)
            else:
                exp = (B, *self._node_shape[v]) if self._node_shape[v] else (B,)
            if tuple(x.shape) != exp:
                raise ValueError(f"Exogenous for '{v}' has shape {tuple(x.shape)}, expected {exp}.")

        # Endogenous (optional), but if present must match shape (only for non-roots)
        if self._fixed_endogenous is not None:
            for v, e in self._fixed_endogenous.items():
                exp = (B, *self._node_shape[v]) if self._node_shape[v] else (B,)
                if tuple(e.shape) != exp:
                    raise ValueError(f"Endogenous noise for '{v}' has shape {tuple(e.shape)}, expected {exp}.")

    def _sample_safe(
        self,
        B: int,
        return_endogenous: bool,
        return_exogenous: bool,
    ):
        self._assert_fixed_noise_safe(B)

        xs: Dict[str, Tensor] = {}
        for v in self._topo:
            if self._is_root[v]:
                if self.use_exogenous_mechanisms:
                    # feed scalar exogenous noise as sole input feature
                    mech = self.mechanisms[v]
                    u = self._fixed_exogenous[v].to(device=self.device, dtype=self.dtype).reshape(B, 1)
                    y = mech(u, eps=None)
                    xs[v] = y
                else:
                    # use exogenous noise as node value directly
                    xs[v] = self._fixed_exogenous[v]  # view
                continue

            mech = self.mechanisms[v]
            parts = [xs[p].reshape(B, -1) for p in self._parents[v]]
            parents_feat = torch.cat(parts, dim=1).to(device=self.device, dtype=self.dtype)

            eps_v = None
            if self._fixed_endogenous is not None and v in self._fixed_endogenous:
                eps_v = self._fixed_endogenous[v].to(device=self.device, dtype=self.dtype)

            y = mech(parents_feat, eps=eps_v)  # BaseMechanism.forward (checks)
            xs[v] = y

        payload = (xs,)
        if return_endogenous:
            payload += (self._fixed_endogenous if self._fixed_endogenous is not None else {},)
        if return_exogenous:
            payload += (self._fixed_exogenous if self._fixed_exogenous is not None else {},)
        return payload if len(payload) > 1 else xs

    # ---------------------- fast mode (no checks) ----------------------

    def _sample_fast(
        self,
        B: int,
        return_endogenous: bool,
        return_exogenous: bool,
    ):
        xs: Dict[str, Tensor] = {}
        for v in self._topo:
            if self._is_root[v]:
                if self.use_exogenous_mechanisms:
                    mech = self.mechanisms[v]
                    u = self._fixed_exogenous[v].reshape(B, 1)
                    y = mech._forward(u, eps=None)
                    xs[v] = y
                else:
                    xs[v] = self._fixed_exogenous[v]  # view
                continue

            mech = self.mechanisms[v]
            D = self._parent_total_dim[v]
            if D == 0:
                parents_feat = torch.empty(B, 0, device=self.device, dtype=self.dtype)
            else:
                parents_feat = torch.empty(B, D, device=self.device, dtype=self.dtype)
                for p, s, e in self._parent_slices[v]:
                    parents_feat[:, s:e] = xs[p].reshape(B, -1)

            eps_v = None
            if self._fixed_endogenous is not None:
                eps_v = self._fixed_endogenous.get(v, None)

            y = mech._forward(parents_feat, eps=eps_v)  # bypass checks
            xs[v] = y

        payload = (xs,)
        if return_endogenous:
            payload += (self._fixed_endogenous if self._fixed_endogenous is not None else {},)
        if return_exogenous:
            payload += (self._fixed_exogenous if self._fixed_exogenous is not None else {},)
        return payload if len(payload) > 1 else xs

    # ---------------------- utilities ----------------------

    @staticmethod
    def _resolve_noise(
        noise: Optional[Mapping[str, Distribution] | Distribution],
        v: str,
    ) -> Optional[Distribution]:
        if noise is None:
            return None
        if isinstance(noise, Mapping):
            return noise.get(v, None)
        return noise


#### example usage

if __name__ == "__main__":
    from priors.causal_prior.causal_graph.GraphSampler import GraphSampler
    from priors.causal_prior.mechanisms.SampleMechanism import SampleMechanism
    from priors.causal_prior.noise_distributions.MixedDist import MixedDist

    NUM_NODES = 5
    p = 0.5 
    SEED = 42
    NOISE_STD = 0.1
    BATCH_SIZE = 16

    # graph
    graph_sampler = GraphSampler(seed=SEED)
    graph = graph_sampler.sample_dag(
        num_nodes=NUM_NODES,
        p=p,
    )
    causal_dag = CausalDAG(
        g=graph,
        check_acyclic=True
    )
    
    # mechanisms 
    mechanisms = {}
    for node in causal_dag.nodes():
        # assign mechanisms
        mechanisms[node] = SampleMechanism(
            input_dim=len(causal_dag.parents(node)),
            node_shape=(1,),  # scalar output
            nonlins="mixed",
            max_hidden_layers=2,
            min_hidden_layers=0,
            hidden_dim=64,
            activation_mode="pre",
            generator=torch.Generator().manual_seed(SEED),
            name=node
        )
    
    # noise distributions

    exogenous_variables = causal_dag.exogenous_variables()
    endogenous_variables = causal_dag.endogenous_variables()

    exo_noise = {}
    for var in exogenous_variables:
        exo_noise[var] = MixedDist(
            std = NOISE_STD
        )
    
    endo_noise = {}
    for var in endogenous_variables:
        endo_noise[var] = MixedDist(
            std = NOISE_STD
        )

    scm = SCM(
        dag = causal_dag,
        mechanisms = mechanisms,
        exogenous_noise = exo_noise,
        endogenous_noise = endo_noise,
        fast=True #False means safe mode with checks
    )

    scm.sample_exogenous(num_samples=BATCH_SIZE)
    scm.sample_endogenous_noise(num_samples=BATCH_SIZE)

    r = scm.propagate(num_samples=BATCH_SIZE)

    print(r)
    print(r[0].shape)

