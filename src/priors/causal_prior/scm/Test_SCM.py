import unittest
import networkx as nx
import torch
from torch import Tensor

# ---------------- Library imports (robust to different layouts) ----------------
try:
    from priors.causal_prior.causal_graph.CausalDAG import CausalDAG
except Exception:
    from causal_graph.CausalDAG import CausalDAG

try:
    from priors.causal_prior.mechanisms.BaseMechanism import BaseMechanism
except Exception:
    from mechanisms.BaseMechanism import BaseMechanism

try:
    from priors.causal_prior.noise_distributions.DistributionInterface import Distribution
except Exception:
    from noise_distributions.DistributionInterface import Distribution

# IMPORTANT: import the *class*, not the module package
try:
    from priors.causal_prior.scm.SCM import SCM
except Exception:
    from SCM import SCM


# ----------------------------- Test helpers -----------------------------------

class ZerosDist(Distribution):
    def sample_one(self) -> Tensor:
        return torch.zeros((), device=self.device, dtype=self.dtype)
    def sample_n(self, n: int) -> Tensor:
        return torch.zeros((n,), device=self.device, dtype=self.dtype)
    def sample_shape(self, shape) -> Tensor:
        return torch.zeros(shape, device=self.device, dtype=self.dtype)

class OnesDist(Distribution):
    def sample_one(self) -> Tensor:
        return torch.ones((), device=self.device, dtype=self.dtype)
    def sample_n(self, n: int) -> Tensor:
        return torch.ones((n,), device=self.device, dtype=self.dtype)
    def sample_shape(self, shape) -> Tensor:
        return torch.ones(shape, device=self.device, dtype=self.dtype)

class StdNormal(Distribution):
    def sample_one(self) -> Tensor:
        return torch.randn((), device=self.device, dtype=self.dtype, generator=self.generator)
    def sample_n(self, n: int) -> Tensor:
        return torch.randn((n,), device=self.device, dtype=self.dtype, generator=self.generator)
    def sample_shape(self, shape) -> Tensor:
        return torch.randn(shape, device=self.device, dtype=self.dtype, generator=self.generator)

class SumMechanism(BaseMechanism):
    """
    y = sum(parents) + eps (if provided).
    node_shape default: scalar ().
    """
    def __init__(self, input_dim: int, node_shape=()):
        super().__init__(input_dim=input_dim, node_shape=node_shape)
    def _forward(self, parents: Tensor, eps: Tensor | None = None) -> Tensor:
        B = parents.shape[0]
        if self.input_dim == 0:
            out = torch.zeros(B, *self.node_shape, device=parents.device, dtype=parents.dtype)
        else:
            if self.node_shape == ():
                out = parents.sum(dim=1)  # (B,)
            else:
                s = parents.sum(dim=1, keepdim=True)  # (B,1)
                out = s.expand(B, *self.node_shape)
        if eps is not None:
            out = out + eps
        return out


# -------------------------------- Test Case -----------------------------------

class TestSCM(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    # ---------- construction & validation ----------

    def test_nonroot_missing_mechanism_raises_in_safe_mode(self):
        # Z -> X; X is non-root and must have a mechanism
        g = nx.DiGraph()
        g.add_edges_from([("Z", "X")])
        dag = CausalDAG(g=g, check_acyclic=True)

        mechs = {"Z": SumMechanism(input_dim=0)}  # X missing
        with self.assertRaises(KeyError):
            SCM(
                dag=dag,
                mechanisms=mechs,
                exogenous_noise=OnesDist(),
                endogenous_noise=ZerosDist(),
                node_shapes={"Z": ()},  # optional; Z has mech anyway
                fast=False,
            )

    def test_root_without_mechanism_defaults_scalar_or_user_shape(self):
        # Case 1: default scalar () root shape if none supplied
        g = nx.DiGraph()
        g.add_edges_from([("Z", "X")])  # Z root -> X
        dag = CausalDAG(g=g, check_acyclic=True)

        # X expects 1-d parent (scalar Z)
        mechs_ok = {"X": SumMechanism(input_dim=1)}
        scm = SCM(
            dag=dag,
            mechanisms=mechs_ok,
            exogenous_noise=OnesDist(),
            endogenous_noise=ZerosDist(),
            node_shapes=None,  # Z has no mech; defaults to scalar ()
            fast=False,
        )
        self.assertIsInstance(scm, SCM)

        # Case 2: require 2-d Z -> must provide node_shapes
        mechs_bad = {"X": SumMechanism(input_dim=2)}  # expects 2 from parent Z
        with self.assertRaises(ValueError):
            SCM(
                dag=dag,
                mechanisms=mechs_bad,
                exogenous_noise=OnesDist(),
                endogenous_noise=ZerosDist(),
                node_shapes=None,  # default scalar -> mismatch (expects 2)
                fast=False,
            )

        # Provide node_shapes to match input_dim
        scm2 = SCM(
            dag=dag,
            mechanisms=mechs_bad,
            exogenous_noise=OnesDist(),
            endogenous_noise=ZerosDist(),
            node_shapes={"Z": (2,)},
            fast=False,
        )
        self.assertIsInstance(scm2, SCM)

    def test_root_with_mechanism_ignored_but_checked_input_dim_zero(self):
        # Provide mechanism for root with input_dim != 0 -> error
        g = nx.DiGraph()
        g.add_nodes_from(["Z"])  # Z root
        dag = CausalDAG(g=g, check_acyclic=True)
        mechs = {"Z": SumMechanism(input_dim=1)}  # invalid for root
        with self.assertRaises(ValueError):
            SCM(
                dag=dag, mechanisms=mechs,
                exogenous_noise=OnesDist(),
                endogenous_noise=None,
                fast=False,
            )

    def test_mechanism_input_dim_mismatch_raises(self):
        # Z -> X -> Y; give Y wrong input_dim
        g = nx.DiGraph()
        g.add_edges_from([("Z", "X"), ("X", "Y")])
        dag = CausalDAG(g=g, check_acyclic=True)
        mechs = {
            "X": SumMechanism(input_dim=1),
            "Y": SumMechanism(input_dim=99),  # wrong
        }
        with self.assertRaises(ValueError):
            SCM(
                dag=dag, mechanisms=mechs,
                exogenous_noise=OnesDist(),
                endogenous_noise=ZerosDist(),
                fast=False,
            )

    def test_cyclic_graph_rejected_by_scm_validation(self):
        # Because CausalDAG.topo_order() is called in SCM.__init__, a cycle triggers
        # networkx.NetworkXUnfeasible (before SCM's own ValueError). Accept either.
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "A")])  # cycle
        dag = CausalDAG(g=g, check_acyclic=False)
        mechs = {"A": SumMechanism(input_dim=1), "B": SumMechanism(input_dim=1)}
        with self.assertRaises((ValueError, nx.NetworkXUnfeasible)):
            SCM(dag=dag, mechanisms=mechs, fast=False)

    # ---------- noise vectorization & views ----------

    def test_exogenous_sampling_flat_buffer_and_views(self):
        # Two independent roots
        g = nx.DiGraph()
        g.add_nodes_from(["Z1", "Z2"])
        dag = CausalDAG(g=g, check_acyclic=True)
        scm = SCM(
            dag=dag,
            mechanisms={},  # no mechanisms for roots
            exogenous_noise=OnesDist(),   # single global distribution -> one flat draw
            endogenous_noise=None,
            node_shapes={"Z1": (), "Z2": ()},
            fast=False,
        )
        B = 8
        exo = scm.sample_exogenous(B)
        self.assertEqual(set(exo.keys()), {"Z1", "Z2"})
        self.assertEqual(tuple(scm._fixed_exogenous_vec.shape), (B, 2))
        self.assertEqual(tuple(exo["Z1"].shape), (B,))
        self.assertEqual(tuple(exo["Z2"].shape), (B,))

        # mutate underlying buffer; views should reflect changes
        scm._fixed_exogenous_vec[:, 0] = 7.0
        self.assertTrue(torch.allclose(exo["Z1"], torch.full((B,), 7.0)))
        self.assertTrue(torch.allclose(exo["Z2"], torch.ones(B)))

    def test_endogenous_sampling_partial_resample_nonroots_only(self):
        # Chain Z -> X (Z root, X non-root)
        g = nx.DiGraph()
        g.add_edges_from([("Z", "X")])
        dag = CausalDAG(g=g, check_acyclic=True)
        scm = SCM(
            dag=dag,
            mechanisms={"X": SumMechanism(input_dim=1)},
            exogenous_noise=OnesDist(),
            endogenous_noise=StdNormal(generator=torch.Generator().manual_seed(123)),
            node_shapes={"Z": ()},
            fast=False,
        )
        B = 10
        scm.sample_exogenous(B)
        endo_all = scm.sample_endogenous_noise(B)
        self.assertEqual(set(endo_all.keys()), {"X"})  # non-roots only

        x_old = endo_all["X"].clone()
        # Resample only X
        scm.sample_endogenous_noise(B, nodes=["X"])
        x_new = scm._fixed_endogenous["X"]
        self.assertFalse(torch.allclose(x_new, x_old))

    # ---------- sampling correctness & mode parity ----------

    def test_safe_and_fast_modes_identical_outputs(self):
        # Z -> X -> Y ; ones exo, zero eps
        g = nx.DiGraph()
        g.add_edges_from([("Z", "X"), ("X", "Y")])
        dag = CausalDAG(g=g, check_acyclic=True)
        mechs = {"X": SumMechanism(input_dim=1), "Y": SumMechanism(input_dim=1)}
        B = 6
        scm = SCM(
            dag=dag, mechanisms=mechs,
            exogenous_noise=OnesDist(),     # Z=1
            endogenous_noise=ZerosDist(),   # eps=0 for non-roots
            node_shapes={"Z": ()},
            fast=False,
        )
        scm.sample_exogenous(B)
        scm.sample_endogenous_noise(B)

        xs_safe = scm.propagate(B)
        self.assertTrue(torch.allclose(xs_safe["Z"], torch.ones(B)))
        self.assertTrue(torch.allclose(xs_safe["X"], torch.ones(B)))  # 1 + 0
        self.assertTrue(torch.allclose(xs_safe["Y"], torch.ones(B)))  # 1 + 0

        scm.fast = True
        xs_fast = scm.propagate(B)
        for k in xs_safe:
            self.assertTrue(torch.allclose(xs_safe[k], xs_fast[k]))

    def test_batch_size_mismatch_raises_in_safe_mode(self):
        # Single root
        g = nx.DiGraph()
        g.add_nodes_from(["Z"])
        dag = CausalDAG(g=g, check_acyclic=True)
        scm = SCM(
            dag=dag,
            mechanisms={},
            exogenous_noise=OnesDist(),
            endogenous_noise=None,
            node_shapes={"Z": ()},
            fast=False,
        )
        scm.sample_exogenous(num_samples=8)
        with self.assertRaises(ValueError):
            _ = scm.propagate(num_samples=4)

    def test_endogenous_none_yields_zero_eps_for_nonroots(self):
        # Z -> X ; endogenous None means zeros eps for non-roots (X)
        g = nx.DiGraph()
        g.add_edges_from([("Z", "X")])
        dag = CausalDAG(g=g, check_acyclic=True)
        scm = SCM(
            dag=dag,
            mechanisms={"X": SumMechanism(input_dim=1)},
            exogenous_noise=OnesDist(),
            endogenous_noise=None,
            node_shapes={"Z": ()},
            fast=False,
        )
        B = 5
        scm.sample_exogenous(B)
        endo = scm.sample_endogenous_noise(B)
        self.assertEqual(set(endo.keys()), {"X"})
        self.assertTrue(torch.allclose(endo["X"], torch.zeros_like(endo["X"])))

    # ---------- internal totals/dims ----------

    def test_total_dims_match_current_semantics(self):
        # Z (2-d root) -> X (scalar non-root)
        g = nx.DiGraph()
        g.add_edges_from([("Z", "X")])
        dag = CausalDAG(g=g, check_acyclic=True)
        scm = SCM(
            dag=dag,
            mechanisms={"X": SumMechanism(input_dim=2, node_shape=())},
            exogenous_noise=ZerosDist(),
            endogenous_noise=ZerosDist(),
            node_shapes={"Z": (2,)},  # root 2-d
            fast=False,
        )
        # exogenous total dim: roots only (Z): 2
        self.assertEqual(scm._total_exo_dim, 2)
        # endogenous total dim: non-roots only (X scalar): 1
        self.assertEqual(scm._total_endo_dim, 1)


if __name__ == "__main__":
    unittest.main()
