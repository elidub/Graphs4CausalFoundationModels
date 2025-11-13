import unittest
import numpy as np
import networkx as nx
from CausalDAG import CausalDAG
from GraphSampler import GraphSampler

class TestGraphSamplerAndCausalGraph(unittest.TestCase):
    """
    Class to test graph sampling and causal dag itself
    """
    def setUp(self) -> None:
        self.n = 25
        self.p = 0.2
        self.seed = 2025
        self.sampler = GraphSampler(seed=self.seed)

    # ---------- GraphSampler ----------
    def test_basic(self):
        G, perm = self.sampler.sample_dag(self.n, self.p, return_perm=True)
        self.assertEqual(G.number_of_nodes(), self.n)
        self.assertTrue(nx.is_directed_acyclic_graph(G))

        # edges respect permutation order
        pos = {node: i for i, node in enumerate(perm)}
        for u, v in G.edges():
            self.assertLess(pos[u], pos[v])

    def test_extremes(self):
        # p = 0 -> empty DAG
        G0 = self.sampler.sample_dag(self.n, 0.0)
        self.assertEqual(G0.number_of_edges(), 0)
        self.assertTrue(nx.is_directed_acyclic_graph(G0))

        # p = 1 -> complete DAG (n choose 2) edges
        G1 = self.sampler.sample_dag(self.n, 1.0)
        self.assertEqual(G1.number_of_edges(), self.n * (self.n - 1) // 2)
        self.assertTrue(nx.is_directed_acyclic_graph(G1))

    # ---------- Reproducibility ----------
    def test_reproducibility(self):
        G1, perm1 = self.sampler.sample_dag(self.n, self.p, return_perm=True)
        sampler2 = GraphSampler(seed=self.seed)
        G2, perm2 = sampler2.sample_dag(self.n, self.p, return_perm=True)

        self.assertTrue(np.array_equal(perm1, perm2))
        self.assertEqual(set(G1.edges()), set(G2.edges()))

    # ---------- CausalGraph integration ----------
    def test_integration_with_causalgraph(self):
        G, _ = self.sampler.sample_dag(self.n, self.p, return_perm=True)
        cg = CausalDAG(g=G, check_acyclic=True)

        self.assertTrue(cg.is_acyclic())
        topo = cg.topo_order()
        self.assertEqual(len(topo), self.n)

        # All edges go forward in topo order
        pos = {node: i for i, node in enumerate(topo)}
        for u, v in cg.edges():
            self.assertLess(pos[u], pos[v])

        # parents/children consistency
        for v in cg.nodes():
            for p in cg.parents(v):
                self.assertIn((p, v), cg.edges())
            for c in cg.children(v):
                self.assertIn((v, c), cg.edges())

        # cache stability (same result on repeated calls)
        topo2 = cg.topo_order()
        self.assertEqual(topo, topo2)

    def test_mutations_and_cache_invalidation(self):
        G, _ = self.sampler.sample_dag(8, 0.3, return_perm=True)
        cg = CausalDAG(g=G, check_acyclic=True)
        old_topo = cg.topo_order()

        # Add a new isolated node, then connect it with a forward edge to preserve DAG
        new_node = max(cg.nodes()) + 1 if len(cg.nodes()) > 0 else 0
        cg.add_node(new_node)
        self.assertTrue(cg.is_acyclic())

        # Connect from a root (node with no parents) to new node (keeps acyclicity)
        roots = [n for n in cg.nodes() if len(cg.parents(n)) == 0 and n != new_node]
        if roots:
            cg.add_edge(roots[0], new_node)
            self.assertTrue(cg.is_acyclic())

        # Topo order should be recomputable and include the new node
        topo = cg.topo_order()
        self.assertIn(new_node, topo)
        self.assertEqual(len(topo), len(old_topo) + 1)

        # Removing an edge keeps acyclicity and allows recomputation
        if cg.edges():
            u, v = cg.edges()[0]
            cg.remove_edge(u, v)
            self.assertTrue(cg.is_acyclic())
            _ = cg.topo_order()  # should not raise

    # ---------- Input validation ----------
    def test_invalid_inputs_sampler(self):
        with self.assertRaises(ValueError):
            self.sampler.sample_dag(-1, 0.2)
        with self.assertRaises(ValueError):
            self.sampler.sample_dag(5, -0.1)
        with self.assertRaises(ValueError):
            self.sampler.sample_dag(5, 1.1)

if __name__ == "__main__":
    unittest.main(verbosity=2)