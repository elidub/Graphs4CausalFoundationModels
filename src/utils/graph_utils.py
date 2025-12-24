"""
Graph utility functions for causal inference.

Utilities for directed graphs (often DAGs), including computing ancestor matrices
(transitive closure / reachability).
"""

from __future__ import annotations

from typing import Literal, Optional
import torch


def adjacency_to_ancestor_matrix(
    adjacency_matrix: torch.Tensor,
    *,
    remove_diagonal: bool = True,
    assume_dag: bool = False,
    method: Literal["auto", "floyd-warshall", "dag-dp"] = "auto",
) -> torch.Tensor:
    """
    Compute the ancestor (reachability) matrix T from an adjacency matrix A.

    Definition:
        T[i, j] = 1  iff  there exists a directed path i -> ... -> j of length >= 1.

    Args:
        adjacency_matrix:
            Tensor of shape (N, N). Nonzero entries indicate directed edges i -> j.
            Can be bool/int/float. Values are treated as edges via (A != 0).
        remove_diagonal:
            If True (default), force T[i, i] = 0 for all i (non-reflexive closure).
            If False, T[i,i] will be 1 iff there is a directed cycle reachable from i
            back to itself (in floyd-warshall), or 0 in dag-dp (since DAG has no cycles).
        assume_dag:
            If True, use a DAG-optimized algorithm (requires acyclic graph).
            If the graph has cycles and assume_dag=True, results are undefined.
        method:
            - "auto": use "dag-dp" if assume_dag else "floyd-warshall"
            - "floyd-warshall": O(N^3), works for any directed graph
            - "dag-dp": O(N^2) typical, requires a DAG and a topological order

    Returns:
        Tensor T of shape (N, N), with dtype equal to adjacency_matrix.dtype
        and on the same device. Entries are 0/1 in that dtype.

    Notes:
        - This computes *reachability*, not necessarily “ancestor” in the strict causal sense
          if your graph has cycles.
        - For a DAG, paths have max length N-1, so this is equivalent to
          sgn(sum_{k=1}^{N-1} A^k) in Boolean sense.
    """
    if not torch.is_tensor(adjacency_matrix):
        raise TypeError("adjacency_matrix must be a torch.Tensor")

    if adjacency_matrix.ndim != 2:
        raise ValueError(f"adjacency_matrix must be 2D, got shape {tuple(adjacency_matrix.shape)}")

    n = adjacency_matrix.shape[0]
    if adjacency_matrix.shape[1] != n:
        raise ValueError(f"adjacency_matrix must be square, got shape {tuple(adjacency_matrix.shape)}")

    if adjacency_matrix.is_floating_point():
        if not torch.isfinite(adjacency_matrix).all():
            raise ValueError("adjacency_matrix contains non-finite values (inf/nan).")

    if method == "auto":
        method = "dag-dp" if assume_dag else "floyd-warshall"
    if method == "dag-dp" and not assume_dag:
        raise ValueError('method="dag-dp" requires assume_dag=True')

    dtype = adjacency_matrix.dtype
    device = adjacency_matrix.device

    # Work in boolean space: edge exists iff entry is nonzero.
    A = (adjacency_matrix != 0)

    if method == "floyd-warshall":
        reachable = A.clone()
        for k in range(n):
            # reachable[i,j] |= reachable[i,k] & reachable[k,j]
            reachable |= (reachable[:, k:k+1] & reachable[k:k+1, :])

        if remove_diagonal:
            reachable.fill_diagonal_(False)

        return reachable.to(dtype=dtype, device=device)

    # DAG DP: compute reachability via reverse topological order
    # We'll derive an order using adjacency; to avoid external deps, do Kahn's algorithm.
    # This is O(N^2) with dense adjacency; good enough and still typically faster than O(N^3).
    indeg = A.sum(dim=0).to(torch.int64)  # indegree of each node
    queue = (indeg == 0).nonzero(as_tuple=False).flatten().tolist()
    topo = []
    indeg_work = indeg.clone()

    # Kahn's algorithm
    while queue:
        v = queue.pop()
        topo.append(v)
        # for each outgoing edge v -> u, decrement indegree[u]
        out = A[v].nonzero(as_tuple=False).flatten()
        if out.numel() > 0:
            indeg_work[out] -= 1
            newly_zero = out[indeg_work[out] == 0]
            if newly_zero.numel() > 0:
                queue.extend(newly_zero.tolist())

    if len(topo) != n:
        raise ValueError("Graph is not acyclic (topological sort failed). Set assume_dag=False to use Floyd–Warshall.")

    reachable = torch.zeros((n, n), dtype=torch.bool, device=device)
    # process in reverse topo: children reachability known when processing parent
    for v in reversed(topo):
        children = A[v].nonzero(as_tuple=False).flatten()
        if children.numel() == 0:
            continue
        reachable[v, children] = True
        # v reaches everything its children reach
        reachable[v] |= reachable[children].any(dim=0)

    if remove_diagonal:
        reachable.fill_diagonal_(False)

    return reachable.to(dtype=dtype, device=device)


if __name__ == "__main__":
    """
    Test suite for ancestor matrix computation.
    """
    print("="*80)
    print("Testing Ancestor Matrix Computation")
    print("="*80)
    
    def test_simple_chain():
        """Test: 0 -> 1 -> 2 (simple chain)"""
        print("\nTest 1: Simple chain (0 -> 1 -> 2)")
        adj = torch.tensor([[0., 1., 0.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        print("Adjacency matrix:")
        print(adj)
        
        anc = adjacency_to_ancestor_matrix(adj)
        
        expected = torch.tensor([[0., 1., 1.],
                                [0., 0., 1.],
                                [0., 0., 0.]])
        
        print("\nAncestor matrix:")
        print(anc)
        print("\nExpected:")
        print(expected)
        
        assert torch.allclose(anc, expected), "Test failed"
        print("✓ Test passed")
    
    def test_fork():
        """Test: 0 -> 1, 0 -> 2 (fork)"""
        print("\nTest 2: Fork (0 -> 1, 0 -> 2)")
        adj = torch.tensor([[0., 1., 1.],
                           [0., 0., 0.],
                           [0., 0., 0.]])
        print("Adjacency matrix:")
        print(adj)
        
        anc = adjacency_to_ancestor_matrix(adj)
        
        expected = torch.tensor([[0., 1., 1.],
                                [0., 0., 0.],
                                [0., 0., 0.]])
        
        print("\nAncestor matrix:")
        print(anc)
        
        assert torch.allclose(anc, expected), "Test failed"
        print("✓ Test passed")
    
    def test_collider():
        """Test: 0 -> 2, 1 -> 2 (collider)"""
        print("\nTest 3: Collider (0 -> 2, 1 -> 2)")
        adj = torch.tensor([[0., 0., 1.],
                           [0., 0., 1.],
                           [0., 0., 0.]])
        print("Adjacency matrix:")
        print(adj)
        
        anc = adjacency_to_ancestor_matrix(adj)
        
        expected = torch.tensor([[0., 0., 1.],
                                [0., 0., 1.],
                                [0., 0., 0.]])
        
        print("\nAncestor matrix:")
        print(anc)
        
        assert torch.allclose(anc, expected), "Test failed"
        print("✓ Test passed")
    
    def test_complex_dag():
        """Test: Complex DAG with multiple paths"""
        print("\nTest 4: Complex DAG")
        print("Structure: 0->1, 0->2, 1->3, 2->3, 3->4")
        adj = torch.tensor([[0., 1., 1., 0., 0.],
                           [0., 0., 0., 1., 0.],
                           [0., 0., 0., 1., 0.],
                           [0., 0., 0., 0., 1.],
                           [0., 0., 0., 0., 0.]])
        print("Adjacency matrix:")
        print(adj)
        
        anc = adjacency_to_ancestor_matrix(adj)
        
        # 0 is ancestor of: 1, 2, 3 (via 1 or 2), 4 (via 1->3 or 2->3)
        # 1 is ancestor of: 3, 4 (via 3)
        # 2 is ancestor of: 3, 4 (via 3)
        # 3 is ancestor of: 4
        expected = torch.tensor([[0., 1., 1., 1., 1.],
                                [0., 0., 0., 1., 1.],
                                [0., 0., 0., 1., 1.],
                                [0., 0., 0., 0., 1.],
                                [0., 0., 0., 0., 0.]])
        
        print("\nAncestor matrix:")
        print(anc)
        print("\nExpected:")
        print(expected)
        
        assert torch.allclose(anc, expected), "Test failed"
        print("✓ Test passed")
    
    def test_no_edges():
        """Test: Graph with no edges"""
        print("\nTest 5: No edges (isolated nodes)")
        adj = torch.zeros(4, 4)
        print("Adjacency matrix:")
        print(adj)
        
        anc = adjacency_to_ancestor_matrix(adj)
        
        expected = torch.zeros(4, 4)
        
        print("\nAncestor matrix:")
        print(anc)
        
        assert torch.allclose(anc, expected), "Test failed"
        print("✓ Test passed")
    
    def test_cuda_support():
        """Test: CUDA support if available"""
        if torch.cuda.is_available():
            print("\nTest 6: CUDA support")
            adj = torch.tensor([[0., 1., 0.],
                               [0., 0., 1.],
                               [0., 0., 0.]], device='cuda')
            print("Adjacency matrix (CUDA):")
            print(adj)
            
            anc = adjacency_to_ancestor_matrix(adj)
            
            expected = torch.tensor([[0., 1., 1.],
                                    [0., 0., 1.],
                                    [0., 0., 0.]], device='cuda')
            
            assert anc.device.type == 'cuda', "Result not on CUDA"
            assert torch.allclose(anc, expected), "Test failed"
            print("✓ CUDA test passed")
        else:
            print("\nTest 6: CUDA not available, skipping")

    def test_cycle():
        # 0 -> 1 -> 0
        adj = torch.tensor([[0., 1.],
                            [1., 0.]])
        anc = adjacency_to_ancestor_matrix(adj, remove_diagonal=True)
        expected = torch.tensor([[0., 1.],
                                [1., 0.]])
        assert torch.allclose(anc, expected)

        anc_diag = adjacency_to_ancestor_matrix(adj, remove_diagonal=False)
        expected_diag = torch.tensor([[1., 1.],
                                    [1., 1.]])
        assert torch.allclose(anc_diag, expected_diag)
    
    # Run all tests
    test_simple_chain()
    test_fork()
    test_collider()
    test_complex_dag()
    test_no_edges()
    test_cuda_support()
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
