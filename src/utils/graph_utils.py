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


import torch
from typing import Tuple, Union, Optional


def propagate_ancestor_knowledge(
    ancestor_matrix: torch.Tensor,
    *,
    max_iterations: int = 64,
    enforce_diagonal: bool = True,
    raise_on_inconsistent: bool = True,
    return_is_consistent: bool = False,
    eps: float = 1e-6,
) -> Union[torch.Tensor, Tuple[torch.Tensor, bool]]:
    """
    Propagate known ancestor relationships in a partial ancestor matrix (PAM).

    Input semantics (strict ancestor / reachability):
      -  1 : i is an ancestor of j (directed path exists i -> ... -> j)
      -  0 : unknown
      - -1 : i is NOT an ancestor of j

    Sound closure rules for DAG ancestor relations:
      (A) Diagonal:        T[i,i] = -1  (strict ancestor; no self-ancestry)
      (B) Antisymmetry:    if T[i,j] = 1 then T[j,i] = -1
      (C) Transitivity:    if T[i,j] = 1 and T[j,k] = 1 then T[i,k] = 1

    The function:
      - supports shapes (N,N) or (B,N,N)
      - uses boolean reachability + batched matmul for transitive closure
      - detects inconsistencies (optionally raises)

    Inconsistencies detected:
      - any T[i,i] == 1
      - any pair with both T[i,j] == 1 and T[j,i] == 1 (cycle implied)
      - any forced ancestor (by transitivity) conflicts with a known -1

    Returns:
      - completed PAM with values in {-1,0,1}, same dtype as input by default
      - optionally also returns a bool flag is_consistent
    """
    if ancestor_matrix.ndim not in (2, 3):
        raise ValueError(f"ancestor_matrix must have shape (N,N) or (B,N,N), got {tuple(ancestor_matrix.shape)}")
    if ancestor_matrix.shape[-1] != ancestor_matrix.shape[-2]:
        raise ValueError(f"ancestor_matrix must be square in the last two dims, got {tuple(ancestor_matrix.shape)}")

    orig_dtype = ancestor_matrix.dtype
    device = ancestor_matrix.device

    # ---- normalize input to int8 in {-1,0,1} robustly ----
    x = ancestor_matrix
    if not x.is_floating_point():
        x_int = x.to(torch.int8)
    else:
        # allow tiny numeric noise
        def close_to(v: float) -> torch.Tensor:
            return (x - v).abs() <= eps

        ok = close_to(-1.0) | close_to(0.0) | close_to(1.0)
        if not bool(ok.all()):
            bad = x[~ok]
            raise ValueError(
                "ancestor_matrix contains values outside {-1,0,1} (within eps). "
                f"Example bad values: {bad.flatten()[:10].tolist()}"
            )
        x_int = torch.zeros_like(x, dtype=torch.int8)
        x_int[close_to(-1.0)] = -1
        x_int[close_to(0.0)] = 0
        x_int[close_to(1.0)] = 1

    # Work on a clone
    T = x_int.clone()

    # Ensure batched shape (B,N,N)
    if T.ndim == 2:
        T = T.unsqueeze(0)  # (1,N,N)
        squeeze_batch = True
    else:
        squeeze_batch = False

    B, N, _ = T.shape

    # ---- (A) diagonal enforcement + check ----
    is_consistent = True
    diag = torch.diagonal(T, dim1=-2, dim2=-1)  # (B,N)
    if bool((diag == 1).any()):
        is_consistent = False
        if raise_on_inconsistent:
            raise ValueError("Inconsistent PAM: found T[i,i] == 1 (a node cannot be its own strict ancestor).")

    if enforce_diagonal:
        # set diagonal to -1 regardless of unknown/-1; keep consistency flag if it was 1
        idx = torch.arange(N, device=device)
        T[:, idx, idx] = -1

    # ---- reachability matrix R (known ancestors) ----
    R = (T == 1)  # (B,N,N), bool

    # ---- (B) antisymmetry immediate contradiction check ----
    if bool((R & R.transpose(-1, -2)).any()):
        is_consistent = False
        if raise_on_inconsistent:
            raise ValueError("Inconsistent PAM: found a 2-cycle implied by T[i,j]==1 and T[j,i]==1.")

    # ---- (C) transitive closure via boolean matmul; detect conflicts with known -1 ----
    known_not_ancestor = (T == -1)

    # helper for batched boolean reachability multiplication
    def boolean_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # (B,N,N) bool -> (B,N,N) bool, where out[i,k] = OR_j a[i,j] & b[j,k]
        # implement via float32 bmm then >0 (CUDA doesn't support int32 for bmm)
        return (torch.bmm(a.to(torch.float32), b.to(torch.float32)) > 0)

    for _ in range(max_iterations):
        R_next = R | boolean_matmul(R, R)

        # transitivity cannot create self-ancestry in a DAG; but if constraints imply it, it's inconsistent
        if bool(torch.diagonal(R_next, dim1=-2, dim2=-1).any()):
            is_consistent = False
            if raise_on_inconsistent:
                raise ValueError("Inconsistent PAM: transitive closure implies self-ancestry (cycle).")

        # conflict: something is forced ancestor but known as NOT ancestor
        if bool((R_next & known_not_ancestor).any()):
            is_consistent = False
            if raise_on_inconsistent:
                raise ValueError("Inconsistent PAM: transitive closure forces an ancestor relation that conflicts with -1.")
            # if not raising, we keep going but flag inconsistency
        if torch.equal(R_next, R):
            break
        R = R_next

    # ---- write back inferred info to T ----
    # Set all inferred ancestors to 1
    T[R] = 1

    # Enforce antisymmetry: if i ancestor of j then j not ancestor of i
    RT = R.transpose(-1, -2)

    # If RT already has 1 where R has 1, that's a cycle (already checked above), but keep safety:
    if bool((R & RT).any()):
        is_consistent = False
        if raise_on_inconsistent:
            raise ValueError("Inconsistent PAM: cycle detected after closure (should not happen).")

    # Fill unknowns on the reverse with -1 (do not overwrite existing 1)
    reverse_should_be_not_ancestor = RT
    overwrite_mask = reverse_should_be_not_ancestor & (T == 0)
    T[overwrite_mask] = -1

    # Keep diagonal as -1 if requested
    if enforce_diagonal:
        idx = torch.arange(N, device=device)
        T[:, idx, idx] = -1

    # restore original batch shape and dtype
    if squeeze_batch:
        T = T.squeeze(0)

    T_out = T.to(orig_dtype) if orig_dtype.is_floating_point else T.to(orig_dtype)

    if return_is_consistent:
        return T_out, is_consistent
    return T_out


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
    
    def test_propagate_simple_chain():
        """Test propagation on simple chain: 0 -> 1 -> 2 where middle edge is unknown."""
        print("\nTest 7: Propagate - Simple chain with unknown edge")
        pam = torch.tensor([
            [-1.,  1.,  0.],  # 0 is ancestor of 1, unknown for 2
            [-1., -1.,  1.],  # 1 is ancestor of 2
            [-1., -1., -1.]   # 2 is sink
        ])
        print("Initial partial ancestor matrix:")
        print(pam)
        
        completed = propagate_ancestor_knowledge(pam)
        print("\nAfter propagation:")
        print(completed)
        
        expected = torch.tensor([
            [-1.,  1.,  1.],  # Now know 0 is ancestor of 2
            [-1., -1.,  1.],
            [-1., -1., -1.]
        ])
        assert torch.allclose(completed, expected), "Failed to infer via transitivity"
        print("✓ Propagation test passed")
    
    def test_propagate_recovery():
        """Test that hidden edges can be recovered via propagation."""
        print("\nTest 8: Propagate - Recovery from hiding")
        # Complete ancestor matrix
        complete = torch.tensor([
            [-1.,  1.,  1.,  1.],
            [-1., -1.,  1.,  1.],
            [-1., -1., -1.,  1.],
            [-1., -1., -1., -1.]
        ])
        print("Complete ancestor matrix:")
        print(complete)
        
        # Hide some edges
        partial = complete.clone()
        partial[0, 2] = 0.0
        partial[0, 3] = 0.0
        partial[1, 3] = 0.0
        print("\nAfter hiding (set to 0):")
        print(partial)
        
        # Propagate to recover
        recovered = propagate_ancestor_knowledge(partial)
        print("\nAfter propagation:")
        print(recovered)
        
        assert torch.allclose(recovered, complete), "Failed to recover all edges"
        print("✓ Successfully recovered all hidden edges via propagation")
    
    # Run all tests
    test_simple_chain()
    test_fork()
    test_collider()
    test_complex_dag()
    test_no_edges()
    test_cuda_support()
    test_propagate_simple_chain()
    test_propagate_recovery()
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
