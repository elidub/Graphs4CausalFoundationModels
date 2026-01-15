"""
Minimal test to isolate the root cause of the ancestor matrix bug.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

try:
    from utils.graph_utils import adjacency_to_ancestor_matrix, propagate_ancestor_knowledge
except ImportError:
    try:
        from src.utils.graph_utils import adjacency_to_ancestor_matrix, propagate_ancestor_knowledge
    except ImportError:
        from pathlib import Path
        utils_path = Path(__file__).resolve().parent / "src" / "utils"
        if str(utils_path) not in sys.path:
            sys.path.insert(0, str(utils_path))
        from graph_utils import adjacency_to_ancestor_matrix, propagate_ancestor_knowledge


def test_conversion_pipeline():
    """
    Test the exact sequence that happens in InterventionalDataset:
    1. Get adjacency matrix
    2. Convert to ancestor matrix
    3. Convert to three-state format {-1, 0, 1}
    4. Call propagate_ancestor_knowledge (with hide_frac=0, so no hiding)
    """
    print("="*80)
    print("Testing Conversion Pipeline")
    print("="*80)
    
    # Create a simple adjacency matrix: 0->1, 1->2
    print("\n1. Starting adjacency matrix (0->1, 1->2):")
    adj = torch.tensor([[0., 1., 0.],
                        [0., 0., 1.],
                        [0., 0., 0.]])
    print(adj)
    
    # Step 2: Convert to ancestor matrix
    print("\n2. Convert to ancestor matrix:")
    ancestor = adjacency_to_ancestor_matrix(adj)
    print(ancestor)
    print(f"Expected: 0 is ancestor of 1 and 2, 1 is ancestor of 2")
    
    # Step 3: Convert binary {0,1} to three-state {-1,1}
    print("\n3. Convert to three-state format {-1, 0, 1}:")
    graph_matrix = ancestor.clone().float()
    graph_matrix = 2.0 * graph_matrix - 1.0  # Map {0,1} to {-1,1}
    print(graph_matrix)
    print(f"  (All values should be -1 or 1, no 0s since hide_frac=0)")
    
    # Count values
    num_neg1 = (graph_matrix == -1).sum().item()
    num_zero = (graph_matrix == 0).sum().item()
    num_pos1 = (graph_matrix == 1).sum().item()
    print(f"  Count: {num_neg1} entries = -1, {num_zero} entries = 0, {num_pos1} entries = 1")
    
    # Step 4: Call propagate_ancestor_knowledge
    print("\n4. Call propagate_ancestor_knowledge:")
    propagated = propagate_ancestor_knowledge(graph_matrix)
    print(propagated)
    
    # Check if it matches
    print("\n5. Comparison:")
    print("  Original three-state matrix:")
    print(graph_matrix)
    print("\n  After propagate_ancestor_knowledge:")
    print(propagated)
    
    # Check differences
    diff_mask = graph_matrix != propagated
    if diff_mask.any():
        print("\n  ❌ FOUND DIFFERENCES!")
        diff_indices = diff_mask.nonzero()
        for idx in diff_indices:
            i, j = idx[0].item(), idx[1].item()
            print(f"    [{i},{j}]: before={graph_matrix[i,j].item():.1f}, after={propagated[i,j].item():.1f}")
    else:
        print("\n  ✓ No differences - propagate_ancestor_knowledge preserved the matrix")
    
    return graph_matrix, propagated


def test_with_real_scm_pattern():
    """
    Test with a pattern that matches what we see in the bug:
    - Node 29 has direct edge to node 6
    - In ancestor matrix: [29,6]=1
    - After three-state conversion: [29,6]=1, [6,29]=-1
    - After propagate_ancestor_knowledge: [29,6] should still be 1!
    """
    print("\n\n" + "="*80)
    print("Testing with Real SCM Pattern")
    print("="*80)
    
    # Simplified: just 3 nodes where node 2 -> node 0
    print("\n1. Adjacency matrix (2->0):")
    adj = torch.tensor([[0., 0., 0.],
                        [0., 0., 0.],
                        [1., 0., 0.]])  # Row 2, col 0 = edge 2->0
    print(adj)
    
    print("\n2. Ancestor matrix:")
    ancestor = adjacency_to_ancestor_matrix(adj)
    print(ancestor)
    assert ancestor[2, 0].item() == 1.0, "2 should be ancestor of 0"
    assert ancestor[0, 2].item() == 0.0, "0 should NOT be ancestor of 2"
    
    print("\n3. Three-state format:")
    graph_matrix = ancestor.clone().float()
    graph_matrix = 2.0 * graph_matrix - 1.0
    print(graph_matrix)
    print(f"  [2,0] = {graph_matrix[2,0].item():.1f} (should be 1)")
    print(f"  [0,2] = {graph_matrix[0,2].item():.1f} (should be -1)")
    
    print("\n4. After propagate_ancestor_knowledge:")
    propagated = propagate_ancestor_knowledge(graph_matrix)
    print(propagated)
    print(f"  [2,0] = {propagated[2,0].item():.1f} (should STILL be 1)")
    print(f"  [0,2] = {propagated[0,2].item():.1f} (should STILL be -1)")
    
    # Check the critical value
    if propagated[2,0].item() != 1.0:
        print("\n  ❌ BUG FOUND: Ancestral relationship [2,0] was incorrectly changed!")
        print(f"     Expected: 1.0, Got: {propagated[2,0].item()}")
    else:
        print("\n  ✓ Ancestral relationship [2,0] preserved correctly")


if __name__ == "__main__":
    graph_matrix, propagated = test_conversion_pipeline()
    test_with_real_scm_pattern()
