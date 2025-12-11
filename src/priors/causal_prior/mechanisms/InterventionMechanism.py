from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor

from priors.causal_prior.mechanisms.BaseMechanism import BaseMechanism


class InterventionMechanism(BaseMechanism):
    """
    Identity mechanism for interventional nodes in a Structural Causal Model.
    
    This mechanism implements Pearl's do-operator by making a node independent
    of its parents. It acts as an identity function that:
    - Returns the noise term (eps) if provided
    - Returns the parent values if noise is not provided
    - Effectively bypasses any causal relationship with parents
    
    This is used internally by SCM.intervene() to perform graph surgery where
    the intervened node becomes exogenous and its value is determined solely
    by its noise distribution (or fixed intervention value).
    
    Parameters
    ----------
    node_shape : Tuple[int, ...], default ()
        Output shape for the node. Empty tuple means scalar output.
        Must match the original node's shape to maintain SCM structure.
    
    Notes
    -----
    - input_dim is always 0 since intervened nodes ignore their parents
    - The mechanism preserves node_shape to maintain tensor dimensions
    - If eps is provided, returns eps directly
    - If eps is None and parents are provided, returns appropriately shaped
      parent values (though parents should be empty after intervention)
    - This is typically used after removing incoming edges in the DAG
    
    Examples
    --------
    >>> # Create intervention mechanism for scalar node
    >>> mech = InterventionMechanism(node_shape=(1,))
    >>> 
    >>> # With noise (typical case after intervention)
    >>> noise = torch.randn(100, 1)  # (batch_size, *node_shape)
    >>> parents = torch.empty(100, 0)  # No parents after intervention
    >>> output = mech(parents, noise)
    >>> assert torch.equal(output, noise)  # Identity on noise
    >>> 
    >>> # In practice, used within SCM.intervene()
    >>> scm.intervene('X2')  # Replaces X2's mechanism with InterventionMechanism
    """
    
    def __init__(self, node_shape: Tuple[int, ...] = ()) -> None:
        """
        Initialize intervention mechanism.
        
        Parameters
        ----------
        node_shape : Tuple[int, ...], default ()
            Output shape for the intervened node. Should match the original
            node's shape to maintain consistency in the SCM.
        """
        super().__init__(input_dim=0, node_shape=node_shape, name="InterventionMechanism")
    
    def _forward(self, parents: Tensor, eps: Optional[Tensor] = None) -> Tensor:
        """
        Identity function: returns noise if provided, otherwise zeros.
        
        Parameters
        ----------
        parents : Tensor, shape (B, 0)
            Parent features (should be empty after intervention).
            
        eps : Optional[Tensor], shape (B, *node_shape)
            Noise term to use as the node's value. If None, returns zeros.
            
        Returns
        -------
        Tensor, shape (B, *node_shape)
            The noise term unchanged (identity function).
            
        Raises
        ------
        ValueError
            If parents dimension is not 0 (intervened nodes should have no parents).
            If eps is None (noise must be provided for intervention nodes).
        """
        B = parents.shape[0]
        
        # Intervened nodes should have minimal parents after graph surgery
        # Allow small parent dimensions (e.g., for exogenous noise passed through mechanism)
        if parents.shape[1] > 2:
            raise ValueError(
                f"InterventionMechanism expects minimal parents (input_dim<=2), "
                f"but received parents with shape {tuple(parents.shape)}. "
                f"Ensure SCM.intervene() was called to remove incoming edges."
            )
        
        if eps is None:
            # If no noise provided, fall back to using (small) parent input as the value.
            # This is to support the case where exogenous noise or a fixed value is
            # passed through the mechanism via the parents tensor.
            #
            # parents has shape (B, D_parent) with D_parent <= 2 (enforced above).
            # We need to reshape it to (B, *node_shape) or (B,) for scalar nodes.

            if self.node_shape:
                # Flatten parents per sample and reshape to node_shape if possible
                flat = parents
                # If parents has fewer entries than the flattened node_shape, we simply
                # broadcast by repeating along the last dimension; if more, we truncate.
                target_flat_dim = int(torch.tensor(self.node_shape).prod().item())
                current_dim = flat.shape[1]

                if current_dim == target_flat_dim:
                    flat_resized = flat
                elif current_dim > target_flat_dim:
                    flat_resized = flat[:, :target_flat_dim]
                else:
                    # Repeat columns to match target dimension
                    reps = (target_flat_dim + current_dim - 1) // current_dim
                    flat_tiled = flat.repeat(1, reps)
                    flat_resized = flat_tiled[:, :target_flat_dim]

                return flat_resized.reshape(B, *self.node_shape)
            else:
                # Scalar node: collapse parents to a single value per sample.
                # If multiple parent entries, take the first.
                return parents[:, 0]
        
        # Identity function: return noise unchanged
        return eps
