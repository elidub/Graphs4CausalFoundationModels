"""
A mechanism that returns the input as output.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from priors.causal_prior.mechanisms.BaseMechanism import BaseMechanism

class IgnoreParentsMechanism(BaseMechanism):
    """
    A mechanism that ignores its parents and returns the noise input as output.
    """

    def __init__(self):
        """
        Initializes the IgnoreParentsMechanism.

        Args:
            node_index (int): The index of the node this mechanism is associated with.
        """
        super().__init__(input_dim=0, node_shape=(), name="IgnoreParentsMechanism")

    def _forward(
        self,
        parents: Optional[Tensor],
        eps: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of the IgnoreParentsMechanism.

        Args:
            parent_values (Optional[Tensor]): The values of the parent nodes. Ignored in this mechanism.
            noise (Optional[Tensor]): The noise term. Ignored in this mechanism.

        Returns:
            Tensor: The output tensor, which is identical to the input noise tensor.
        """
        if eps is None:
            raise ValueError("Noise must be provided for IgnoreParentsMechanism.")
        return eps