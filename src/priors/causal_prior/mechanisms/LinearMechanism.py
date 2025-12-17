from __future__ import annotations
from typing import Optional, Tuple, Callable
import torch
from torch import nn, Tensor

from priors.causal_prior.mechanisms.BaseMechanism import BaseMechanism


class LinearMechanism(BaseMechanism):
    """
    Simple linear mechanism that:
    1. Optionally standardizes parent values (z-score normalization)
    2. Computes a weighted linear combination of parent values
    3. Applies a nonlinearity
    4. Adds noise
    
    Output: nonlinearity(weights @ standardized_parents) + noise
    
    Args:
        input_dim: Number of parent features (D)
        weights: List or tensor of weights for the linear combination.
                 Must have length equal to input_dim.
        nonlinearity: Function to apply after linear combination.
                     Can be:
                     - A string: 'tanh', 'relu', 'sigmoid', 'identity', 'leaky_relu', 'elu'
                     - A callable: Any function that takes a tensor and returns a tensor
                     Default: 'identity' (no nonlinearity)
        standardize: If True, standardizes inputs to zero mean and unit variance
                    before applying linear combination. Default: False
        node_shape: Output shape per sample, default () for scalar output
        name: Optional name for the mechanism
        
    Example:
        >>> # Linear mechanism with 3 parents: 0.5*x1 + 0.3*x2 - 0.2*x3, then tanh
        >>> mech = LinearMechanism(
        ...     input_dim=3,
        ...     weights=[0.5, 0.3, -0.2],
        ...     nonlinearity='tanh'
        ... )
        >>> parents = torch.randn(10, 3)  # 10 samples, 3 parents
        >>> noise = torch.randn(10)        # 10 noise samples
        >>> output = mech(parents, noise)  # shape: (10,)
    """

    def __init__(
        self,
        *,
        input_dim: int,
        weights: list | Tensor,
        nonlinearity: str | Callable[[Tensor], Tensor] = 'identity',
        standardize: bool = False,
        node_shape: Tuple[int, ...] = (),
        name: Optional[str] = None,
    ) -> None:
        super().__init__(input_dim=input_dim, node_shape=node_shape, name=name)
        
        # Store standardization flag
        self.standardize = standardize
        
        # Convert weights to tensor
        if isinstance(weights, list):
            weights = torch.tensor(weights, dtype=torch.float32)
        else:
            weights = torch.as_tensor(weights, dtype=torch.float32)
        
        # Validate weights shape
        if weights.numel() != input_dim:
            raise ValueError(
                f"weights must have {input_dim} elements (one per parent), "
                f"got {weights.numel()}"
            )
        
        # Register weights as a buffer (not a parameter, so it won't be trained)
        self.register_buffer('weights', weights.reshape(-1))
        
        # Set up nonlinearity
        if isinstance(nonlinearity, str):
            self.nonlinearity = self._get_nonlinearity(nonlinearity)
        elif callable(nonlinearity):
            self.nonlinearity = nonlinearity
        else:
            raise ValueError(
                f"nonlinearity must be a string or callable, got {type(nonlinearity)}"
            )
        
        self.nonlinearity_name = nonlinearity if isinstance(nonlinearity, str) else 'custom'
    
    @staticmethod
    def _get_nonlinearity(name: str) -> Callable[[Tensor], Tensor]:
        """Get nonlinearity function by name."""
        nonlinearities = {
            'identity': lambda x: x,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'relu': torch.relu,
            'leaky_relu': lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.1),
            'elu': lambda x: torch.nn.functional.elu(x),
            'softplus': torch.nn.functional.softplus,
            'abs': torch.abs,
            'square': lambda x: x ** 2,
            'sin': torch.sin,
            'cos': torch.cos,
        }
        
        if name not in nonlinearities:
            raise ValueError(
                f"Unknown nonlinearity '{name}'. "
                f"Available: {list(nonlinearities.keys())}"
            )
        
        return nonlinearities[name]
    
    def _forward(self, parents: Tensor, eps: Optional[Tensor] = None) -> Tensor:
        """
        Compute: nonlinearity(weights @ standardized_parents) + noise
        
        Args:
            parents: (B, D) where D == input_dim
            eps: (B,) or (B, *node_shape) or None
            
        Returns:
            (B,) or (B, *node_shape) depending on node_shape
        """
        B = parents.shape[0]
        
        # Standardize inputs if requested
        if self.standardize and self.input_dim > 0:
            # Compute mean and std along batch dimension
            mean = parents.mean(dim=0, keepdim=True)  # (1, D)
            std = parents.std(dim=0, keepdim=True, unbiased=False)  # (1, D)
            # Avoid division by zero
            std = torch.where(std > 1e-8, std, torch.ones_like(std))
            parents = (parents - mean) / std
        
        # Compute weighted sum: (B, D) @ (D,) -> (B,)
        linear_combination = torch.matmul(parents, self.weights)
        
        # Apply nonlinearity
        output = self.nonlinearity(linear_combination)
        
        # Add noise if provided
        if eps is not None:
            output = output + eps
        
        # Reshape if node_shape is not scalar
        if self.node_shape:
            output = output.reshape(B, *self.node_shape)
        
        return output
    
    def __repr__(self) -> str:
        weights_str = ', '.join([f'{w:.3f}' for w in self.weights[:5]])
        if len(self.weights) > 5:
            weights_str += ', ...'
        
        return (
            f"LinearMechanism("
            f"input_dim={self.input_dim}, "
            f"weights=[{weights_str}], "
            f"nonlinearity='{self.nonlinearity_name}', "
            f"standardize={self.standardize}, "
            f"node_shape={self.node_shape}"
            f"{f', name={self.name}' if self.name else ''})"
        )


if __name__ == "__main__":
    print("Testing LinearMechanism")
    print("=" * 80)
    
    # Test 1: Simple linear mechanism with identity (no nonlinearity)
    print("\n1. Linear mechanism with identity nonlinearity")
    mech1 = LinearMechanism(
        input_dim=3,
        weights=[1.0, 2.0, -0.5],
        nonlinearity='identity',
        name='simple_linear'
    )
    print(f"   Mechanism: {mech1}")
    
    parents1 = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]])  # (2, 3)
    noise1 = torch.tensor([0.1, -0.1])  # (2,)
    output1 = mech1(parents1, noise1)
    print(f"   Parents: {parents1}")
    print(f"   Expected linear: [1*1 + 2*2 + (-0.5)*3 = 3.5, 0.5*1 + 2*1 + (-0.5)*1.5 = 1.75]")
    print(f"   With noise [0.1, -0.1]: {output1}")
    print(f"   Output shape: {output1.shape}")
    
    # Test 2: Linear mechanism with tanh nonlinearity
    print("\n2. Linear mechanism with tanh nonlinearity")
    mech2 = LinearMechanism(
        input_dim=2,
        weights=[0.5, -1.0],
        nonlinearity='tanh'
    )
    print(f"   Mechanism: {mech2}")
    
    parents2 = torch.tensor([[2.0, 1.0], [0.0, 0.0], [-2.0, -1.0]])  # (3, 2)
    noise2 = torch.tensor([0.0, 0.0, 0.0])  # (3,)
    output2 = mech2(parents2, noise2)
    print(f"   Parents: {parents2}")
    print(f"   Linear combinations: [0.5*2 + (-1)*1 = 0, 0, 0.5*(-2) + (-1)*(-1) = 0]")
    print(f"   After tanh: {output2}")
    
    # Test 3: Different nonlinearities
    print("\n3. Testing different nonlinearities")
    test_parents = torch.tensor([[1.0, -1.0]])  # (1, 2)
    test_weights = [1.0, 1.0]
    
    for nonlin in ['identity', 'tanh', 'sigmoid', 'relu', 'leaky_relu', 'elu', 'abs', 'square']:
        mech = LinearMechanism(input_dim=2, weights=test_weights, nonlinearity=nonlin)
        output = mech(test_parents, eps=None)
        linear_val = 1.0 * 1.0 + 1.0 * (-1.0)  # = 0.0
        print(f"   {nonlin:15s}: input={linear_val:.2f} -> output={output.item():.4f}")
    
    # Test 4: With custom callable nonlinearity
    print("\n4. Custom callable nonlinearity (x^3)")
    def cubic(x: Tensor) -> Tensor:
        return x ** 3
    
    mech4 = LinearMechanism(
        input_dim=1,
        weights=[2.0],
        nonlinearity=cubic
    )
    print(f"   Mechanism: {mech4}")
    
    parents4 = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
    output4 = mech4(parents4, eps=None)
    print(f"   Parents: {parents4.squeeze()}")
    print(f"   After 2*x then x^3: {output4}")
    
    # Test 5: No parents (root node)
    print("\n5. Root node (no parents, only noise)")
    mech5 = LinearMechanism(
        input_dim=0,
        weights=[],
        nonlinearity='identity',
        name='root'
    )
    print(f"   Mechanism: {mech5}")
    
    parents5 = torch.empty(5, 0)  # (5, 0)
    noise5 = torch.randn(5)
    output5 = mech5(parents5, noise5)
    print(f"   Output (just noise): {output5}")
    
    # Test 6: Large number of parents
    print("\n6. Many parents (10 parents)")
    import numpy as np
    weights6 = np.random.randn(10).tolist()
    mech6 = LinearMechanism(
        input_dim=10,
        weights=weights6,
        nonlinearity='tanh'
    )
    print(f"   Mechanism: {mech6}")
    
    parents6 = torch.randn(3, 10)
    noise6 = torch.randn(3)
    output6 = mech6(parents6, noise6)
    print(f"   Output shape: {output6.shape}")
    print(f"   Output range: [{output6.min():.3f}, {output6.max():.3f}]")
    
    # Test 7: Error handling
    print("\n7. Error handling")
    try:
        bad_mech = LinearMechanism(input_dim=3, weights=[1.0, 2.0])  # Wrong number of weights
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    try:
        bad_mech = LinearMechanism(input_dim=2, weights=[1.0, 2.0], nonlinearity='invalid')
    except ValueError as e:
        print(f"   Caught expected error: {e}")
    
    # Test 8: Deterministic behavior (no noise)
    print("\n8. Deterministic behavior (eps=None)")
    mech8 = LinearMechanism(
        input_dim=2,
        weights=[1.0, 1.0],
        nonlinearity='relu'
    )
    parents8 = torch.tensor([[-1.0, 0.5], [1.0, 2.0]])
    output8 = mech8(parents8, eps=None)
    print(f"   Parents: {parents8}")
    print(f"   Linear: [-0.5, 3.0], after ReLU: {output8}")
    
    # Test 9: Standardization
    print("\n9. With standardization")
    mech9_no_std = LinearMechanism(
        input_dim=2,
        weights=[1.0, 1.0],
        nonlinearity='identity',
        standardize=False
    )
    mech9_std = LinearMechanism(
        input_dim=2,
        weights=[1.0, 1.0],
        nonlinearity='identity',
        standardize=True
    )
    
    # Create data with different scales
    parents9 = torch.tensor([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0]
    ])
    
    output9_no_std = mech9_no_std(parents9, eps=None)
    output9_std = mech9_std(parents9, eps=None)
    
    print(f"   Parents:\n{parents9}")
    print(f"   Without standardization: {output9_no_std}")
    print(f"   With standardization: {output9_std}")
    print(f"   (Standardization equalizes the contribution of each parent)")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
