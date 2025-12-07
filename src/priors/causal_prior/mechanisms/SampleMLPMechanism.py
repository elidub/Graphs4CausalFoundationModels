from __future__ import annotations
from typing import List, Optional, Tuple, Literal
import torch
from torch import nn, Tensor
from priors.causal_prior.mechanisms.BaseMechanism import BaseMechanism
from priors.causal_prior.mechanisms.RandomActivation import RandomActivation  # your existing class

# ------------ SampledMechanism ------------
class SampleMLPMechanism(BaseMechanism):
    """
    Randomly-sampled MLP mechanism with a fixed (sampled) activation module.

    y =
      - if activation_mode == 'pre' : act( f(parents) ) + eps
      - if activation_mode == 'post': act( f(parents) + eps )

    Args
    ----
    input_dim : int
        Number of parent features D (can be 0).
    node_shape : tuple, default ()
        Output per-sample shape. Empty => scalar.
    nonlins : str
        Activation function type. Options include:
        - "mixed": combines multiple sampling strategies
        - "tabicl": uses TabICL activation functions (diverse set including RBF, sine, random functions, etc.)
        - "sophisticated_sampling_1": complex sampling strategy
        - "tanh", "sin", "neg", "id", "elu": specific activation functions
        See RandomActivation for all options.
    num_hidden_layers : int
        Fixed number of hidden layers.
        hidden_dim : int, default 64
        Width of hidden layers.
    activation_mode : {'pre','post','mixed_in_noise'}, default 'pre'
        'pre': act(f(parents)) + eps
        'post': act(f(parents) + eps)
        'mixed_in_noise': concatenate eps to parents and feed through MLP (no additive noise on output)
    generator : torch.Generator, optional
        RNG for reproducibility of activation sampling.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        node_shape: Tuple[int, ...] = (),
        nonlins: str = "mixed",
        num_hidden_layers: int = 2,
        hidden_dim: int = 64,
        activation_mode: Literal["pre", "post", "mixed_in_noise"] = "pre",
        generator: Optional[torch.Generator] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(input_dim=input_dim, node_shape=node_shape, name=name)
        # Normalize alias
        if activation_mode == "mixed_in":
            activation_mode = "mixed_in_noise"
        self.activation_mode = activation_mode
        self.gen = generator

        out_dim = int(torch.tensor(node_shape).prod().item()) if node_shape else 1
        D = max(1, input_dim)  # allow D=0 via learned token
        # If we're mixing noise into the input, the effective input dimension doubles (parents + noise)
        effective_input_dim = D * 2 if self.activation_mode == "mixed_in_noise" else D

        # use fixed number of hidden layers
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")
        n_hidden = num_hidden_layers

        layers: List[nn.Module] = []
        if n_hidden == 0:
            layers.append(nn.Linear(effective_input_dim, out_dim, bias=False))
            act = RandomActivation(nonlins=nonlins, generator=self.gen)
            layers.append(act)
            
        else:
            d = effective_input_dim
            act = RandomActivation(nonlins=nonlins, generator=self.gen)
            for _ in range(n_hidden):
                layers += [nn.Linear(d, hidden_dim, bias=False), act]
                d = hidden_dim
            layers.append(nn.Linear(d, out_dim, bias=False))
        self.net = nn.Sequential(*layers)

        # no-parent support
        self._zero_parent_token = nn.Parameter(torch.zeros(1, 1), requires_grad=(input_dim == 0))

        # if we need a separate activation for 'post' but n_hidden==0, build one
        self.post_activation = RandomActivation(nonlins=nonlins, generator=self.gen) if self.activation_mode == "post" else None

        # Tiny runtime indicator to confirm mixed-in-noise wiring
        if self.activation_mode == "mixed_in_noise":
            print("[SampleMLPMechanism] Using mixed-in-noise: concatenating eps to input features.")

    def _forward(self, parents: Tensor, eps: Optional[Tensor] = None) -> Tensor:
        B = parents.shape[0]
        x = parents if self.input_dim > 0 else self._zero_parent_token.expand(B, 1)

        # Handle mixed-in-noise by concatenating eps to the input features
        if self.activation_mode == "mixed_in_noise":
            # eps is expected to be shaped like the parents (B, D). If not provided, use zeros.
            if eps is None:
                eps_in = torch.zeros_like(x)
            else:
                # If eps has different shape (e.g., output-shaped), try to adapt: use a zeros_like(x) fallback
                eps_in = eps
                if eps_in.dim() != x.dim() or eps_in.shape != x.shape:
                    try:
                        eps_in = torch.zeros_like(x)
                    except Exception:
                        eps_in = torch.zeros_like(x)
            x_aug = torch.cat([x, eps_in], dim=-1)
            out = self.net(x_aug)  # (B, out_dim)
        else:
            out = self.net(x)  # (B, out_dim)
        if self.node_shape:
            out = out.view(B, *self.node_shape)

        if eps is None:
            eps = torch.zeros_like(out)

        if self.activation_mode == "pre":
            out = out + eps
            out = out.reshape(B, -1)
            return out 
        elif self.activation_mode == "post":
            act = self.post_activation if self.post_activation is not None else (lambda t: t)
            out = act(out + eps)
            return out.reshape(B, *self.node_shape)  # reshape back to original shape
        else:  # mixed_in_noise already applied at input
            return out.reshape(B, *self.node_shape) if self.node_shape else out
