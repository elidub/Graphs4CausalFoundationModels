from __future__ import annotations
from typing import List, Optional, Tuple, Sequence
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Import TabICL activations
try:
    from priors.causal_prior.mechanisms.TabICL_Activations import get_activations
    _HAS_TABICL = True
except ImportError:
    _HAS_TABICL = False



# --- tiny helpers for RNG on a device ---
def _randint(n: int, gen: Optional[torch.Generator], device: torch.device) -> int:
    return int(torch.randint(n, (1,), generator=gen, device=device).item())

def _rand(shape, gen: Optional[torch.Generator], device: torch.device) -> Tensor:
    return torch.rand(shape, generator=gen, device=device)

class ToModule(nn.Module):
    def __init__(self, f): super().__init__(); self.f = f
    def forward(self, x: Tensor) -> Tensor: return self.f(x)

class ApplyBothAndAverage(nn.Module):
    """Return (f1(x) + f2(x)) / 2 on the SAME x."""
    def __init__(self, f1: nn.Module, f2: nn.Module):
        super().__init__()
        self.f1, self.f2 = f1, f2
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (self.f1(x) + self.f2(x))

class WeightedSum(nn.Module):
    """Sum_i w_i f_i(x) on the SAME x. Weights are fixed and normalized."""
    def __init__(self, funcs: Sequence[nn.Module], weights: Tensor):
        super().__init__()
        if len(funcs) != int(weights.numel()):
            raise ValueError("funcs and weights length must match.")
        self.funcs = nn.ModuleList(funcs)
        w = weights.reshape(-1).to(dtype=torch.float32)
        w = w / (w.sum() + 1e-12)
        self.register_buffer("w", w)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.zeros_like(x)
        # apply each nonlinearity to the SAME input and weight-sum
        for wi, fi in zip(self.w, self.funcs):
            out = out + wi * fi(x)
        return out

class LayerNormStatic(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        dims = x.shape[1:]
        # if input is 1d (scalar per batch element), normalize it
        if len(dims) == 0:
            mean = x.mean()
            std = x.std(unbiased=False)
            return (x - mean) / (std + 1e-12)
        # no affine parameters
        return x if not dims else F.layer_norm(x, dims, weight=None, bias=None)

class Affine(nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale, self.shift = float(scale), float(shift)
    def forward(self, x: Tensor) -> Tensor:
        return self.scale * (x + self.shift)

class RandomActivation(nn.Module):
    """
    Randomly activate different non-linearities during training.
    options for nonlines:
        mixed: combines multiple sampling strategies
        sophisticated_sampling_1: a more complex sampling strategy
        sophisticated_sampling_1_normalization
        sophisticated_sampling_1_normalization_rescaled
        tabicl: uses TabICL activation functions (diverse set including RBF, sine, random functions, etc.)
        tanh: hyperbolic tangent
        sin: sine function
        neg: negation
        id: identity
        elu: exponential linear unit
        summed: average of two randomly selected simple activations
    """


    def __init__(
        self,
        nonlins: str = "mixed",
        clamp: Tuple[float, float] = (-1000.0, 1000.0),
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.clamp = clamp
        self.gen = generator
        self.register_buffer("_tok", torch.empty(0) if device is None else torch.empty(0, device=device))
        self._module = self._sample(nonlins)

    def forward(self, x: Tensor) -> Tensor:
        y = self._module(x)
        if self.clamp is not None:
            y = torch.clamp(y, self.clamp[0], self.clamp[1])
        return y

    def _simple_pool(self) -> List[nn.Module]:
        return [ToModule(lambda x: x*x), ToModule(F.relu), ToModule(torch.tanh), ToModule(lambda x: x)]

    def _rich_pool(self) -> List[nn.Module]:
        return [
            nn.ReLU(), nn.ReLU6(), nn.SELU(), nn.SiLU(),
            nn.Softplus(), nn.Hardtanh(),
            ToModule(torch.sign), ToModule(torch.sin),
            ToModule(lambda x: torch.exp(-(x**2))),
            ToModule(torch.exp),
            ToModule(lambda x: torch.sqrt(torch.clamp(x.abs(), min=0.0))),
            ToModule(lambda x: (x.abs() < 1).to(x.dtype)),
            ToModule(lambda x: x**2),
            ToModule(lambda x: x.abs()),
        ]

    def _sample(self, kind: str) -> nn.Module:
        dev = self._tok.device

        def summed():
            pool = self._simple_pool()
            ids = torch.randperm(len(pool), generator=self.gen, device=dev)[:2]
            return ApplyBothAndAverage(pool[int(ids[0])], pool[int(ids[1])])

        def soph1():
            pool = self._rich_pool()
            r = float(_rand((), self.gen, dev))
            if r < 1/3:
                return pool[_randint(len(pool), self.gen, dev)]
            elif r < 2/3:
                ids = torch.randperm(len(pool), generator=self.gen, device=dev)[:2]
                w = _rand((2,), self.gen, dev); w = w / w.sum()
                return WeightedSum([pool[int(ids[0])], pool[int(ids[1])]], w)
            else:
                ids = torch.randperm(len(pool), generator=self.gen, device=dev)[:3]
                w = _rand((3,), self.gen, dev); w = w / w.sum()
                return WeightedSum([pool[int(ids[0])], pool[int(ids[1])], pool[int(ids[2])]], w)

        def soph1_norm():
            return nn.Sequential(soph1(), LayerNormStatic())

        def soph1_rescale_norm():
            a = torch.randn((), generator=self.gen, device=dev)
            b = torch.randn((), generator=self.gen, device=dev)
            return nn.Sequential(soph1(), LayerNormStatic(), Affine(float(torch.exp(2*a)), float(b)))

        def tabicl_activation():
            """Sample from TabICL activation functions."""
            if not _HAS_TABICL:
                raise ImportError("TabICL_Activations not available. Please ensure TabICL_Activations.py is in the mechanisms folder.")
            
            # Get the TabICL activations with various configurations
            activations = get_activations(random=True, scale=True, diverse=True)
            
            # Randomly select one activation function
            idx = _randint(len(activations), self.gen, dev)
            activation_factory = activations[idx]
            
            # Instantiate the activation (some are classes, some are factories)
            if hasattr(activation_factory, '__call__') and not isinstance(activation_factory, type):
                # It's a factory function
                return activation_factory()
            else:
                # It's a class
                return activation_factory()
        
        # routing
        if kind in ("mixed", "post"):  # legacy names
            pool = self._simple_pool()
            return pool[_randint(len(pool), self.gen, dev)]
        if kind == "tabicl":
            return tabicl_activation()
        if kind == "tanh":
            return ToModule(torch.tanh)
        if kind == "sin":
            return ToModule(torch.sin)
        if kind == "neg":
            return ToModule(lambda x: -x)
        if kind == "id":
            return ToModule(lambda x: x)
        if kind == "elu":
            return ToModule(F.elu)
        if kind == "summed":
            return summed()
        if kind == "sophisticated_sampling_1":
            return soph1()
        if kind == "sophisticated_sampling_1_normalization":
            return soph1_norm()
        if kind == "sophisticated_sampling_1_rescaling_normalization":
            return soph1_rescale_norm()
        # default
        pool = self._simple_pool()
        return pool[_randint(len(pool), self.gen, dev)]


class ToModule(nn.Module):
    def __init__(self, func): super().__init__(); self.func = func
    def forward(self, x: Tensor) -> Tensor: return self.func(x)

class _Avg2(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # expects Sequential(ToModule(f1), ToModule(f2), _Avg2())
        # compute f2(f1(x)) then average input & output? In original they averaged f1 + f2 on SAME x.
        # We'll implement (f1(x)+f2(x))/2 by capturing via closure is complex; simpler: re-build below if needed.
        # For now treat input already as f1(x)+f2(x); keep identity.
        return torch.clamp(x / 2, -1000.0, 1000.0)

class _WeightedSum(nn.Module):
    def __init__(self, funcs: List, weights: Tensor) -> None:
        super().__init__()
        self.funcs = nn.ModuleList([ToModule(f) if not isinstance(f, nn.Module) else f for f in funcs])
        self.register_buffer("w", weights.reshape(-1))

    def forward(self, x: Tensor) -> Tensor:
        out = 0.0
        for wi, fi in zip(self.w, self.funcs):
            out = out + wi * fi(x)
        return torch.clamp(out, -1000.0, 1000.0)

class _LayerNormDynamic(nn.Module):
    """Applies LayerNorm over all non-batch dims of the input tensor without affine parameters."""
    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape[1:]
        if len(shape) == 0:
            return x  # scalar per batch
        return F.layer_norm(x, shape, weight=None, bias=None)

class _Affine(nn.Module):
    def __init__(self, scale: float, shift: float) -> None:
        super().__init__()
        self.scale = float(scale)
        self.shift = float(shift)
    def forward(self, x: Tensor) -> Tensor:
        return self.scale * (x + self.shift)