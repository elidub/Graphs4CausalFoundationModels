from __future__ import annotations
from abc import ABC, abstractmethod
from torch import Tensor


class PosteriorPredictive(ABC):
    @abstractmethod
    def average_log_prob(self, pred: Tensor, y: Tensor) -> Tensor:
        pass

    def mode(self, pred: Tensor) -> Tensor:
        raise NotImplementedError

    def mean(self, pred: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, pred: Tensor, num_samples: int) -> Tensor:
        raise NotImplementedError