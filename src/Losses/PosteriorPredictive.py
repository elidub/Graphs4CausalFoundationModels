"""
Interface to parametrize a posterior predictive distribution for num_points testing points. 
The distribution is parametrized by num_params parameters for each of the num_points testing points.
"""

from abc import ABC, abstractmethod
import torch  

class PosteriorPredictive(ABC):

    @abstractmethod
    def average_log_prob(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the average log-probability of the posterior predictive distribution at the given points.

        Args:
            pred (torch.Tensor): Parameters of the posterior predictive distribution. Shape: (batch_size, num_points, num_params)
            y (torch.Tensor): Points at which to evaluate the log-probability. Shape: (batch_size, num_points)

        Returns:
            torch.Tensor: Log-probabilities at the given points. Shape: (batch_size,)
        """
        pass

    def mode(pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the mode of the posterior predictive distribution.

        Args:
            pred (torch.Tensor): Parameters of the posterior predictive distribution. Shape: (batch_size, num_points, num_params)

        Returns:
            torch.Tensor: Mode of the distribution. Shape: (batch_size, num_points)
        """
        raise NotImplementedError("Mode computation is not implemented for this posterior predictive distribution.")
    
    def mean(pred: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean of the posterior predictive distribution.

        Args:
            pred (torch.Tensor): Parameters of the posterior predictive distribution. Shape: (batch_size, num_points, num_params)

        Returns:
            torch.Tensor: Mean of the distribution. Shape: (batch_size, num_points)
        """
        raise NotImplementedError("Mean computation is not implemented for this posterior predictive distribution.")
    
    def sample(self, pred: torch.Tensor, num_samples: int) -> torch.Tensor:
        """
        Generate samples from the posterior predictive distribution.

        Args:
            pred (torch.Tensor): Parameters of the posterior predictive distribution. Shape: (batch_size, num_points, num_params)
            num_samples (int): Number of samples to generate.

        Returns:
            torch.Tensor: Samples from the distribution. Shape: (batch_size, num_samples, num_points)
        """
        raise NotImplementedError("Sampling is not implemented for this posterior predictive distribution.")