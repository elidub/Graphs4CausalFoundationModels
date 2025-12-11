"""Resampling distribution.

Provides sampling *without replacement* from a provided 1D tensor of values.
When the reservoir is exhausted:
  - If ``shuffle_each_epoch=True`` (default) it reshuffles and continues (multiple epochs).
  - Otherwise it raises ``StopIteration`` on further sampling attempts.

Implements the ``Distribution`` interface (see ``DistributionInterface.py``).

Typical usage:
	dist = ResamplingDist(data_tensor)  # data_tensor shape (N,)
	x = dist.sample_one()               # scalar ()
	xs = dist.sample_n(10)              # shape (10,)

Edge cases handled:
  - Empty data raises ``ValueError``.
  - Requesting more than remaining samples in current epoch will transparently
	start a new epoch if shuffling is enabled.
  - ``sample_n`` with n==0 returns an empty tensor of shape (0,).
"""

from __future__ import annotations

from typing import Tuple
import torch
from torch import Tensor

from .DistributionInterface import Distribution


class ResamplingDist(Distribution):
	"""Resample from a fixed set of scalar values without replacement.

	Parameters
	----------
	data : Tensor | list | tuple
		Source values. Will be converted to a 1D tensor (flattened if necessary).
	device, dtype, generator : see ``Distribution`` base class.
	shuffle_each_epoch : bool, default True
		If True, after all items have been drawn once, a new random permutation
		is generated and sampling continues. If False, further sampling after
		exhaustion raises ``StopIteration``.
	"""

	def __init__(
		self,
		data: Tensor | list | tuple,
		*,
		device: torch.device | str = "cpu",
		dtype: torch.dtype = torch.float32,
		generator: torch.Generator | None = None,
		shuffle_each_epoch: bool = True,
	) -> None:
		super().__init__(device=device, dtype=dtype, generator=generator)

		data_tensor = torch.as_tensor(data, dtype=dtype, device=self.device)
		if data_tensor.numel() == 0:
			raise ValueError("ResamplingDist requires a non-empty data tensor.")
		if data_tensor.dim() != 1:
			data_tensor = data_tensor.flatten()
		self.data = data_tensor  # shape (N,)
		self.shuffle_each_epoch = shuffle_each_epoch
		self._N = self.data.shape[0]
		self._i = 0  # cursor into permutation
		self._perm = torch.empty(0, dtype=torch.long, device=self.device)
		self._reshuffle()  # initialize permutation

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------
	def _reshuffle(self) -> None:
		if self.shuffle_each_epoch:
			self._perm = torch.randperm(
				self._N, generator=self.generator, device=self.device
			)
		else:
			# deterministic single epoch order
			self._perm = torch.arange(self._N, device=self.device)
		self._i = 0

	def reset(self) -> None:
		"""Manual reset to start a fresh epoch (reshuffles if enabled)."""
		self._reshuffle()

	def __len__(self) -> int:  # size of reservoir
		return self._N

	# ------------------------------------------------------------------
	# Distribution interface implementations
	# ------------------------------------------------------------------
	def sample_one(self) -> Tensor:
		if self._i >= self._N:  # exhausted current epoch
			if not self.shuffle_each_epoch:
				raise StopIteration("ResamplingDist exhausted (no reshuffle).")
			self._reshuffle()
		idx = self._perm[self._i]
		self._i += 1
		# Returning a scalar tensor ()
		return self.data[idx]

	def sample_n(self, n: int) -> Tensor:
		if n < 0:
			raise ValueError("n must be non-negative.")
		if n == 0:
			return torch.empty(0, dtype=self.dtype, device=self.device)
		out_parts: list[Tensor] = []
		remaining_request = n
		while remaining_request > 0:
			remaining_epoch = self._N - self._i
			if remaining_epoch == 0:
				if not self.shuffle_each_epoch:
					raise StopIteration(
						"ResamplingDist exhausted during sample_n (no reshuffle)."
					)
				self._reshuffle()
				remaining_epoch = self._N
			take = min(remaining_request, remaining_epoch)
			idx_slice = self._perm[self._i : self._i + take]
			out_parts.append(self.data[idx_slice])
			self._i += take
			remaining_request -= take
		return torch.cat(out_parts, dim=0)

	def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
		# Flatten sampling then reshape. Uses sample_n for epoch logic.
		total = 1
		for s in shape:
			total *= int(s)
		if total == 0:
			return torch.empty(*shape, dtype=self.dtype, device=self.device)
		return self.sample_n(total).reshape(shape)

	# ------------------------------------------------------------------
	# Convenience / representation
	# ------------------------------------------------------------------
	def __repr__(self) -> str:  # pragma: no cover (simple)
		return (
			f"ResamplingDist(N={self._N}, shuffle_each_epoch={self.shuffle_each_epoch},"
			f" device={self.device}, dtype={self.dtype})"
		)
