from __future__ import annotations
from typing import List, Tuple, Union
import torch


class BatchSplitCollator:
	"""
	Custom collator that performs a per-batch train/test split based on a
	sampled number of test samples, then slices each item in the batch to
	match the sampled split.

	This avoids zero-padding by keeping only the appropriate number of
	train/test samples for each dataset element in the batch.

	Supports both dataset item formats:
	- Observational (4-tuple): (X_train, Y_train, X_test, Y_test)
	- Interventional (6-tuple): (X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv)

	Args:
		max_number_samples_per_dataset: Maximum total samples per dataset element (train + test).
		max_number_train_samples_per_dataset: Upper cap for train samples available per element.
		max_number_test_samples_per_dataset: Upper cap for test samples available per element.
		n_test_samples_distribution: A torch.distributions.Distribution or int specifying
			how many test samples to use per batch. If a distribution is provided, it will
			be sampled once per batch.
		device: Optional device for potential distribution sampling context.
	"""

	def __init__(
		self,
		max_number_samples_per_dataset: int,
		max_number_train_samples_per_dataset: int,
		max_number_test_samples_per_dataset: int,
		n_test_samples_distribution: Union[int, torch.distributions.Distribution],
		device: Union[torch.device, str, None] = None,
	):
		self.max_total = int(max_number_samples_per_dataset)
		self.max_train_cap = int(max_number_train_samples_per_dataset)
		self.max_test_cap = int(max_number_test_samples_per_dataset)
		self.n_test_dist = n_test_samples_distribution
		self.device = torch.device(device) if isinstance(device, str) else device

	def _sample_n_test(self) -> int:
		# Sample once per batch
		if isinstance(self.n_test_dist, torch.distributions.Distribution):
			val = self.n_test_dist.sample()
			# Handle tensor return
			n_test = int(val.item()) if hasattr(val, 'item') else int(val)
		else:
			n_test = int(self.n_test_dist)

		# Cap by provided maxima and total
		n_test = max(0, min(n_test, self.max_test_cap, self.max_total))
		return n_test

	def __call__(self, batch: List[Union[Tuple[torch.Tensor, ...], list]]):
		# Determine per-batch split
		n_test = self._sample_n_test()
		n_train = max(0, self.max_total - n_test)

		# Helper to slice first dimension safely
		def _slice_first_dim(t: torch.Tensor, n: int) -> torch.Tensor:
			return t[: min(n, t.shape[0])]

		# Observational format
		if len(batch) > 0 and len(batch[0]) == 4:
			X_tr_list, Y_tr_list, X_te_list, Y_te_list = [], [], [], []
			for item in batch:
				X_train, Y_train, X_test, Y_test = item
				# Slice to requested counts (bounded by available per-item lengths)
				X_train_s = _slice_first_dim(X_train, min(n_train, self.max_train_cap))
				Y_train_s = _slice_first_dim(Y_train, min(n_train, self.max_train_cap))
				X_test_s = _slice_first_dim(X_test, min(n_test, self.max_test_cap))
				Y_test_s = _slice_first_dim(Y_test, min(n_test, self.max_test_cap))

				X_tr_list.append(X_train_s)
				Y_tr_list.append(Y_train_s)
				X_te_list.append(X_test_s)
				Y_te_list.append(Y_test_s)

			# Stack into batched tensors: (B, N, ...)
			X_tr = torch.stack(X_tr_list, dim=0)
			Y_tr = torch.stack(Y_tr_list, dim=0)
			X_te = torch.stack(X_te_list, dim=0)
			Y_te = torch.stack(Y_te_list, dim=0)
			return [X_tr, Y_tr, X_te, Y_te]

		# Interventional format
		if len(batch) > 0 and len(batch[0]) == 6:
			X_obs_list, T_obs_list, Y_obs_list = [], [], []
			X_intv_list, T_intv_list, Y_intv_list = [], [], []
			for item in batch:
				X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv = item
				# Slice per split
				X_obs_s = _slice_first_dim(X_obs, min(n_train, self.max_train_cap))
				T_obs_s = _slice_first_dim(T_obs, min(n_train, self.max_train_cap))
				Y_obs_s = _slice_first_dim(Y_obs, min(n_train, self.max_train_cap))

				X_intv_s = _slice_first_dim(X_intv, min(n_test, self.max_test_cap))
				T_intv_s = _slice_first_dim(T_intv, min(n_test, self.max_test_cap))
				Y_intv_s = _slice_first_dim(Y_intv, min(n_test, self.max_test_cap))

				X_obs_list.append(X_obs_s)
				T_obs_list.append(T_obs_s)
				Y_obs_list.append(Y_obs_s)

				X_intv_list.append(X_intv_s)
				T_intv_list.append(T_intv_s)
				Y_intv_list.append(Y_intv_s)

			# Stack into batched tensors
			X_obs_b = torch.stack(X_obs_list, dim=0)
			T_obs_b = torch.stack(T_obs_list, dim=0)
			Y_obs_b = torch.stack(Y_obs_list, dim=0)
			X_intv_b = torch.stack(X_intv_list, dim=0)
			T_intv_b = torch.stack(T_intv_list, dim=0)
			Y_intv_b = torch.stack(Y_intv_list, dim=0)
			return [X_obs_b, T_obs_b, Y_obs_b, X_intv_b, T_intv_b, Y_intv_b]

		# Fallback: default PyTorch stacking (should not happen in our pipeline)
		return torch.utils.data.default_collate(batch)


if __name__ == "__main__":
	# Simple self-test for the collator
	import torch
	from torch.distributions import Uniform

	# Config-like params
	max_total = 100
	max_train_cap = 80
	max_test_cap = 80
	n_test_dist = Uniform(low=20, high=60)  # sample a float; we'll cast to int

	collator = BatchSplitCollator(
		max_number_samples_per_dataset=max_total,
		max_number_train_samples_per_dataset=max_train_cap,
		max_number_test_samples_per_dataset=max_test_cap,
		n_test_samples_distribution=n_test_dist,
	)

	# Build a fake observational batch of size B=3
	B = 3
	N_avail = 90  # available train samples per item
	M_avail = 90  # available test samples per item
	F = 5

	obs_batch = []
	for _ in range(B):
		X_train = torch.randn(N_avail, F)
		Y_train = torch.randn(N_avail, 1)
		X_test = torch.randn(M_avail, F)
		Y_test = torch.randn(M_avail, 1)
		obs_batch.append((X_train, Y_train, X_test, Y_test))

	out_obs = collator(obs_batch)
	X_tr, Y_tr, X_te, Y_te = out_obs
	print("[Observational] Shapes:", X_tr.shape, Y_tr.shape, X_te.shape, Y_te.shape)

	# Build a fake interventional batch of size B=2
	B2 = 2
	N2_avail = 85
	M2_avail = 85
	F2 = 7

	intv_batch = []
	for _ in range(B2):
		X_obs = torch.randn(N2_avail, F2)
		T_obs = torch.randn(N2_avail, 1)
		Y_obs = torch.randn(N2_avail, 1)
		X_intv = torch.randn(M2_avail, F2)
		T_intv = torch.randn(M2_avail, 1)
		Y_intv = torch.randn(M2_avail, 1)
		intv_batch.append((X_obs, T_obs, Y_obs, X_intv, T_intv, Y_intv))

	out_intv = collator(intv_batch)
	Xo, To, Yo, Xi, Ti, Yi = out_intv
	print("[Interventional] Shapes:", Xo.shape, To.shape, Yo.shape, Xi.shape, Ti.shape, Yi.shape)


