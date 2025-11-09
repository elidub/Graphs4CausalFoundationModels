from __future__ import annotations
from typing import Any, Optional, Callable, Union
import torch
from torch.utils.data import Dataset

# Keep the type hint for compatibility, but we won't rely on concrete class features
try:
    from priors.causal_prior.scm.SCMSampler import SCMSampler  # type: ignore
except Exception:  # pragma: no cover - sampler may be an inline/alternative implementation
    SCMSampler = Any  # fallback for typing

from priordata_processing.BasicProcessing import BasicProcessing


class CurriculumPurelyObservationalDataset(Dataset):
    """
    Curriculum version of PurelyObservationalDataset with a notion of time t in [0, 1].

    Assumptions:
    - We map dataset index to time via t = idx / (size - 1). Thus t=0 -> first index, t=1 -> last index.
    - Time t is forwarded to:
        * SCM sampler: we attempt sampler.sample(seed=..., t=t) and fallback to sampler.sample(seed=...)
        * Train/test sample count distributions: if they accept t, we pass it (supports torch distributions,
          custom samplers with .sample, FixedSampler, or callables).
        * Max counts and features: may be ints, torch distributions, custom samplers, or callables; we sample with t
          per item.
        * BasicProcessing: if the provided processor is callable or has .with_time(...), we pass t and the sampled
          sizes; otherwise we construct a BasicProcessing mirroring PurelyObservationalDataset's behavior.

    Parameters mirror PurelyObservationalDataset, but allow time-aware inputs:
    - number_train_samples_per_dataset_distribution: Union[torch.distributions.Distribution, FixedSampler, Callable[[float], int]]
    - number_test_samples_per_dataset_distribution: same as above
    - max_number_train_samples / max_number_test_samples / max_number_features: Union[int, torch.distributions.Distribution,
      FixedSampler, Callable[[float], int]]
    - scm_sampler: any object exposing sample(seed, t?) -> SCM
    - priordata_processor: BasicProcessing or a callable factory or an object with .with_time(...)
    """

    def __init__(
        self,
        scm_sampler: Any,
        priordata_processor: Union[BasicProcessing, Callable[..., BasicProcessing], Any],
        number_train_samples_per_dataset_distribution: Any,
        number_test_samples_per_dataset_distribution: Any,
        size: int = 10_000,
        max_number_train_samples: Any = 500,
        max_number_test_samples: Any = 500,
        max_number_features: Any = 100,
        seed: int = 123,
    ):
        self.scm_sampler = scm_sampler
        self.priodata_processor = priordata_processor
        self.number_train_samples_per_dataset_distribution = (
            number_train_samples_per_dataset_distribution
        )
        self.number_test_samples_per_dataset_distribution = (
            number_test_samples_per_dataset_distribution
        )

        self.size = size
        self.max_number_train_samples_spec = max_number_train_samples
        self.max_number_test_samples_spec = max_number_test_samples
        self.max_number_features_spec = max_number_features
        self.seed = seed

    def __len__(self):
        return self.size

    def _time_for_index(self, idx: int) -> float:
        if self.size <= 1:
            return 1.0
        return float(idx) / float(self.size - 1)

    @staticmethod
    def _sample_time_aware(spec: Any, t: float, generator: Optional[torch.Generator] = None) -> Any:
        """
        Sample or resolve a value from a time-aware spec.

        Supported spec types:
        - int/float: returned as-is
        - torch.distributions.Distribution: calls .sample() (no t support there)
        - objects with .sample(...): tries sample(t=t), then sample(generator), then sample()
        - callable: tries fn(t), then fn()
        - torch.Tensor scalar: converts to Python scalar
        """
        # Primitive types
        if isinstance(spec, (int, float)):
            return spec
        if torch.is_tensor(spec):
            return spec.item() if spec.numel() == 1 else spec

        # torch Distribution
        if isinstance(spec, torch.distributions.Distribution):
            out = spec.sample()
            if torch.is_tensor(out):
                out = out.item() if out.numel() == 1 else out
            return out

        # Sampler-like with .sample
        if hasattr(spec, "sample") and callable(spec.sample):
            sample_fn = spec.sample
            try:
                return sample_fn(t=t)
            except TypeError:
                pass
            try:
                if generator is not None:
                    return sample_fn(generator)
            except TypeError:
                pass
            out = sample_fn()
            if torch.is_tensor(out):
                out = out.item() if hasattr(out, "numel") and out.numel() == 1 else out
            return out

        # Callable
        if callable(spec):
            try:
                return spec(t)
            except TypeError:
                return spec()

        # Fallback: return as-is (caller may handle)
        return spec

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")

        t = self._time_for_index(idx)
        seed = self.seed + idx
        torch.manual_seed(seed)  # Ensure reproducible sampling per item

        # SCM: try passing time, then fallback
        scm = None
        if hasattr(self.scm_sampler, "sample") and callable(self.scm_sampler.sample):
            try:
                scm = self.scm_sampler.sample(seed=seed, t=t)
            except TypeError:
                scm = self.scm_sampler.sample(seed=seed)
        else:
            raise TypeError("scm_sampler must provide a callable .sample method")

        # Resolve max specs possibly dependent on time
        max_n_train_samples = int(self._sample_time_aware(self.max_number_train_samples_spec, t))
        max_n_test_samples = int(self._sample_time_aware(self.max_number_test_samples_spec, t))
        max_features = int(self._sample_time_aware(self.max_number_features_spec, t))

        # Sample train/test sizes w.r.t. time
        number_train_samples = self._sample_time_aware(
            self.number_train_samples_per_dataset_distribution, t
        )
        number_test_samples = self._sample_time_aware(
            self.number_test_samples_per_dataset_distribution, t
        )

        # Convert to ints if tensors/scalars
        if hasattr(number_train_samples, "item"):
            number_train_samples = int(number_train_samples.item())
        else:
            number_train_samples = int(number_train_samples)

        if hasattr(number_test_samples, "item"):
            number_test_samples = int(number_test_samples.item())
        else:
            number_test_samples = int(number_test_samples)

        # Total samples to generate from SCM
        total_samples = number_train_samples + number_test_samples

        # Generate data from SCM
        scm.sample_exogenous(num_samples=total_samples)
        scm.sample_endogenous_noise(num_samples=total_samples)
        dataset = scm.propagate(num_samples=total_samples)

        # Create a time-aware BasicProcessing instance
        actual_processor: BasicProcessing

        # Case 1: a factory/callable provided by user
        if callable(self.priodata_processor):
            # Try passing rich context first, then fallback to fewer args
            try:
                actual_processor = self.priodata_processor(
                    t=t,
                    n_features=max_features,
                    max_n_features=max_features,
                    n_train_samples=number_train_samples,
                    max_n_train_samples=max_n_train_samples,
                    n_test_samples=number_test_samples,
                    max_n_test_samples=max_n_test_samples,
                )
            except TypeError:
                actual_processor = self.priodata_processor(t=t)

        # Case 2: object exposes .with_time(...) to derive a processor for this t
        elif hasattr(self.priodata_processor, "with_time") and callable(
            getattr(self.priodata_processor, "with_time")
        ):
            try:
                actual_processor = self.priodata_processor.with_time(
                    t=t,
                    n_train_samples=number_train_samples,
                    n_test_samples=number_test_samples,
                    max_n_train_samples=max_n_train_samples,
                    max_n_test_samples=max_n_test_samples,
                    max_n_features=max_features,
                )
            except TypeError:
                actual_processor = self.priodata_processor.with_time(t=t)

        # Case 3: fallback to constructing a BasicProcessing mirroring the non-curriculum dataset
        else:
            base = self.priodata_processor
            actual_processor = BasicProcessing(
                n_features=getattr(base, "n_features", max_features),
                max_n_features=max_features,
                n_train_samples=number_train_samples,
                max_n_train_samples=max_n_train_samples,
                n_test_samples=number_test_samples,
                max_n_test_samples=max_n_test_samples,
                dropout_prob=getattr(base, "dropout_prob", 0.0),
                target_feature=getattr(base, "target_feature", None),
                random_seed=getattr(base, "random_seed", None),
                negative_one_one_scaling=getattr(base, "negative_one_one_scaling", False),
                standardize=getattr(base, "standardize", False),
                yeo_johnson=getattr(base, "yeo_johnson", False),
                remove_outliers=getattr(base, "remove_outliers", False),
                outlier_quantile=getattr(base, "outlier_quantile", 0.0),
                shuffle_samples=getattr(base, "shuffle_samples", True),
                shuffle_features=getattr(base, "shuffle_features", True),
                y_clip_quantile=getattr(base, "y_clip_quantile", None),
                eps=getattr(base, "eps", 1e-8),
                device=getattr(base, "device", None),
                dtype=getattr(base, "dtype", None),
            )

        X_train, Y_train, X_test, Y_test = actual_processor.process(dataset)
        return X_train, Y_train, X_test, Y_test
