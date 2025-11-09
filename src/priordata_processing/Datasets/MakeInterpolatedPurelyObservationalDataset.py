from __future__ import annotations
from typing import Dict, Any, Callable, Optional
import math

import torch
import torch.distributions as dist

from priordata_processing.Datasets.CurriculumPurelyObservationalDataset import CurriculumPurelyObservationalDataset
from priordata_processing.BasicProcessing import BasicProcessing
from priordata_processing.Datasets.MakePurelyObservationalDataset import (
    FixedSampler,
    TorchDistributionSampler,
    CategoricalSampler,
    DiscreteUniformSampler,
    SCMBuilder,
)


# Top-level helpers (picklable) to avoid local/closure functions
def _top_level_sample_pair(sampler_t0: Any, sampler_t1: Any, generator: torch.Generator):
    v0 = sampler_t0.sample(generator)
    v1 = sampler_t1.sample(generator)
    if torch.is_tensor(v0) and v0.numel() == 1:
        v0 = v0.item()
    if torch.is_tensor(v1) and v1.numel() == 1:
        v1 = v1.item()
    return v0, v1

def _choose_by_alpha(v0: Any, v1: Any, a: float) -> Any:
    if torch.rand(()) < a:
        return v1
    return v0


def _normalize_scm_param(name: str, value: Any) -> Any:
    """Normalize SCMBuilder parameters to expected types.
    - node shapes must be tuples of positive ints; allow list/int input.
    """
    if name in ("mlp_node_shape", "xgb_node_shape"):
        # Accept list/int/tuple and normalize to tuple of positive ints
        if isinstance(value, list):
            value = tuple(int(x) for x in value)
        elif isinstance(value, tuple):
            value = tuple(int(x) for x in value)
        elif isinstance(value, int):
            value = (int(value),)
        else:
            raise ValueError(f"{name} must be tuple of positive integers, got {value}")
        if len(value) == 0 or not all(isinstance(x, int) and x > 0 for x in value):
            raise ValueError(f"{name} must be tuple of positive integers, got {value}")
        return value
    return value


class MakeInterpolatedPurelyObservationalDataset:
    """
    Factory that creates a curriculum dataset whose hyperparameters interpolate
    between two configurations (t0 and t1) using a probability alpha(t).

    Naive interpolation rule (per user request):
      For each hyperparameter h:
        - Sample h0 from config_t0
        - Sample h1 from config_t1
        - With probability alpha(t) choose h1 else h0

    This selection happens independently per hyperparameter for each dataset index.

    The resulting dataset is returned as a CurriculumPurelyObservationalDataset instance
    with time-aware sampling logic embedded in wrapper samplers.
    """

    DISTRIBUTION_FACTORIES = {
        "fixed": lambda params: FixedSampler(params["value"]),
        "uniform": lambda params: TorchDistributionSampler(
            dist.Uniform(low=params["low"], high=params["high"])
        ),
        "normal": lambda params: TorchDistributionSampler(
            dist.Normal(loc=params["mean"], scale=params["std"])
        ),
        "lognormal": lambda params: TorchDistributionSampler(
            dist.LogNormal(loc=params["mean"], scale=params["std"])
        ),
        "exponential": lambda params: TorchDistributionSampler(
            dist.Exponential(rate=params["lambd"])
        ),
        "gamma": lambda params: TorchDistributionSampler(
            dist.Gamma(concentration=params["alpha"], rate=params["beta"])
        ),
        "beta": lambda params: TorchDistributionSampler(
            dist.Beta(concentration1=params["alpha"], concentration0=params["beta"])
        ),
        "categorical": lambda params: CategoricalSampler(
            params["choices"], params.get("probabilities")
        ),
        "discrete_uniform": lambda params: DiscreteUniformSampler(params["low"], params["high"]),
    }

    def __init__(
        self,
        scm_config_t0: Dict[str, Any],
        scm_config_t1: Dict[str, Any],
        preprocessing_config_t0: Dict[str, Any],
        preprocessing_config_t1: Dict[str, Any],
        dataset_config_t0: Dict[str, Any],
        dataset_config_t1: Dict[str, Any],
        interpolation_function: str = "sigmoid",
        seed: Optional[int] = None,
    ) -> None:
        self.scm_config_t0 = scm_config_t0
        self.scm_config_t1 = scm_config_t1
        self.preprocessing_config_t0 = preprocessing_config_t0
        self.preprocessing_config_t1 = preprocessing_config_t1
        self.dataset_config_t0 = dataset_config_t0
        self.dataset_config_t1 = dataset_config_t1
        self.interpolation_function = interpolation_function
        self.seed = seed

        # Basic validation (same keys per category)
        self._validate_matching_keys(self.scm_config_t0, self.scm_config_t1, "scm")
        self._validate_matching_keys(self.preprocessing_config_t0, self.preprocessing_config_t1, "preprocessing")
        self._validate_matching_keys(self.dataset_config_t0, self.dataset_config_t1, "dataset")

        # Build sampler dicts for t0 and t1
        self.scm_samplers_t0 = self._build_samplers(self.scm_config_t0, "scm")
        self.scm_samplers_t1 = self._build_samplers(self.scm_config_t1, "scm")
        self.preproc_samplers_t0 = self._build_samplers(self.preprocessing_config_t0, "preprocessing")
        self.preproc_samplers_t1 = self._build_samplers(self.preprocessing_config_t1, "preprocessing")
        self.dataset_samplers_t0 = self._build_samplers(self.dataset_config_t0, "dataset")
        self.dataset_samplers_t1 = self._build_samplers(self.dataset_config_t1, "dataset")

        # Extract dataset_size (must match) from dataset configs
        ds_size0 = self.dataset_config_t0.get("dataset_size", {}).get("value")
        ds_size1 = self.dataset_config_t1.get("dataset_size", {}).get("value")
        if ds_size0 != ds_size1:
            raise ValueError("dataset_size must match between t0 and t1 configs for simplicity.")
        self.dataset_size = ds_size0

    @staticmethod
    def _validate_matching_keys(c0: Dict[str, Any], c1: Dict[str, Any], name: str) -> None:
        k0 = set(c0.keys())
        k1 = set(c1.keys())
        if k0 != k1:
            missing0 = k1 - k0
            missing1 = k0 - k1
            raise ValueError(
                f"Config key mismatch in {name}: missing_in_t0={missing0}, missing_in_t1={missing1}"
            )

    def _build_samplers(self, config: Dict[str, Any], category: str) -> Dict[str, Any]:
        samplers: Dict[str, Any] = {}
        for param_name, spec in config.items():
            # Fixed value simple form {"value": v}
            if "value" in spec and "distribution" not in spec:
                samplers[param_name] = FixedSampler(spec["value"])
            elif "distribution" in spec:
                dist_type = spec["distribution"]
                if dist_type not in self.DISTRIBUTION_FACTORIES:
                    raise ValueError(f"Unknown distribution '{dist_type}' for {category}.{param_name}")
                params = spec.get("distribution_parameters", {})
                if dist_type == "fixed":
                    if "value" not in spec:
                        raise ValueError(f"Fixed distribution for {param_name} requires 'value'")
                    params = {"value": spec["value"]}
                try:
                    samplers[param_name] = self.DISTRIBUTION_FACTORIES[dist_type](params)
                except Exception as e:
                    raise ValueError(f"Failed to create sampler for {category}.{param_name}: {e}")
            else:
                raise ValueError(f"Invalid spec for {category}.{param_name}: {spec}")
        return samplers

    # Removed old pair sampling and choose methods (replaced by top-level helpers)

    def create_dataset(self) -> CurriculumPurelyObservationalDataset:
        """Return a CurriculumPurelyObservationalDataset that performs per-index interpolation."""

        # Wrap SCM sampling in a time-aware object
        scm_wrapper = _InterpolatedSCMSampler(
            self.scm_samplers_t0,
            self.scm_samplers_t1,
            seed=self.seed,
        )

        # Build a prototype preprocessing processor from t0 to copy defaults (max sizes will vary per index)
        # We'll let the dataset create BasicProcessing each __getitem__ based on selected parameters.
        # Provide a lightweight factory capturing original template values.
        preproc_factory = _InterpolatedPreprocessingFactory(
            self.preproc_samplers_t0,
            self.preproc_samplers_t1,
            seed=self.seed,
        )

        # Prepare train/test count samplers + max specs (interpolated per index)
        train_count_sampler = _InterpolatedValueSampler(
            self.dataset_samplers_t0["number_train_samples_per_dataset"],
            self.dataset_samplers_t1["number_train_samples_per_dataset"],
            seed=self.seed,
        )
        test_count_sampler = _InterpolatedValueSampler(
            self.dataset_samplers_t0["number_test_samples_per_dataset"],
            self.dataset_samplers_t1["number_test_samples_per_dataset"],
            seed=self.seed,
        )
        max_train_sampler = _InterpolatedValueSampler(
            self.dataset_samplers_t0["max_number_train_samples"],
            self.dataset_samplers_t1["max_number_train_samples"],
            seed=self.seed,
        )
        max_test_sampler = _InterpolatedValueSampler(
            self.dataset_samplers_t0["max_number_test_samples"],
            self.dataset_samplers_t1["max_number_test_samples"],
            seed=self.seed,
        )
        max_feat_sampler = _InterpolatedValueSampler(
            self.dataset_samplers_t0["max_number_features"],
            self.dataset_samplers_t1["max_number_features"],
            seed=self.seed,
        )

        dataset = _InterpolatedCurriculumDataset(
            scm_sampler=scm_wrapper,
            preprocessing_factory=preproc_factory,
            train_count_sampler=train_count_sampler,
            test_count_sampler=test_count_sampler,
            max_train_sampler=max_train_sampler,
            max_test_sampler=max_test_sampler,
            max_feat_sampler=max_feat_sampler,
            size=self.dataset_size,
            base_seed=self.seed or 0,
            interpolation_function=self.interpolation_function,
        )
        return dataset


class _InterpolatedSCMSampler:
    def __init__(
        self,
        samplers_t0: Dict[str, Any],
        samplers_t1: Dict[str, Any],
        seed: Optional[int],
    ):
        self.samplers_t0 = samplers_t0
        self.samplers_t1 = samplers_t1
        self.seed = seed

    def sample(self, seed: Optional[int] = None, t: Optional[float] = None, a: Optional[float] = None):
        if t is None:
            t = 0.0
        # Per-parameter sampling
        generator = torch.Generator()
        effective_seed = (self.seed or 0) + (seed or 0)
        generator.manual_seed(effective_seed)
        final_params: Dict[str, Any] = {}
        a = 0.0 if a is None else float(a)
        for name in self.samplers_t0.keys():
            v0, v1 = _top_level_sample_pair(self.samplers_t0[name], self.samplers_t1[name], generator)
            chosen = _choose_by_alpha(v0, v1, a)
            chosen = _normalize_scm_param(name, chosen)
            final_params[name] = chosen
        builder = SCMBuilder(**final_params)
        return builder.build()


class _InterpolatedPreprocessingFactory:
    def __init__(
        self,
        samplers_t0: Dict[str, Any],
        samplers_t1: Dict[str, Any],
        seed: Optional[int],
    ):
        self.samplers_t0 = samplers_t0
        self.samplers_t1 = samplers_t1
        self.seed = seed

    def __call__(
        self,
        t: float,
        a: float,
        n_features: int,
        max_n_features: int,
        n_train_samples: int,
        max_n_train_samples: int,
        n_test_samples: int,
        max_n_test_samples: int,
    ) -> BasicProcessing:
        # Sample preprocessing params
        generator = torch.Generator()
        generator.manual_seed((self.seed or 0) + int(t * 1_000_003))
        params: Dict[str, Any] = {}
        a = float(a) if a is not None else 0.0
        for name in self.samplers_t0.keys():
            v0, v1 = _top_level_sample_pair(self.samplers_t0[name], self.samplers_t1[name], generator)
            params[name] = _choose_by_alpha(v0, v1, a)

        return BasicProcessing(
            n_features=n_features,
            max_n_features=max_n_features,
            n_train_samples=n_train_samples,
            max_n_train_samples=max_n_train_samples,
            n_test_samples=n_test_samples,
            max_n_test_samples=max_n_test_samples,
            dropout_prob=params.get("dropout_prob", 0.0),
            target_feature=params.get("target_feature", None),
            random_seed=params.get("random_seed", None),
            negative_one_one_scaling=params.get("negative_one_one_scaling", False),
            standardize=params.get("standardize", False),
            yeo_johnson=params.get("yeo_johnson", False),
            remove_outliers=params.get("remove_outliers", False),
            outlier_quantile=params.get("outlier_quantile", 1.0),
            shuffle_samples=params.get("shuffle_data", True),
            shuffle_features=True,
        )


class _InterpolatedValueSampler:
    def __init__(
        self,
        sampler_t0: Any,
        sampler_t1: Any,
        seed: Optional[int],
    ):
        self.sampler_t0 = sampler_t0
        self.sampler_t1 = sampler_t1
        self.seed = seed

    def _sample_pair_with_seed(self, t: float, seed_offset: int = 0) -> Any:
        generator = torch.Generator()
        generator.manual_seed((self.seed or 0) + int(t * 7_17_31) + int(seed_offset))
        v0, v1 = _top_level_sample_pair(self.sampler_t0, self.sampler_t1, generator)
        return v0, v1

    def sample(self, t: float, a: float) -> Any:
        v0, v1 = self._sample_pair_with_seed(t)
        out = _choose_by_alpha(v0, v1, a)
        if torch.is_tensor(out) and out.numel() == 1:
            out = out.item()
        return out

    def sample_group(self, t: float, side: int, seed_offset: int = 0) -> Any:
        """Select from t0/t1 using a provided side (0->t0, 1->t1) for correlated choices."""
        v0, v1 = self._sample_pair_with_seed(t, seed_offset=seed_offset)
        out = v1 if side == 1 else v0
        if torch.is_tensor(out) and out.numel() == 1:
            out = out.item()
        return out


class _InterpolatedCurriculumDataset(CurriculumPurelyObservationalDataset):
    def __init__(
        self,
        scm_sampler: _InterpolatedSCMSampler,
        preprocessing_factory: _InterpolatedPreprocessingFactory,
        train_count_sampler: _InterpolatedValueSampler,
        test_count_sampler: _InterpolatedValueSampler,
        max_train_sampler: _InterpolatedValueSampler,
        max_test_sampler: _InterpolatedValueSampler,
        max_feat_sampler: _InterpolatedValueSampler,
        size: int,
        base_seed: int,
        interpolation_function: str,
    ):
        self.scm_sampler = scm_sampler
        self.preproc_factory = preprocessing_factory
        self.train_count_sampler = train_count_sampler
        self.test_count_sampler = test_count_sampler
        self.max_train_sampler = max_train_sampler
        self.max_test_sampler = max_test_sampler
        self.max_feat_sampler = max_feat_sampler
        self.size = size
        self.seed = base_seed
        self.interp = interpolation_function or "sigmoid"

    def __len__(self):
        return self.size

    def _time_for_index(self, idx: int) -> float:
        if self.size <= 1:
            return 1.0
        return float(idx) / float(self.size - 1)

    def _alpha(self, t: float) -> float:
        s = (self.interp or "sigmoid").strip().lower()
        if s.endswith('.'):
            s = s[:-1]
        if s == "linear":
            return float(max(0.0, min(1.0, t)))
        if s.startswith("step"):
            thr = 0.5
            if '_' in s:
                try:
                    thr = float(s.split('_', 1)[1])
                except ValueError:
                    thr = 0.5
            return 0.0 if t < thr else 1.0
        if s.startswith("constant"):
            p = 1.0
            if '_' in s:
                try:
                    p = float(s.split('_', 1)[1])
                except ValueError:
                    p = 1.0
            return float(max(0.0, min(1.0, p)))
        if s.startswith("sigmoid"):
            # New default: slow start, slow end; clamp first/last 10%
            if t <= 0.1:
                return 0.0
            if t >= 0.9:
                return 1.0
            # Normalize t to [0,1] within (0.1, 0.9)
            x = (t - 0.1) / 0.8
            # Sigmoid centered at 0.5 with moderate slope (k)
            k = 8.0
            z = k * (x - 0.5)
            raw = 1.0 / (1.0 + math.exp(-z))
            low = 1.0 / (1.0 + math.exp(k * 0.5 * 1.0))   # sigma(-k/2)
            high = 1.0 / (1.0 + math.exp(-k * 0.5 * 1.0)) # sigma(+k/2)
            denom = (high - low) if (high - low) != 0.0 else 1.0
            a = (raw - low) / denom
            return float(max(0.0, min(1.0, a)))
        # default
        # Use sigmoid as safe default
        if t <= 0.1:
            return 0.0
        if t >= 0.9:
            return 1.0
        x = (t - 0.1) / 0.8
        k = 8.0
        z = k * (x - 0.5)
        raw = 1.0 / (1.0 + math.exp(-z))
        low = 1.0 / (1.0 + math.exp(k * 0.5))
        high = 1.0 / (1.0 + math.exp(-k * 0.5))
        denom = (high - low) if (high - low) != 0.0 else 1.0
        a = (raw - low) / denom
        return float(max(0.0, min(1.0, a)))

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for size {self.size}")
        t = self._time_for_index(idx)
        seed = self.seed + idx
        torch.manual_seed(seed)
        a = self._alpha(t)
        scm = self.scm_sampler.sample(seed=seed, t=t, a=a)

        # Correlated choices for train/test groups
        side_train = 1 if torch.rand(()) < a else 0
        side_test = 1 if torch.rand(()) < a else 0

        max_n_train = int(self.max_train_sampler.sample_group(t, side=side_train, seed_offset=11))
        n_train = int(self.train_count_sampler.sample_group(t, side=side_train, seed_offset=13))

        max_n_test = int(self.max_test_sampler.sample_group(t, side=side_test, seed_offset=17))
        n_test = int(self.test_count_sampler.sample_group(t, side=side_test, seed_offset=19))

        max_n_features = int(self.max_feat_sampler.sample(t, a))

        total = n_train + n_test
        scm.sample_exogenous(num_samples=total)
        scm.sample_endogenous_noise(num_samples=total)
        raw = scm.propagate(num_samples=total)

        processor = self.preproc_factory(
            t=t,
            a=a,
            n_features=max_n_features,
            max_n_features=max_n_features,
            n_train_samples=n_train,
            max_n_train_samples=max_n_train,
            n_test_samples=n_test,
            max_n_test_samples=max_n_test,
        )

        X_train, Y_train, X_test, Y_test = processor.process(raw)
        return (
            X_train,
            Y_train,
            X_test,
            Y_test,
            torch.tensor(t, dtype=torch.float32),
            torch.tensor(a, dtype=torch.float32),
        )
