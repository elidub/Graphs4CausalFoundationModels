from __future__ import annotations

from typing import Any, Dict, Optional

from priordata_processing.Datasets.ObservationalDataset import ObservationalDataset
from priordata_processing.Reg2ClsProcessor import Reg2ClsProcessor


class ObservationalDatasetTabICLNorm(ObservationalDataset):
    """ObservationalDataset with TabICL's Reg2Cls normalisation instead of BasicProcessing.

    Thin subclass — all logic lives in ObservationalDataset + Reg2ClsProcessor.
    """

    def __init__(
        self,
        scm_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        tabicl_hp: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            scm_config=scm_config,
            preprocessing_config=None,
            dataset_config=dataset_config,
            seed=seed,
            processor_class=Reg2ClsProcessor,
            processor_kwargs={"tabicl_hp": tabicl_hp},
        )
