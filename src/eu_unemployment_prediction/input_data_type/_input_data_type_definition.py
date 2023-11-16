from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from pandas._typing import InterpolateOptions

from eu_unemployment_prediction.input_data_type._data_periodicity import DataPeriodicity


def _not_implemented_yet(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    raise NotImplementedError("Conversion function hasn't been implemented for this input data type")


@dataclass
class InputDataTypeDefinition:
    identifier: str
    file_base_name: str
    column_name: str
    data_loader: Callable[[Path, str, str], pd.Series[float]]
    periodicity: Optional[DataPeriodicity]
    normalizer: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]] = _not_implemented_yet
    denormalizer: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]] = _not_implemented_yet
    interpolation_method: InterpolateOptions = "cubicspline"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, InputDataTypeDefinition):
            return False
        return self.identifier == other.identifier
