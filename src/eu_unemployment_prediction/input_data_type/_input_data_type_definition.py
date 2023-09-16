from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy import typing as npt


def _not_implemented_yet(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    raise NotImplementedError("Conversion function hasn't been implemented for this input data type")


@dataclass
class InputDataTypeDefinition:
    file_base_name: str
    column_name: str
    data_loader: Callable[[Path, str, str], pd.Series[float]]
    normalizer: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]] = _not_implemented_yet
    interpolation_method: str = "cubic"
