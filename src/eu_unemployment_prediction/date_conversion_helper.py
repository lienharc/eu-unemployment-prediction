from __future__ import annotations

import re
from typing import Callable, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd

_QUARTERLY_PATTERN = re.compile(r"^(\d{4})(Q[1234])$")

_QUARTER_TRANSFORM = {
    "Q1": lambda year: pd.to_datetime(f"{year}-03-31"),
    "Q2": lambda year: pd.to_datetime(f"{year}-06-30"),
    "Q3": lambda year: pd.to_datetime(f"{year}-09-30"),
    "Q4": lambda year: pd.to_datetime(f"{year}-12-31"),
}  # type: Dict[str, Callable[[int,], pd.Timestamp]]


def convert_quarterly_format_to_date(input_date: str) -> pd.Timestamp:
    """
    converts a date format like "2022Q3" into a proper date "2022-09-30" (at the end of the quarter).
    """
    match = _QUARTERLY_PATTERN.match(input_date)
    if match is None:
        raise ValueError(f"{input_date} has the wrong format, expected {_QUARTERLY_PATTERN.pattern}")
    year = int(match.group(1))
    quarter = match.group(2)

    return _QUARTER_TRANSFORM[quarter](year)


def convert_timestamp_index_to_float(index: pd.Index[pd.Timestamp]) -> npt.NDArray[np.float32]:
    return index.astype(int).to_numpy(dtype=np.float32) * 1.0e-19
