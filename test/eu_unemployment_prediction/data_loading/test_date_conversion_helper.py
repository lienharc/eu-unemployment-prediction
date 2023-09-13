from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from eu_unemployment_prediction.date_conversion_helper import (
    convert_quarterly_format_to_date,
    convert_timestamp_index_to_float,
)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("2022Q1", pd.to_datetime("2022-03-31")),
        ("2022Q2", pd.to_datetime("2022-06-30")),
        ("2022Q3", pd.to_datetime("2022-09-30")),
        ("2022Q4", pd.to_datetime("2022-12-31")),
    ],
)
def test_parses_quater(test_input: str, expected: date) -> None:
    converted_date = convert_quarterly_format_to_date(test_input)

    assert converted_date == expected


def test_raises_on_wrong_format() -> None:
    input_date = "2023Q5"
    with pytest.raises(ValueError) as exc_info:
        convert_quarterly_format_to_date(input_date)

    assert input_date in str(exc_info.value)
    assert "wrong format" in str(exc_info.value)


def test_convert_timestamp_index_to_float() -> None:
    input_index = pd.DatetimeIndex(
        [pd.Timestamp(f"2020-0{month}-01") for month in range(1, 5)]
    )  # type: pd.Index[pd.Timestamp]

    float_version = convert_timestamp_index_to_float(input_index)

    assert float_version.dtype == np.float32
    assert (float_version < 10.0).all(), "Floating timestamps are not in the range of (1, 10)"
    assert (float_version > 1.0).all(), "Floating timestamps are not in the range of (1, 10)"
