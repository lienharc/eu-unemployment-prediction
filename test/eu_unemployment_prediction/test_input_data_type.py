from pathlib import Path

import numpy as np
import pandas as pd

from eu_unemployment_prediction.input_data_type import InputDataType

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def test_add_normalized_column_with_default_data_set() -> None:
    data_set = InputDataType.UNEMPLOYMENT.load_default(DATA_DIR).to_frame()

    data_set_with_normalized_column = InputDataType.UNEMPLOYMENT.add_normalized_column(data_set)

    new_column = data_set_with_normalized_column.loc[:, InputDataType.UNEMPLOYMENT.normalized_column_name].to_numpy()
    np.testing.assert_almost_equal(
        new_column * 100.0, data_set[InputDataType.UNEMPLOYMENT.column_name].to_numpy(), decimal=6
    )


def test_add_normalized_column_if_data_missing_for_index() -> None:
    data_set = pd.DataFrame(
        {InputDataType.UNEMPLOYMENT.column_name: [0.0, 1.0, 2.0]},
        index=[pd.Timestamp(f"2020-01-0{i}") for i in range(1, 4)],
    )
    nan_index_value = "2020-01-31"
    data_set.loc[pd.Timestamp(nan_index_value), "another column"] = 13.0

    data_set_with_normalized_column = InputDataType.UNEMPLOYMENT.add_normalized_column(data_set)

    normalized_nan_cell = data_set_with_normalized_column.loc[
        nan_index_value, InputDataType.UNEMPLOYMENT.normalized_column_name
    ]
    np.testing.assert_equal(normalized_nan_cell, np.nan)
