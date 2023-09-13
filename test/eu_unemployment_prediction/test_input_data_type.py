from pathlib import Path

import numpy as np
import pandas as pd

from eu_unemployment_prediction.input_data_type import InputDataType

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def test_add_normalized_column_with_default_data_set() -> None:
    data_set_with_normalized_column = InputDataType.UNEMPLOYMENT.load_with_normalized_column(DATA_DIR)

    new_column = data_set_with_normalized_column.loc[:, InputDataType.UNEMPLOYMENT.normalized_column_name].to_numpy()
    original_column = data_set_with_normalized_column.loc[:, InputDataType.UNEMPLOYMENT.column_name].to_numpy()

    np.testing.assert_almost_equal(new_column * 100.0, original_column, decimal=6)
