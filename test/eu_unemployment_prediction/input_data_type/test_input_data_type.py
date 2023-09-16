from pathlib import Path

import numpy as np

from eu_unemployment_prediction.input_data_type import InputDataType


def test_add_normalized_column_with_default_data_set(data_dir: Path) -> None:
    data_set_with_normalized_column = InputDataType.UNEMPLOYMENT.load_with_normalized_column(data_dir)

    new_column = data_set_with_normalized_column.loc[:, InputDataType.UNEMPLOYMENT.normalized_column_name].to_numpy()
    original_column = data_set_with_normalized_column.loc[:, InputDataType.UNEMPLOYMENT.column_name].to_numpy()

    np.testing.assert_almost_equal(new_column * 100.0, original_column, decimal=6)
