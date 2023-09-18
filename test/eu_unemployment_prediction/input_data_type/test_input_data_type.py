from pathlib import Path

import numpy as np
import pandas as pd

from eu_unemployment_prediction.input_data_type import InputDataType, DataPeriodicity


def test_add_normalized_column_with_default_data_set(data_dir: Path) -> None:
    data_set_with_normalized_column = InputDataType.UNEMPLOYMENT.load_with_normalized_column(data_dir)

    new_column = data_set_with_normalized_column.loc[:, InputDataType.UNEMPLOYMENT.normalized_column_name].to_numpy()
    original_column = data_set_with_normalized_column.loc[:, InputDataType.UNEMPLOYMENT.column_name].to_numpy()

    np.testing.assert_almost_equal(new_column * 100.0, original_column, decimal=6)


def test_load_normalized_interpolated(data_dir: Path) -> None:
    result_df = InputDataType.load_normalized_interpolated(
        [InputDataType.UNEMPLOYMENT, InputDataType.GOV_DEBT], data_dir
    )

    assert InputDataType.UNEMPLOYMENT.column_name in result_df.columns
    assert InputDataType.UNEMPLOYMENT.normalized_column_name in result_df.columns
    assert InputDataType.GOV_DEBT.column_name in result_df.columns
    assert InputDataType.GOV_DEBT.normalized_column_name in result_df.columns

    result_index_freq = pd.infer_freq(result_df.index)  # type: ignore
    assert result_index_freq == InputDataType.UNEMPLOYMENT.periodicity.frequency
    assert (
        ~np.isnan(result_df.to_numpy())
    ).all(), "Expected no nan values in final dataframe (missing values should be interpolated)"


def test_load_normalized_interpolated_other_periodicity(data_dir: Path) -> None:
    result_df = InputDataType.load_normalized_interpolated(
        [InputDataType.UNEMPLOYMENT, InputDataType.GOV_DEBT], data_dir, periodicity=DataPeriodicity.QUARTERLY
    )

    assert InputDataType.UNEMPLOYMENT.column_name in result_df.columns
    assert InputDataType.UNEMPLOYMENT.normalized_column_name in result_df.columns
    assert InputDataType.GOV_DEBT.column_name in result_df.columns
    assert InputDataType.GOV_DEBT.normalized_column_name in result_df.columns

    result_index_freq = pd.infer_freq(result_df.index)  # type: ignore
    assert result_index_freq == "Q-DEC"
    assert (
        ~np.isnan(result_df.to_numpy())
    ).all(), "Expected no nan values in final dataframe (missing values should be interpolated)"
    print(result_df)
