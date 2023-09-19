from pathlib import Path

import numpy as np
import pandas as pd

from eu_unemployment_prediction.input_data_type import InputDataType, DataPeriodicity
from eu_unemployment_prediction.lstm import DataLoader


def test_add_normalized_column_with_default_data_set(data_dir: Path) -> None:
    data_loader = DataLoader(data_dir, [InputDataType.UNEMPLOYMENT])

    new_column = data_loader.data_frame[InputDataType.UNEMPLOYMENT.normalized_column_name].to_numpy()
    original_column = data_loader.data_frame[InputDataType.UNEMPLOYMENT.column_name].to_numpy()

    np.testing.assert_almost_equal(new_column * 100.0, original_column, decimal=6)


def test_load_normalized_interpolated(data_dir: Path) -> None:
    data_loader = DataLoader(data_dir, [InputDataType.UNEMPLOYMENT, InputDataType.GOV_DEBT])
    result_df = data_loader.data_frame

    result_df = data_loader.data_frame
    np.testing.assert_almost_equal(result_df["float date"].to_numpy(), 0.1, decimal=1)
    assert InputDataType.UNEMPLOYMENT.column_name in result_df.columns
    assert InputDataType.UNEMPLOYMENT.normalized_column_name in result_df.columns
    assert InputDataType.GOV_DEBT.column_name in result_df.columns
    assert InputDataType.GOV_DEBT.normalized_column_name in result_df.columns

    result_index_freq = pd.infer_freq(result_df.index)  # type: ignore
    periodicity = InputDataType.UNEMPLOYMENT.periodicity
    assert periodicity is not None
    assert result_index_freq == periodicity.frequency
    assert (
        ~np.isnan(result_df.to_numpy())
    ).all(), "Expected no nan values in final dataframe (missing values should be interpolated)"


def test_load_normalized_interpolated_other_periodicity(data_dir: Path) -> None:
    data_loader = DataLoader(
        data_dir, [InputDataType.UNEMPLOYMENT, InputDataType.GOV_DEBT], periodicity=DataPeriodicity.QUARTERLY
    )

    result_df = data_loader.data_frame
    np.testing.assert_almost_equal(result_df["float date"].to_numpy(), 0.1, decimal=1)
    assert InputDataType.UNEMPLOYMENT.column_name in result_df.columns
    assert InputDataType.UNEMPLOYMENT.normalized_column_name in result_df.columns
    assert InputDataType.GOV_DEBT.column_name in result_df.columns
    assert InputDataType.GOV_DEBT.normalized_column_name in result_df.columns

    result_index_freq = pd.infer_freq(result_df.index)  # type: ignore
    assert result_index_freq == "Q-DEC"
    assert (
        ~np.isnan(result_df.to_numpy())
    ).all(), "Expected no nan values in final dataframe (missing values should be interpolated)"
