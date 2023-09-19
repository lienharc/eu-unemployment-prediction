from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from eu_unemployment_prediction.input_data_type import InputDataType, DataPeriodicity
from eu_unemployment_prediction.lstm import DataLoader


def test_add_normalized_column_with_default_data_set(data_dir: Path) -> None:
    data_loader = DataLoader(data_dir, [InputDataType.UNEMPLOYMENT])

    new_column = data_loader.full[InputDataType.UNEMPLOYMENT.normalized_column_name].to_numpy()
    original_column = data_loader.full[InputDataType.UNEMPLOYMENT.column_name].to_numpy()

    np.testing.assert_almost_equal(new_column * 100.0, original_column, decimal=6)


def test_load_normalized_interpolated(data_dir: Path) -> None:
    data_types = [InputDataType.UNEMPLOYMENT, InputDataType.GOV_DEBT]
    data_loader = DataLoader(data_dir, data_types)

    expected_column_names = [
        x for data_type in data_types for x in [data_type.column_name, data_type.normalized_column_name]
    ]

    result_df = data_loader.full
    np.testing.assert_almost_equal(result_df["float date"].to_numpy(), 0.1, decimal=1)
    for expected_column_name in expected_column_names:
        assert expected_column_name in result_df.columns

    result_index_freq = pd.infer_freq(result_df.index)  # type: ignore
    periodicity = InputDataType.UNEMPLOYMENT.periodicity
    assert periodicity is not None
    assert result_index_freq == periodicity.frequency
    assert (
        ~np.isnan(result_df.loc[:, expected_column_names].to_numpy())
    ).all(), "Expected no nan values in final dataframe (missing values should be interpolated)"


def test_load_normalized_interpolated_other_periodicity(data_dir: Path) -> None:
    data_types = [InputDataType.UNEMPLOYMENT, InputDataType.GOV_DEBT]
    expected_column_names = [
        x for data_type in data_types for x in [data_type.column_name, data_type.normalized_column_name]
    ]
    data_loader = DataLoader(data_dir, data_types, periodicity=DataPeriodicity.QUARTERLY)

    result_df = data_loader.full
    np.testing.assert_almost_equal(result_df["float date"].to_numpy(), 0.1, decimal=1)
    for expected_column_name in expected_column_names:
        assert expected_column_name in result_df.columns

    result_index_freq = pd.infer_freq(result_df.index)  # type: ignore
    assert result_index_freq == "Q-DEC"
    assert (
        ~np.isnan(result_df.loc[:, expected_column_names].to_numpy())
    ).all(), "Expected no nan values in final dataframe (missing values should be interpolated)"


def test_type_labels_train_only(data_dir: Path) -> None:
    data_loader = DataLoader(data_dir, [InputDataType.UNEMPLOYMENT])

    assert data_loader.test.shape[0] == 0
    assert data_loader.train.shape[0] == data_loader.full.shape[0]


def test_type_labels_data_mask(data_dir: Path) -> None:
    expected_data_points = 4

    def before_march(index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        return index <= "2000-04-30"  # type: ignore

    data_loader = DataLoader(data_dir, [InputDataType.UNEMPLOYMENT], test_data_masker=before_march)

    assert data_loader.test.shape[0] == expected_data_points
    assert data_loader.train.shape[0] == data_loader.full.shape[0] - expected_data_points


def test_type_labels_data_mask_after_reindexing(data_dir: Path) -> None:
    expected_data_points = 91  # three months with ~ 30 days

    def before_march(index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        return index <= "2000-04-30"  # type: ignore

    data_loader = DataLoader(data_dir, [InputDataType.UNEMPLOYMENT], test_data_masker=before_march)
    data_loader.apply_new_index(DataPeriodicity.DAILY)
    print(data_loader.full)

    assert data_loader.test.shape[0] == expected_data_points
    assert data_loader.train.shape[0] == data_loader.full.shape[0] - expected_data_points


def test_chunk_generator(data_dir: Path) -> None:
    chunk_size = 50
    data_loader = DataLoader(data_dir, [InputDataType.UNEMPLOYMENT, InputDataType.KEY_INTEREST_RATE])
    chunks = list(data_loader.chunks(chunk_size))

    first_train_chunk = chunks[0][0]
    first_target_chunk = chunks[0][1]
    assert first_train_chunk.shape == (chunk_size, 1, 2)
    assert first_target_chunk.shape == (chunk_size, 2)
    np.testing.assert_equal(first_train_chunk.view(chunk_size, 2).numpy()[1:], first_target_chunk.numpy()[:-1])

    first_element_in_second_train_chunk = chunks[1][0].view(chunk_size, 2).numpy()[0]
    last_element_in_first_target_chunk = first_target_chunk.numpy()[-1]
    np.testing.assert_equal(first_element_in_second_train_chunk, last_element_in_first_target_chunk)
