from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eu_unemployment_prediction.input_data_type import InputDataType
from eu_unemployment_prediction.lstm import UnemploymentLstmTrainer, UnemploymentLstm, DataLoader


@pytest.fixture
def unemployment_data(data_dir: Path) -> pd.DataFrame:
    return DataLoader(data_dir, [InputDataType.UNEMPLOYMENT]).data_frame


def test_constructor_raises_if_data_frame_is_missing_the_right_columns(unemployment_data: pd.DataFrame) -> None:
    with pytest.raises(ValueError) as exc_info:
        UnemploymentLstmTrainer(
            UnemploymentLstm(2, input_features=[InputDataType.UNEMPLOYMENT, InputDataType.GOV_DEBT]),
            unemployment_data,
        )

    assert f'Expected a column named "{InputDataType.GOV_DEBT.normalized_column_name}"' in str(exc_info.value)


def test_chunk_generator(unemployment_data: pd.DataFrame) -> None:
    chunk_size = 50
    trainer = UnemploymentLstmTrainer(UnemploymentLstm(8), unemployment_data, chunk_size=chunk_size)

    chunks = list(trainer._generate_chunks())

    first_train_chunk = chunks[0][0]
    first_target_chunk = chunks[0][1]
    assert first_train_chunk.shape == (chunk_size, 1, 1)
    assert first_target_chunk.shape == (chunk_size, 1)
    np.testing.assert_equal(first_train_chunk.numpy().flatten()[1:], first_target_chunk.numpy().flatten()[:-1])
    first_element_in_second_train_chunk = chunks[1][0].numpy().flatten()[0]
    last_element_in_first_target_chunk = first_target_chunk.numpy().flatten()[-1]

    assert first_element_in_second_train_chunk == last_element_in_first_target_chunk


def test_no_test_data_masker(unemployment_data: pd.DataFrame) -> None:
    trainer = UnemploymentLstmTrainer(UnemploymentLstm(8), unemployment_data)

    assert trainer.train_data.shape[0] == unemployment_data.shape[0]
    assert trainer.test_data.shape[0] == 0


def test_data_masker_works(unemployment_data: pd.DataFrame) -> None:
    test_data_masker = lambda index: index <= "2000-04-30"
    expected_test_data_size = 4  # ECB data starts at 2000-01-31
    trainer = UnemploymentLstmTrainer(UnemploymentLstm(8), unemployment_data, test_data_masker=test_data_masker)

    assert trainer.train_data.shape[0] == unemployment_data.shape[0] - expected_test_data_size
    assert trainer.test_data.shape[0] == expected_test_data_size
