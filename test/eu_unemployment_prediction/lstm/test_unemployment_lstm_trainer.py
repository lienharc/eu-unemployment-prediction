from pathlib import Path
from typing import Callable
import numpy as np
import numpy.typing as npt
import pandas as pd

import numpy as np

from eu_unemployment_prediction.input_data_type import InputDataType
from eu_unemployment_prediction.lstm import UnemploymentLstmTrainer, UnemploymentLstm


def test_chunk_generator(data_dir: Path) -> None:
    chunk_size = 50
    input_data = InputDataType.UNEMPLOYMENT.load_with_normalized_column(data_dir)
    trainer = UnemploymentLstmTrainer(UnemploymentLstm(8), input_data, chunk_size=chunk_size)

    chunks = list(trainer._generate_chunks())

    first_train_chunk = chunks[0][0]
    first_target_chunk = chunks[0][1]
    assert first_train_chunk.shape == (chunk_size, 1, 1)
    assert first_target_chunk.shape == (chunk_size, 1)
    np.testing.assert_equal(first_train_chunk.numpy().flatten()[1:], first_target_chunk.numpy().flatten()[:-1])
    first_element_in_second_train_chunk = chunks[1][0].numpy().flatten()[0]
    last_element_in_first_target_chunk = first_target_chunk.numpy().flatten()[-1]

    assert first_element_in_second_train_chunk == last_element_in_first_target_chunk


def test_no_test_data_masker(data_dir: Path) -> None:
    input_data = InputDataType.UNEMPLOYMENT.load_with_normalized_column(data_dir)
    trainer = UnemploymentLstmTrainer(UnemploymentLstm(8), input_data)

    assert trainer.train_data.shape[0] == input_data.shape[0]
    assert trainer.test_data.shape[0] == 0


def test_data_masker_works(data_dir: Path) -> None:
    test_data_masker = lambda index: index < "2000-04-10"
    expected_test_data_size = 4  # ECB data starts at 2000-01-01
    input_data = InputDataType.UNEMPLOYMENT.load_with_normalized_column(data_dir)
    trainer = UnemploymentLstmTrainer(UnemploymentLstm(8), input_data, test_data_masker=test_data_masker)

    assert trainer.train_data.shape[0] == input_data.shape[0] - expected_test_data_size
    assert trainer.test_data.shape[0] == expected_test_data_size
