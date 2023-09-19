from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eu_unemployment_prediction.input_data_type import InputDataType
from eu_unemployment_prediction.lstm import UnemploymentLstmTrainer, UnemploymentLstm, DataLoader


@pytest.fixture
def unemployment_data(data_dir: Path) -> pd.DataFrame:
    return DataLoader(data_dir, [InputDataType.UNEMPLOYMENT]).full


def test_constructor_raises_if_data_frame_is_missing_the_right_columns(unemployment_data: pd.DataFrame) -> None:
    with pytest.raises(ValueError) as exc_info:
        UnemploymentLstmTrainer(
            UnemploymentLstm(2, input_features=[InputDataType.UNEMPLOYMENT, InputDataType.GOV_DEBT]),
            unemployment_data,
        )

    assert f'Expected a column named "{InputDataType.GOV_DEBT.normalized_column_name}"' in str(exc_info.value)
