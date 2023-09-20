from pathlib import Path

import pytest

from eu_unemployment_prediction.input_data_type import InputDataType
from eu_unemployment_prediction.lstm import UnemploymentLstmTrainer, UnemploymentLstm, DataLoader


def test_constructor_raises_if_data_frame_is_missing_the_right_columns(data_dir: Path) -> None:
    data = DataLoader(data_dir, [InputDataType.UNEMPLOYMENT])
    with pytest.raises(ValueError) as exc_info:
        UnemploymentLstmTrainer(
            UnemploymentLstm(2, input_features=[InputDataType.UNEMPLOYMENT, InputDataType.GOV_DEBT]),
            data,
        )

    assert f'Expected a column named "{InputDataType.GOV_DEBT.normalized_column_name}"' in str(exc_info.value)
