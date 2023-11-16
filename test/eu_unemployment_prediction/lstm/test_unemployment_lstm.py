from pathlib import Path

import numpy as np
import pytest

from eu_unemployment_prediction.lstm import UnemploymentLstm


def test_constructor_raises_if_unemployment_not_a_feature(data_dir: Path) -> None:
    with pytest.raises(ValueError) as exc_info:
        UnemploymentLstm(2, input_features=[])

    assert "input features" in str(exc_info.value)
    assert "UNEMPLOYMENT" in str(exc_info.value)


def test_save_and_reload_model(tmpdir: Path) -> None:
    file_path = Path(tmpdir) / "test_model.pt"

    model = UnemploymentLstm(8)
    model.save(file_path)
    loaded_model = UnemploymentLstm.load(file_path)

    model_state = model.state_dict()
    loaded_model_state = loaded_model.state_dict()

    assert model_state.keys() == loaded_model_state.keys()
    for key in model_state:
        np.testing.assert_equal(model_state[key].numpy(), loaded_model_state[key].numpy())


def test_load_with_pretrained_model_without_error(module_dir: Path) -> None:
    pretrained_model_path = module_dir / "lstm" / "U_lstm.pt"
    UnemploymentLstm.load(pretrained_model_path)
