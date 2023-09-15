from pathlib import Path

from eu_unemployment_prediction.lstm import UnemploymentLstm


def test_load_with_pretrained_model_without_error(data_dir: Path, module_dir: Path) -> None:
    UnemploymentLstm.load(module_dir / "lstm" / "unemployment_lstm.pt")
