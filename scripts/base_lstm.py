import logging
from pathlib import Path
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from eu_unemployment_prediction.input_data_type import InputDataType
from eu_unemployment_prediction.lstm import UnemploymentLstmTrainer, UnemploymentLstm, DataLoader


def train_and_plot(
    features: List[InputDataType],
    project_dir: Path = Path(__file__).parent.parent,
    lstm_hidden_dim: int = 32,
    test_data_cut_off_date: str | pd.Timestamp = "2023-03-01",
    learning_rates: float | List[float] = 0.001,
    epochs: int = 10000,
    plot_zoom_date: str | pd.Timestamp = "2022-01-01",
    train_from_scratch: bool = True,
) -> None:
    """Trains an LSTM with a specified set of input data provided in the data folder.
    Saves the trained models to the model folder and generates images for each feature in the image folder

    :param features: The set of features to be used for training
    :param project_dir: The project root directory
    :param lstm_hidden_dim: Dimension of the hidden layer (and the cell) of the LSTM
    :param test_data_cut_off_date: Devides the training data into a train section and a test section.
        The train section is used to train the LSTM, the test section is used to test the prediction
        capabilities of the LSTM.
    :param learning_rates:
        Defines the learning rates for the optimizer during training.
        If a list is provided the training is done several times with the specified learning rates.
        Each iteration will have the number of epochs specified for :param epochs.
    :param epochs: Number of epochs for training
    :param plot_zoom_date: The script will also create a 'zoomed' plot to better visiualize the predictive capabilities of the LSTM. Ideally, the zoom date should be a few data
    points before the :param test_data_cut_off_date.
    :param train_from_scratch: Whether to start the training from the beginning or from the pretrained models in the model folder.
        Be aware that the pretrained models might have a different :param hidden_dim than specified for this method.
    """
    torch.manual_seed(42)
    if not isinstance(learning_rates, list):
        learning_rates = [learning_rates]
    file_prefix = "".join(feature.value.identifier for feature in features)
    model_path = project_dir / "model" / "lstm" / f"lstm_{file_prefix}.pt"
    data_dir = project_dir / "data"
    img_dir = project_dir / "img"

    lstm = UnemploymentLstm(lstm_hidden_dim, features) if train_from_scratch else UnemploymentLstm.load(model_path)

    def data_masker(index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        return index > test_data_cut_off_date  # type: ignore

    data = DataLoader(data_dir, features, test_data_masker=data_masker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = UnemploymentLstmTrainer(lstm, data, chunk_size=300, device=device)
    for lr in learning_rates:
        trainer.learning_rate = lr
        trainer.run(epochs)
    lstm.save(model_path)
    for feature in features:
        trainer.plot(img_dir / f"lstm_{file_prefix}_{feature.file_base_name}.png", feature)
        trainer.plot(
            img_dir / f"lstm_{file_prefix}_{feature.file_base_name}_zoom.png",
            feature,
            plot_mask=data.full.index > plot_zoom_date,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_plot([InputDataType.UNEMPLOYMENT])
