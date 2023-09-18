import logging
from pathlib import Path
from typing import Tuple, Generator, Optional, Callable, List

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor

from eu_unemployment_prediction.input_data_type import InputDataType
from eu_unemployment_prediction.lstm import UnemploymentLstm


class UnemploymentLstmTrainer:
    _LOGGER = logging.getLogger("UnemploymentLstmTrainer")

    def __init__(
        self,
        lstm: UnemploymentLstm,
        input_data: pd.DataFrame,
        input_features: Optional[List[InputDataType]] = None,
        test_data_masker: Optional[Callable[[pd.DatetimeIndex], npt.NDArray[np.bool_]]] = None,
        learning_rate: float = 0.001,
        chunk_size: int = 40,
    ) -> None:
        """
        :param lstm: An instance of the unemployment LSTM to be trained
        :param input_data: A data frame containing the raw input data for training.
            It should have the same structure as the one provided by
            :func:`~InputDataType.UNEMPLOYMENT.load_with_normalized_column`
        :param input_features: List of features that should be considedered for the LSTM model.
            Default: only unemployment
        :param test_data_masker: A function which returns a numpy array specifying which data points of the input_data
            should be used for testing the trained model.
            These data points are not used for training.
            The resulting boolean array has to have the same length as there are rows in input_data.

            Example:
            >>> test_data_masker = lambda index: index > "2020-01-01"

            Default: All data points are considered for training, none for testing.
        :param learning_rate: The learning rate during training
        :param chunk_size: The size of the chunks of the time series in between which the optimization is happening.
        """
        self._model = lstm
        self._raw_data = input_data
        self._input_features = input_features if input_features is not None else [InputDataType.UNEMPLOYMENT]

        self._consistency_check()

        if test_data_masker is not None:
            self._test_data_mask = test_data_masker(self._raw_data.index)  # type: ignore
        else:
            self._test_data_mask = np.full_like(self._raw_data.index, False, dtype=np.bool_)
        self._train_data = self._raw_data.loc[~self._test_data_mask]
        self._test_data = self._raw_data.loc[self._test_data_mask]

        self._chunk_size = chunk_size
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._loss_function = nn.MSELoss()

    @property
    def model(self) -> UnemploymentLstm:
        return self._model

    @property
    def train_data(self) -> pd.DataFrame:
        return self._train_data

    @property
    def test_data(self) -> pd.DataFrame:
        return self._test_data

    def run(self, epochs: int = 1000) -> None:
        epoch = 0
        try:
            for epoch in range(epochs):
                loss = self._run_epoch()
                if loss is None:
                    self._LOGGER.warning(
                        f"Epoch {epoch:03d}/{epochs} | No training happened in this epoch. No training data?"
                    )
                self._LOGGER.info(f"Epoch {epoch:03d}/{epochs} | loss: {loss}")
        except KeyboardInterrupt:
            self._LOGGER.warning(f"Learning process interrupted by user at epoch {epoch}/{epochs}")

    def _generate_chunks(self) -> Generator[Tuple[Tensor, Tensor], None, None]:
        """Generates chunks of (training_data, target_data) tuples in the right shape"""
        column_id = self._train_data.columns.get_loc(InputDataType.UNEMPLOYMENT.normalized_column_name)
        input_data = self._train_data.iloc[:-1, column_id]
        target_data = self._train_data.iloc[1:, column_id]
        for start_index in range(0, self._train_data.shape[0], self._chunk_size):
            stop_index = start_index + self._chunk_size
            input_chunk = torch.tensor(input_data.iloc[start_index:stop_index].to_numpy())
            target_chunk = torch.tensor(target_data.iloc[start_index:stop_index].to_numpy())
            lstm_input = input_chunk.view(input_chunk.shape[0], 1, -1)
            targets = target_chunk.unsqueeze(-1)
            yield lstm_input, targets

    def _run_epoch(self) -> Optional[Tensor]:
        loss = None
        hidden, cell = (torch.zeros(1, 1, self._model.hidden_dim), torch.zeros(1, 1, self._model.hidden_dim))
        for train_chunk, target_chunk in self._generate_chunks():
            self._LOGGER.debug(f"Training with chunk of size {train_chunk.shape}")
            self._optimizer.zero_grad()
            hidden = hidden.detach()
            cell = cell.detach()

            out, (hidden, cell) = self._model(train_chunk, (hidden, cell))
            loss = self._loss_function(out, target_chunk)

            loss.backward()
            self._optimizer.step()

        return loss

    def plot(self, file_path: Path) -> None:
        sns.set_theme(style="whitegrid")
        plot_df = self._raw_data.copy()
        plot_df["type"] = "train"
        plot_df.loc[self._test_data_mask, "type"] = "test"
        ax = sns.scatterplot(
            plot_df,
            x="date",
            y="unemployment rate",
            hue="type",
            palette={"train": "black", "test": "grey"},
        )  # type: plt.Axes
        predictions = self._predict_future()
        ax.plot(self._raw_data.index[1:], predictions, c="red", label="LSTM out")
        ax.legend()
        plt.savefig(file_path, dpi=500)
        plt.clf()

    def _predict_future(self) -> npt.NDArray[np.float32]:
        hidden, cell = (torch.zeros(1, 1, self._model.hidden_dim), torch.zeros(1, 1, self._model.hidden_dim))
        trained_input = torch.tensor(self._train_data.loc[:, InputDataType.UNEMPLOYMENT.normalized_column_name]).view(
            self._train_data.shape[0], 1, -1
        )
        predictions = []
        with torch.no_grad():
            trained_out, (hidden, cell) = self._model(trained_input, (hidden, cell))
            prediction = trained_out[-1].unsqueeze(-1)
            for i in range(self._test_data.shape[0] - 1):
                prediction, (hidden, cell) = self._model(prediction.unsqueeze(-1), (hidden, cell))
                predictions.append(prediction.numpy())
        full_prediction = np.concatenate(
            [trained_out.numpy().flatten(), np.array(predictions).flatten()]
        )  # type: npt.NDArray[np.float32]
        return full_prediction * 100.0

    def _consistency_check(self) -> None:
        if InputDataType.UNEMPLOYMENT not in self._input_features:
            raise ValueError("input_features needs to include the 'UNEMPLOYMENT' feature")
        if len(self._input_features) != self._model.input_dim:
            raise ValueError(
                f"The model's input_dim ({self._model.input_dim}) needs to be the same as "
                f"the length of input_features ({len(self._input_features)}). But it isn't."
            )
        for data_type in self._input_features:
            if data_type.normalized_column_name not in self._raw_data.columns:
                raise ValueError(
                    f'Expected a column named "{data_type.normalized_column_name}" in the input_data.'
                    f"Existing column names: {self._raw_data.columns}"
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    project_dir = Path(__file__).parent.parent.parent.parent
    data_dir = project_dir / "data"
    img_dir = project_dir / "img"
    model_path = project_dir / "model" / "lstm" / "unemployment_lstm.pt"

    lstm_model = UnemploymentLstm(32)
    # lstm_model = UnemploymentLstm.load(model_path)
    unemployment_data = InputDataType.UNEMPLOYMENT.load_with_normalized_column(data_dir)

    def data_masker(index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        return index > "2022-01-01"  # type: ignore

    trainer = UnemploymentLstmTrainer(
        lstm_model,
        unemployment_data,
        test_data_masker=data_masker,
        learning_rate=0.001,
        chunk_size=50,
    )
    trainer.run(epochs=5000)

    trainer.model.save(model_path)
    trainer.plot(img_dir / "lstm_unemployment.png")
