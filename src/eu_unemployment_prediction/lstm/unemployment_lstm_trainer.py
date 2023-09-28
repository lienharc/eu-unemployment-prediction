import logging
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from matplotlib import pyplot as plt
from matplotlib.dates import ConciseDateFormatter
from torch import nn, Tensor

from eu_unemployment_prediction.input_data_type import InputDataType
from eu_unemployment_prediction.lstm import UnemploymentLstm, DataLoader


class UnemploymentLstmTrainer:
    def __init__(
        self,
        lstm: UnemploymentLstm,
        input_data: DataLoader,
        learning_rate: float = 0.001,
        chunk_size: int = 40,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        :param lstm: An instance of the unemployment LSTM to be trained
        :param input_data: A data frame containing the raw input data for training.
            It should have the same structure as the one provided by
            :func:`~InputDataType.UNEMPLOYMENT.load_with_normalized_column`
        :param learning_rate: The learning rate during training
        :param chunk_size: The size of the chunks of the time series in between which the optimization is happening.
        """
        self._LOGGER = logging.getLogger(self.__class__.__name__)
        self._device = device
        self._model = lstm
        self._data = input_data

        self._consistency_check()

        self._chunk_size = chunk_size
        self._learning_rate = learning_rate
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._loss_function = nn.MSELoss()

    @property
    def model(self) -> UnemploymentLstm:
        return self._model

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: int) -> None:
        self._learning_rate = value
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)

    def run(self, epochs: int = 1000) -> None:
        self._model.to(self._device)
        self._model.train()
        for epoch in range(epochs):
            loss = self._run_epoch()
            if loss is None:
                self._LOGGER.warning(
                    f"Epoch {epoch:03d}/{epochs} | No training happened in this epoch. No training data?"
                )
            if epoch % 50 == 0:
                self._LOGGER.info(f"Epoch {epoch:03d}/{epochs} | loss: {loss:.3e}; lr: {self.learning_rate:.2e}")
        self._model.eval()

    def _run_epoch(self) -> Optional[Tensor]:
        loss = None
        hidden = tuple(torch.zeros(1, self._model.hidden_dim, device=self._device) for _ in range(2))
        for train_chunk, target_chunk in self._data.chunks(self._chunk_size):
            lstm_input = torch.tensor(train_chunk, device=self._device)
            self._LOGGER.debug(f"Training with chunk of size {train_chunk.shape}")
            self._optimizer.zero_grad()

            out, hidden = self._model(lstm_input, hidden)
            loss = self._loss_function(out, torch.tensor(target_chunk, device=self._device))

            loss.backward()
            self._optimizer.step()

        return loss

    def plot(self, file_path: Path, feature: InputDataType, plot_mask: Optional[npt.NDArray[np.bool_]] = None) -> None:
        self._LOGGER.info(f"Creating plot {file_path.name} for feature {feature}")
        if feature not in self._model.input_features:
            raise ValueError(
                f"Cannot plot results for {feature} since it is not part of the model. "
                f"Available features: {', '.join(str(f) for f in self._model.input_features)}"
            )
        if plot_mask is None:
            plot_mask = np.full_like(self._data.full.index, fill_value=True, dtype=np.bool_)
        ax = self._data.plot(feature, plot_mask)
        predictions = self._predict_future()
        ax.plot(
            self._data.full.index[1:].to_numpy()[plot_mask[1:]],
            predictions[plot_mask[1:], self._model.input_features.index(feature)],
            c="red",
            label="LSTM out",
        )
        ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend()
        plt.savefig(file_path, dpi=500)
        self._LOGGER.info(f"Saved plot to {file_path.as_uri()}")
        plt.clf()

    def _predict_future(self) -> npt.NDArray[np.float32]:
        steps = self._data.test.shape[0] - 1
        hidden_dim = self._model.hidden_dim
        hidden = torch.zeros(1, hidden_dim, device=self._device), torch.zeros(1, hidden_dim, device=self._device)
        columns = [input_feature.normalized_column_name for input_feature in self._model.input_features]
        columns.append(self._data.FLOAT_DATE_NAME)
        train_data = self._data.train.loc[:, columns].to_numpy()
        trained_input = torch.tensor(train_data, device=self._device)
        with torch.no_grad():
            trained_out, hidden = self._model(trained_input, hidden)
            predictions = torch.stack(list(self._model.predict_future(steps, trained_out[-1], hidden)))
        full_prediction = np.concatenate(
            [trained_out.cpu().numpy(), predictions.cpu().numpy()]
        )  # type: npt.NDArray[np.float32]

        for index, feature in enumerate(self._model.input_features):
            full_prediction[:, index] = feature.value.denormalizer(full_prediction[:, index])
        return full_prediction

    def _consistency_check(self) -> None:
        for data_type in self._model.input_features:
            if data_type.normalized_column_name not in self._data.columns:
                raise ValueError(
                    f'Expected a column named "{data_type.normalized_column_name}" in the input_data.'
                    f"Existing column names: {self._data.columns}"
                )
