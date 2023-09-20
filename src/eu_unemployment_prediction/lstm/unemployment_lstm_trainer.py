import logging
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor

from eu_unemployment_prediction.input_data_type import InputDataType
from eu_unemployment_prediction.lstm import UnemploymentLstm, DataLoader


class UnemploymentLstmTrainer:
    _LOGGER = logging.getLogger("UnemploymentLstmTrainer")

    def __init__(
        self,
        lstm: UnemploymentLstm,
        input_data: DataLoader,
        learning_rate: float = 0.001,
        chunk_size: int = 40,
    ) -> None:
        """
        :param lstm: An instance of the unemployment LSTM to be trained
        :param input_data: A data frame containing the raw input data for training.
            It should have the same structure as the one provided by
            :func:`~InputDataType.UNEMPLOYMENT.load_with_normalized_column`
        :param learning_rate: The learning rate during training
        :param chunk_size: The size of the chunks of the time series in between which the optimization is happening.
        """
        self._model = lstm
        self._data = input_data

        self._consistency_check()

        self._chunk_size = chunk_size
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._loss_function = nn.MSELoss()

    @property
    def model(self) -> UnemploymentLstm:
        return self._model

    def run(self, epochs: int = 1000) -> None:
        epoch = 0
        try:
            for epoch in range(epochs):
                loss = self._run_epoch()
                if loss is None:
                    self._LOGGER.warning(
                        f"Epoch {epoch:03d}/{epochs} | No training happened in this epoch. No training data?"
                    )
                if epoch % 50 == 0:
                    self._LOGGER.info(f"Epoch {epoch:03d}/{epochs} | loss: {loss:.3e}")
        except KeyboardInterrupt:
            self._LOGGER.warning(f"Learning process interrupted by user at epoch {epoch}/{epochs}")

    def _run_epoch(self) -> Optional[Tensor]:
        loss = None
        hidden = torch.zeros(1, 1, self._model.hidden_dim)
        cell = torch.zeros(1, 1, self._model.hidden_dim)
        for train_chunk, target_chunk in self._data.chunks(self._chunk_size):
            self._LOGGER.debug(f"Training with chunk of size {train_chunk.shape}")
            self._optimizer.zero_grad()
            hidden = hidden.detach()
            cell = cell.detach()

            out, (hidden, cell) = self._model(train_chunk, (hidden, cell))
            loss = self._loss_function(out, target_chunk)

            loss.backward()
            self._optimizer.step()

        return loss

    def plot(self, file_path: Path, feature: InputDataType, plot_mask: Optional[npt.NDArray[np.bool_]] = None) -> None:
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
        ax.legend()
        plt.savefig(file_path, dpi=500)
        plt.clf()

    def _predict_future(self) -> npt.NDArray[np.float32]:
        hidden, cell = (torch.zeros(1, 1, self._model.hidden_dim), torch.zeros(1, 1, self._model.hidden_dim))
        columns = [input_feature.normalized_column_name for input_feature in self._model.input_features]
        columns.append(self._data.FLOAT_DATE_NAME)
        testi = self._data.train.loc[:, columns].to_numpy()
        trained_input = torch.tensor(testi).view(self._data.train.shape[0], 1, -1)
        predictions = []
        with torch.no_grad():
            trained_out, (hidden, cell) = self._model(trained_input, (hidden, cell))
            prediction = trained_out[-1]
            for i in range(self._data.test.shape[0] - 1):
                prediction, (hidden, cell) = self._model(prediction.view(1, 1, self._model.input_dim), (hidden, cell))
                predictions.append(prediction.view(self._model.input_dim).numpy())
        full_prediction = np.concatenate([trained_out.numpy(), np.array(predictions)])  # type: npt.NDArray[np.float32]

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    project_dir = Path(__file__).parent.parent.parent.parent
    data_dir = project_dir / "data"
    img_dir = project_dir / "img"
    input_types = [InputDataType.UNEMPLOYMENT, InputDataType.EURO_STOXX_50, InputDataType.KEY_INTEREST_RATE]
    # input_types = [InputDataType.UNEMPLOYMENT]
    file_name_prefix = "_".join(data_type.file_base_name for data_type in input_types)
    model_path = project_dir / "model" / "lstm" / f"{file_name_prefix}_lstm.pt"

    lstm_model = UnemploymentLstm(64, input_features=input_types)
    # lstm_model = UnemploymentLstm.load(model_path)

    def data_masker(index: pd.DatetimeIndex) -> npt.NDArray[np.bool_]:
        return index > "2023-03-01"  # type: ignore

    data = DataLoader(data_dir, input_types, test_data_masker=data_masker)
    trainer = UnemploymentLstmTrainer(
        lstm_model,
        data,
        learning_rate=0.001,
        chunk_size=100,
    )
    trainer.run(epochs=10000)

    trainer.model.save(model_path)
    for input_type in input_types:
        trainer.plot(img_dir / f"{file_name_prefix}_lstm_{input_type.file_base_name}.png", input_type)
        trainer.plot(
            img_dir / f"{file_name_prefix}_lstm_{input_type.file_base_name}_zoom.png",
            input_type,
            plot_mask=data.full.index > "2022-01-01",
        )
