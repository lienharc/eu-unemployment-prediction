import logging
from pathlib import Path
from typing import Tuple, Generator, Optional

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
    def __init__(
        self, lstm: UnemploymentLstm, input_data: pd.DataFrame, learning_rate: float = 0.001, chunk_size: int = 40
    ) -> None:
        """
        :param lstm: An instance of the unemployment LSTM to be trained
        :param input_data: A data frame containing the raw input data for training.
            It should have the same structure as the one provided by
            :func:`~InputDataType.UNEMPLOYMENT.load_with_normalized_column`
        """
        self._model = lstm
        # todo: consider checking input_data properly (columns "float time", "unemployment rate",
        # todo: "unemployment rate norm"
        self._input_data = input_data

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
                    logging.warning(
                        f"Epoch {epoch:03d}/{epochs} | No training happened in this epoch. No training data?"
                    )
                logging.info(f"Epoch {epoch:03d}/{epochs} | loss: {loss}")
        except KeyboardInterrupt:
            logging.warning(f"Learning process interrupted by user at epoch {epoch}/{epochs}")

    def _generate_chunks(self) -> Generator[pd.DataFrame, None, None]:
        for i in range(0, self._input_data.shape[0], self._chunk_size):
            yield self._input_data.iloc[i : i + self._chunk_size]

    def _run_epoch(self) -> Optional[Tensor]:
        loss = None
        hidden, cell = (torch.randn(1, 1, self._model.hidden_dim), torch.randn(1, 1, self._model.hidden_dim))
        for train_chunk in self._generate_chunks():
            self._optimizer.zero_grad()
            hidden = hidden.detach()
            cell = cell.detach()

            lstm_input, targets = self._get_input_and_target_from_chunk(train_chunk)
            out, (hidden, cell) = self._model(lstm_input, (hidden, cell))
            loss = self._loss_function(out, targets)

            loss.backward()
            self._optimizer.step()

        return loss

    @staticmethod
    def _get_input_and_target_from_chunk(train_chunk: pd.DataFrame) -> Tuple[Tensor, Tensor]:
        # todo: we are losing one datapoint at the edges for each chunk.
        # todo: consider letting the generator generate the slices already which would remove these edge cases
        unemployment_data = train_chunk.loc[:, InputDataType.UNEMPLOYMENT.normalized_column_name].to_numpy()
        training_slice = torch.tensor(unemployment_data[:-1].copy())
        lstm_input = training_slice.view(training_slice.shape[0], 1, -1)
        target_slice = unemployment_data[1:]
        targets = torch.tensor(target_slice.copy()).unsqueeze(-1)
        return lstm_input, targets

    def plot(self, file_path: Path) -> None:
        sns.set_theme(style="whitegrid")
        ax = sns.lineplot(
            data=self._input_data,
            x=self._input_data.index,
            y="unemployment rate",
            # palette={"train": "black", "test": "grey"},
        )  # type: plt.Axes
        with torch.no_grad():
            lstm_input = torch.tensor(self._input_data.loc[:, InputDataType.UNEMPLOYMENT.normalized_column_name]).view(
                self._input_data.shape[0], 1, -1
            )
            out = self._model(lstm_input).numpy().flatten()  # type: npt.NDArray[np.float32]
        ax.scatter(self._input_data.index, out * 100, c="red", marker=".", label="lstm out")
        ax.legend()
        plt.savefig(file_path, dpi=500)
        plt.clf()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    project_dir = Path(__file__).parent.parent.parent.parent
    data_dir = project_dir / "data"
    img_dir = project_dir / "img"
    model_path = project_dir / "model" / "lstm" / "unemployment_lstm.pt"

    unemployment_data = InputDataType.UNEMPLOYMENT.load_with_normalized_column(data_dir)
    trainer = UnemploymentLstmTrainer(UnemploymentLstm(64), unemployment_data, learning_rate=0.0001, chunk_size=30)

    trainer.run(epochs=5000)

    trainer.model.save(model_path)
    trainer.plot(img_dir / "lstm_unemployment.png")
