import logging
from typing import Tuple

import numpy.typing as npt
import numpy as np
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor

from eu_unemployment_prediction.input_data_type import InputDataType
from eu_unemployment_prediction.lstm import UnemploymentLstm


class UnemploymentLstmTrainer:
    def __init__(
        self,
        lstm: UnemploymentLstm,
        input_data: pd.DataFrame,
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
        self._input_data = input_data.iloc[::-1]

    def run(self, epochs: int = 1000, chunk_size: int = 10, learning_rate: float = 0.001) -> None:
        train_chunks = [
            self._input_data.iloc[i : i + chunk_size] for i in range(0, self._input_data.shape[0], chunk_size)
        ]
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()

        try:
            for epoch in range(epochs):
                hidden, cell = (torch.randn(1, 1, self._model.hidden_dim), torch.randn(1, 1, self._model.hidden_dim))
                for train_chunk in train_chunks:
                    optimizer.zero_grad()
                    hidden = hidden.detach()
                    cell = cell.detach()

                    unemployment_data = train_chunk.loc[:, InputDataType.UNEMPLOYMENT.normalized_column_name].to_numpy()
                    # todo: float time sollte eine constante an geeigneter stelle sein
                    training_slice = torch.tensor(unemployment_data[:-1].copy())
                    lstm_input = training_slice.view(training_slice.shape[0], 1, -1)
                    target_slice = unemployment_data[1:]
                    targets = torch.tensor(target_slice.copy()).unsqueeze(-1)

                    return_val = self._model(lstm_input, (hidden, cell))  # type: Tuple[Tensor, Tuple[Tensor, Tensor]]
                    out, (hidden, cell) = return_val

                    loss = loss_function(out, targets)

                    loss.backward()
                    optimizer.step()
                logging.info(f"Epoch {epoch:03d}/{epochs} | loss: {loss}")
        except KeyboardInterrupt:
            logging.warning(f"Learning process interrupted by user at epoch {epoch}/{epochs}")

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
    project_dir = Path(__file__).parent.parent.parent.parent
    data_dir = project_dir / "data"
    img_dir = project_dir / "img"
    input_data = InputDataType.UNEMPLOYMENT.load_with_normalized_column(data_dir)
    trainer = UnemploymentLstmTrainer(UnemploymentLstm(64), input_data)

    trainer.run(epochs=2000, chunk_size=50, learning_rate=0.001)

    trainer.plot(img_dir / "lstm_unemployment.png")
