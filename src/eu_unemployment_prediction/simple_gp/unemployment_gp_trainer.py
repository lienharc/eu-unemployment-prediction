from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import gpytorch
import numpy.typing as npt
import pandas
import pandas as pd
import seaborn as sns
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from tueplots.constants.color.palettes import muted

from eu_unemployment_prediction.data_loading import load_unemployment_data
from eu_unemployment_prediction.simple_gp.simple_gp_unemployment_model import SimpleGpUnemploymentModel

_PROJECT_DIR = Path(__file__).parent.parent.parent.parent
_DATA_DIR = _PROJECT_DIR / "data"
_IMG_DIR = _PROJECT_DIR / "img"


class UnemploymentGpTrainer:
    def __init__(
        self, model: SimpleGpUnemploymentModel, unemployment_data: pd.Series[float], test_data_mask: npt.NDArray[bool]
    ) -> None:
        self._unemployment_data = unemployment_data
        self._test_data_mask = test_data_mask

        self._data_mean = unemployment_data.mean()  # type: float
        gp_adjusted_input_data = pandas.Series(self._unemployment_data.values - self._data_mean)  # type: ignore
        self._test_data = gp_adjusted_input_data.loc[self._test_data_mask]
        self._train_data = gp_adjusted_input_data.loc[self._test_data_mask == False]

        self._train_x = torch.tensor(self._train_data.index, dtype=torch.float32)
        self._train_y = torch.tensor(self._train_data.values, dtype=torch.float32)
        self._model = model
        self._model.train_inputs = (self._train_x.unsqueeze(-1) if self._train_x.ndimension() == 1 else self._train_x,)
        self._model.train_targets = self._train_y

        self._result_data_frame = None  # type: Optional[pd.DataFrame]

    @property
    def result_data_frame(self) -> pd.DataFrame:
        if self._result_data_frame is None:
            raise RuntimeError("Run the training first before expecting results.")
        return self._result_data_frame

    def run(self, train_steps: int = 50, learning_rate: float = 0.1) -> None:
        self._model.train()
        self._model.likelihood.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        for i in range(train_steps):
            self._execute_training_step(i, train_steps, mll, optimizer)
        self._result_data_frame = self._create_result_data_frame()

    def _execute_training_step(
        self, iteration: int, train_steps: int, mll: ExactMarginalLogLikelihood, optimizer: Optimizer
    ) -> None:
        optimizer.zero_grad()
        output = self._model(self._train_x)
        loss = -mll(output, self._train_y)
        loss.backward()
        optimizer.step()
        logging.info(
            f"Iter {iteration + 1}/{train_steps} - Loss: {loss.item():.3f}   "
            f"lengthscale: {self._model.covariance_module.base_kernel.lengthscale.item():.3f}   "
            f"noise: {self._model.likelihood.noise.item():.3f}"
        )

    def _create_result_data_frame(self) -> pd.DataFrame:
        self._model.eval()
        self._model.likelihood.eval()

        data_length = self._unemployment_data.index.shape[0]
        whole_data_set_x = torch.linspace(0, data_length, steps=data_length, dtype=torch.float32)

        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model = self._model(whole_data_set_x)
            observed_pred = self._model.likelihood(model)
            gp_mean = observed_pred.mean.numpy()
            confidence_region = observed_pred.confidence_region()
            confidence_min, confidence_max = confidence_region[0].numpy(), confidence_region[1].numpy()

        result_df = self._unemployment_data.to_frame()
        result_df["gp_mean"] = gp_mean + self._data_mean
        result_df["gp_confidence_min"] = confidence_min + self._data_mean
        result_df["gp_confidence_max"] = confidence_max + self._data_mean
        result_df["data_type"] = "test"
        result_df.loc[self._test_data_mask == False, "data_type"] = "train"

        return result_df

    def plot(self, file_path: Path) -> None:
        sns.set_theme(style="whitegrid")
        ax = sns.scatterplot(
            data=self.result_data_frame,
            x=self.result_data_frame.index,
            y="unemployment rate",
            hue="data_type",
            palette={"train": "black", "test": "grey"},
        )  # type: plt.Axes
        ax.plot(self.result_data_frame.index, self.result_data_frame["gp_mean"], f"#{muted[0]}", label="GP Mean")
        ax.fill_between(
            self.result_data_frame.index,
            self.result_data_frame["gp_confidence_min"],
            self.result_data_frame["gp_confidence_max"],
            color=f"#{muted[0]}",
            alpha=0.2,
            label="GP Confidence",
        )
        ax.legend()
        plt.savefig(file_path, dpi=500)
        plt.clf()


if __name__ == "__main__":
    # case = "interpolate"
    case = "extrapolate"

    logging.basicConfig(level=logging.INFO)
    time_series_raw = load_unemployment_data(_DATA_DIR)
    if case == "interpolate":
        data_mask = ((time_series_raw.index > "2009-05-01") & (time_series_raw.index < "2010-01-01")) | (
            (time_series_raw.index > "2021-05-01") & (time_series_raw.index < "2022-01-01")
        )
    else:
        data_mask = time_series_raw.index > "2022-10-01"
    gp_model = SimpleGpUnemploymentModel(GaussianLikelihood())
    trainer = UnemploymentGpTrainer(gp_model, time_series_raw, data_mask)
    trainer.run()
    trainer.plot(_IMG_DIR / f"gp_unemployment_prediction_{case}.png")
