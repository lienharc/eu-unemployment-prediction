from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from tueplots.constants.color.palettes import muted

from eu_unemployment_prediction.data_loading import (
    load_data_named_month_index,
    load_data_valid_date,
    load_data_quarterly_index,
    load_data_yearly_index,
)
from eu_unemployment_prediction.date_conversion_helper import convert_timestamp_index_to_float

sns.set_theme(style="whitegrid")


def _not_implemented_yet(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    raise NotImplementedError("Conversion function hasn't been implemented for this input data type")


def _normalize_percentage_rate(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 100.0


@dataclass
class InputDataTypeDefinition:
    file_base_name: str
    column_name: str
    data_loader: Callable[[Path, str, str], pd.Series[float]]
    normalizer: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]] = _not_implemented_yet
    interpolation_method: str = "cubic"


class InputDataType(Enum):
    UNEMPLOYMENT = InputDataTypeDefinition(
        "unemployment_seasonadjusted",
        "unemployment rate",
        load_data_named_month_index,
        _normalize_percentage_rate,
    )
    DOLLAR_EURO_EXCHANGE_RATE = InputDataTypeDefinition(
        "dollar_euro_exchange_rate",
        "exchange rate",
        load_data_valid_date,
    )
    GDP = InputDataTypeDefinition(
        "gdp_at_market_price",
        "gdp at market price",
        load_data_quarterly_index,
    )
    GOV_DEBT = InputDataTypeDefinition(
        "government_debt",
        "government debt",
        load_data_quarterly_index,
    )
    INFLATION_RATE = InputDataTypeDefinition(
        "inflation_rate",
        "inflation rate",
        load_data_named_month_index,
    )
    LABOUR_PRODUCTIVITY = InputDataTypeDefinition(
        "labour_productivity",
        "labour productivity",
        load_data_quarterly_index,
    )
    MONETARY_AGGREGATE_M3 = (
        "monetary_aggregate_m3",
        "m3",
        load_data_named_month_index,
    )
    POPULATION = InputDataTypeDefinition(
        "population",
        "population",
        load_data_yearly_index,
    )
    LABOUR_COSTS = InputDataTypeDefinition(
        "unit_labour_costs",
        "labour costs",
        load_data_quarterly_index,
    )

    @property
    def column_name(self) -> str:
        return self.value.column_name  # type: ignore

    @property
    def normalized_column_name(self) -> str:
        return self.column_name + " norm"

    def load(self, data_dir: Path, file_name: str) -> pd.Series[float]:
        return self.value.data_loader(data_dir, file_name, self.column_name)  # type: ignore

    def load_default(self, data_dir: Path) -> pd.Series[float]:
        return self.load(data_dir, self.value.file_base_name + ".csv")

    def plot(self, data_dir: Path, img_dir: Path) -> None:
        sns.lineplot(data=self.load_default(data_dir), palette=muted, linewidth=1.5)
        plt.savefig(img_dir / (self.value.file_base_name + ".png"), dpi=500)
        plt.clf()

    def load_with_normalized_column(self, data_dir: Path) -> pd.DataFrame:
        data_frame = self.load_default(data_dir).to_frame()
        new_column = data_frame.loc[:, self.column_name].to_numpy(dtype=np.float32)
        data_frame[self.normalized_column_name] = self.value.normalizer(new_column)
        data_frame.insert(0, "float time", convert_timestamp_index_to_float(data_frame.index))
        return data_frame

    @classmethod
    def plot_all(cls, data_dir: Path, img_dir: Path) -> None:
        for input_data_type in cls:
            input_data_type.plot(data_dir, img_dir)


if __name__ == "__main__":
    project_dir = Path(__file__).parent.parent.parent
    project_data_dir = project_dir / "data"
    project_img_dir = project_dir / "img"
    InputDataType.plot_all(project_data_dir, project_img_dir)
