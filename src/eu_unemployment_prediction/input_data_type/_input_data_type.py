from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

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
from eu_unemployment_prediction.input_data_type._data_periodicity import DataPeriodicity
from eu_unemployment_prediction.input_data_type._input_data_type_definition import InputDataTypeDefinition

__all__ = ["InputDataType", "InputDataTypeDefinition", "DataPeriodicity"]

sns.set_theme(style="whitegrid")


def _normalize_percentage_rate(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 100.0


def _denormalize_percentage_rate(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x * 100.0


def _normalize_eurostoxx(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 10000.0


def _denormalize_eurostoxx(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x * 10000.0


class InputDataType(Enum):
    UNEMPLOYMENT = InputDataTypeDefinition(
        "unemployment_seasonadjusted",
        "unemployment rate",
        load_data_named_month_index,
        DataPeriodicity.MONTHLY,
        _normalize_percentage_rate,
        _denormalize_percentage_rate,
    )
    KEY_INTEREST_RATE = InputDataTypeDefinition(
        "key_interest_rate",
        "key interest rate",
        load_data_valid_date,
        None,
        _normalize_percentage_rate,
        _denormalize_percentage_rate,
        interpolation_method="pad",
    )
    DOLLAR_EURO_EXCHANGE_RATE = InputDataTypeDefinition(
        "dollar_euro_exchange_rate",
        "exchange rate",
        load_data_valid_date,
        DataPeriodicity.DAILY,
    )
    GDP = InputDataTypeDefinition(
        "gdp_at_market_price",
        "gdp at market price",
        load_data_quarterly_index,
        DataPeriodicity.QUARTERLY,
    )
    GOV_DEBT = InputDataTypeDefinition(
        "government_debt",
        "government debt",
        load_data_quarterly_index,
        DataPeriodicity.QUARTERLY,
        lambda x: x / 1000.0,
    )
    INFLATION_RATE = InputDataTypeDefinition(
        "inflation_rate",
        "inflation rate",
        load_data_named_month_index,
        DataPeriodicity.MONTHLY,
    )
    LABOUR_PRODUCTIVITY = InputDataTypeDefinition(
        "labour_productivity",
        "labour productivity",
        load_data_quarterly_index,
        DataPeriodicity.QUARTERLY,
    )
    MONETARY_AGGREGATE_M3 = InputDataTypeDefinition(
        "monetary_aggregate_m3",
        "m3",
        load_data_named_month_index,
        DataPeriodicity.MONTHLY,
    )
    POPULATION = InputDataTypeDefinition(
        "population",
        "population",
        load_data_yearly_index,
        DataPeriodicity.YEARLY,
    )
    LABOUR_COSTS = InputDataTypeDefinition(
        "unit_labour_costs",
        "labour costs",
        load_data_quarterly_index,
        DataPeriodicity.QUARTERLY,
    )
    EURO_STOXX_50 = InputDataTypeDefinition(
        "euro_stoxx_50",
        "euro stoxx 50",
        load_data_named_month_index,
        DataPeriodicity.MONTHLY,
        _normalize_eurostoxx,
        _denormalize_eurostoxx,
    )

    @property
    def file_base_name(self) -> str:
        return self.value.file_base_name

    @property
    def column_name(self) -> str:
        return self.value.column_name

    @property
    def normalized_column_name(self) -> str:
        return self.column_name + " norm"

    @property
    def periodicity(self) -> Optional[DataPeriodicity]:
        return self.value.periodicity

    def load(self, data_dir: Path, file_name: str) -> pd.Series[float]:
        return self.value.data_loader(data_dir, file_name, self.column_name)

    def load_default(self, data_dir: Path) -> pd.Series[float]:
        return self.load(data_dir, self.value.file_base_name + ".csv")

    def plot(self, data_dir: Path, img_dir: Path) -> None:
        sns.lineplot(data=self.load_default(data_dir), palette=muted, linewidth=1.5)
        plt.savefig(img_dir / (self.value.file_base_name + ".png"), dpi=500)
        plt.clf()

    @classmethod
    def plot_all(cls, data_dir: Path, img_dir: Path) -> None:
        for input_data_type in cls:
            input_data_type.plot(data_dir, img_dir)


if __name__ == "__main__":
    project_dir = Path(__file__).parent.parent.parent.parent
    project_data_dir = project_dir / "data"
    project_img_dir = project_dir / "img"
    InputDataType.plot_all(project_data_dir, project_img_dir)
