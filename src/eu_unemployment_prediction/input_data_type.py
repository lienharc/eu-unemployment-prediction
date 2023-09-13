from enum import Enum
from typing import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd


def _not_implemented_yet(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    raise NotImplementedError("Conversion function hasn't been implemented for this input data type")


def _normalize_percentage_rate(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 100.0


class InputDataType(Enum):
    UNEMPLOYMENT = ("unemployment_seasonadjusted", "unemployment rate", _normalize_percentage_rate)
    DOLLAR_EURO_EXCHANGE_RATE = ("dollar_euro_exchange_rate", "exchange rate", _not_implemented_yet)
    GDP = ("gdp_at_market_price", "gdp at market price", _not_implemented_yet)
    GOV_DEBT = ("government_debt", "government debt", _not_implemented_yet)
    INFLATION_RATE = ("inflation_rate", "inflation rate", _not_implemented_yet)
    LABOUR_PRODUCTIVITY = ("labour_productivity", "labour productivity", _not_implemented_yet)
    MONETARY_AGGREGATE_M3 = ("monetary_aggregate_m3", "m3", _not_implemented_yet)
    POPULATION = ("population", "population", _not_implemented_yet)
    LABOUR_COSTS = ("unit_labour_costs", "labour costs", _not_implemented_yet)

    @property
    def file_base_name(self) -> str:
        return self.value[0]

    @property
    def column_name(self) -> str:
        return self.value[1]

    @property
    def normalize_function(self) -> Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]:
        return self.value[2]

    @property
    def normalized_column_name(self) -> str:
        return self.column_name + " norm"

    @property
    def default_file_name(self) -> str:
        return self.value[0] + ".csv"

    @property
    def png_name(self) -> str:
        return self.value[0] + ".png"

    def add_normalized_column(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        new_column = data_frame.loc[:, self.column_name].to_numpy(dtype=np.float32)
        data_frame[self.normalized_column_name] = self.normalize_function(new_column)
        return data_frame
