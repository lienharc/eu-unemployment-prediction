from enum import Enum
from typing import Any

import pandas as pd


class DataPeriodicity(Enum):
    DAILY = (0, "D")
    MONTHLY = (2, "MS")
    QUARTERLY = (3, "3M")
    YEARLY = (4, "YS")

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, DataPeriodicity):
            return self.value[0] < other.value[0]
        raise TypeError(f"Cannot compare DataPeriodicity to {type(other)}")

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, DataPeriodicity):
            return self.value[0] > other.value[0]
        raise TypeError(f"Cannot compare DataPeriodicity to {type(other)}")

    def date_range(self, start_time: str | pd.Timestamp, stop_time: str | pd.Timestamp) -> pd.DatetimeIndex:
        return pd.date_range(start_time, stop_time, freq=self.value[1])
