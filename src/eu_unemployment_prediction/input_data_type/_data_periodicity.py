from enum import Enum
from typing import Any, Optional

import pandas as pd


class DataPeriodicity(Enum):
    DAILY = (0, "D")
    MONTHLY = (2, "M")
    QUARTERLY = (3, "Q")
    YEARLY = (4, "Y")

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, DataPeriodicity):
            return self.order < other.order
        raise TypeError(f"Cannot compare DataPeriodicity to {type(other)}")

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, DataPeriodicity):
            return self.order > other.order
        raise TypeError(f"Cannot compare DataPeriodicity to {type(other)}")

    @property
    def order(self) -> int:
        return self.value[0]

    @property
    def frequency(self) -> str:
        return self.value[1]

    def date_range(self, start_time: str | pd.Timestamp, end_time: str | pd.Timestamp) -> pd.DatetimeIndex:
        return pd.date_range(start_time, end_time, freq=self.frequency)
