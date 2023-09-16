from enum import Enum
from typing import Any


class DataPeriodicity(Enum):
    DAILY = 0
    WEEKLY = 1
    MONTHLY = 2
    QUARTERLY = 3
    YEARLY = 4

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, DataPeriodicity):
            return self.value < other.value
        raise TypeError(f"Cannot compare DataPeriodicity to {type(other)}")

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, DataPeriodicity):
            return self.value > other.value
        raise TypeError(f"Cannot compare DataPeriodicity to {type(other)}")
