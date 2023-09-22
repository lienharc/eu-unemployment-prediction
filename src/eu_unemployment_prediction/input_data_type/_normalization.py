"""
Normalization functions
I tried using a function creating functions with the signature
>>> def _power10_normalizer(power: int) -> Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]:
which would boil down all the below functions to just two and then use this function generator
in the InputDataType.
However, this makes problems when pickling the LSTM model.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy import typing as npt


def _normalize_max10(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 10.0


def _denormalize_max10(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x * 10.0


def _normalize_max100(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 100.0


def _denormalize_max100(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x * 100.0


def _normalize_max1e3(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 1000.0


def _denormalize_max1e3(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x * 1000.0


def _normalize_max1e5(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 1.0e5


def _denormalize_max1e5(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x * 1.0e5


def _normalize_max1e6(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 1.0e6


def _denormalize_max1e6(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x * 1.0e6


def _normalize_max1e7(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x / 1.0e7


def _denormalize_max1e7(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return x * 1.0e7
