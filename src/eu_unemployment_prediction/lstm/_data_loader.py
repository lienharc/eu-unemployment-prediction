from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Callable, Generator, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from eu_unemployment_prediction.input_data_type import InputDataType, DataPeriodicity


@dataclass
class DataLoader:
    """
    Attributes
    ---------
    data_dir:
        The directory in which ECB-style csv's are located
    data_types:
        The features to be loaded
    periodicity:
        The periodicity or frequency to which all data is 'normalized'.
        Default: Smalles periodicity available among data_types
    test_data_masker:
        A function which returns a numpy array specifying which data points of the input_data
        should be used for testing the trained model.
        These data points are not used for training.
        The resulting boolean array has to have the same length as there are rows in input_data.

        Example:
        >>> test_data_masker = lambda index: index > "2020-01-01"

        Default: All data points are considered for training, none for testing.
    """

    FLOAT_DATE_NAME = "float date"
    data_dir: Path
    data_types: List[InputDataType]
    periodicity: Optional[DataPeriodicity] = None
    test_data_masker: Optional[Callable[[pd.DatetimeIndex], npt.NDArray[np.bool_]]] = None
    _raw_data: Dict[InputDataType, pd.DataFrame] = field(init=False)
    _data: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        if self.periodicity is None:
            self.periodicity = min(
                data_type.periodicity for data_type in self.data_types if data_type.periodicity is not None
            )
        self._raw_data = {data_type: self._load_with_normalized_column(data_type) for data_type in self.data_types}
        self.apply_new_index(self.periodicity)

    @property
    def columns(self) -> List[str]:
        return list(self._data.columns)

    @property
    def full(self) -> pd.DataFrame:
        return self._data

    @property
    def train(self) -> pd.DataFrame:
        return self._data[self._data["type"] == "train"]

    @property
    def test(self) -> pd.DataFrame:
        return self._data[self._data["type"] == "test"]

    def apply_new_index(self, periodicity: DataPeriodicity) -> pd.DatetimeIndex:
        start_date = max(min(data_frame.index) for data_frame in self._raw_data.values())
        end_date = min(max(data_frame.index) for data_frame in self._raw_data.values())
        new_index = periodicity.date_range(start_date, end_date)
        new_index_dfs = [
            self._reindex_interpolated(data_type, new_index) for data_type, loaded_data in self._raw_data.items()
        ]
        self._data = pd.concat(new_index_dfs, axis=1)
        self._data.index.name = "date"
        self._add_float_time()
        self._add_type_labels()
        return new_index

    def chunks(self, chunk_size: int) -> Generator[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]], None, None]:
        """Generates chunks of (training_data, target_data) tuples in the right shape"""
        columns = [data_type.normalized_column_name for data_type in self.data_types]
        columns.append(self.FLOAT_DATE_NAME)
        input_data = self.train.iloc[:-1].loc[:, columns]  # type: pd.DataFrame
        target_data = self.train.iloc[1:].loc[:, columns]  # type: pd.DataFrame
        for start_index in range(0, self.train.shape[0], chunk_size):
            stop_index = start_index + chunk_size
            train_chunk = input_data.iloc[start_index:stop_index].to_numpy(dtype=np.float32)
            target_chunk = target_data.iloc[start_index:stop_index].to_numpy(dtype=np.float32)
            yield train_chunk.reshape((train_chunk.shape[0], 1, -1)), target_chunk

    def plot(self, feature: InputDataType, plot_mask: Optional[npt.NDArray[np.bool_]] = None) -> plt.Axes:
        if plot_mask is None:
            plot_mask = np.full_like(self._data.full.index, fill_value=True, dtype=np.bool_)
        sns.set_theme(style="whitegrid")
        plot_df = self._data.copy()
        return sns.scatterplot(
            plot_df.loc[plot_mask],
            x="date",
            y=feature.column_name,
            hue="type",
            palette={"train": "black", "test": "grey"},
        )

    def _load_with_normalized_column(self, feature: InputDataType) -> pd.DataFrame:
        df = feature.load_default(self.data_dir).to_frame()
        new_column = df.loc[:, feature.column_name].to_numpy(dtype=np.float32)
        df[feature.normalized_column_name] = feature.value.normalizer(new_column)
        return df

    def _reindex_interpolated(self, feature: InputDataType, new_index: pd.DatetimeIndex) -> pd.DataFrame:
        interpolated_df = self._raw_data[feature].reindex(self._raw_data[feature].index.union(new_index))
        interpolated_df = interpolated_df.interpolate(method=feature.value.interpolation_method)
        interpolated_df = interpolated_df.reindex(new_index)
        return interpolated_df

    def _add_float_time(self) -> None:
        self._data[self.FLOAT_DATE_NAME] = self._data.index.astype(int).to_numpy(dtype=np.float32) * 1.0e-19

    def _add_type_labels(self) -> None:
        self._data["type"] = "train"
        if self.test_data_masker is not None:
            mask = self.test_data_masker(self._data.index)  # type: ignore
            self._data.loc[mask, "type"] = "test"
