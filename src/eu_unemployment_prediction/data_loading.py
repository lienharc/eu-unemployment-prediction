from pathlib import Path

import pandas as pd

from eu_unemployment_prediction.date_conversion_helper import convert_quarterly_format_to_date


def _reverse_order(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.iloc[::-1]


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_data_valid_date(data_dir: Path, file_name: str, column_name: str) -> pd.Series:  # type: ignore
    file_path = data_dir / file_name
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
        parse_dates=[0],
        na_values={column_name: "-"},
    )
    return _reverse_order(data_frame).squeeze()  # type: ignore


def load_data_named_month_index(data_dir: Path, file_name: str, column_name: str) -> pd.Series:  # type: ignore
    file_path = data_dir / file_name
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
        date_format="%Y%b",
        na_values={column_name: "-"},
    )
    return _reverse_order(data_frame).squeeze()  # type: ignore


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_data_quarterly_index(data_dir: Path, file_name: str, column_name: str) -> pd.Series:  # type: ignore
    file_path = data_dir / file_name
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
        na_values={column_name: "-"},
    )
    data_frame.index = data_frame.index.map(convert_quarterly_format_to_date)
    return _reverse_order(data_frame).squeeze()  # type: ignore


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_data_yearly_index(data_dir: Path, file_name: str, column_name: str) -> pd.Series:  # type: ignore
    file_path = data_dir / file_name
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
        date_format="%Y",
        na_values={column_name: "-"},
    )
    return _reverse_order(data_frame).squeeze()  # type: ignore
