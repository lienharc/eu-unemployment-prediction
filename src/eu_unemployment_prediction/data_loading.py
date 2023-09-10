from pathlib import Path

import pandas as pd

from eu_unemployment_prediction.date_conversion_helper import convert_quarterly_format_to_date
from eu_unemployment_prediction.input_data_type import InputDataType


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_unemployment_data(
    data_dir: Path, file_name: str = InputDataType.UNEMPLOYMENT.default_file_name
) -> pd.Series:  # type: ignore
    file_path = data_dir / file_name
    return pd.read_csv(  # type: ignore
        file_path,
        header=5,
        names=["date", "unemployment rate"],
        usecols=[0, 1],
        index_col=0,
        date_format="%Y%b",
    ).squeeze()


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_dollar_euro_exchange_rate(
    data_dir: Path, file_name: str = InputDataType.DOLLAR_EURO_EXCHANGE_RATE.default_file_name
) -> pd.Series:  # type: ignore
    column_name = "exchange rate"
    file_path = data_dir / file_name
    return pd.read_csv(  # type: ignore
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
        parse_dates=[0],
        na_values={column_name: "-"},
    ).squeeze()


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_gdp(data_dir: Path, file_name: str = InputDataType.GDP.default_file_name) -> pd.Series:  # type: ignore
    column_name = "gdp at market price"
    file_path = data_dir / file_name
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
    )
    data_frame.index = data_frame.index.map(convert_quarterly_format_to_date)
    return data_frame.squeeze()  # type: ignore


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_gov_debt(
    data_dir: Path, file_name: str = InputDataType.GOV_DEBT.default_file_name
) -> pd.Series:  # type: ignore
    column_name = "government debt"
    file_path = data_dir / file_name
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
    )
    data_frame.index = data_frame.index.map(convert_quarterly_format_to_date)
    return data_frame.squeeze()  # type: ignore


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_inflation_rate(
    data_dir: Path, file_name: str = InputDataType.INFLATION_RATE.default_file_name
) -> pd.Series:  # type: ignore
    column_name = "inflation rate"
    file_path = data_dir / file_name
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
        date_format="%Y%b",
    )
    return data_frame.squeeze()  # type: ignore


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_labour_productivity(
    data_dir: Path, file_name: str = InputDataType.LABOUR_PRODUCTIVITY.default_file_name
) -> pd.Series:  # type: ignore
    column_name = "labour productivity"
    file_path = data_dir / file_name
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
    )
    data_frame.index = data_frame.index.map(convert_quarterly_format_to_date)
    return data_frame.squeeze()  # type: ignore


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_monetary_aggregate_m3(
    data_dir: Path, file_name: str = InputDataType.MONETARY_AGGREGATE_M3.default_file_name
) -> pd.Series:  # type: ignore
    column_name = "m3"
    file_path = data_dir / file_name
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
        date_format="%Y%b",
    )
    return data_frame.squeeze()  # type: ignore
