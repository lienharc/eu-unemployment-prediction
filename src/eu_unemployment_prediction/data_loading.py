from pathlib import Path

import pandas as pd

from eu_unemployment_prediction.date_conversion_helper import convert_quarterly_format_to_date


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_unemployment_data(file_path: Path) -> pd.Series:  # type: ignore
    return pd.read_csv(  # type: ignore
        file_path,
        header=5,
        names=["date", "unemployment rate"],
        usecols=[0, 1],
        index_col=0,
        date_format="%Y%b",
    ).squeeze()


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_dollar_euro_exchange_rate(file_path: Path) -> pd.Series:  # type: ignore
    column_name = "exchange rate"
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
def load_gdp(file_path: Path) -> pd.Series:  # type: ignore
    column_name = "gdp at market price"
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
def load_gov_debt(file_path: Path) -> pd.Series:  # type: ignore
    column_name = "government debt"
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
def load_inflation_rate(file_path: Path) -> pd.Series:  # type: ignore
    column_name = "inflation rate"
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
def load_labour_productivity(file_path: Path) -> pd.Series:  # type: ignore
    column_name = "labour productivity"
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
def load_monetary_aggregate_m3(file_path: Path) -> pd.Series:  # type: ignore
    column_name = "m3"
    data_frame = pd.read_csv(
        file_path,
        header=5,
        names=["date", column_name],
        usecols=[0, 1],
        index_col=0,
        date_format="%Y%b",
    )
    return data_frame.squeeze()  # type: ignore
