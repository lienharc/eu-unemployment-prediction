from pathlib import Path

import pandas as pd


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_unemployment_data(file_path: Path) -> pd.Series:  # type: ignore
    return pd.read_csv(  # type: ignore
        file_path,
        header=5,
        names=["date", "unemployment rate"],
        usecols=[0, 1],
        index_col=0,
        parse_dates=[0],
        date_format="%Y%b",
    ).squeeze()


# Type is ignored since mypy won't take pd.Series and intellij won't take pd.Series[float]
def load_dollar_euro_exchange_rate(file_path: Path) -> pd.Series:  # type: ignore
    return pd.read_csv(  # type: ignore
        file_path,
        header=5,
        names=["date", "exchange rate"],
        usecols=[0, 1],
        index_col=0,
        parse_dates=[0],
        na_values={"exchange rate": "-"},
    ).squeeze()
