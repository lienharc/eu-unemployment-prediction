import math
from pathlib import Path

import pandas as pd
from pandas import Timestamp

from eu_unemployment_prediction.input_data_type import InputDataType

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def test_load_unemployment_data() -> None:
    output_df = InputDataType.UNEMPLOYMENT.load_default(DATA_DIR)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-07-01T00:00")
    assert output_df[0] == 6.442155
    assert output_df.index[-1] == Timestamp.fromisoformat("2000-01-01T00:00")
    assert output_df[-1] == 9.43814
    assert output_df.shape == (283,)


def test_load_dollar_euro_exchange_rate() -> None:
    output_df = InputDataType.DOLLAR_EURO_EXCHANGE_RATE.load_default(DATA_DIR)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-09-08T00:00")
    assert output_df[0] == 1.0704
    assert output_df.index[-1] == Timestamp.fromisoformat("1999-01-04T00:00")
    assert output_df[-1] == 1.1789
    assert math.isnan(output_df.at["2012-04-09"])
    assert output_df.shape == (6386,)


def test_load_gdp() -> None:
    output_df = InputDataType.GDP.load_default(DATA_DIR)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-06-30T00:00")
    assert output_df[0] == 3540248.30
    assert output_df.index[-1] == Timestamp.fromisoformat("1995-03-31T00:00")
    assert output_df[-1] == 1380494.80
    assert output_df.shape == (114,)


def test_load_government_debt() -> None:
    output_df = InputDataType.GOV_DEBT.load_default(DATA_DIR)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-03-31T00:00")
    assert output_df[0] == 91.230
    assert output_df.index[-1] == Timestamp.fromisoformat("2000-03-31T00:00")
    assert output_df[-1] == 71.460
    assert output_df.shape == (93,)


def test_load_inflation_rate() -> None:
    output_df = InputDataType.INFLATION_RATE.load_default(DATA_DIR)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-08-01T00:00")
    assert output_df[0] == 5.3
    assert output_df.index[-1] == Timestamp.fromisoformat("1997-01-01T00:00")
    assert output_df[-1] == 2.0
    assert output_df.shape == (320,)


def test_load_labour_productivity() -> None:
    output_df = InputDataType.LABOUR_PRODUCTIVITY.load_default(DATA_DIR)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-06-30T00:00")
    assert output_df[0] == -0.9
    assert output_df.index[-1] == Timestamp.fromisoformat("1996-03-31T00:00")
    assert output_df[-1] == 0.9
    assert output_df.shape == (110,)


def test_load_monetary_aggregate_m3() -> None:
    output_df = InputDataType.MONETARY_AGGREGATE_M3.load_default(DATA_DIR)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-07-01T00:00")
    assert output_df[0] == -0.4
    assert output_df.index[-1] == Timestamp.fromisoformat("1981-01-01T00:00")
    assert output_df[-1] == 10.2
    assert output_df.shape == (511,)


def test_load_population() -> None:
    output_df = InputDataType.POPULATION.load_default(DATA_DIR)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2022-01-01T00:00")
    assert output_df[0] == 348458.70
    assert output_df.index[-1] == Timestamp.fromisoformat("1995-01-01T00:00")
    assert output_df[-1] == 321215.41
    assert output_df.shape == (28,)


def test_load_labour_costs() -> None:
    output_df = InputDataType.LABOUR_COSTS.load_default(DATA_DIR)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-06-30T00:00")
    assert output_df[0] == 6.5
    assert output_df.index[-1] == Timestamp.fromisoformat("1996-03-31T00:00")
    assert output_df[-1] == 2.1
    assert output_df.shape == (110,)
