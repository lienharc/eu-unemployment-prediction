import math
from pathlib import Path

import pandas as pd
from pandas import Timestamp

from eu_unemployment_prediction.data_loading import load_unemployment_data, load_dollar_euro_exchange_rate


def test_load_unemployment_data() -> None:
    input_file = Path(__file__).parent.parent.parent.parent / "data" / "unemployment_seasonadjusted.csv"

    output_df = load_unemployment_data(input_file)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-07-01T00:00")
    assert output_df[0] == 6.442155
    assert output_df.index[-1] == Timestamp.fromisoformat("2000-01-01T00:00")
    assert output_df[-1] == 9.43814
    assert output_df.shape == (283,)


def test_load_dollar_euro_exchange_rate() -> None:
    input_file = Path(__file__).parent.parent.parent.parent / "data" / "dollar_euro_exchange_rate.csv"

    output_df = load_dollar_euro_exchange_rate(input_file)

    assert type(output_df) == pd.Series
    assert output_df.index[0] == Timestamp.fromisoformat("2023-09-08T00:00")
    assert output_df[0] == 1.0704
    assert output_df.index[-1] == Timestamp.fromisoformat("1999-01-04T00:00")
    assert output_df[-1] == 1.1789
    assert math.isnan(output_df.at["2012-04-09"])
    assert output_df.shape == (6386,)
