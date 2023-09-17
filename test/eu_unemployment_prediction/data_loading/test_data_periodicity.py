import numpy as np

from eu_unemployment_prediction.input_data_type import DataPeriodicity


def test_max_enabled() -> None:
    max_periodicity = max([DataPeriodicity.YEARLY, DataPeriodicity.MONTHLY, DataPeriodicity.QUARTERLY])

    assert max_periodicity == DataPeriodicity.YEARLY


def test_min_enabled() -> None:
    max_periodicity = min([DataPeriodicity.YEARLY, DataPeriodicity.MONTHLY, DataPeriodicity.QUARTERLY])

    assert max_periodicity == DataPeriodicity.MONTHLY


def test_daily_date_range() -> None:
    expected_date_range = [f"2020-01-0{i}" for i in range(1, 10)]
    date_range = DataPeriodicity.DAILY.date_range(expected_date_range[0], expected_date_range[-1])

    np.testing.assert_equal(date_range.to_numpy(), np.array(expected_date_range, dtype=np.datetime64))


def test_monthly_date_range() -> None:
    expected_date_range = [f"2020-0{i}-01" for i in range(1, 10)]
    date_range = DataPeriodicity.MONTHLY.date_range(expected_date_range[0], expected_date_range[-1])

    np.testing.assert_equal(date_range.to_numpy(), np.array(expected_date_range, dtype=np.datetime64))


def test_quarterly_date_range() -> None:
    expected_date_range = ["2018-09-30", "2018-12-31", "2019-03-31", "2019-06-30"]
    date_range = DataPeriodicity.QUARTERLY.date_range(expected_date_range[0], expected_date_range[-1])

    np.testing.assert_equal(date_range.to_numpy(), np.array(expected_date_range, dtype=np.datetime64))


def test_yearly_date_range() -> None:
    expected_date_range = [f"200{i}-01-01" for i in range(1, 10)]
    date_range = DataPeriodicity.YEARLY.date_range(expected_date_range[0], expected_date_range[-1])

    np.testing.assert_equal(date_range.to_numpy(), np.array(expected_date_range, dtype=np.datetime64))
