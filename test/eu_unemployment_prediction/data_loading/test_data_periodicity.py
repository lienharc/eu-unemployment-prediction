from eu_unemployment_prediction.input_data_type import DataPeriodicity


def test_max_enabled() -> None:
    max_periodicity = max([DataPeriodicity.YEARLY, DataPeriodicity.MONTHLY, DataPeriodicity.QUARTERLY])

    assert max_periodicity == DataPeriodicity.YEARLY


def test_min_enabled() -> None:
    max_periodicity = min([DataPeriodicity.YEARLY, DataPeriodicity.MONTHLY, DataPeriodicity.QUARTERLY])

    assert max_periodicity == DataPeriodicity.MONTHLY
