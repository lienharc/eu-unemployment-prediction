from enum import Enum


class InputDataType(Enum):
    UNEMPLOYMENT = "unemployment_seasonadjusted"
    DOLLAR_EURO_EXCHANGE_RATE = "dollar_euro_exchange_rate"
    GDP = "gdp_at_market_price"
    GOV_DEBT = "government_debt"
    INFLATION_RATE = "inflation_rate"
    LABOUR_PRODUCTIVITY = "labour_productivity"
    MONETARY_AGGREGATE_M3 = "monetary_aggregate_m3"
    POPULATION = "population"

    @property
    def default_file_name(self) -> str:
        return self.value + ".csv"

    @property
    def png_name(self) -> str:
        return self.value + ".png"
