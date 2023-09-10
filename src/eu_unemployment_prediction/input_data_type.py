from enum import Enum


class InputDataType(Enum):
    UNEMPLOYMENT = "unemployment_seasonadjusted.csv"
    DOLLAR_EURO_EXCHANGE_RATE = "dollar_euro_exchange_rate.csv"
    GDP = "gdp_at_market_price.csv"
    GOV_DEBT = "government_debt.csv"
    INFLATION_RATE = "inflation_rate.csv"
    LABOUR_PRODUCTIVITY = "labour_productivity.csv"
    MONETARY_AGGREGATE_M3 = "monetary_aggregate_m3.csv"

    @property
    def default_file_name(self) -> str:
        return self.value