from enum import Enum


class InputDataType(Enum):
    UNEMPLOYMENT = ("unemployment_seasonadjusted", "unemployment rate")
    DOLLAR_EURO_EXCHANGE_RATE = ("dollar_euro_exchange_rate", "exchange rate")
    GDP = ("gdp_at_market_price", "gdp at market price")
    GOV_DEBT = ("government_debt", "government debt")
    INFLATION_RATE = ("inflation_rate", "inflation rate")
    LABOUR_PRODUCTIVITY = ("labour_productivity", "labour productivity")
    MONETARY_AGGREGATE_M3 = ("monetary_aggregate_m3", "m3")
    POPULATION = ("population", "population")
    LABOUR_COSTS = ("unit_labour_costs", "labour costs")

    @property
    def file_base_name(self) -> str:
        return self.value[0]

    @property
    def column_name(self) -> str:
        return self.value[1]

    @property
    def default_file_name(self) -> str:
        return self.value[0] + ".csv"

    @property
    def png_name(self) -> str:
        return self.value[0] + ".png"
