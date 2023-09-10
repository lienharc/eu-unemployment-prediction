# Predicting unemployment in the euro zone

This project tries to predict the unemployment rate in the euro
zone based on macroeconomic data issued by ECB.

The main motivation for this project is to familiarize ourselves with machine learning on time series data.

## Data source

The source for our data sets is the [ECB's statistics portal](https://sdw.ecb.europa.eu/), exclusively.

We use a variety of macroeconomic metrics such as the unemployment data

![unemployment](img/unemployment_seasonadjusted.png)

or the GDP (at market price)

![gdp](img/gdp_at_market_price.png)

## Contributing

The repo uses black and mypy among other things.
Make sure 

### Setup

Install dev dependencies:

```shell
pip install .[dev]
```

