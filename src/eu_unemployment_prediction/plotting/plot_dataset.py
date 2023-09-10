from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from tueplots.constants.color.palettes import muted

from eu_unemployment_prediction.data_loading import (
    load_unemployment_data,
    load_dollar_euro_exchange_rate,
    load_gdp,
    load_gov_debt,
    load_inflation_rate,
    load_labour_productivity,
    load_monetary_aggregate_m3,
    load_population,
)
from eu_unemployment_prediction.input_data_type import InputDataType

sns.set_theme(style="whitegrid")

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
IMG_DIR = Path(__file__).parent.parent.parent.parent / "img"


def plot_unemployment_data() -> None:
    data = load_unemployment_data(DATA_DIR)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / InputDataType.UNEMPLOYMENT.png_name, dpi=500)
    plt.clf()


def plot_dollar_exchange_rate() -> None:
    data = load_dollar_euro_exchange_rate(DATA_DIR)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / InputDataType.DOLLAR_EURO_EXCHANGE_RATE.png_name, dpi=500)
    plt.clf()


def plot_gdp() -> None:
    data = load_gdp(DATA_DIR)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / InputDataType.GDP.png_name, dpi=500)
    plt.clf()


def plot_gov_debt() -> None:
    data = load_gov_debt(DATA_DIR)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / InputDataType.GOV_DEBT.png_name, dpi=500)
    plt.clf()


def plot_inflation_rate() -> None:
    data = load_inflation_rate(DATA_DIR)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / InputDataType.INFLATION_RATE.png_name, dpi=500)
    plt.clf()


def plot_labour_productivity() -> None:
    data = load_labour_productivity(DATA_DIR)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / InputDataType.LABOUR_PRODUCTIVITY.png_name, dpi=500)
    plt.clf()


def plot_monetary_aggregate_m3() -> None:
    data = load_monetary_aggregate_m3(DATA_DIR)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / InputDataType.MONETARY_AGGREGATE_M3.png_name, dpi=500)
    plt.clf()


def plot_population() -> None:
    data = load_population(DATA_DIR)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / InputDataType.POPULATION.png_name, dpi=500)
    plt.clf()


if __name__ == "__main__":
    plot_unemployment_data()
    plot_dollar_exchange_rate()
    plot_gdp()
    plot_population()
    plot_inflation_rate()
    plot_labour_productivity()
    plot_gov_debt()
    plot_monetary_aggregate_m3()
