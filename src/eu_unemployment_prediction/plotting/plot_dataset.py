from pathlib import Path
import matplotlib.pyplot as plt
from tueplots.constants.color.palettes import muted
import seaborn as sns
sns.set_theme(style="whitegrid")


from eu_unemployment_prediction.data_loading import load_unemployment_data, load_dollar_euro_exchange_rate, load_gdp

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
IMG_DIR = Path(__file__).parent.parent.parent.parent / "img"


def plot_unemployment_data():
    csv_ = DATA_DIR / "unemployment_seasonadjusted.csv"
    data = load_unemployment_data(csv_)
    sns.lineplot(data=data, palette=muted, linewidth=2.5)
    plt.savefig(IMG_DIR / "unemployment_seasonadjusted.png", dpi=500)
    plt.clf()


def plot_dollar_exchange_rate():
    csv_ = DATA_DIR / "dollar_euro_exchange_rate.csv"
    data = load_dollar_euro_exchange_rate(csv_)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / "dollar_euro_exchange_rate.png", dpi=500)
    plt.clf()


def plot_gdp():
    csv_ = DATA_DIR / "gdp_at_market_price.csv"
    data = load_gdp(csv_)
    sns.lineplot(data=data, palette=muted, linewidth=1.5)
    plt.savefig(IMG_DIR / "gdp_at_market_price.png", dpi=500)
    plt.clf()
