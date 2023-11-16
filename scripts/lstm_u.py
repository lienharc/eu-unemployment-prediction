import logging

from base_lstm import train_and_plot
from eu_unemployment_prediction.input_data_type import InputDataType

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_plot(
        [InputDataType.UNEMPLOYMENT],
        test_data_cut_off_date="2022-01-01",
        plot_zoom_date="2021-07-01",
    )
