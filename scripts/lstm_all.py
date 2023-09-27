import logging

from base_lstm import train_and_plot  # type: ignore
from eu_unemployment_prediction.input_data_type import InputDataType

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler("lstm_all.log")],
    )
    train_and_plot(
        # exclude POPULATION because it's latest data point is the end of 2022
        [feature for feature in InputDataType if feature is not InputDataType.POPULATION],
        epochs=20000,
        learning_rates=[0.01, 0.001, 0.0001, 0.0001],
        test_data_cut_off_date="2022-09-15",
        plot_zoom_date="2022-01-01",
    )
