import logging

from base_lstm import train_and_plot  # type: ignore
from eu_unemployment_prediction.input_data_type import InputDataType

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_plot(
        [InputDataType.UNEMPLOYMENT, InputDataType.EURO_STOXX_50, InputDataType.KEY_INTEREST_RATE],
        epochs=10000,
        learning_rates=[0.001, 0.0001, 0.00001],
    )
