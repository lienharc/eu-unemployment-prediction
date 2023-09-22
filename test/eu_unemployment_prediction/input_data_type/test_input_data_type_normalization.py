from pathlib import Path

import numpy as np
import pytest

from eu_unemployment_prediction.input_data_type import InputDataType


@pytest.mark.parametrize("input_type", list(InputDataType))
def test_normalization_in_range(data_dir: Path, input_type: InputDataType) -> None:
    data = input_type.load_default(data_dir)
    output = input_type.value.normalizer(data.to_numpy())
    assert np.nanmax(output) < 1.0, f"Expected all values of {type} to be less than 1.0 after normalization"
    assert np.nanmax(output) > 0.01, f"Expected max value of {type} to be greater than 0.1 after normalization"


@pytest.mark.parametrize("input_type", list(InputDataType))
def test_normalization_and_denormalization_is_identity(data_dir: Path, input_type: InputDataType) -> None:
    data = input_type.load_default(data_dir)
    input_array = data.to_numpy()
    normalized = input_type.value.normalizer(input_array)
    output_array = input_type.value.denormalizer(normalized)

    np.testing.assert_almost_equal(output_array, input_array, decimal=9)
