from _ast import In
from pathlib import Path
from typing import Optional, Tuple, overload, Dict, Any, List

import torch
from torch import nn, Tensor

from eu_unemployment_prediction.input_data_type import InputDataType


class UnemploymentLstm(nn.Module):
    def __init__(self, hidden_dim: int, input_features: Optional[List[InputDataType]] = None) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._input_features = self._parse_input_features(input_features)
        self._input_dim = len(self._input_features)
        self._lstm = nn.LSTM(input_size=self._input_dim, hidden_size=self._hidden_dim)  # type: ignore
        self._output_layer = nn.Linear(in_features=self._hidden_dim, out_features=self._input_dim, bias=True)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def input_features(self) -> List[InputDataType]:
        return self._input_features

    @property
    def input_dim(self):
        return self._input_dim

    @overload
    def forward(self, x: Tensor) -> Tensor:
        ...

    @overload
    def forward(self, x: Tensor, hidden: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        ...

    def forward(
        self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tensor | Tuple[Tensor, Tuple[Tensor, Tensor]]:
        lstm_out, new_hidden = self._lstm(x, hidden)
        output_input = lstm_out.view(len(x), -1)
        result = self._output_layer(output_input)  # type: Tensor
        if hidden is None:
            return result
        return result, new_hidden

    def save(self, file_path: Path) -> None:
        base_state = {"init_vars": {"hidden_dim": self._hidden_dim, "input_features": self._input_features}}
        state_dict = self.state_dict(destination=base_state)
        torch.save(state_dict, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "UnemploymentLstm":
        state_dict = torch.load(file_path)  # type: Dict[str, Any]
        init_vars = state_dict.pop("init_vars")
        loaded_model = cls(**init_vars)
        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()
        return loaded_model

    @staticmethod
    def _parse_input_features(input_features: Optional[List[InputDataType]]) -> List[InputDataType]:
        if input_features is None:
            return [InputDataType.UNEMPLOYMENT]
        if InputDataType.UNEMPLOYMENT not in input_features:
            raise ValueError("One of the input features has to be 'UNEMPLOYMENT'.")
        return input_features


if __name__ == "__main__":
    pretrained_model_path = (
        Path(__file__).parent.parent.parent.parent / "model" / "lstm" / "unemployment_seasonadjusted_lstm.pt"
    )
    lstm = UnemploymentLstm.load(pretrained_model_path)

    inputs = [torch.tensor([i / 10.0]) for i in range(9)]
    lstm_inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    result = lstm(lstm_inputs)
