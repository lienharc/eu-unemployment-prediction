from typing import Optional, Tuple, Union, overload

import torch
from torch import nn, Tensor


class UnemploymentLstm(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        input_dim = 1
        self._hidden_dim = hidden_dim
        self._lstm = nn.LSTM(input_size=input_dim, hidden_size=self._hidden_dim)  # type: ignore
        self._output_layer = nn.Linear(in_features=self._hidden_dim, out_features=input_dim, bias=True)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

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


if __name__ == "__main__":
    my_lstm = UnemploymentLstm(hidden_dim=16)
    inputs = [torch.tensor([i / 10.0]) for i in range(9)]
    lstm_inputs = torch.cat(inputs).view(len(inputs), 1, -1)
    result2 = my_lstm(lstm_inputs)