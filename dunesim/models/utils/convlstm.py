import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self) -> None:
        super(ConvLSTMCell, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class ConvLSTM(nn.Module):
    def __init__(self) -> None:
        super(ConvLSTM, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input
