"""Configurable implementation of the HubmiRNet residual architecture."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Residual block with names matching the archived PyTorch checkpoint."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.dropout(self.relu(self.bn1(self.fc1(inputs))))
        output = self.bn2(self.fc2(output))
        return self.relu(inputs + output)


class HubmiRNet(nn.Module):
    """Map 977 L1000 genes to an inferred miRNA-expression vector."""

    def __init__(
        self,
        input_size: int = 977,
        hidden_size: int = 4096,
        output_size: int = 414,
        num_blocks: int = 3,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if not 1 <= num_blocks <= 7:
            raise ValueError("num_blocks must be between 1 and 7")
        self.num_blocks = num_blocks
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        for index in range(1, num_blocks + 1):
            setattr(self, f"layer{index}", ResidualBlock(hidden_size, dropout))
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.dropout(self.relu(self.bn(self.input_layer(inputs))))
        for index in range(1, self.num_blocks + 1):
            output = getattr(self, f"layer{index}")(output)
        return self.output_layer(output)


class LegacyResNetMLP(HubmiRNet):
    """Seven-block object layout used by the historical full-object pickle."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float) -> None:
        super().__init__(input_size, hidden_size, output_size, num_blocks=7, dropout=dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.dropout(self.relu(self.bn(self.input_layer(inputs))))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.output_layer(output)


# The original checkpoint was serialized as ``__main__.ResNetMLP``.
ResNetMLP = LegacyResNetMLP
