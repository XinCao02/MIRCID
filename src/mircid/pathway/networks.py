from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KANLayer(nn.Module):
    """Frozen legacy KAN approximation used in the submitted code."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        self.spline_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.base_weight) + F.linear(F.silu(x), self.spline_weight)


class SimpleKAN(nn.Module):
    def __init__(self, input_dim: int, hidden: int, classes: int):
        super().__init__()
        self.layer1 = KANLayer(input_dim, hidden)
        self.layer2 = KANLayer(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.relu(out + x)


class ResNetMLP(nn.Module):
    """Legacy concat ResNet with optional light dropout."""

    def __init__(self, input_dim: int, hidden: int, classes: int, blocks: int, dropout: float):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.layers = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(blocks)])
        self.classifier = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.layers(F.relu(self.bn0(self.input_fc(x)))))


def model_from_name(name: str, input_dim: int, classes: int, config: dict) -> nn.Module:
    if name == "mlp":
        return SimpleMLP(input_dim, int(config["hidden"]), classes)
    if name == "kan":
        return SimpleKAN(input_dim, int(config["hidden"]), classes)
    if name == "resnet":
        return ResNetMLP(input_dim, int(config["hidden"]), classes, int(config["blocks"]), float(config["dropout"]))
    raise ValueError(name)

