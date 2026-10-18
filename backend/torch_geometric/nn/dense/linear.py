# -*- coding: utf-8 -*-
"""
torch_geometric/nn/dense/linear.py
=====================================
Stub for ``torch_geometric.nn.dense.linear.Linear``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Linear(nn.Module):
    """PyTorch Geometric Linear layer stub (weight-receiving only)."""

    def __init__(
        self,
        in_channels: int = -1,
        out_channels: int = 64,
        bias: bool = True,
        weight_initializer: str = "glorot",
        **kwargs,
    ) -> None:
        super().__init__()
        if in_channels > 0:
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        else:
            self.weight = nn.Parameter(torch.empty(out_channels, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return torch.nn.functional.linear(x, self.weight,
                                          getattr(self, 'bias', None))
