# -*- coding: utf-8 -*-
"""
torch_geometric/nn/aggr/basic.py  (dedicated module file)
===========================================================
Stub for ``torch_geometric.nn.aggr.basic.SumAggregation``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SumAggregation(nn.Module):
    """Sum aggregation stub (weight-receiving only)."""

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # pragma: no cover
        return x.sum(dim=0, keepdim=True) if x.dim() > 1 else x
